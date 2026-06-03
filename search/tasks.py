from celery import shared_task
from django.conf import settings
from django.utils import timezone


def _merge_label_tags(
    detected: list[dict],
    vlm_phrases: list,
    vlm_conf: float,
) -> list[dict]:
    """Union model tags with VLM phrases; keep max confidence per label."""
    by_key: dict[str, dict] = {}

    for item in detected or []:
        label = str(item.get('label', '')).strip()
        if not label:
            continue
        key = label.lower()
        conf = float(item.get('confidence', 0))
        entry = {
            'label': label,
            'confidence': round(conf, 3),
            'source': item.get('source', 'model'),
        }
        if key not in by_key or conf > by_key[key]['confidence']:
            by_key[key] = entry

    for phrase in vlm_phrases or []:
        if isinstance(phrase, str):
            label = phrase.strip()
            conf = vlm_conf
            source = 'vlm'
        elif isinstance(phrase, dict):
            label = str(phrase.get('label', phrase.get('text', ''))).strip()
            conf = float(phrase.get('confidence', vlm_conf))
            source = phrase.get('source', 'vlm')
        else:
            continue
        if not label:
            continue
        key = label.lower()
        entry = {'label': label, 'confidence': round(conf, 3), 'source': source}
        if key not in by_key or conf > by_key[key]['confidence']:
            by_key[key] = entry

    return sorted(by_key.values(), key=lambda x: x['confidence'], reverse=True)


def _merge_ocr_blocks(detected: list[dict], vlm_texts: list, vlm_conf: float) -> list[dict]:
    by_key: dict[str, dict] = {}

    for block in detected or []:
        text = str(block.get('text', '')).strip()
        if not text:
            continue
        key = text.lower()
        conf = float(block.get('confidence', 0))
        entry = {
            'text': text,
            'confidence': round(conf, 3),
            'source': block.get('source', 'ocr'),
        }
        if key not in by_key or conf > by_key[key]['confidence']:
            by_key[key] = entry

    for text in vlm_texts or []:
        text = str(text).strip()
        if not text:
            continue
        key = text.lower()
        entry = {'text': text, 'confidence': round(vlm_conf, 3), 'source': 'vlm'}
        if key not in by_key or vlm_conf > by_key[key]['confidence']:
            by_key[key] = entry

    return sorted(by_key.values(), key=lambda x: x['confidence'], reverse=True)


@shared_task(bind=True)
def index_video(self, video_id: int):
    from .models import Video, Clip, Keyframe, IndexingJob
    from .pipeline import segmenter, extractor, embedder

    job = IndexingJob.objects.get(video_id=video_id)
    video = Video.objects.get(id=video_id)
    ml = settings.ML
    vlm_conf = ml.get('VLM_TAG_CONF', 0.6)

    def update(pct: int, step: str):
        job.progress_pct = pct
        job.current_step = step
        job.save(update_fields=['progress_pct', 'current_step'])

    try:
        job.status = 'running'
        job.save(update_fields=['status'])
        video.status = 'processing'
        video.save(update_fields=['status'])

        video_path = settings.VIDEOS_DIR / video.filename

        update(2, 'detecting scenes')
        video.duration_sec = segmenter.get_video_duration(video_path)
        video.save(update_fields=['duration_sec'])

        scenes = segmenter.get_scenes(video_path, threshold=ml['SCENE_THRESHOLD'])
        total = len(scenes)

        old_clips = Clip.objects.filter(video=video)
        for c in old_clips:
            embedder.delete_clip(c.chroma_id)
        old_clips.delete()

        for i, scene in enumerate(scenes):
            base_pct = int(5 + (i / total) * 90)
            update(base_pct, f'clip {i + 1}/{total} — extracting keyframe')

            start_sec = scene['start_sec']
            end_sec = scene['end_sec']
            mid_sec = (start_sec + end_sec) / 2.0

            kf_dir = settings.KEYFRAMES_DIR / str(video.id)
            kf_path = kf_dir / f'clip_{i:04d}.jpg'
            ok = segmenter.extract_keyframe(video_path, mid_sec, kf_path)
            if not ok:
                continue

            update(base_pct, f'clip {i + 1}/{total} — sampling frames')
            n_sample = max(
                ml.get('XCLIP_FRAMES', 8),
                ml.get('VLM_NUM_FRAMES', 3),
                ml.get('OBJECT_FRAMES', 3),
                ml.get('OCR_FRAMES', 3),
            )
            frames = segmenter.sample_clip_frames(
                video_path, start_sec, end_sec, n=n_sample
            )
            if not frames:
                continue

            vlm_frames = segmenter.pick_frames(frames, ml.get('VLM_NUM_FRAMES', 3))
            obj_frames = segmenter.pick_frames(frames, ml.get('OBJECT_FRAMES', 3))
            ocr_frames = segmenter.pick_frames(frames, ml.get('OCR_FRAMES', 3))

            update(base_pct, f'clip {i + 1}/{total} — VLM describe')
            try:
                vlm = extractor.describe(vlm_frames)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).error(
                    'describe() failed: %s — restart Celery worker if CUDA errors persist',
                    exc,
                )
                vlm = {
                    'caption': '',
                    'actions': [],
                    'objects': [],
                    'on_screen_text': [],
                }

            update(base_pct, f'clip {i + 1}/{total} — object detection')
            objects = extractor.detect_objects(obj_frames)
            objects = _merge_label_tags(objects, vlm.get('objects', []), vlm_conf)

            update(base_pct, f'clip {i + 1}/{total} — action recognition')
            xclip_frames = segmenter.pick_frames(frames, ml.get('XCLIP_FRAMES', 8))
            actions = extractor.recognize_actions(xclip_frames)
            actions = _merge_label_tags(actions, vlm.get('actions', []), vlm_conf)

            update(base_pct, f'clip {i + 1}/{total} — OCR')
            ocr_list = extractor.ocr_blocks(ocr_frames)
            ocr_list = _merge_ocr_blocks(ocr_list, vlm.get('on_screen_text', []), vlm_conf)
            ocr_text = ' '.join(b['text'] for b in ocr_list)

            caption = vlm.get('caption', '') or extractor.caption_image(kf_path)
            update(base_pct, f'clip {i + 1}/{total} — caption confidence')
            caption_confidence = embedder.clip_image_text_similarity(kf_path, caption)

            clip = Clip.objects.create(
                video=video,
                clip_index=i,
                start_sec=start_sec,
                end_sec=end_sec,
                caption=caption,
                caption_confidence=caption_confidence,
                ocr_text=ocr_text,
                ocr_blocks=ocr_list,
                action_label=actions[0]['label'] if actions else '',
                action_confidence=actions[0]['confidence'] if actions else None,
                actions_detected=actions,
                objects_detected=objects,
            )

            kf_rel = kf_path.relative_to(settings.DATA_DIR)
            Keyframe.objects.create(
                clip=clip,
                file_path=str(kf_rel),
                timestamp_sec=mid_sec,
            )

            obj_labels = ' '.join(o['label'] for o in objects)
            action_labels = ' '.join(a['label'] for a in actions)
            text_blob = ' '.join(filter(None, [caption, obj_labels, action_labels, ocr_text]))

            update(base_pct, f'clip {i + 1}/{total} — embedding')
            embedder.index_clip(
                clip_id=clip.chroma_id,
                text_blob=text_blob,
                keyframe_path=kf_path,
                metadata={
                    'video_id': str(video.id),
                    'video_filename': video.filename,
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                },
            )

        video.status = 'indexed'
        video.indexed_at = timezone.now()
        video.save(update_fields=['status', 'indexed_at'])

        job.status = 'done'
        job.progress_pct = 100
        job.current_step = f'done — {total} clips indexed'
        job.completed_at = timezone.now()
        job.save()

    except Exception as exc:
        video.status = 'error'
        video.save(update_fields=['status'])
        job.status = 'error'
        job.error_msg = str(exc)
        job.save(update_fields=['status', 'error_msg'])
        raise
