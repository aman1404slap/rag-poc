import re
from pathlib import Path

from django.conf import settings
from django.db.models import Count, Sum
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from .models import Video, Clip, IndexingJob
from .pipeline import embedder


def index(request):
    return render(request, 'search/index.html')


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokens, min length 3."""
    return {t for t in re.findall(r'[a-z0-9]{3,}', text.lower())}


def _matched_tag_confidence(clip: Clip, query_tokens: set[str]) -> float:
    """Max detection confidence among tags whose text overlaps query tokens."""
    if not query_tokens:
        return 0.0

    best = 0.0

    for action in clip.actions_detected or []:
        if _tokenize(action.get('label', '')) & query_tokens:
            best = max(best, float(action.get('confidence', 0)))

    # Legacy single action field
    if clip.action_label and _tokenize(clip.action_label) & query_tokens:
        if clip.action_confidence is not None:
            best = max(best, float(clip.action_confidence))

    for obj in clip.objects_detected or []:
        if _tokenize(obj.get('label', '')) & query_tokens:
            best = max(best, float(obj.get('confidence', 0)))

    for block in clip.ocr_blocks or []:
        if _tokenize(block.get('text', '')) & query_tokens:
            best = max(best, float(block.get('confidence', 0)))

    # Legacy OCR string
    if clip.ocr_text and _tokenize(clip.ocr_text) & query_tokens:
        best = max(best, 0.5)

    if clip.caption and _tokenize(clip.caption) & query_tokens:
        if clip.caption_confidence is not None:
            best = max(best, float(clip.caption_confidence))

    return round(best, 4)


def _overall_score(semantic: float, tag_match: float) -> float:
    boost = settings.ML['TAG_BOOST_WEIGHT']
    overall = semantic + (1.0 - semantic) * tag_match * boost
    return round(min(1.0, max(0.0, overall)), 4)


def _serialize_video(v):
    job = getattr(v, 'job', None)
    return {
        'id': v.id,
        'filename': v.filename,
        'status': v.status,
        'clip_count': v.clips.count(),
        'duration_sec': v.duration_sec,
        'indexed_at': v.indexed_at.isoformat() if v.indexed_at else None,
        'created_at': v.created_at.isoformat() if v.created_at else None,
        'progress_pct': job.progress_pct if job else 0,
        'current_step': job.current_step if job else '',
        'task_id': job.celery_task_id if job else '',
    }


@require_GET
def api_search(request):
    query = request.GET.get('q', '').strip()
    try:
        n_results = min(int(request.GET.get('n', 50)), 100)
    except (TypeError, ValueError):
        n_results = 50

    if not query:
        return JsonResponse({'results': [], 'total': 0})

    try:
        hits = embedder.search(query, n_results=n_results)
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)

    query_tokens = _tokenize(query)
    results = []
    for hit in hits:
        try:
            clip = (
                Clip.objects.select_related('video')
                .prefetch_related('keyframes')
                .get(id=int(hit['clip_id']))
            )
        except Clip.DoesNotExist:
            continue

        semantic = hit['semantic']
        tag_match = _matched_tag_confidence(clip, query_tokens)
        overall = _overall_score(semantic, tag_match)

        actions = clip.actions_detected or []
        if not actions and clip.action_label:
            actions = [{
                'label': clip.action_label,
                'confidence': clip.action_confidence or 0,
            }]

        ocr = clip.ocr_blocks or []
        if not ocr and clip.ocr_text:
            ocr = [{'text': clip.ocr_text, 'confidence': 0.5}]

        kf = clip.keyframes.first()
        results.append({
            'clip_id': clip.id,
            'video_id': clip.video.id,
            'video_filename': clip.video.filename,
            'video_url': clip.video.video_url,
            'start_sec': clip.start_sec,
            'end_sec': clip.end_sec,
            'score': overall,
            'semantic': semantic,
            'text_sim': hit['text_sim'],
            'visual_sim': hit['visual_sim'],
            'tag_match': tag_match,
            'caption': clip.caption,
            'caption_confidence': clip.caption_confidence,
            'ocr_text': clip.ocr_text,
            'ocr': ocr,
            'action_label': clip.action_label,
            'action_confidence': clip.action_confidence,
            'actions': actions,
            'objects': clip.objects_detected or [],
            'keyframe_url': kf.url if kf else '',
        })

    results.sort(key=lambda r: r['score'], reverse=True)
    return JsonResponse({'results': results, 'total': len(results)})


@require_POST
def api_upload(request):
    video_file = request.FILES.get('video')
    if not video_file:
        return JsonResponse({'error': 'No video file provided'}, status=400)

    # Deduplicate filename
    videos_dir = settings.VIDEOS_DIR
    stem = Path(video_file.name).stem
    suffix = Path(video_file.name).suffix.lower()
    filename = video_file.name
    counter = 1
    while (videos_dir / filename).exists():
        filename = f'{stem}_{counter}{suffix}'
        counter += 1

    dest = videos_dir / filename
    with open(dest, 'wb') as f:
        for chunk in video_file.chunks():
            f.write(chunk)

    video = Video.objects.create(filename=filename, status='pending')
    job = IndexingJob.objects.create(video=video)

    from .tasks import index_video
    task = index_video.delay(video.id)

    job.celery_task_id = task.id
    job.status = 'queued'
    job.save(update_fields=['celery_task_id', 'status'])

    video.status = 'processing'
    video.save(update_fields=['status'])

    return JsonResponse({'task_id': task.id, 'video_id': video.id})


@require_GET
def api_job_status(request, task_id):
    try:
        job = IndexingJob.objects.select_related('video').get(celery_task_id=task_id)
    except IndexingJob.DoesNotExist:
        return JsonResponse({'error': 'Job not found'}, status=404)

    return JsonResponse({
        'status': job.status,
        'progress_pct': job.progress_pct,
        'current_step': job.current_step,
        'error_msg': job.error_msg,
        'video_filename': job.video.filename,
    })


@require_GET
def api_videos(request):
    try:
        page = max(int(request.GET.get('page', 1)), 1)
    except (TypeError, ValueError):
        page = 1
    try:
        page_size = min(max(int(request.GET.get('page_size', 10)), 1), 50)
    except (TypeError, ValueError):
        page_size = 10

    status = request.GET.get('status', '').strip()
    q = request.GET.get('q', '').strip()
    sort = request.GET.get('sort', '-created_at').strip()

    allowed_sorts = {
        'filename', '-filename',
        'status', '-status',
        'created_at', '-created_at',
        'indexed_at', '-indexed_at',
        'duration_sec', '-duration_sec',
    }
    if sort not in allowed_sorts:
        sort = '-created_at'

    qs = Video.objects.prefetch_related('clips', 'job').order_by(sort)
    if status:
        qs = qs.filter(status=status)
    if q:
        qs = qs.filter(filename__icontains=q)

    total = qs.count()
    start = (page - 1) * page_size
    page_qs = qs[start:start + page_size]

    data = [_serialize_video(v) for v in page_qs]
    pages = (total + page_size - 1) // page_size if total else 0

    return JsonResponse({
        'videos': data,
        'total': total,
        'page': page,
        'page_size': page_size,
        'pages': pages,
    })


@require_GET
def api_stats(request):
    status_counts = {
        row['status']: row['n']
        for row in Video.objects.values('status').annotate(n=Count('id'))
    }
    total_duration = Video.objects.aggregate(s=Sum('duration_sec'))['s'] or 0

    return JsonResponse({
        'videos_total': Video.objects.count(),
        'videos_indexed': status_counts.get('indexed', 0),
        'videos_processing': (
            status_counts.get('processing', 0) + status_counts.get('pending', 0)
        ),
        'videos_error': status_counts.get('error', 0),
        'clips_total': Clip.objects.count(),
        'duration_sec': total_duration,
    })
