"""
Feature extraction: YOLO object detection, X-CLIP / VideoMAE actions,
Qwen2.5-VL captioning, PaddleOCR / EasyOCR.

Models are lazy-loaded per worker process. CUDA used when available.
Failures degrade to lighter fallbacks (BLIP, EasyOCR, VideoMAE).
"""
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Per-process model cache ───────────────────────────────────────────────────
_yolo = None
_vlm_model = None
_vlm_processor = None
_vlm_load_failed = False
_blip_processor = None
_blip_model = None
_xclip_processor = None
_xclip_model = None
_xclip_load_failed = False
_videomae_processor = None
_videomae_model = None
_paddle_ocr = None
_paddle_failed = False
_ocr_reader = None


def _cfg():
    from django.conf import settings
    return settings.ML


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_vlm_json(text: str) -> dict:
    """Extract JSON object from VLM output; tolerate markdown fences."""
    text = text.strip()
    fence = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start, end = text.find('{'), text.rfind('}')
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {'caption': text, 'actions': [], 'objects': [], 'on_screen_text': []}


def _as_str_list(val) -> list[str]:
    if not val:
        return []
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    out = []
    for item in val:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            s = item.get('label') or item.get('text') or item.get('name') or ''
            if str(s).strip():
                out.append(str(s).strip())
    return out


def _pil_to_bgr_array(img: Image.Image) -> np.ndarray:
    import cv2
    rgb = np.array(img.convert('RGB'))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _is_cuda_error(exc: BaseException) -> bool:
    msg = f'{type(exc).__name__} {exc}'.lower()
    return 'cuda' in msg or 'device-side assert' in msg or 'acceleratorerror' in msg


def _cuda_reset():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(_device)


def _move_inputs_to_device(inputs, device: torch.device):
    """Move BatchFeature tensors without relying on model.device."""
    for key, val in inputs.items():
        if isinstance(val, torch.Tensor):
            inputs[key] = val.to(device, non_blocking=False)
    return inputs


def _resize_for_vlm(img: Image.Image, max_side: int) -> Image.Image:
    img = img.convert('RGB')
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.Resampling.LANCZOS,
    )


def _pick_vlm_frame(frames: list[Image.Image]) -> Image.Image:
    """Use middle frame; resize to stay within VLM vision token budget."""
    cfg = _cfg()
    max_side = cfg.get('VLM_MAX_SIDE', 768)
    idx = len(frames) // 2
    return _resize_for_vlm(frames[idx], max_side)


# ── VLM (Qwen2.5-VL) ──────────────────────────────────────────────────────────

def _get_vlm():
    global _vlm_model, _vlm_processor, _vlm_load_failed
    if _vlm_load_failed:
        return None, None
    if _vlm_model is None:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

            cfg = _cfg()
            model_id = cfg['VLM_MODEL']
            # Cap vision resolution to avoid device-side asserts / OOM on 12GB GPUs
            _vlm_processor = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=256 * 28 * 28,
                max_pixels=512 * 28 * 28,
            )

            load_kwargs = {'attn_implementation': 'sdpa'}
            if cfg.get('VLM_LOAD_4BIT') and _device == 'cuda':
                load_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type='nf4',
                )
                load_kwargs['device_map'] = {'': 0}
            else:
                load_kwargs['torch_dtype'] = torch.float16 if _device == 'cuda' else torch.float32
                load_kwargs['device_map'] = 'auto' if _device == 'cuda' else None

            _vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                **load_kwargs,
            )
            _vlm_model.eval()
            logger.info('Loaded VLM %s', model_id)
        except Exception as exc:
            logger.warning('VLM load failed, will use BLIP fallback: %s', exc)
            _vlm_load_failed = True
            return None, None
    return _vlm_processor, _vlm_model


def _run_vlm_describe(processor, model, frame: Image.Image) -> dict:
    cfg = _cfg()
    prompt = (
        'Describe this video frame from a short clip. Reply with JSON only, no markdown, keys: '
        '"caption" (1-2 sentences), "actions" (array of verb phrases), '
        '"objects" (array of visible object nouns), '
        '"on_screen_text" (array of readable on-screen text).'
    )

    from qwen_vl_utils import process_vision_info

    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': frame},
            {'type': 'text', 'text': prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    )
    device = _model_device(model)
    inputs = _move_inputs_to_device(inputs, device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.get('VLM_MAX_NEW_TOKENS', 256),
            do_sample=False,
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    raw = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    data = _parse_vlm_json(raw)
    return {
        'caption': str(data.get('caption', raw)).strip(),
        'actions': _as_str_list(data.get('actions')),
        'objects': _as_str_list(data.get('objects')),
        'on_screen_text': _as_str_list(data.get('on_screen_text')),
    }


def describe(frames: list[Image.Image]) -> dict:
    """
    Structured clip description via VLM (single resized frame for stability).
    Falls back to BLIP on CPU if VLM fails or CUDA is in a bad state.
    """
    empty = {'caption': '', 'actions': [], 'objects': [], 'on_screen_text': []}
    if not frames:
        return empty

    frame = _pick_vlm_frame(frames)

    processor, model = _get_vlm()
    if processor is None or model is None:
        return _describe_blip_fallback(frame, force_cpu=False)

    try:
        return _run_vlm_describe(processor, model, frame)
    except Exception as exc:
        cuda_bad = _is_cuda_error(exc)
        logger.warning('VLM describe failed (%s), BLIP fallback: %s', cuda_bad, exc)
        if cuda_bad:
            _cuda_reset()
        return _describe_blip_fallback(frame, force_cpu=cuda_bad)


def _describe_blip_fallback(frame: Image.Image, force_cpu: bool = False) -> dict:
    try:
        caption = caption_image_from_pil(frame, force_cpu=force_cpu)
    except Exception as exc:
        logger.warning('BLIP fallback failed: %s', exc)
        if _is_cuda_error(exc):
            _cuda_reset()
        caption = ''
    return {
        'caption': caption,
        'actions': [],
        'objects': [],
        'on_screen_text': [],
    }


# ── BLIP fallback caption ─────────────────────────────────────────────────────

_blip_device = None


def _get_blip(force_cpu: bool = False):
    global _blip_processor, _blip_model, _blip_device
    dev = 'cpu' if force_cpu else _device
    if _blip_model is not None and _blip_device != dev:
        del _blip_model
        _blip_model = None
        _cuda_reset()
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        model_id = _cfg()['BLIP_MODEL']
        if _blip_processor is None:
            _blip_processor = BlipProcessor.from_pretrained(model_id)
        dtype = torch.float32 if dev == 'cpu' else torch.float16
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(dev)
        _blip_model.eval()
        _blip_device = dev
    return _blip_processor, _blip_model


def caption_image_from_pil(image: Image.Image, force_cpu: bool = False) -> str:
    processor, model = _get_blip(force_cpu=force_cpu)
    dev = _blip_device
    inputs = processor(_resize_for_vlm(image, 384), return_tensors='pt').to(dev)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=80)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def caption_image(keyframe_path: Path) -> str:
    return caption_image_from_pil(Image.open(keyframe_path).convert('RGB'))


# ── Object Detection (YOLO) ───────────────────────────────────────────────────

def _get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(_cfg()['YOLO_MODEL'])
    return _yolo


def _detect_objects_single(image: Image.Image) -> list[dict]:
    model = _get_yolo()
    arr = np.array(image.convert('RGB'))
    min_conf = _cfg()['OBJECT_MIN_CONF']
    objects = []
    for r in model(arr, verbose=False):
        for box in r.boxes:
            conf = float(box.conf)
            if conf < min_conf:
                continue
            objects.append({
                'label': r.names[int(box.cls)],
                'confidence': round(conf, 3),
                'source': 'yolo',
            })
    return objects


def detect_objects(frames_or_path) -> list[dict]:
    """
    Run YOLO on one or more frames; merge by max confidence per label.
    Accepts Path, str, single Image, or list of Images.
    """
    frames: list[Image.Image] = []
    if isinstance(frames_or_path, (Path, str)):
        frames = [Image.open(frames_or_path).convert('RGB')]
    elif isinstance(frames_or_path, Image.Image):
        frames = [frames_or_path]
    elif isinstance(frames_or_path, list):
        for item in frames_or_path:
            if isinstance(item, Image.Image):
                frames.append(item.convert('RGB'))
            elif isinstance(item, (Path, str)):
                frames.append(Image.open(item).convert('RGB'))

    if not frames:
        return []

    merged: dict[str, dict] = {}
    for img in frames:
        for obj in _detect_objects_single(img):
            key = obj['label'].lower()
            if key not in merged or obj['confidence'] > merged[key]['confidence']:
                merged[key] = obj

    return sorted(merged.values(), key=lambda x: x['confidence'], reverse=True)


# ── Action Recognition (X-CLIP primary, VideoMAE fallback) ──────────────────

def _get_xclip():
    global _xclip_processor, _xclip_model, _xclip_load_failed
    if _xclip_load_failed:
        return None, None
    if _xclip_model is None:
        try:
            from transformers import XCLIPModel, XCLIPProcessor
            model_id = _cfg()['XCLIP_MODEL']
            _xclip_processor = XCLIPProcessor.from_pretrained(model_id)
            _xclip_model = XCLIPModel.from_pretrained(model_id).to(_device)
            _xclip_model.eval()
            logger.info('Loaded X-CLIP %s', model_id)
        except Exception as exc:
            logger.warning('X-CLIP load failed: %s', exc)
            _xclip_load_failed = True
            return None, None
    return _xclip_processor, _xclip_model


def _recognize_actions_xclip(frames: list[Image.Image]) -> list[dict]:
    processor, model = _get_xclip()
    if processor is None or model is None:
        return _recognize_actions_videomae(frames)

    cfg = _cfg()
    vocab = cfg['ACTION_VOCAB']
    texts = [f'a video of a person {action}' for action in vocab]

    try:
        inputs = processor(
            text=texts,
            videos=[frames],
            return_tensors='pt',
            padding=True,
        ).to(_device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits_per_video[0]
        probs = torch.softmax(logits, dim=-1)
        k = min(cfg['ACTION_TOP_K'], len(vocab))
        min_conf = cfg['ACTION_MIN_CONF']
        top = torch.topk(probs, k)

        actions = []
        for p, idx in zip(top.values, top.indices):
            conf = float(p)
            if conf < min_conf:
                continue
            actions.append({
                'label': vocab[int(idx)],
                'confidence': round(conf, 3),
                'source': 'xclip',
            })
        return actions
    except Exception as exc:
        logger.warning('X-CLIP inference failed, VideoMAE fallback: %s', exc)
        return _recognize_actions_videomae(frames)


def _get_videomae():
    global _videomae_processor, _videomae_model
    if _videomae_model is None:
        from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
        model_id = _cfg()['VIDEOMAE_MODEL']
        _videomae_processor = VideoMAEImageProcessor.from_pretrained(model_id)
        _videomae_model = VideoMAEForVideoClassification.from_pretrained(model_id).to(_device)
        _videomae_model.eval()
    return _videomae_processor, _videomae_model


def _recognize_actions_videomae(frames: list[Image.Image]) -> list[dict]:
    if not frames:
        return []

    cfg = _cfg()
    processor, model = _get_videomae()
    inputs = processor(frames, return_tensors='pt').to(_device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    k = min(cfg['ACTION_TOP_K'], probs.shape[0])
    min_conf = cfg['ACTION_MIN_CONF']
    top = torch.topk(probs, k)

    actions = []
    for p, idx in zip(top.values, top.indices):
        conf = float(p)
        if conf < min_conf:
            continue
        actions.append({
            'label': model.config.id2label[int(idx)],
            'confidence': round(conf, 3),
            'source': 'videomae',
        })
    return actions


def recognize_actions(frames: list[Image.Image]) -> list[dict]:
    """Top-k actions with confidence; X-CLIP open-vocab or VideoMAE fallback."""
    if not frames:
        return []
    engine = _cfg().get('ACTION_ENGINE', 'xclip')
    if engine == 'videomae':
        return _recognize_actions_videomae(frames)
    return _recognize_actions_xclip(frames)


def recognize_action(frames: list[Image.Image]) -> dict:
    actions = recognize_actions(frames)
    if not actions:
        return {'label': '', 'confidence': 0.0}
    return actions[0]


# ── OCR (Paddle primary, EasyOCR fallback) ────────────────────────────────────

def _get_paddle_ocr():
    global _paddle_ocr, _paddle_failed
    if _paddle_failed:
        return None
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            use_gpu = _device == 'cuda'
            _paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=use_gpu,
                show_log=False,
            )
            logger.info('Loaded PaddleOCR (gpu=%s)', use_gpu)
        except Exception as exc:
            logger.warning('PaddleOCR load failed: %s', exc)
            _paddle_failed = True
            return None
    return _paddle_ocr


def _get_easyocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=(_device == 'cuda'), verbose=False)
    return _ocr_reader


def _ocr_paddle_single(image: Image.Image) -> list[dict]:
    ocr = _get_paddle_ocr()
    if ocr is None:
        return []
    min_conf = _cfg()['OCR_MIN_CONF']
    bgr = _pil_to_bgr_array(image)
    try:
        result = ocr.ocr(bgr, cls=True)
    except TypeError:
        result = ocr.ocr(bgr)

    blocks = []
    if not result:
        return blocks
    lines = result[0] if result and isinstance(result[0], list) else result
    for line in lines or []:
        if not line or len(line) < 2:
            continue
        text_info = line[1]
        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
            text, conf = text_info[0], text_info[1]
        else:
            continue
        text = str(text).strip()
        if not text or float(conf) < min_conf:
            continue
        blocks.append({'text': text, 'confidence': round(float(conf), 3), 'source': 'paddle'})
    return blocks


def _ocr_easy_single(image: Image.Image) -> list[dict]:
    reader = _get_easyocr()
    min_conf = _cfg()['OCR_MIN_CONF']
    import tempfile
    import os

    blocks = []
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        image.save(tmp.name, 'JPEG')
        path = tmp.name
    try:
        out = reader.readtext(path, detail=1, paragraph=False)
    finally:
        os.unlink(path)

    for _box, text, conf in out:
        text = text.strip()
        if not text or float(conf) < min_conf:
            continue
        blocks.append({'text': text, 'confidence': round(float(conf), 3), 'source': 'easyocr'})
    return blocks


def _ocr_single(image: Image.Image) -> list[dict]:
    engine = _cfg().get('OCR_ENGINE', 'paddle')
    if engine == 'paddle':
        blocks = _ocr_paddle_single(image)
        if blocks or not _paddle_failed:
            return blocks
    return _ocr_easy_single(image)


def ocr_blocks(frames_or_path) -> list[dict]:
    """OCR with per-block confidence; multi-frame merge by max confidence per text."""
    frames: list[Image.Image] = []
    if isinstance(frames_or_path, (Path, str)):
        frames = [Image.open(frames_or_path).convert('RGB')]
    elif isinstance(frames_or_path, Image.Image):
        frames = [frames_or_path]
    elif isinstance(frames_or_path, list):
        for item in frames_or_path:
            if isinstance(item, Image.Image):
                frames.append(item.convert('RGB'))
            elif isinstance(item, (Path, str)):
                frames.append(Image.open(item).convert('RGB'))

    merged: dict[str, dict] = {}
    for img in frames:
        for block in _ocr_single(img):
            key = block['text'].lower()
            if key not in merged or block['confidence'] > merged[key]['confidence']:
                merged[key] = block

    return sorted(merged.values(), key=lambda x: x['confidence'], reverse=True)


def ocr_image(keyframe_path: Path) -> str:
    return ' '.join(b['text'] for b in ocr_blocks(keyframe_path)).strip()
