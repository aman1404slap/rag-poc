"""
Video segmentation and keyframe extraction.
Uses PySceneDetect for scene boundaries, ffmpeg for keyframes and frame sampling.
"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)['format']['duration'])


def get_scenes(video_path: Path, threshold: float = 30.0) -> list[dict]:
    """
    Returns list of {'start_sec', 'end_sec'} dicts.
    Falls back to fixed windows if PySceneDetect finds fewer than 3 scenes.
    """
    from django.conf import settings
    cfg = settings.ML

    try:
        from scenedetect import detect, ContentDetector
        raw = detect(str(video_path), ContentDetector(threshold=threshold))
        scenes = [
            {'start_sec': s[0].get_seconds(), 'end_sec': s[1].get_seconds()}
            for s in raw
            if s[1].get_seconds() - s[0].get_seconds() >= cfg['MIN_CLIP_SEC']
        ]
        if len(scenes) >= 3:
            return scenes
    except Exception:
        pass

    # Fixed-window fallback
    duration = get_video_duration(video_path)
    window = cfg['FALLBACK_WINDOW_SEC']
    scenes = []
    t = 0.0
    while t < duration:
        end = min(t + window, duration)
        if end - t >= cfg['MIN_CLIP_SEC']:
            scenes.append({'start_sec': t, 'end_sec': end})
        t += window
    return scenes


def extract_keyframe(video_path: Path, timestamp_sec: float, output_path: Path) -> bool:
    """Extract a single frame at timestamp_sec to output_path. Returns True on success."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            'ffmpeg', '-y',
            '-ss', f'{timestamp_sec:.3f}',
            '-i', str(video_path),
            '-vframes', '1',
            '-q:v', '2',
            str(output_path),
        ],
        capture_output=True,
    )
    return result.returncode == 0 and output_path.exists()


def sample_clip_frames(video_path: Path, start_sec: float, end_sec: float, n: int = 16) -> list[Image.Image]:
    """
    Sample n frames uniformly from [start_sec, end_sec] for VideoMAE.
    Returns list of PIL Images in RGB.
    """
    duration = max(end_sec - start_sec, 0.1)
    timestamps = [start_sec + i * duration / max(n - 1, 1) for i in range(n)]

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    # Pad with last frame if we didn't get enough
    if frames and len(frames) < n:
        frames += [frames[-1]] * (n - len(frames))

    return frames[:n]


def pick_frames(frames: list[Image.Image], n: int) -> list[Image.Image]:
    """Pick n evenly spaced frames from a list (includes first and last when n > 1)."""
    if not frames:
        return []
    if len(frames) <= n:
        return list(frames)
    indices = [int(i * (len(frames) - 1) / max(n - 1, 1)) for i in range(n)]
    return [frames[i] for i in indices]
