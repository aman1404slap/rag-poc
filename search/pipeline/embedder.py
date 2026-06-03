# WSL2 / older Linux ships SQLite < 3.35.0 which ChromaDB rejects.
# pysqlite3-binary bundles a modern SQLite; swap it in before chromadb loads.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # pysqlite3-binary not installed, hope system sqlite3 is new enough

"""
Embedding and vector search.

Two ChromaDB collections (cosine similarity):
  - text_clips:   sentence-transformer embedding of caption + objects + action + OCR
  - visual_clips: CLIP visual embedding of keyframe image

Search uses absolute calibrated cosine similarity over text + visual channels.
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image

_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Model cache ───────────────────────────────────────────────────────────────
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_text_model = None
_chroma_client = None
_text_col = None
_visual_col = None


def _get_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        import open_clip
        from django.conf import settings
        cfg = settings.ML
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            cfg['CLIP_MODEL'], pretrained=cfg['CLIP_PRETRAINED']
        )
        _clip_tokenizer = open_clip.get_tokenizer(cfg['CLIP_MODEL'])
        _clip_model = _clip_model.to(_device).eval()
    return _clip_model, _clip_preprocess, _clip_tokenizer


def _get_text_model():
    global _text_model
    if _text_model is None:
        from sentence_transformers import SentenceTransformer
        from django.conf import settings
        _text_model = SentenceTransformer(settings.ML['TEXT_EMB_MODEL'], device=_device)
    return _text_model


def _get_chroma():
    global _chroma_client, _text_col, _visual_col
    if _chroma_client is None:
        import chromadb
        from django.conf import settings
        _chroma_client = chromadb.PersistentClient(path=str(settings.INDEX_DIR))
        _text_col = _chroma_client.get_or_create_collection(
            'text_clips', metadata={'hnsw:space': 'cosine'}
        )
        _visual_col = _chroma_client.get_or_create_collection(
            'visual_clips', metadata={'hnsw:space': 'cosine'}
        )
    return _text_col, _visual_col


# ── Embedding helpers ─────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    model = _get_text_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_image_clip(image_path: Path) -> list[float]:
    model, preprocess, _ = _get_clip()
    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(_device)
    with torch.no_grad():
        feat = model.encode_image(image)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().numpy().tolist()


def embed_text_clip(text: str) -> list[float]:
    """CLIP text encoder — used at query time to match against visual embeddings."""
    model, _, tokenizer = _get_clip()
    tokens = tokenizer([text]).to(_device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().numpy().tolist()


def clip_image_text_similarity(image_path: Path, text: str) -> float:
    """Cosine similarity between CLIP image and text embeddings (both L2-normalized)."""
    if not text or not text.strip():
        return 0.0
    img_vec = np.array(embed_image_clip(image_path), dtype=np.float32)
    txt_vec = np.array(embed_text_clip(text), dtype=np.float32)
    return round(float(np.dot(img_vec, txt_vec)), 4)


# ── Index (write) ─────────────────────────────────────────────────────────────

def index_clip(
    clip_id: str,
    text_blob: str,
    keyframe_path: Path,
    metadata: dict,
):
    """Store text + visual embeddings for one clip in ChromaDB."""
    text_col, visual_col = _get_chroma()

    text_vec = embed_text(text_blob)
    visual_vec = embed_image_clip(keyframe_path)

    text_col.upsert(ids=[clip_id], embeddings=[text_vec], metadatas=[metadata])
    visual_col.upsert(ids=[clip_id], embeddings=[visual_vec], metadatas=[metadata])


def delete_clip(clip_id: str):
    """Remove a clip from both collections (used when re-indexing)."""
    text_col, visual_col = _get_chroma()
    for col in (text_col, visual_col):
        try:
            col.delete(ids=[clip_id])
        except Exception:
            pass


# ── Search ────────────────────────────────────────────────────────────────────

def _calibrate(sim: float, lo: float, hi: float) -> float:
    """Map raw cosine similarity to [0, 1] using configurable bounds."""
    if hi <= lo:
        return max(0.0, min(1.0, sim))
    return max(0.0, min(1.0, (sim - lo) / (hi - lo)))


def _distance_to_sim(distance: float) -> float:
    """Chroma cosine distance -> similarity in [0, 1]."""
    return max(0.0, min(1.0, 1.0 - distance))


def search(query: str, n_results: int = 10) -> list[dict]:
    """
    Returns top-n clips by absolute calibrated semantic similarity.
    Each item: {clip_id, semantic, text_sim, visual_sim}
    """
    from django.conf import settings

    cfg = settings.ML
    text_lo, text_hi = cfg['TEXT_SIM_CAL']
    vis_lo, vis_hi = cfg['VISUAL_SIM_CAL']
    w = cfg['SCORE_WEIGHTS']

    text_col, visual_col = _get_chroma()
    fetch = n_results

    text_vec = embed_text(query)
    visual_vec = embed_text_clip(query)

    text_res = text_col.query(
        query_embeddings=[text_vec], n_results=fetch, include=['distances']
    )
    visual_res = visual_col.query(
        query_embeddings=[visual_vec], n_results=fetch, include=['distances']
    )

    candidates: dict[str, dict[str, float]] = {}

    if text_res['ids'] and text_res['ids'][0]:
        for cid, dist in zip(text_res['ids'][0], text_res['distances'][0]):
            candidates.setdefault(cid, {'text_sim': 0.0, 'visual_sim': 0.0})
            candidates[cid]['text_sim'] = _distance_to_sim(dist)

    if visual_res['ids'] and visual_res['ids'][0]:
        for cid, dist in zip(visual_res['ids'][0], visual_res['distances'][0]):
            candidates.setdefault(cid, {'text_sim': 0.0, 'visual_sim': 0.0})
            candidates[cid]['visual_sim'] = _distance_to_sim(dist)

    out = []
    for cid, ch in candidates.items():
        t_cal = _calibrate(ch['text_sim'], text_lo, text_hi)
        v_cal = _calibrate(ch['visual_sim'], vis_lo, vis_hi)
        semantic = w['text'] * t_cal + w['visual'] * v_cal
        out.append({
            'clip_id': cid,
            'semantic': round(semantic, 4),
            'text_sim': round(t_cal, 4),
            'visual_sim': round(v_cal, 4),
        })

    out.sort(key=lambda x: x['semantic'], reverse=True)
    return out[:n_results]
