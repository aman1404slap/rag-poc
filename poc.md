# Video RAG POC — Architecture Overview

## Stack

| Layer | Tech |
|---|---|
| Web / API | Django (REST JSON views) |
| Async workers | Celery |
| Metadata store | SQLite (`data/db.sqlite3`) |
| Vector store | ChromaDB (`data/index/`, HNSW, cosine) |
| Keyframes | JPEG files (`data/keyframes/`) |

---

## Indexing Pipeline

Triggered on upload (or via `manage.py index_videos`). Each video runs as a Celery task.

```
Video file
  └─ Scene detection (PySceneDetect ContentDetector → fixed-window fallback)
       └─ Per clip:
            ├─ Extract keyframe (ffmpeg, midpoint)
            ├─ Sample N frames uniformly across clip
            ├─ VLM describe (Qwen2.5-VL 4-bit) → caption, actions, objects, on-screen text
            │     └─ fallback: BLIP
            ├─ Object detection (YOLOv8) → merged with VLM objects
            ├─ Action recognition (X-CLIP open-vocab) → merged with VLM actions
            │     └─ fallback: VideoMAE
            ├─ OCR (PaddleOCR) → merged with VLM on-screen text
            │     └─ fallback: EasyOCR
            ├─ Caption confidence = CLIP image↔text cosine similarity
            ├─ Save to SQLite (Video / Clip / Keyframe models)
            └─ Embed + index in ChromaDB (two collections):
                 ├─ text_clips  → sentence-transformer embedding of
                 │                 caption + objects + actions + OCR
                 └─ visual_clips → OpenCLIP image embedding of keyframe
```

---

## Search

```
Query string
  ├─ sentence-transformer embed → query text_clips collection
  └─ CLIP text encode          → query visual_clips collection

For each candidate clip:
  text_sim   = 1 - chroma_cosine_distance  (calibrated to [0,1])
  visual_sim = 1 - chroma_cosine_distance  (calibrated to [0,1])
  semantic   = w_text * text_sim + w_visual * visual_sim

  tag_match  = max detection confidence of tags whose text overlaps query tokens
               (actions, objects, OCR blocks, caption)

  final_score = semantic + (1 - semantic) * tag_match * TAG_BOOST_WEIGHT

Results sorted by final_score descending.
```

---

## Models Used

| Purpose | Primary | Fallback |
|---|---|---|
| VLM captioning | Qwen2.5-VL (4-bit int4) | BLIP |
| Object detection | YOLOv8 | — |
| Action recognition | X-CLIP (open-vocab) | VideoMAE (Kinetics labels) |
| OCR | PaddleOCR | EasyOCR |
| Text embedding | sentence-transformers (`all-MiniLM-L6-v2`) | — |
| Visual embedding | OpenCLIP (`ViT-B-32`) | — |

---

## Key Design Decisions

- **Dual-channel retrieval**: text (semantic description) and visual (keyframe appearance) are indexed and queried separately, then blended. This lets a query like "person running" hit both the caption and the visual embedding.
- **VLM as the spine**: Qwen2.5-VL produces structured JSON (caption + tags) from a single resized frame per clip. Dedicated models (YOLO, X-CLIP, PaddleOCR) run alongside and their outputs are merged by max-confidence per label — VLM provides context, specialized models provide precision.
- **Tag-match boost**: keyword overlap between query and stored tags (weighted by detection confidence) is used as a non-semantic re-rank signal on top of vector similarity, improving recall for exact noun/action queries.
- **Fallback chain everywhere**: every expensive model (VLM, X-CLIP, PaddleOCR) has a lighter fallback so indexing never hard-fails on a model load error.

---

## Production Upgrade Path

Each row below is an independent swap — you don't need to change everything at once.

### Indexing / Feature Extraction

| Component | POC (current) | Production replacement | Why it matters |
|---|---|---|---|
| **Scene segmentation** | PySceneDetect pixel-diff threshold | [TransNetV2](https://github.com/soCzech/TransNetV2) or a VLM-based semantic boundary detector | Pixel-diff misses cuts within similar-looking scenes and over-splits on motion blur. Semantic segmentation produces coherent clips. |
| **VLM (captioning + tagging)** | Qwen2.5-VL 7B @ 4-bit, single midpoint frame | GPT-4o / Claude 3.5 Sonnet / Gemini 1.5 Pro via API, fed 4–8 frames per clip | API-grade VLMs produce significantly richer, more accurate captions. Multi-frame input catches motion and temporal context a single frame misses. |
| **Object detection** | YOLOv8 (COCO 80-class closed vocab) | [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) or Florence-2 | Open-vocabulary detection — finds any object described in text, not just 80 COCO classes. Critical for domain-specific content. |
| **Action recognition** | X-CLIP (fixed vocab list) | Drop in favour of VLM-only, or use [InternVideo2](https://github.com/OpenGVLab/InternVideo) | X-CLIP is constrained to a predefined vocab. A capable VLM can describe actions in natural language; InternVideo2 handles temporal understanding natively. |
| **OCR** | PaddleOCR | Google Vision API / AWS Textract | Higher accuracy on stylised text, low-contrast overlays, and non-standard fonts common in sports/broadcast video. |
| **Text embedding** | `all-MiniLM-L6-v2` (384-dim) | `text-embedding-3-large` (OpenAI) or `voyage-3` (Voyage AI) | MiniLM is fast but weak on nuanced semantic similarity. Larger API embeddings close the gap between how users phrase queries and how clips are described. |
| **Visual embedding** | OpenCLIP `ViT-B-32` | OpenCLIP `ViT-bigG-14` or [SigLIP2](https://huggingface.co/google/siglip2-so400m-patch14-384) | ViT-B-32 is the smallest CLIP variant. ViT-bigG-14 and SigLIP2 score significantly higher on zero-shot visual retrieval benchmarks. |
| **Clip representation** | Single keyframe (midpoint) | Encode the clip as a short video using [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind) or InternVideo2 video encoder | Keyframe-only indexing loses motion and temporal context entirely. Video encoders produce a single embedding for the whole clip that captures what happens over time. |

---

### Retrieval / Search

| Component | POC (current) | Production replacement | Why it matters |
|---|---|---|---|
| **Hybrid search** | Dense-only (two vector channels) | Dense + sparse (BM25 / TF-IDF) merged via RRF or weighted sum | Sparse retrieval handles exact-match queries (names, product codes, UI text) that dense embeddings score poorly on. |
| **Re-ranking** | Simple weighted score formula | Cross-encoder re-ranker (e.g. `ms-marco-MiniLM-L-12-v2`) or a VLM zero-shot ranker | The dual-channel score is a coarse proximity measure. A cross-encoder reads query and clip description together and produces a much more accurate relevance score. |
| **Query understanding** | Raw query string embedded as-is | Query expansion via LLM (synonyms, rephrasing) or HyDE (generate a hypothetical clip description, embed that) | Short user queries are often ambiguous. Expanding them before embedding improves recall, especially for conceptual or abstract queries. |
| **Vector store** | ChromaDB (single-process, no sharding) | Qdrant, Weaviate, or pgvector (Postgres) | ChromaDB is in-process and single-node. Qdrant/Weaviate support distributed deployment, payload filtering, and quantized HNSW at scale. pgvector keeps vectors in Postgres and simplifies the stack if you're already on Postgres. |
| **Tag-match boost** | Keyword token overlap (exact match only) | BM25 over stored tags, or a small cross-encoder scoring query against tag bag | Current token overlap misses synonyms ("car" vs "vehicle") and penalises multi-word phrases. BM25 handles partial matches and term frequency naturally. |
