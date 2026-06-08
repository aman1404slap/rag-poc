"""
LLM-enhanced search via AWS Bedrock.

Pipeline:
  1. plan_query()   — LLM expands the user query into multiple ChromaDB queries + infers intent
  2. _multi_query_retrieve() — runs all queries against both ChromaDB collections, fuses with RRF
  3. rerank()       — LLM re-orders the ~30 candidates by true relevance to the original intent
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

_bedrock_client = None

def _build_schema_context() -> str:
    from django.conf import settings
    vocab = settings.ML.get('ACTION_VOCAB', [])
    vocab_line = ', '.join(vocab) if vocab else '(not configured)'
    return f"""
You are helping search a video clip database. Each video clip has been analyzed and indexed with:

TEXT CHANNEL (sentence-transformer embeddings):
- caption: AI-generated natural language description of what happens visually
- objects_detected: physical objects visible in the frame (e.g. "laptop", "whiteboard", "person", "car")
- actions_detected: activities occurring — detected using this exact vocabulary:
  {vocab_line}
- ocr_text: text visible on screen (code, slides, UI elements, subtitles)

VISUAL CHANNEL (CLIP image embeddings of keyframe):
- the visual appearance and scene composition of a representative still frame

SEARCH STRATEGY:
- Text channel: best for semantic content, context, what is happening
- Visual channel: best for visual appearance, scene type, colors, composition
- For action-based queries, prefer the exact vocabulary terms listed above over synonyms
- Generating multiple query variants with different phrasings improves recall significantly
"""


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        from anthropic import AnthropicBedrock
        kwargs = {"aws_region": os.environ.get("AWS_REGION", "us-east-1")}
        # Explicit .env keys take precedence; otherwise fall back to ~/.aws/credentials / CLI profile
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            kwargs["aws_access_key"] = os.environ["AWS_ACCESS_KEY_ID"]
            kwargs["aws_secret_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
        if os.environ.get("AWS_PROFILE"):
            kwargs["aws_profile"] = os.environ["AWS_PROFILE"]
        _bedrock_client = AnthropicBedrock(**kwargs)
    return _bedrock_client


def _reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists of clip IDs into a single ordering.
    Score = sum of 1/(k + rank) across all lists the clip appears in.
    Clips appearing consistently across many query variants rank highest.
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, clip_id in enumerate(lst):
            scores[clip_id] = scores.get(clip_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _multi_query_retrieve(
    text_queries: list[str],
    visual_queries: list[str],
    n_per_query: int = 15,
) -> list[str]:
    """
    Run all LLM-generated queries against both ChromaDB collections and fuse with RRF.
    Returns ~25-30 clip IDs ordered by fused relevance score.
    """
    from search.pipeline import embedder as emb

    text_col, visual_col = emb._get_chroma()

    text_count = text_col.count()
    visual_count = visual_col.count()
    n_text = min(n_per_query, text_count) if text_count > 0 else 0
    n_visual = min(n_per_query, visual_count) if visual_count > 0 else 0

    ranked_lists: list[list[str]] = []

    if n_text > 0:
        for q in text_queries:
            try:
                vec = emb.embed_text(q)
                res = text_col.query(
                    query_embeddings=[vec],
                    n_results=n_text,
                    include=["distances"],
                )
                if res["ids"] and res["ids"][0]:
                    ranked_lists.append(res["ids"][0])
            except Exception:
                logger.warning("Text query failed: %r", q, exc_info=True)

    if n_visual > 0:
        for q in visual_queries:
            try:
                vec = emb.embed_text_clip(q)
                res = visual_col.query(
                    query_embeddings=[vec],
                    n_results=n_visual,
                    include=["distances"],
                )
                if res["ids"] and res["ids"][0]:
                    ranked_lists.append(res["ids"][0])
            except Exception:
                logger.warning("Visual query failed: %r", q, exc_info=True)

    if not ranked_lists:
        return []

    fused = _reciprocal_rank_fusion(ranked_lists)
    return [clip_id for clip_id, _ in fused[:30]]


class BedrockSearchEnhancer:

    def plan_query(self, query: str) -> dict:
        """
        Ask the LLM to understand the query and produce multiple ChromaDB search variants.

        Returns dict with keys:
          intent (str), text_queries (list[str]), visual_queries (list[str]),
          channel_weights (dict with 'text' and 'visual' floats summing to 1.0)
        """
        client = _get_bedrock_client()
        model = os.environ.get("BEDROCK_PLANNER_MODEL_ID", "")
        if not model:
            raise ValueError("BEDROCK_PLANNER_MODEL_ID is not set in .env")

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_build_schema_context(),
            messages=[{
                "role": "user",
                "content": (
                    f'Generate a search plan for this video clip query: "{query}"\n\n'
                    "Respond with valid JSON only, no markdown, no explanation. Format:\n"
                    "{\n"
                    '  "intent": "<one sentence: what the user is actually looking for>",\n'
                    '  "text_queries": ["<3-5 query variants for the text embedding channel>"],\n'
                    '  "visual_queries": ["<2-3 variants describing visual appearance for CLIP>"],\n'
                    '  "channel_weights": {"text": <0.0-1.0>, "visual": <0.0-1.0>}\n'
                    "}\n\n"
                    "Rules:\n"
                    "- text_queries: use synonyms, more/less specific terms, related concepts\n"
                    "- visual_queries: describe what the scene LOOKS like, not what happens\n"
                    "- channel_weights must sum to 1.0; weight text higher for content queries, "
                    "visual higher for appearance queries"
                ),
            }],
        )

        raw = next(b.text for b in response.content if b.type == "text").strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        plan = json.loads(raw)

        # Normalise: ensure required keys exist with sane defaults
        plan.setdefault("intent", query)
        plan.setdefault("text_queries", [query])
        plan.setdefault("visual_queries", [query])
        plan.setdefault("channel_weights", {"text": 0.6, "visual": 0.4})

        return plan

    def _format_candidate(self, clip) -> str:
        """Render a Clip ORM object as a text block for the reranker prompt."""
        actions = list(clip.actions_detected or [])
        if not actions and clip.action_label:
            actions = [{"label": clip.action_label, "confidence": clip.action_confidence or 0}]

        objects = list(clip.objects_detected or [])

        ocr_blocks = list(clip.ocr_blocks or [])
        if not ocr_blocks and clip.ocr_text:
            ocr_blocks = [{"text": clip.ocr_text}]

        top_actions = sorted(actions, key=lambda x: x.get("confidence", 0), reverse=True)[:3]
        top_objects = sorted(objects, key=lambda x: x.get("confidence", 0), reverse=True)[:5]

        action_str = ", ".join(
            f"{a['label']} ({a.get('confidence', 0):.2f})" for a in top_actions
        )
        object_str = ", ".join(
            f"{o['label']} ({o.get('confidence', 0):.2f})" for o in top_objects
        )
        ocr_str = " | ".join(b.get("text", "") for b in ocr_blocks[:3] if b.get("text"))

        lines = [f"Clip ID: {clip.id}"]
        lines.append(f"Caption: {clip.caption or '(none)'}")
        if action_str:
            lines.append(f"Actions: {action_str}")
        if object_str:
            lines.append(f"Objects: {object_str}")
        if ocr_str:
            lines.append(f"On-screen text: {ocr_str}")

        return "\n".join(lines)

    def rerank(self, query: str, intent: str, candidates: list) -> list[dict]:
        """
        Re-order candidate Clip objects by relevance to the original query + intent.

        Returns list of {"clip_id": str, "reasoning": str}, max 10 items.
        Hallucinated clip IDs (not in candidates) are silently dropped by the caller.
        """
        if not candidates:
            return []

        client = _get_bedrock_client()
        model = os.environ.get("BEDROCK_RERANKER_MODEL_ID", "")
        if not model:
            raise ValueError("BEDROCK_RERANKER_MODEL_ID is not set in .env")

        candidate_text = "\n\n---\n\n".join(
            self._format_candidate(c) for c in candidates
        )

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system="You are a video search relevance expert. Rank video clips by how well they match the user's search intent.",
            messages=[{
                "role": "user",
                "content": (
                    f"ORIGINAL QUERY: {query}\n"
                    f"SEARCH INTENT: {intent}\n\n"
                    "Rank the following video clips from most to least relevant.\n"
                    "Return the top 10 most relevant clips only. Omit clips that are not relevant.\n\n"
                    "Respond with valid JSON only, no markdown. Format:\n"
                    '{"ranked_clips": [{"clip_id": "<id>", "reasoning": "<one sentence why it matches>"}, ...]}\n\n'
                    f"CANDIDATE CLIPS:\n{candidate_text}"
                ),
            }],
        )

        raw = next(b.text for b in response.content if b.type == "text").strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)
        return result.get("ranked_clips", [])[:10]
