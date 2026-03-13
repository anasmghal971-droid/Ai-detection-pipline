"""
============================================================
DETECT-AI: AI Ensemble Labeler  (Stage 3)
============================================================
Runs as a cron Python process (every 2 minutes).

  Flow:
    1. Pull batch of 'staged' samples from Supabase
    2. Route each sample to its modality ensemble
    3. Call HF Inference API for each model in parallel
    4. Compute weighted average → final label
    5. Flag 2% of UNCERTAIN for human verification
    6. Write results to samples_processed
    7. Trigger shard export when 200k threshold hit

  Ensembles:
    TEXT:  roberta-openai(0.40) + chatgpt-roberta(0.35) + ai-detector(0.25)
    IMAGE: ai-image-det(0.40)   + vit-deepfake(0.35)    + sdxl-det(0.25)
    AUDIO: wav2vec2(0.60)       + resemblyzer(0.40)
    VIDEO: llama-3.2-90b-vision (primary) → vit fallback
============================================================
"""

import os
import time
import uuid
import random
import logging
import asyncio
import aiohttp
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.INFO,
)
log = logging.getLogger("detect-ai.labeler")

# ── Config ─────────────────────────────────────────────────────
SUPABASE_URL        = os.environ["SUPABASE_URL"]
SUPABASE_KEY        = os.environ["SUPABASE_SERVICE_KEY"]
HF_TOKEN            = os.environ["HF_TOKEN"]
BATCH_SIZE          = int(os.environ.get("LABELER_BATCH_SIZE", "50"))
SHARD_THRESHOLD     = int(os.environ.get("SHARD_THRESHOLD", "200000"))
UNCERTAIN_VERIFY_PCT = 0.02   # 2% of UNCERTAIN → human queue
AI_THRESHOLD        = 0.75
HUMAN_THRESHOLD     = 0.35
HF_API_BASE         = "https://api-inference.huggingface.co/models"
MAX_HF_RETRIES      = 3
HF_TIMEOUT_SEC      = 30

# ── Model definitions ──────────────────────────────────────────
@dataclass
class ModelDef:
    name:       str
    hf_id:      str
    weight:     float
    modality:   str
    input_type: str   # "text" | "image_url" | "audio_url" | "image_bytes"

TEXT_ENSEMBLE = [
    ModelDef("roberta-openai",    "roberta-base-openai-detector",                  0.40, "text", "text"),
    ModelDef("chatgpt-roberta",   "Hello-SimpleAI/chatgpt-detector-roberta",       0.35, "text", "text"),
    ModelDef("ai-content-det",    "openai-community/roberta-base-openai-detector", 0.25, "text", "text"),
]

IMAGE_ENSEMBLE = [
    ModelDef("ai-image-det",  "umm-maybe/AI-image-detector",                   0.40, "image", "image_url"),
    ModelDef("vit-deepfake",  "Wvolf/ViT-Deepfake-Detection",                  0.35, "image", "image_url"),
    ModelDef("sdxl-detector", "haywoodsloan/autotrain-ai-or-not-diffusion-20240906",  0.25, "image", "image_url"),
]

AUDIO_ENSEMBLE = [
    ModelDef("wav2vec2-deepfake", "mo-thecreator/deepfake-audio-detector",   0.60, "audio", "audio_url"),
    ModelDef("resemblyzer",       "speechbrain/spkrec-ecapa-voxceleb",        0.40, "audio", "audio_url"),
]

VIDEO_ENSEMBLE = [
    ModelDef("llama-vision",  "meta-llama/Llama-3.2-11B-Vision-Instruct",   1.00, "video", "image_url"),
    # Fallback if quota exceeded:
    ModelDef("vit-fallback",  "Wvolf/ViT-Deepfake-Detection",               1.00, "video", "image_url"),
]

ENSEMBLE_MAP = {
    "text":  TEXT_ENSEMBLE,
    "image": IMAGE_ENSEMBLE,
    "audio": AUDIO_ENSEMBLE,
    "video": VIDEO_ENSEMBLE,
}

# ── Label result ───────────────────────────────────────────────
@dataclass
class LabelResult:
    sample_id:        str
    label:            str   # AI_GENERATED | HUMAN | UNCERTAIN
    final_confidence: float
    model_scores:     dict = field(default_factory=dict)
    error:            Optional[str] = None

# ── Supabase REST client ───────────────────────────────────────
class SupabaseClient:
    def __init__(self, url: str, key: str, session: aiohttp.ClientSession):
        self.url     = url.rstrip("/")
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        self.session = session

    async def select(self, table: str, params: dict) -> list:
        async with self.session.get(
            f"{self.url}/rest/v1/{table}",
            headers=self.headers,
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def insert(self, table: str, rows: list) -> None:
        async with self.session.post(
            f"{self.url}/rest/v1/{table}",
            headers=self.headers,
            json=rows,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()

    async def update(self, table: str, filters: dict, values: dict) -> None:
        params = {k: f"eq.{v}" for k, v in filters.items()}
        async with self.session.patch(
            f"{self.url}/rest/v1/{table}",
            headers=self.headers,
            params=params,
            json=values,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()

    async def count(self, table: str, params: dict) -> int:
        hdrs = {**self.headers, "Prefer": "count=exact"}
        hdrs["Range"] = "0-0"
        async with self.session.get(
            f"{self.url}/rest/v1/{table}",
            headers=hdrs,
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            content_range = resp.headers.get("Content-Range", "0/0")
            try:
                return int(content_range.split("/")[1])
            except (IndexError, ValueError):
                return 0

# ── HF Inference API caller ────────────────────────────────────
async def call_hf_model(
    session:    aiohttp.ClientSession,
    model:      ModelDef,
    content:    str,           # text body OR URL string for image/audio
) -> Optional[float]:
    """
    Call HF Inference API and return AI probability [0.0–1.0].
    Returns None on failure.
    """
    url = f"{HF_API_BASE}/{model.hf_id}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type":  "application/json",
    }

    if model.input_type == "text":
        payload = {"inputs": content[:2048]}   # truncate to 2048 chars
    elif model.input_type == "image_url":
        payload = {"inputs": {"image": content}}
    elif model.input_type == "audio_url":
        payload = {"inputs": content}
    else:
        payload = {"inputs": content}

    for attempt in range(1, MAX_HF_RETRIES + 1):
        try:
            async with session.post(
                url, headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=HF_TIMEOUT_SEC),
            ) as resp:
                if resp.status == 429 or resp.status == 503:
                    # Model loading or rate-limited — wait and retry
                    wait = 10 * attempt
                    log.warning(f"[{model.name}] {resp.status} — waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                if resp.status == 404:
                    log.warning(f"[{model.name}] Model not found on HF Inference API")
                    return None
                if not resp.ok:
                    log.warning(f"[{model.name}] HTTP {resp.status}")
                    return None

                data = await resp.json()

                # Parse classification response
                # HF returns: [[{"label": "LABEL_1", "score": 0.9}, ...]]
                # or: [{"label": "LABEL_1", "score": 0.9}]
                if isinstance(data, list):
                    inner = data[0] if isinstance(data[0], list) else data
                    # Find the AI/FAKE/LABEL_1 class probability
                    for item in inner:
                        lbl = str(item.get("label", "")).upper()
                        if any(k in lbl for k in ["AI", "FAKE", "LABEL_1", "MACHINE", "GENERATED", "DEEPFAKE"]):
                            return float(item.get("score", 0.5))
                    # If no AI label found, return 1 - highest human label score
                    for item in inner:
                        lbl = str(item.get("label", "")).upper()
                        if any(k in lbl for k in ["HUMAN", "REAL", "LABEL_0", "AUTHENTIC"]):
                            return 1.0 - float(item.get("score", 0.5))
                    # Fallback: return 0.5 (uncertain)
                    return 0.5

        except asyncio.TimeoutError:
            log.warning(f"[{model.name}] Timeout on attempt {attempt}")
            if attempt == MAX_HF_RETRIES:
                return None
            await asyncio.sleep(5 * attempt)
        except Exception as e:
            log.warning(f"[{model.name}] Error: {e}")
            if attempt == MAX_HF_RETRIES:
                return None
            await asyncio.sleep(5 * attempt)

    return None

# ── Ensemble runner ────────────────────────────────────────────
async def run_ensemble(
    session:      aiohttp.ClientSession,
    content_type: str,
    content:      str,
) -> LabelResult:
    """
    Run all models in the ensemble concurrently.
    Compute weighted average. Apply thresholds.
    """
    models = ENSEMBLE_MAP.get(content_type, TEXT_ENSEMBLE)

    # For video: use first representative frame URL or description
    if content_type == "video":
        # content is the raw_content (video URL or description)
        # We pass it as text to llama-vision via image prompt
        effective_content = content
        effective_models  = [VIDEO_ENSEMBLE[0]]  # Try primary first
    else:
        effective_content = content
        effective_models  = models

    # Run all models concurrently
    tasks = [
        call_hf_model(session, model, effective_content)
        for model in effective_models
    ]
    scores = await asyncio.gather(*tasks, return_exceptions=True)

    model_scores: dict[str, float] = {}
    weighted_sum  = 0.0
    weight_total  = 0.0

    for model, score in zip(effective_models, scores):
        if isinstance(score, float):
            model_scores[model.name] = round(score, 4)
            weighted_sum  += score * model.weight
            weight_total  += model.weight
        else:
            log.warning(f"Model {model.name} returned no score")

    if weight_total == 0:
        # All models failed — mark uncertain, flag for human review
        return LabelResult(
            sample_id="",
            label="UNCERTAIN",
            final_confidence=0.5,
            model_scores={},
            error="all_models_failed",
        )

    final_confidence = round(weighted_sum / weight_total, 4)

    if final_confidence >= AI_THRESHOLD:
        label = "AI_GENERATED"
    elif final_confidence <= HUMAN_THRESHOLD:
        label = "HUMAN"
    else:
        label = "UNCERTAIN"

    return LabelResult(
        sample_id="",
        label=label,
        final_confidence=final_confidence,
        model_scores=model_scores,
    )

# ── Main labeling loop ─────────────────────────────────────────
async def label_batch(db: SupabaseClient, session: aiohttp.ClientSession) -> int:
    """
    Pull a batch of staged samples, label them, move to processed.
    Returns number of samples labeled.
    """
    # Fetch staged samples
    rows = await db.select("samples_staging", {
        "status":  "eq.staged",
        "select":  "sample_id,source_id,source_url,content_type,language,raw_content,metadata,scraped_at,worker_id",
        "limit":   str(BATCH_SIZE),
        "order":   "scraped_at.asc",
    })

    if not rows:
        return 0

    # Mark as 'labeling' to prevent double-processing
    sample_ids = [r["sample_id"] for r in rows]
    log.info(f"Labeling batch of {len(rows)} samples...")

    # Mark all as labeling
    for sid in sample_ids:
        await db.update("samples_staging", {"sample_id": sid}, {"status": "labeling"})

    # Run ensemble labeling concurrently (max 10 parallel)
    semaphore = asyncio.Semaphore(10)

    async def label_one(row: dict) -> tuple[dict, LabelResult]:
        async with semaphore:
            content = row.get("raw_content") or ""
            if not content:
                result = LabelResult("", "UNCERTAIN", 0.5, {}, "empty_content")
            else:
                result = await run_ensemble(session, row["content_type"], content)
            result.sample_id = row["sample_id"]
            return row, result

    pairs = await asyncio.gather(*[label_one(r) for r in rows])

    # Build processed records
    now = datetime.now(timezone.utc).isoformat()
    processed_rows = []
    uncertain_for_review = []

    for row, result in pairs:
        processed = {
            "sample_id":        row["sample_id"],
            "source_id":        row["source_id"],
            "source_url":       row["source_url"],
            "content_type":     row["content_type"],
            "language":         row["language"],
            "raw_content":      row.get("raw_content"),
            "storage_path":     row.get("storage_path"),
            "metadata":         row.get("metadata", {}),
            "scraped_at":       row["scraped_at"],
            "worker_id":        row["worker_id"],
            "label":            result.label,
            "final_confidence": result.final_confidence,
            "model_scores":     result.model_scores,
            "verified":         False,
            "labeled_at":       now,
        }
        processed_rows.append(processed)

        # Flag 2% of UNCERTAIN samples for human verification
        if result.label == "UNCERTAIN" and random.random() < UNCERTAIN_VERIFY_PCT:
            uncertain_for_review.append(row["sample_id"])

    # Insert into samples_processed (batch)
    CHUNK = 50
    for i in range(0, len(processed_rows), CHUNK):
        await db.insert("samples_processed", processed_rows[i:i+CHUNK])

    # Add UNCERTAIN samples to verification queue
    if uncertain_for_review:
        verify_rows = [
            {"id": str(uuid.uuid4()), "sample_id": sid, "status": "pending", "queued_at": now}
            for sid in uncertain_for_review
        ]
        await db.insert("verification_queue", verify_rows)
        log.info(f"  Queued {len(verify_rows)} samples for human verification")

    # Mark staging records as labeled
    for sid in sample_ids:
        await db.update("samples_staging", {"sample_id": sid}, {"status": "labeled"})

    label_counts = {}
    for _, r in pairs:
        label_counts[r.label] = label_counts.get(r.label, 0) + 1

    log.info(
        f"  ✅ Labeled {len(rows)} samples — "
        f"AI_GENERATED: {label_counts.get('AI_GENERATED',0)}, "
        f"HUMAN: {label_counts.get('HUMAN',0)}, "
        f"UNCERTAIN: {label_counts.get('UNCERTAIN',0)}"
    )

    return len(rows)

# ── Shard trigger check ────────────────────────────────────────
async def check_shard_trigger(db: SupabaseClient) -> None:
    """
    For each (content_type, language) combo, if unshareded count
    >= SHARD_THRESHOLD, trigger the HF push manager.
    """
    rows = await db.select("samples_processed", {
        "select":   "content_type,language",
        "shard_id": "is.null",
    })

    if not rows:
        return

    # Count per (content_type, language)
    counts: dict[tuple, int] = {}
    for r in rows:
        key = (r["content_type"], r["language"])
        counts[key] = counts.get(key, 0) + 1

    for (ct, lang), count in counts.items():
        if count >= SHARD_THRESHOLD:
            log.info(f"🔥 Shard threshold reached: {ct}/{lang} has {count:,} samples — triggering export")
            # Import and run shard export
            try:
                import subprocess
                subprocess.Popen([
                    "python3", "hf_push/hf_push_manager.py",
                    "export", ct, lang,
                    str(count // SHARD_THRESHOLD),
                ])
            except Exception as e:
                log.error(f"Failed to trigger shard export: {e}")

# ── Pipeline metrics snapshot ──────────────────────────────────
async def record_metrics(db: SupabaseClient) -> None:
    """Write a pipeline_metrics snapshot every run."""
    try:
        staged    = await db.count("samples_staging",   {"status": "eq.staged"})
        labeled   = await db.count("samples_processed", {})
        exported  = await db.count("samples_processed", {"exported_at": "not.is.null"})
        shards    = await db.count("shard_registry",    {"push_status": "eq.pushed"})
        verify_q  = await db.count("verification_queue",{"status": "eq.pending"})

        await db.insert("pipeline_metrics", [{
            "measured_at":       datetime.now(timezone.utc).isoformat(),
            "total_staged":      staged,
            "total_labeled":     labeled,
            "total_exported":    exported,
            "total_shards_pushed": shards,
            "active_workers":    0,
        }])
        log.info(f"📊 Metrics — staged:{staged} labeled:{labeled} exported:{exported} shards:{shards} verify_queue:{verify_q}")
    except Exception as e:
        log.warning(f"Failed to record metrics: {e}")

# ── Entry point ────────────────────────────────────────────────
async def main():
    log.info("🚀 DETECT-AI Labeler starting...")
    connector = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)

    async with aiohttp.ClientSession(connector=connector) as session:
        db = SupabaseClient(SUPABASE_URL, SUPABASE_KEY, session)

        total_labeled = 0
        run_count     = 0

        while True:
            run_count += 1
            try:
                labeled = await label_batch(db, session)
                total_labeled += labeled

                # Every 10 runs, check shard triggers + record metrics
                if run_count % 10 == 0:
                    await check_shard_trigger(db)
                    await record_metrics(db)

                if labeled == 0:
                    log.info("No staged samples — sleeping 30s...")
                    await asyncio.sleep(30)
                else:
                    # Brief pause between batches to avoid HF rate limits
                    await asyncio.sleep(2)

            except KeyboardInterrupt:
                log.info(f"Shutting down. Total labeled this session: {total_labeled:,}")
                break
            except Exception as e:
                log.error(f"Labeler loop error: {e}")
                await asyncio.sleep(15)

if __name__ == "__main__":
    asyncio.run(main())
