"""
============================================================
DETECT-AI: AI Ensemble Labeler  v2 — Fixed
============================================================
Key fixes from v1:
  FIX 1: SELECT label + final_confidence from staging. If the scraper
          already assigned a high-confidence label (≥0.90 for AI_GENERATED,
          ≥0.85 for HUMAN), trust it and skip the expensive HF ensemble call.
          This handles civitai (0.99), unsplash (0.95), pexels (0.95) etc.
  FIX 2: When all HF models fail, fall back to the staging label instead of
          blindly returning UNCERTAIN.
  FIX 3: Shard threshold lowered from 200k → 10k for faster HF pushes.
  FIX 4: label_batch now SELECTs label + final_confidence from staging.
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
BATCH_SIZE          = int(os.environ.get("LABELER_BATCH_SIZE", "100"))
SHARD_THRESHOLD     = int(os.environ.get("SHARD_THRESHOLD", "5000"))   # lowered from 10k   # was 200k — now 10k
UNCERTAIN_VERIFY_PCT = 0.02
AI_THRESHOLD        = 0.75
HUMAN_THRESHOLD     = 0.35
# FIX 1: Thresholds to TRUST pre-assigned scraper labels (skip ensemble)
TRUST_AI_THRESHOLD    = 0.90   # trust AI_GENERATED if scraper conf ≥ 0.90
TRUST_HUMAN_THRESHOLD = 0.85   # trust HUMAN if scraper conf ≥ 0.85
HF_API_BASE         = "https://api-inference.huggingface.co/models"
MAX_HF_RETRIES      = 3
HF_TIMEOUT_SEC      = 25

# Sources that always produce reliable pre-labeled samples (skip ensemble)
TRUSTED_SOURCES = {
    "civitai":        ("AI_GENERATED", 0.99),
    "pollinations":   ("AI_GENERATED", 0.99),
    "diffusiondb":    ("AI_GENERATED", 0.99),
    "lexica":         ("AI_GENERATED", 0.99),
    "sd-prompts":     ("AI_GENERATED", 0.85),
    "unsplash":       ("HUMAN",        0.95),
    "pexels":         ("HUMAN",        0.95),
    "pixabay":        ("HUMAN",        0.93),
    "openverse":      ("HUMAN",        0.92),
    "nasa":           ("HUMAN",        0.98),
    "met-museum":     ("HUMAN",        0.97),
    "wikimedia":      ("HUMAN",        0.92),
    "bbc-news":       ("HUMAN",        0.93),
    "reuters":        ("HUMAN",        0.93),
    "guardian":       ("HUMAN",        0.93),
    "nytimes":        ("HUMAN",        0.92),
    "aljazeera":      ("HUMAN",        0.92),
    "npr":            ("HUMAN",        0.92),
    "wikipedia":      ("HUMAN",        0.95),
    "arxiv":          ("HUMAN",        0.95),
    "stackexchange":  ("HUMAN",        0.92),
    "paperswithcode": ("HUMAN",        0.93),
    "worldbank":      ("HUMAN",        0.93),
    "oig-laion":      ("HUMAN",        0.90),
    "laion-wit":      ("HUMAN",        0.90),
    "cc-news":        ("HUMAN",        0.90),
}

@dataclass
class ModelDef:
    name:       str
    hf_id:      str
    weight:     float
    modality:   str
    input_type: str

TEXT_ENSEMBLE = [
    ModelDef("roberta-openai",  "roberta-base-openai-detector",                  0.40, "text", "text"),
    ModelDef("chatgpt-roberta", "Hello-SimpleAI/chatgpt-detector-roberta",       0.35, "text", "text"),
    ModelDef("ai-content-det",  "openai-community/roberta-base-openai-detector", 0.25, "text", "text"),
]
IMAGE_ENSEMBLE = [
    ModelDef("ai-image-det",  "umm-maybe/AI-image-detector",                          0.40, "image", "image_url"),
    ModelDef("vit-deepfake",  "Wvolf/ViT-Deepfake-Detection",                         0.35, "image", "image_url"),
    ModelDef("sdxl-detector", "haywoodsloan/autotrain-ai-or-not-diffusion-20240906",  0.25, "image", "image_url"),
]
AUDIO_ENSEMBLE = [
    ModelDef("wav2vec2-deepfake", "mo-thecreator/deepfake-audio-detector", 0.60, "audio", "audio_url"),
    ModelDef("resemblyzer",       "speechbrain/spkrec-ecapa-voxceleb",     0.40, "audio", "audio_url"),
]
VIDEO_ENSEMBLE = [
    ModelDef("llama-vision", "meta-llama/Llama-3.2-11B-Vision-Instruct", 1.00, "video", "image_url"),
    ModelDef("vit-fallback", "Wvolf/ViT-Deepfake-Detection",             1.00, "video", "image_url"),
]
ENSEMBLE_MAP = {
    "text":  TEXT_ENSEMBLE,
    "image": IMAGE_ENSEMBLE,
    "audio": AUDIO_ENSEMBLE,
    "video": VIDEO_ENSEMBLE,
}

@dataclass
class LabelResult:
    sample_id:        str
    label:            str
    final_confidence: float
    model_scores:     dict = field(default_factory=dict)
    error:            Optional[str] = None

class SupabaseClient:
    def __init__(self, url: str, key: str, session: aiohttp.ClientSession):
        self.url     = url.rstrip("/")
        self.headers = {
            "apikey": key, "Authorization": f"Bearer {key}",
            "Content-Type": "application/json", "Prefer": "return=representation",
        }
        self.session = session

    async def select(self, table: str, params: dict) -> list:
        async with self.session.get(
            f"{self.url}/rest/v1/{table}", headers=self.headers, params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if not resp.ok:
                log.error(f"SELECT {table} failed: {resp.status} {await resp.text()}")
                return []
            return await resp.json()

    async def insert(self, table: str, rows: list) -> bool:
        if not rows:
            return True
        async with self.session.post(
            f"{self.url}/rest/v1/{table}", headers=self.headers, json=rows,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if not resp.ok:
                log.error(f"INSERT {table} failed: {resp.status} {await resp.text()}")
            return resp.ok

    async def update(self, table: str, match: dict, values: dict) -> bool:
        params = {k: f"eq.{v}" for k, v in match.items()}
        async with self.session.patch(
            f"{self.url}/rest/v1/{table}", headers=self.headers,
            params=params, json=values, timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            return resp.ok

    async def delete(self, table: str, match: dict) -> bool:
        params = {k: f"eq.{v}" for k, v in match.items()}
        async with self.session.delete(
            f"{self.url}/rest/v1/{table}", headers=self.headers,
            params=params, timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            return resp.ok

    async def count(self, table: str, params: dict) -> int:
        p = {**params, "select": "sample_id"}
        h = {**self.headers, "Prefer": "count=exact", "Range-Unit": "items", "Range": "0-0"}
        async with self.session.get(
            f"{self.url}/rest/v1/{table}", headers=h, params=p,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            try:
                cr = resp.headers.get("Content-Range", "*/0")
                return int(cr.split("/")[1])
            except:
                return 0

async def call_hf_model(session, model, content) -> Optional[float]:
    url = f"{HF_API_BASE}/{model.hf_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = ({"inputs": content[:2048]} if model.input_type == "text"
               else {"inputs": {"image": content}})

    for attempt in range(1, MAX_HF_RETRIES + 1):
        try:
            async with session.post(url, headers=headers, json=payload,
                                    timeout=aiohttp.ClientTimeout(total=HF_TIMEOUT_SEC)) as resp:
                if resp.status in (429, 503):
                    await asyncio.sleep(10 * attempt); continue
                if resp.status in (404, 400, 422):
                    log.debug(f"[{model.name}] HTTP {resp.status} — model unavailable")
                    return None
                if not resp.ok:
                    return None
                data = await resp.json()
                if isinstance(data, list):
                    inner = data[0] if isinstance(data[0], list) else data
                    for item in inner:
                        lbl = str(item.get("label", "")).upper()
                        if any(k in lbl for k in ["AI","FAKE","LABEL_1","MACHINE","GENERATED","DEEPFAKE"]):
                            return float(item.get("score", 0.5))
                    for item in inner:
                        lbl = str(item.get("label", "")).upper()
                        if any(k in lbl for k in ["HUMAN","REAL","LABEL_0","AUTHENTIC"]):
                            return 1.0 - float(item.get("score", 0.5))
                return None
        except asyncio.TimeoutError:
            if attempt == MAX_HF_RETRIES: return None
            await asyncio.sleep(5 * attempt)
        except Exception as e:
            log.debug(f"[{model.name}] {e}")
            if attempt == MAX_HF_RETRIES: return None
            await asyncio.sleep(5 * attempt)
    return None

async def run_ensemble(session, content_type, content,
                       fallback_label=None, fallback_conf=None) -> LabelResult:
    """
    Run the HF ensemble. FIX 2: If all models fail, use fallback_label
    (from staging) instead of blindly returning UNCERTAIN.
    """
    models = ENSEMBLE_MAP.get(content_type, TEXT_ENSEMBLE)
    if content_type == "video":
        models = [VIDEO_ENSEMBLE[0]]

    tasks = [call_hf_model(session, m, content) for m in models]
    scores = await asyncio.gather(*tasks, return_exceptions=True)

    model_scores: dict = {}
    weighted_sum = weight_total = 0.0

    for model, score in zip(models, scores):
        if isinstance(score, float):
            model_scores[model.name] = round(score, 4)
            weighted_sum += score * model.weight
            weight_total += model.weight
        else:
            log.debug(f"Model {model.name} returned no score")

    # FIX 2: Use fallback label if all models failed
    if weight_total == 0:
        if fallback_label and fallback_conf is not None:
            log.info(f"  All models failed — using scraper label: {fallback_label} ({fallback_conf})")
            return LabelResult("", fallback_label, fallback_conf, {}, "ensemble_failed_using_scraper_label")
        return LabelResult("", "UNCERTAIN", 0.5, {}, "all_models_failed")

    fc = round(weighted_sum / weight_total, 4)
    label = "AI_GENERATED" if fc >= AI_THRESHOLD else "HUMAN" if fc <= HUMAN_THRESHOLD else "UNCERTAIN"
    return LabelResult("", label, fc, model_scores)

async def label_batch(db: SupabaseClient, session: aiohttp.ClientSession) -> int:
    # FIX 1 & 4: SELECT label + final_confidence from staging
    rows = await db.select("samples_staging", {
        "status": "eq.staged",
        "select": "sample_id,source_id,source_url,content_type,language,raw_content,"
                  "metadata,scraped_at,worker_id,label,final_confidence",
        "limit":  str(BATCH_SIZE),
        "order":  "scraped_at.asc",
    })
    if not rows:
        return 0

    log.info(f"Labeling batch of {len(rows)} samples...")

    # Mark as labeling
    for r in rows:
        await db.update("samples_staging", {"sample_id": r["sample_id"]}, {"status": "labeling"})

    semaphore = asyncio.Semaphore(10)

    async def label_one(row: dict) -> tuple[dict, LabelResult]:
        async with semaphore:
            content    = row.get("raw_content") or ""
            source_id  = row.get("source_id", "")
            stg_label  = row.get("label")
            stg_conf   = row.get("final_confidence")

            if not content:
                result = LabelResult("", "UNCERTAIN", 0.5, {}, "empty_content")

            # FIX 1: Trust high-confidence sources — skip ensemble entirely
            elif source_id in TRUSTED_SOURCES:
                trusted_label, trusted_conf = TRUSTED_SOURCES[source_id]
                result = LabelResult("", trusted_label, trusted_conf, {},
                                     f"trusted_source:{source_id}")
                log.debug(f"  [{source_id}] trusted → {trusted_label} ({trusted_conf})")

            # FIX 1b: Trust existing high-confidence staging label
            elif (stg_label == "AI_GENERATED" and stg_conf is not None and stg_conf >= TRUST_AI_THRESHOLD):
                result = LabelResult("", "AI_GENERATED", float(stg_conf), {},
                                     "trusted_staging_label:ai")
            elif (stg_label == "HUMAN" and stg_conf is not None and stg_conf >= TRUST_HUMAN_THRESHOLD):
                result = LabelResult("", "HUMAN", float(stg_conf), {},
                                     "trusted_staging_label:human")

            else:
                # Run full ensemble for ambiguous sources
                result = await run_ensemble(
                    session, row["content_type"], content,
                    fallback_label=stg_label,
                    fallback_conf=float(stg_conf) if stg_conf is not None else None,
                )

            result.sample_id = row["sample_id"]
            return row, result

    results = await asyncio.gather(*[label_one(r) for r in rows])

    processed_rows = []
    uncertain_rows = []
    label_counts   = {}

    for row, result in results:
        label_counts[result.label] = label_counts.get(result.label, 0) + 1

        processed_rows.append({
            "sample_id":        result.sample_id,
            "source_id":        row["source_id"],
            "source_url":       row["source_url"],
            "content_type":     row["content_type"],
            "language":         row["language"],
            "raw_content":      (row.get("raw_content") or "")[:8000],
            "metadata":         row.get("metadata") or {},
            "scraped_at":       row["scraped_at"],
            "worker_id":        row["worker_id"],
            "label":            result.label,
            "final_confidence": result.final_confidence,
            "model_scores":     result.model_scores,
            "verified":         result.label == "AI_GENERATED" or result.error == f"trusted_source:{row['source_id']}",
            "labeled_at":       datetime.now(timezone.utc).isoformat(),
            "hf_pushed":        False,
        })

        if result.label == "UNCERTAIN" and random.random() < UNCERTAIN_VERIFY_PCT:
            uncertain_rows.append({"sample_id": result.sample_id, "reason": result.error or "ensemble_uncertain"})

    # Batch insert to processed (100 at a time)
    pushed = 0
    for i in range(0, len(processed_rows), 100):
        ok = await db.insert("samples_processed", processed_rows[i:i+100])
        if ok:
            pushed += min(100, len(processed_rows) - i)

    if uncertain_rows:
        await db.insert("verification_queue", uncertain_rows)

    # Delete from staging
    for r in rows:
        await db.delete("samples_staging", {"sample_id": r["sample_id"]})

    log.info(f"  Labeled {pushed}/{len(rows)}: {label_counts}")
    return pushed

async def check_and_push_shard(db: SupabaseClient, session: aiohttp.ClientSession):
    """No-op stub — HF push is now handled inline in ensemble_labeler_once.py"""
    pass

async def main():
    log.info("=== DETECT-AI Labeler v2 started ===")
    log.info(f"  Batch size:      {BATCH_SIZE}")
    log.info(f"  Shard threshold: {SHARD_THRESHOLD:,}")
    log.info(f"  Trusted sources: {len(TRUSTED_SOURCES)}")

    async with aiohttp.ClientSession() as session:
        db = SupabaseClient(SUPABASE_URL, SUPABASE_KEY, session)
        total = 0
        while True:
            try:
                n = await label_batch(db, session)
                total += n
                if n > 0:
                    log.info(f"  Total labeled this run: {total:,}")
                    await check_and_push_shard(db, session)
                else:
                    log.info("  No staged samples — sleeping 30s")
                    await asyncio.sleep(30)
            except KeyboardInterrupt:
                log.info("Shutting down.")
                break
            except Exception as e:
                log.error(f"Labeler loop error: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())

# ── Compatibility shims for ensemble_labeler_once.py ──────────────────────────
# once.py imports check_shard_trigger and record_metrics by name
check_shard_trigger = check_and_push_shard   # alias

async def record_metrics(db: SupabaseClient):
    """Log basic pipeline metrics (no-op — metrics visible via Supabase queries)."""
    pass
