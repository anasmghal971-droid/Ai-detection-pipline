"""
DETECT-AI — Pipeline Runner (label + push, single process)
==========================================================
Runs as GitHub Actions cron every 2 minutes.
Does two things in one process (no subprocess tricks):
  1. Label up to BATCH_SIZE staged samples → samples_processed
  2. If enough processed rows exist → push Parquet shard to HF and delete from Supabase

Fixes vs previous version:
  - No subprocess.Popen — HF push runs inline in the same process
  - check_shard_trigger always called (not gated on labeled > 0)
  - _register_shard uses correct DB column names
  - Full error logging to stdout (visible in GH Actions)
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import aiohttp
import pandas as pd
import requests
from huggingface_hub import HfApi, CommitOperationAdd

sys.path.insert(0, os.path.dirname(__file__))
from ensemble_labeler import (
    label_batch,
    SupabaseClient,
    TRUSTED_SOURCES,
    SHARD_THRESHOLD,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline-runner")

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HF_REPO      = os.environ.get("HF_DATASET_REPO", "anas775/DETECT-AI-Dataset")

SB_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=minimal",
}

# ── Supabase helpers (sync — for HF push section) ────────────────────────────
def sb_get(path: str, params: dict, timeout=20) -> list:
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{path}",
        headers=SB_HEADERS, params=params, timeout=timeout
    )
    if not r.ok:
        log.error(f"GET {path} failed: {r.status_code} {r.text[:200]}")
        return []
    return r.json()

def sb_count(path: str, params: dict) -> int:
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{path}",
        headers={**SB_HEADERS, "Prefer": "count=exact", "Range": "0-0"},
        params=params, timeout=15
    )
    if not r.ok:
        return 0
    return int(r.headers.get("Content-Range", "0-0/0").split("/")[-1])

def sb_patch(path: str, params: dict, body: dict) -> bool:
    r = requests.patch(
        f"{SUPABASE_URL}/rest/v1/{path}",
        headers=SB_HEADERS, params=params, json=body, timeout=20
    )
    return r.ok

def sb_delete(path: str, params: dict) -> bool:
    r = requests.delete(
        f"{SUPABASE_URL}/rest/v1/{path}",
        headers=SB_HEADERS, params=params, timeout=20
    )
    return r.ok

def sb_post(path: str, body: dict) -> bool:
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{path}",
        headers=SB_HEADERS, json=body, timeout=15
    )
    return r.ok

# ── HF push (inline, sync) ────────────────────────────────────────────────────
def push_shards_to_hf() -> int:
    """
    Pull ready processed samples → Parquet → HuggingFace.
    Runs inline (not subprocess). Returns number of shards pushed.
    """
    if not HF_TOKEN:
        log.error("HF_TOKEN not set — cannot push to HuggingFace")
        return 0

    api = HfApi(token=HF_TOKEN)
    pushed = 0

    for content_type in ["text", "image", "video", "audio"]:
        # Get languages that have unpushed rows
        rows_all = sb_get("samples_processed", {
            "select": "language",
            "content_type": f"eq.{content_type}",
            "hf_pushed": "eq.false",
            "limit": "1000",
        })
        languages = list({r["language"] for r in rows_all if r.get("language")}) or []
        if not languages:
            continue

        for lang in languages:
            count = sb_count("samples_processed", {
                "content_type": f"eq.{content_type}",
                "language":     f"eq.{lang}",
                "hf_pushed":    "eq.false",
            })
            log.info(f"  {content_type}/{lang}: {count:,} unpushed (threshold {SHARD_THRESHOLD:,})")

            if count < SHARD_THRESHOLD:
                continue

            # Fetch the shard
            rows = sb_get("samples_processed", {
                "select": "*",
                "content_type": f"eq.{content_type}",
                "language":     f"eq.{lang}",
                "hf_pushed":    "eq.false",
                "order":        "labeled_at.asc",
                "limit":        str(SHARD_THRESHOLD),
            }, timeout=60)

            if not rows:
                continue

            log.info(f"  Pushing {len(rows):,} rows as {content_type}/{lang} shard...")

            # Build Parquet
            try:
                df = pd.DataFrame(rows)
                keep = [
                    "sample_id", "source_id", "source_url", "content_type",
                    "language", "raw_content", "label", "final_confidence",
                    "model_scores", "verified", "scraped_at", "labeled_at", "metadata",
                ]
                df = df[[c for c in keep if c in df.columns]]
                for col in ["metadata", "model_scores"]:
                    if col in df.columns:
                        df[col] = df[col].apply(
                            lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x or "")
                        )
                buf = io.BytesIO()
                df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
                parquet_bytes = buf.getvalue()
            except Exception as e:
                log.error(f"  Parquet build failed: {e}")
                continue

            # Find next shard number (use timestamp-based name to avoid collisions)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            shard_path = f"data/{content_type}/{lang}/shard_{ts}.parquet"
            sha256 = hashlib.sha256(parquet_bytes).hexdigest()[:16]

            # Push to HF
            try:
                api.create_commit(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    commit_message=f"[shard] {content_type}/{lang} — {len(rows):,} samples",
                    operations=[CommitOperationAdd(shard_path, parquet_bytes)],
                )
                log.info(f"  ✅ Pushed: {shard_path} ({len(parquet_bytes)//1024}KB)")
            except Exception as e:
                log.error(f"  HF push failed: {e}")
                continue

            # Mark pushed in DB
            sample_ids = [r["sample_id"] for r in rows]
            for i in range(0, len(sample_ids), 500):
                batch = sample_ids[i:i+500]
                sb_patch("samples_processed",
                    {"sample_id": f"in.({','.join(batch)})"},
                    {"hf_pushed": True}
                )

            # Register shard — use ACTUAL shard_registry column names
            sb_post("shard_registry", {
                "shard_id":          f"{content_type}-{lang}-{ts}",
                "content_type":      content_type,
                "language":          lang,
                "sample_count":      len(rows),          # ← correct column name
                "size_bytes":        len(parquet_bytes),  # ← correct column name
                "sha256_hash":       sha256,              # ← correct column name
                "hf_url":            f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{shard_path}",
                "push_status":       "pushed",            # ← correct column name
                "source_distribution": {},
            })

            # Delete from Supabase (buffer cleared)
            for i in range(0, len(sample_ids), 500):
                batch = sample_ids[i:i+500]
                sb_delete("samples_processed",
                    {"sample_id": f"in.({','.join(batch)})"})
            log.info(f"  ✅ Deleted {len(rows):,} rows from Supabase (pushed to HF)")

            pushed += 1
            time.sleep(3)  # brief pause between shards

    return pushed


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    log.info("=== DETECT-AI Pipeline Runner ===")
    log.info(f"  Supabase: {SUPABASE_URL}")
    log.info(f"  HF Repo:  {HF_REPO}")
    log.info(f"  Trusted sources: {len(TRUSTED_SOURCES)}")
    log.info(f"  Shard threshold: {SHARD_THRESHOLD:,}")

    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        db = SupabaseClient(SUPABASE_URL, SUPABASE_KEY, session)

        # Step 1: Label a batch
        labeled = await label_batch(db, session)
        log.info(f"Labeled: {labeled} samples")

    # Step 2: ALWAYS check if we should push to HF (not gated on labeled > 0)
    log.info("Checking HF push threshold...")
    shards = push_shards_to_hf()

    if shards > 0:
        log.info(f"✅ Pushed {shards} shard(s) to HuggingFace!")
    else:
        log.info("No shards pushed this run (below threshold or no HF token)")

    print(f"DONE: labeled={labeled} shards_pushed={shards}")


asyncio.run(main())
