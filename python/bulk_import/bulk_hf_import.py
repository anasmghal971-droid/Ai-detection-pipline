#!/usr/bin/env python3
"""
DETECT-AI — HuggingFace Bulk Dataset Importer
===============================================
Streams millions of pre-labeled samples from public HF datasets and
writes them DIRECTLY to anas775/DETECT-AI-Dataset as Parquet shards.
Bypasses Supabase entirely — this is the 10M sample engine.

Usage (called by GitHub Actions matrix):
  python bulk_hf_import.py --source diffusiondb --offset 0 --limit 1000000
  python bulk_hf_import.py --source wikipedia   --offset 0 --limit 1000000
  python bulk_hf_import.py --source pollinations --offset 0 --limit 500000
  python bulk_hf_import.py --source laion-wit   --offset 0 --limit 500000
  python bulk_hf_import.py --source oig         --offset 0 --limit 500000
  python bulk_hf_import.py --source sd-prompts  --offset 0 --limit 500000

Sources:
  diffusiondb   → poloclub/diffusiondb 2M SD images (AI_GENERATED, conf 0.99)
  wikipedia     → laion/Wikipedia-Abstract multilingual text (HUMAN, conf 0.95)
  pollinations  → Gustavosta/Stable-Diffusion-Prompts + Pollinations AI (AI_GENERATED, conf 0.99)
  laion-wit     → laion/filtered-wit image-text pairs (HUMAN, conf 0.95)
  oig           → laion/OIG instruction text (HUMAN, conf 0.95)
  sd-prompts    → Gustavosta/Stable-Diffusion-Prompts text only (AI_GENERATED prompts)
  cc-news       → cc_news dataset human news text (HUMAN, conf 0.95)
"""

import os
import sys
import json
import uuid
import time
import hashlib
import logging
import argparse
import requests
import io
from datetime import datetime, timezone
from typing import Iterator, List, Dict, Any, Optional

import pandas as pd
from huggingface_hub import HfApi, CommitOperationAdd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bulk-import")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID  = os.environ.get("HF_DATASET_REPO", "anas775/DETECT-AI-Dataset")
WORKER_ID = "bulk-import-v1"

SHARD_SIZE = 50_000   # push every 50k rows (was 200k — now much more frequent)

HF_DS_API = "https://datasets-server.huggingface.co"
MAX_BATCH = 100       # HF datasets server max rows per request
RATE_SLEEP = 0.5      # polite delay between HF API calls

# ─── Parquet schema columns (must match samples_staging + HF dataset) ────────
COLUMNS = [
    "sample_id", "source_id", "source_url", "content_type", "language",
    "raw_content", "label", "final_confidence", "verified", "metadata",
    "scraped_at", "worker_id", "status",
]

def make_sample(source_id: str, source_url: str, content_type: str,
                language: str, raw_content: str, label: str,
                confidence: float, metadata: dict) -> dict:
    return {
        "sample_id":        str(uuid.uuid4()),
        "source_id":        source_id,
        "source_url":       source_url[:1000],
        "content_type":     content_type,
        "language":         language[:10],
        "raw_content":      str(raw_content)[:8000],
        "label":            label,
        "final_confidence": confidence,
        "verified":         label == "AI_GENERATED",
        "metadata":         json.dumps(metadata),
        "scraped_at":       datetime.now(timezone.utc).isoformat(),
        "worker_id":        WORKER_ID,
        "status":           "staged",
    }

# ─── HF Datasets Server streaming ────────────────────────────────────────────
def hf_stream_rows(dataset: str, config: str, split: str,
                   offset: int, total_limit: int) -> Iterator[dict]:
    """Stream rows from HF datasets server API, handling pagination."""
    fetched = 0
    current_offset = offset
    consecutive_errors = 0

    while fetched < total_limit:
        batch = min(MAX_BATCH, total_limit - fetched)
        url = (f"{HF_DS_API}/rows"
               f"?dataset={dataset}&config={config}&split={split}"
               f"&offset={current_offset}&length={batch}")
        try:
            r = requests.get(url, timeout=30,
                             headers={"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {})
            if r.status_code == 429:
                log.warning("  HF rate limit — sleeping 60s")
                time.sleep(60)
                continue
            if r.status_code == 404:
                log.error(f"  Dataset {dataset}/{config}/{split} not found")
                return
            if not r.ok:
                log.warning(f"  HTTP {r.status_code} — {r.text[:200]}")
                consecutive_errors += 1
                if consecutive_errors > 5:
                    log.error("  Too many errors, stopping stream")
                    return
                time.sleep(5 * consecutive_errors)
                continue

            data = r.json()
            rows = data.get("rows", [])
            if not rows:
                log.info(f"  No more rows at offset {current_offset}")
                return

            consecutive_errors = 0
            for row_obj in rows:
                yield row_obj.get("row", {})
            fetched += len(rows)
            current_offset += len(rows)

            if fetched % 10_000 == 0:
                log.info(f"  Streamed {fetched:,}/{total_limit:,} rows")

            time.sleep(RATE_SLEEP)

        except Exception as e:
            log.warning(f"  Error fetching rows: {e}")
            consecutive_errors += 1
            if consecutive_errors > 5:
                return
            time.sleep(10)

# ─── Pollinations.ai bulk generation ─────────────────────────────────────────
def pollinations_generate(prompt: str, seed: int) -> Optional[str]:
    """Generate one AI image URL via Pollinations (no rate limit, free)."""
    safe_prompt = prompt[:200].replace("\n", " ")
    return (f"https://image.pollinations.ai/prompt/{requests.utils.quote(safe_prompt)}"
            f"?width=512&height=512&model=flux&seed={seed}&nologo=true")

# ─── Source implementations ───────────────────────────────────────────────────

def stream_diffusiondb(offset: int, limit: int) -> Iterator[dict]:
    """Stream from DiffusionDB 2M subset (AI_GENERATED images)."""
    log.info(f"  Source: DiffusionDB | offset={offset:,} | limit={limit:,}")
    # Use the 2m_first_5k config for fast access, then random_5k configs
    configs = [
        ("2m_first_5k", "train"),
        ("2m_random_5k_0", "train"),
        ("2m_random_5k_1", "train"),
        ("2m_random_5k_2", "train"),
        ("2m_random_5k_3", "train"),
        ("2m_random_5k_4", "train"),
        ("2m_random_5k_5", "train"),
        ("2m_random_5k_6", "train"),
        ("2m_random_5k_7", "train"),
        ("2m_random_5k_8", "train"),
        ("2m_random_5k_9", "train"),
        ("2m_random_5k_10", "train"),
        ("2m_random_5k_11", "train"),
        ("2m_random_5k_12", "train"),
        ("2m_random_5k_13", "train"),
        ("2m_random_5k_14", "train"),
        ("2m_random_5k_15", "train"),
        ("2m_random_5k_16", "train"),
        ("2m_random_5k_17", "train"),
        ("2m_random_5k_18", "train"),
        ("2m_random_5k_19", "train"),
    ]
    yielded = 0
    for config, split in configs:
        if yielded >= limit:
            break
        chunk_limit = min(5000, limit - yielded)
        for row in hf_stream_rows("poloclub/diffusiondb", config, split,
                                   0, chunk_limit):
            if yielded >= limit:
                break
            # Row has: image (dict with 'src'), prompt, seed, step, cfg, sampler_name
            img = row.get("image", {})
            img_url = img.get("src", "") if isinstance(img, dict) else ""
            prompt = str(row.get("prompt", ""))[:400]
            if not img_url and not prompt:
                continue
            # If no direct image url, generate via pollinations using the prompt
            if not img_url and prompt:
                img_url = pollinations_generate(prompt, row.get("seed", yielded))

            yield make_sample(
                source_id="diffusiondb",
                source_url="https://huggingface.co/datasets/poloclub/diffusiondb",
                content_type="image",
                language="en",
                raw_content=img_url,
                label="AI_GENERATED",
                confidence=0.99,
                metadata={
                    "prompt": prompt[:300],
                    "seed": row.get("seed"),
                    "steps": row.get("step"),
                    "sampler": row.get("sampler_name"),
                    "cfg": row.get("cfg"),
                    "model": "stable-diffusion",
                    "license": "CC0 1.0",
                    "tags": ["diffusiondb", "stable-diffusion", "ai-generated"],
                    "is_ai_generated": True,
                },
            )
            yielded += 1

def stream_wikipedia(offset: int, limit: int) -> Iterator[dict]:
    """Stream from laion/Wikipedia-Abstract (multilingual HUMAN text)."""
    log.info(f"  Source: Wikipedia-Abstract | offset={offset:,} | limit={limit:,}")
    # This dataset has multiple language configs
    lang_map = {
        "en": "english", "fr": "french", "de": "german", "es": "spanish",
        "ar": "arabic", "zh": "chinese", "ja": "japanese", "pt": "portuguese",
        "ru": "russian", "it": "italian", "ko": "korean", "nl": "dutch",
        "pl": "polish", "tr": "turkish", "vi": "vietnamese", "hi": "hindi",
    }
    yielded = 0
    per_lang = limit // len(lang_map) + 100

    for lang_code, config_name in lang_map.items():
        if yielded >= limit:
            break
        chunk_limit = min(per_lang, limit - yielded)
        chunk_offset = offset // len(lang_map)
        for row in hf_stream_rows("laion/Wikipedia-Abstract", config_name,
                                   "train", chunk_offset, chunk_limit):
            if yielded >= limit:
                break
            title = str(row.get("title", ""))[:200]
            abstract = str(row.get("abstract", row.get("text", "")))[:3000]
            if len(abstract) < 50:
                continue
            url = row.get("url", f"https://{lang_code}.wikipedia.org/wiki/{requests.utils.quote(title)}")

            yield make_sample(
                source_id="wikipedia",
                source_url=url[:1000],
                content_type="text",
                language=lang_code,
                raw_content=f"{title}\n\n{abstract}",
                label="HUMAN",
                confidence=0.95,
                metadata={
                    "title": title,
                    "language": lang_code,
                    "license": "CC BY-SA 4.0",
                    "tags": ["wikipedia", "encyclopedia", "human-content"],
                    "is_ai_generated": False,
                },
            )
            yielded += 1

def stream_pollinations(offset: int, limit: int) -> Iterator[dict]:
    """Generate AI images using SD prompts + Pollinations FLUX API."""
    log.info(f"  Source: Pollinations AI gen | offset={offset:,} | limit={limit:,}")
    yielded = 0
    prompt_offset = offset

    for row in hf_stream_rows("Gustavosta/Stable-Diffusion-Prompts",
                               "default", "train", prompt_offset, limit * 2):
        if yielded >= limit:
            break
        prompt = str(row.get("Prompt", row.get("text", "")))[:300]
        if len(prompt) < 10:
            continue
        seed = (offset + yielded) % 999999
        img_url = pollinations_generate(prompt, seed)

        yield make_sample(
            source_id="pollinations",
            source_url="https://pollinations.ai",
            content_type="image",
            language="en",
            raw_content=img_url,
            label="AI_GENERATED",
            confidence=0.99,
            metadata={
                "prompt": prompt,
                "model": "FLUX",
                "seed": seed,
                "width": 512,
                "height": 512,
                "tags": ["pollinations", "flux", "ai-generated", "synthetic"],
                "is_ai_generated": True,
                "generation_source": "Pollinations.ai FLUX model",
            },
        )
        yielded += 1

def stream_laion_wit(offset: int, limit: int) -> Iterator[dict]:
    """Stream from laion/filtered-wit image-text pairs (HUMAN real images)."""
    log.info(f"  Source: LAION filtered-WIT | offset={offset:,} | limit={limit:,}")
    yielded = 0
    for row in hf_stream_rows("laion/filtered-wit", "default", "train",
                               offset, limit):
        if yielded >= limit:
            break
        url = str(row.get("image_url", ""))
        caption = str(row.get("caption", row.get("caption_title_and_reference_description", "")))[:2000]
        if not url or len(url) < 10:
            continue
        lang = str(row.get("language", "en"))[:5]

        yield make_sample(
            source_id="laion-wit",
            source_url=url,
            content_type="image",
            language=lang if len(lang) <= 5 else "en",
            raw_content=url,
            label="HUMAN",
            confidence=0.95,
            metadata={
                "caption": caption[:300],
                "license": "CC BY-SA 4.0",
                "tags": ["laion", "wit", "real-image", "human-content"],
                "is_ai_generated": False,
            },
        )
        yielded += 1

def stream_oig(offset: int, limit: int) -> Iterator[dict]:
    """Stream from laion/OIG instruction dataset (HUMAN text)."""
    log.info(f"  Source: LAION OIG | offset={offset:,} | limit={limit:,}")
    yielded = 0
    for row in hf_stream_rows("laion/OIG", "unified_chip2", "train",
                               offset, limit):
        if yielded >= limit:
            break
        text = str(row.get("text", ""))[:5000]
        if len(text) < 50:
            continue

        yield make_sample(
            source_id="oig-laion",
            source_url="https://huggingface.co/datasets/laion/OIG",
            content_type="text",
            language="en",
            raw_content=text,
            label="HUMAN",
            confidence=0.90,
            metadata={
                "source": "laion-OIG",
                "license": "Apache 2.0",
                "tags": ["oig", "instruction", "human-content"],
                "is_ai_generated": False,
            },
        )
        yielded += 1

def stream_sd_prompts(offset: int, limit: int) -> Iterator[dict]:
    """Stream SD prompts as text samples (AI GENERATED prompts = synthetic)."""
    log.info(f"  Source: SD Prompts text | offset={offset:,} | limit={limit:,}")
    yielded = 0
    for row in hf_stream_rows("Gustavosta/Stable-Diffusion-Prompts",
                               "default", "train", offset, limit):
        if yielded >= limit:
            break
        prompt = str(row.get("Prompt", ""))[:1000]
        if len(prompt) < 10:
            continue

        yield make_sample(
            source_id="sd-prompts",
            source_url="https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts",
            content_type="text",
            language="en",
            raw_content=prompt,
            label="AI_GENERATED",
            confidence=0.85,
            metadata={
                "source": "Gustavosta/Stable-Diffusion-Prompts",
                "license": "MIT",
                "tags": ["sd-prompts", "synthetic", "ai-generated"],
                "is_ai_generated": True,
                "note": "SD prompt text — considered AI/synthetic content",
            },
        )
        yielded += 1

def stream_cc_news(offset: int, limit: int) -> Iterator[dict]:
    """Stream from cc_news dataset (HUMAN news text, multiple langs)."""
    log.info(f"  Source: CC-News | offset={offset:,} | limit={limit:,}")
    yielded = 0
    # cc_news uses date-based configs; use a broad set
    for row in hf_stream_rows("cc_news", "plain_text", "train",
                               offset, limit):
        if yielded >= limit:
            break
        title = str(row.get("title", ""))[:200]
        text = str(row.get("text", ""))[:4000]
        domain = str(row.get("domain", ""))[:100]
        date = str(row.get("publish_date", ""))[:30]
        if len(text) < 100:
            continue

        yield make_sample(
            source_id="cc-news",
            source_url=f"https://{domain}" if domain else "https://commoncrawl.org/connect/blog/",
            content_type="text",
            language="en",
            raw_content=f"{title}\n\n{text}",
            label="HUMAN",
            confidence=0.90,
            metadata={
                "title": title,
                "domain": domain,
                "publish_date": date,
                "license": "Common Crawl ToS",
                "tags": ["cc-news", "news", "human-content"],
                "is_ai_generated": False,
            },
        )
        yielded += 1

SOURCE_MAP = {
    "diffusiondb": stream_diffusiondb,
    "wikipedia":   stream_wikipedia,
    "pollinations": stream_pollinations,
    "laion-wit":   stream_laion_wit,
    "oig":         stream_oig,
    "sd-prompts":  stream_sd_prompts,
    "cc-news":     stream_cc_news,
}

# ─── HF Push ─────────────────────────────────────────────────────────────────
def push_shard(api: HfApi, rows: List[dict], shard_num: int,
               source: str, content_type: str) -> bool:
    """Push one Parquet shard directly to HuggingFace dataset repo."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow",
                  compression="snappy")
    buf.seek(0)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = f"data/{content_type}/{source}/shard_{shard_num:06d}_{ts}.parquet"
    op = CommitOperationAdd(path_in_repo=path, path_or_fileobj=buf)

    for attempt in range(5):
        try:
            api.create_commit(
                repo_id=REPO_ID,
                repo_type="dataset",
                operations=[op],
                commit_message=f"bulk: {source} shard {shard_num} ({len(rows):,} rows)",
            )
            log.info(f"  ✅ Pushed shard {shard_num}: {len(rows):,} rows → {path}")
            return True
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 60 * (attempt + 1)
                log.warning(f"  HF rate limit — waiting {wait}s")
                time.sleep(wait)
            else:
                log.warning(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(30)
    log.error(f"  ❌ Shard {shard_num} push failed after 5 attempts")
    return False

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DETECT-AI Bulk HF Importer")
    parser.add_argument("--source",  required=True, choices=list(SOURCE_MAP.keys()))
    parser.add_argument("--offset",  type=int, default=0)
    parser.add_argument("--limit",   type=int, default=500_000)
    parser.add_argument("--shard-start", type=int, default=0,
                        help="Starting shard number (avoid collisions in parallel jobs)")
    args = parser.parse_args()

    if not HF_TOKEN:
        log.error("HF_TOKEN env var not set")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)
    stream_fn = SOURCE_MAP[args.source]

    log.info(f"=== DETECT-AI Bulk Import ===")
    log.info(f"Source:  {args.source}")
    log.info(f"Offset:  {args.offset:,}")
    log.info(f"Limit:   {args.limit:,}")
    log.info(f"Shard:   starting at {args.shard_start}")
    log.info(f"Repo:    {REPO_ID}")

    # Guess content_type from source
    ct_map = {
        "diffusiondb": "image", "pollinations": "image", "laion-wit": "image",
        "wikipedia": "text", "oig": "text", "sd-prompts": "text", "cc-news": "text",
    }
    content_type = ct_map.get(args.source, "text")

    buffer: List[dict] = []
    total_pushed = 0
    shard_num = args.shard_start

    try:
        for sample in stream_fn(args.offset, args.limit):
            buffer.append(sample)

            if len(buffer) >= SHARD_SIZE:
                ok = push_shard(api, buffer, shard_num, args.source, content_type)
                if ok:
                    total_pushed += len(buffer)
                    shard_num += 1
                buffer = []
                time.sleep(5)  # breathing room between commits

        # Push final partial shard
        if buffer:
            ok = push_shard(api, buffer, shard_num, args.source, content_type)
            if ok:
                total_pushed += len(buffer)

    except KeyboardInterrupt:
        log.info("Interrupted — pushing remaining buffer...")
        if buffer:
            push_shard(api, buffer, shard_num, args.source, content_type)

    log.info(f"=== Complete: {total_pushed:,} samples pushed to {REPO_ID} ===")

if __name__ == "__main__":
    main()
