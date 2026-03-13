"""
============================================================
DETECT-AI: Hugging Face Push Manager
============================================================
Handles ALL pushes to Hugging Face with:

  THROTTLE PROTECTION:
  - Token bucket rate limiter (max 50 commits/hour)
  - Exponential backoff on 429 / 503 errors
  - Max 100 files per commit (HF hard recommendation)
  - 5-second minimum gap between commits
  - Automatic retry up to 5 times per shard

  FOLDER STRUCTURE:
  anas775/DETECT-AI-Dataset/
  ├── text/{lang}/part-{NNNN}.parquet
  ├── image/{lang}/metadata/part-{NNNN}.parquet
  ├── image/{lang}/frames/{image_id}.jpg
  ├── video/{lang}/metadata/part-{NNNN}.parquet
  ├── video/{lang}/frames/{video_id}/full/frame_{N}.png
  ├── video/{lang}/frames/{video_id}/faces/face_{N}.jpg
  ├── audio/{lang}/metadata/part-{NNNN}.parquet
  └── _metadata/
      ├── shard_registry.json
      ├── push_log.jsonl
      └── schema_v1.json

  SHARD RULES (per HF repo limits):
  - Max 100k files per repo → split by content_type+lang
  - Max 5GB per file → parquet shards capped at 500k rows
  - Max 100 files per commit → batch upload in chunks of 50
  - Parquet compressed with snappy (~0.7x ratio)
============================================================
"""

import os
import time
import json
import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import (
    HfApi,
    CommitOperationAdd,
    create_repo,
)
from huggingface_hub.utils import HfHubHTTPError

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.INFO,
)
log = logging.getLogger("detect-ai.hf-push")

# ── Configuration ─────────────────────────────────────────────
HF_REPO_ID          = os.environ.get("HF_DATASET_REPO", "anas775/DETECT-AI-Dataset")
HF_TOKEN            = os.environ.get("HF_TOKEN", "")
SUPABASE_URL        = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY        = os.environ.get("SUPABASE_SERVICE_KEY", "")

SCHEMA_VERSION      = "1.0"
SHARD_ROW_LIMIT     = 200_000       # Rows per parquet shard
MAX_FILES_PER_COMMIT = 50           # Well under HF's 100-file recommendation
MIN_COMMIT_GAP_SEC  = 5            # Minimum seconds between commits
MAX_COMMITS_PER_HR  = 50           # Token bucket capacity
MAX_RETRY_ATTEMPTS  = 5
BASE_BACKOFF_SEC    = 10           # Base for exponential backoff

# ── 90+ Languages supported ───────────────────────────────────
SUPPORTED_LANGUAGES = [
    "en","ar","fr","de","es","zh","ja","ko","pt","ru","hi","it","nl","pl",
    "sv","tr","vi","fa","uk","he","cs","ro","hu","el","da","fi","no","sk",
    "hr","bg","sr","lt","lv","et","sl","sq","mk","bs","ca","eu","gl","cy",
    "ga","is","mt","af","sw","ms","id","tl","th","my","km","lo","ka","hy",
    "az","kk","uz","tk","ky","tg","mn","si","ne","ur","bn","pa","gu","mr",
    "ta","te","kn","ml","am","yo","ig","ha","zu","xh","sn","st","tn","so",
    "rw","mg","ln","wo","ff","bm","tw","ee","ak","ny","lg","kg","ki","sg",
]

# ── Token Bucket Rate Limiter ─────────────────────────────────
class TokenBucketLimiter:
    """
    Limits commits to MAX_COMMITS_PER_HR per hour.
    Thread-safe. Blocks until a token is available.
    """
    def __init__(self, capacity: int = MAX_COMMITS_PER_HR, refill_period_sec: int = 3600):
        self.capacity         = capacity
        self.tokens           = capacity
        self.refill_period    = refill_period_sec
        self.refill_rate      = capacity / refill_period_sec  # tokens/sec
        self.last_refill_time = time.monotonic()
        self._lock            = threading.Lock()

    def _refill(self):
        now     = time.monotonic()
        elapsed = now - self.last_refill_time
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill_time = now

    def acquire(self, timeout_sec: int = 3600) -> bool:
        """Block until a token is available. Returns True if acquired."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
            wait_sec = 1.0 / self.refill_rate
            log.info(f"[THROTTLE] Rate limit reached. Waiting {wait_sec:.1f}s for token...")
            time.sleep(min(wait_sec, 60))
        return False

# ── Shard Record ──────────────────────────────────────────────
@dataclass
class ShardRecord:
    shard_id:            str
    content_type:        str
    language:            str
    sample_count:        int
    size_bytes:          int
    sha256_hash:         str
    created_at:          str
    schema_version:      str = SCHEMA_VERSION
    hf_path:             str = ""
    push_status:         str = "pending"   # pending | pushed | failed
    source_distribution: dict = field(default_factory=dict)
    push_attempts:       int = 0
    last_error:          str = ""

# ── HF Push Manager ───────────────────────────────────────────
class HFPushManager:
    """
    Manages all Hugging Face dataset pushes for DETECT-AI.
    Thread-safe, throttle-protected, retry-enabled.
    """

    def __init__(self):
        self.api       = HfApi(token=HF_TOKEN)
        self.limiter   = TokenBucketLimiter()
        self.push_log  = []
        self._lock     = threading.Lock()
        self._last_commit_time = 0.0

        # Local shard staging directory
        self.staging_dir = Path("/tmp/detect-ai-shards")
        self.staging_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"HFPushManager initialized → repo: {HF_REPO_ID}")

    # ── Ensure repo exists ────────────────────────────────────
    def ensure_repo(self):
        """Create HF dataset repo if it doesn't exist."""
        try:
            create_repo(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                private=False,
                exist_ok=True,
                token=HF_TOKEN,
            )
            log.info(f"✅ Dataset repo ready: https://huggingface.co/datasets/{HF_REPO_ID}")
        except Exception as e:
            log.error(f"Failed to create/verify repo: {e}")
            raise

    # ── HF path builder ───────────────────────────────────────
    @staticmethod
    def build_hf_path(content_type: str, language: str, shard_id: str, subpath: str = "") -> str:
        """
        Build the correct HF repo path for a given shard.

        text   → text/{lang}/part-{NNNN}.parquet
        image  → image/{lang}/metadata/part-{NNNN}.parquet
        video  → video/{lang}/metadata/part-{NNNN}.parquet
        audio  → audio/{lang}/metadata/part-{NNNN}.parquet
        frame  → video/{lang}/frames/{video_id}/full/frame_{N}.png
        face   → video/{lang}/frames/{video_id}/faces/face_{N}.jpg
        """
        lang = language if language in SUPPORTED_LANGUAGES else "unknown"

        if subpath:
            return f"{content_type}/{lang}/{subpath}"

        if content_type == "text":
            return f"text/{lang}/{shard_id}.parquet"
        else:
            return f"{content_type}/{lang}/metadata/{shard_id}.parquet"

    # ── Build frame path ──────────────────────────────────────
    @staticmethod
    def build_frame_path(language: str, video_id: str, frame_index: int,
                         face: bool = False, texture: bool = False) -> str:
        """
        video/{lang}/frames/{video_id}/full/frame_{NNNNN}.png
        video/{lang}/frames/{video_id}/faces/face_{NNNNN}.jpg
        video/{lang}/frames/{video_id}/textures/mask_{NNNNN}.png
        """
        lang = language if language in SUPPORTED_LANGUAGES else "unknown"
        idx  = str(frame_index).zfill(5)
        if texture:
            return f"video/{lang}/frames/{video_id}/textures/mask_{idx}.png"
        elif face:
            return f"video/{lang}/frames/{video_id}/faces/face_{idx}.jpg"
        else:
            return f"video/{lang}/frames/{video_id}/full/frame_{idx}.png"

    # ── Compute SHA256 ────────────────────────────────────────
    @staticmethod
    def sha256_of_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    # ── Serialize to Parquet ──────────────────────────────────
    def to_parquet_bytes(self, df: pd.DataFrame) -> bytes:
        """Convert DataFrame to snappy-compressed Parquet bytes."""
        table = pa.Table.from_pandas(df, preserve_index=False)
        import io
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        return buf.getvalue()

    # ── Core commit with throttle + retry ─────────────────────
    def _commit_with_throttle(
        self,
        operations: list,
        commit_message: str,
        shard_id: str,
    ) -> bool:
        """
        Push a list of CommitOperationAdd to HF with:
        - Token bucket rate limiting
        - Minimum 5s gap between commits
        - Exponential backoff on errors
        - Up to MAX_RETRY_ATTEMPTS retries
        """
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            # ── Enforce minimum gap between commits ───────────
            with self._lock:
                elapsed = time.time() - self._last_commit_time
                if elapsed < MIN_COMMIT_GAP_SEC:
                    sleep_for = MIN_COMMIT_GAP_SEC - elapsed
                    time.sleep(sleep_for)

            # ── Acquire rate limit token ───────────────────────
            if not self.limiter.acquire():
                log.error(f"[{shard_id}] Could not acquire rate limit token after 1hr")
                return False

            try:
                log.info(f"[{shard_id}] Pushing {len(operations)} file(s) — attempt {attempt}/{MAX_RETRY_ATTEMPTS}")

                self.api.create_commit(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=commit_message,
                    token=HF_TOKEN,
                )

                with self._lock:
                    self._last_commit_time = time.time()

                log.info(f"[{shard_id}] ✅ Pushed successfully on attempt {attempt}")
                return True

            except HfHubHTTPError as e:
                status = e.response.status_code if e.response else 0

                if status == 429:
                    # Hard rate limit from HF — back off much longer
                    backoff = BASE_BACKOFF_SEC * (4 ** attempt)
                    log.warning(f"[{shard_id}] HF 429 rate limit. Backing off {backoff}s...")
                    time.sleep(min(backoff, 900))  # Cap at 15 min

                elif status in (500, 502, 503, 504):
                    backoff = BASE_BACKOFF_SEC * (2 ** attempt)
                    log.warning(f"[{shard_id}] HF server error {status}. Retry in {backoff}s...")
                    time.sleep(backoff)

                elif status == 401:
                    log.error(f"[{shard_id}] ❌ HF auth error — check HF_TOKEN")
                    return False

                elif status == 413:
                    log.error(f"[{shard_id}] ❌ File too large for HF commit. Reduce shard size.")
                    return False

                else:
                    backoff = BASE_BACKOFF_SEC * (2 ** attempt)
                    log.warning(f"[{shard_id}] HF error {status}: {e}. Retry in {backoff}s...")
                    time.sleep(backoff)

            except Exception as e:
                backoff = BASE_BACKOFF_SEC * (2 ** attempt)
                log.warning(f"[{shard_id}] Unexpected error: {e}. Retry in {backoff}s...")
                time.sleep(backoff)

        log.error(f"[{shard_id}] ❌ All {MAX_RETRY_ATTEMPTS} attempts failed")
        return False

    # ── Push a Parquet shard ──────────────────────────────────
    def push_shard(self, df: pd.DataFrame, shard: ShardRecord) -> bool:
        """
        Push a single Parquet shard to HF.
        Automatically batches into ≤50 file commits.
        """
        # Validate language
        if shard.language not in SUPPORTED_LANGUAGES:
            log.warning(f"Unknown language '{shard.language}' — routing to 'unknown' folder")
            shard.language = "unknown"

        # Serialize to parquet
        parquet_bytes = self.to_parquet_bytes(df)
        shard.size_bytes  = len(parquet_bytes)
        shard.sha256_hash = self.sha256_of_bytes(parquet_bytes)

        hf_path = self.build_hf_path(shard.content_type, shard.language, shard.shard_id)
        shard.hf_path = hf_path

        operations = [
            CommitOperationAdd(
                path_in_repo=hf_path,
                path_or_fileobj=parquet_bytes,
            )
        ]

        commit_msg = (
            f"[{shard.content_type}/{shard.language}] Add shard {shard.shard_id} "
            f"({shard.sample_count:,} samples, schema v{SCHEMA_VERSION})"
        )

        shard.push_attempts += 1
        success = self._commit_with_throttle(operations, commit_msg, shard.shard_id)

        shard.push_status = "pushed" if success else "failed"
        self._log_push(shard, success)

        return success

    # ── Push video frames (batched) ───────────────────────────
    def push_frames(
        self,
        video_id: str,
        language: str,
        frames: list[dict],   # [{ "frame_index": int, "full_bytes": bytes, "face_bytes": list[bytes] }]
    ) -> bool:
        """
        Push extracted video frames to HF in batched commits.
        Batches of MAX_FILES_PER_COMMIT files per commit.
        """
        all_operations = []

        for frame in frames:
            idx = frame["frame_index"]

            # Full frame (PNG lossless)
            if frame.get("full_bytes"):
                path = self.build_frame_path(language, video_id, idx, face=False)
                all_operations.append(
                    CommitOperationAdd(path_in_repo=path, path_or_fileobj=frame["full_bytes"])
                )

            # Face crops (JPEG 95%)
            for face_idx, face_bytes in enumerate(frame.get("face_bytes", [])):
                face_path = self.build_frame_path(language, video_id, idx * 100 + face_idx, face=True)
                all_operations.append(
                    CommitOperationAdd(path_in_repo=face_path, path_or_fileobj=face_bytes)
                )

            # Face texture masks
            for mask_idx, mask_bytes in enumerate(frame.get("mask_bytes", [])):
                mask_path = self.build_frame_path(language, video_id, idx * 100 + mask_idx, texture=True)
                all_operations.append(
                    CommitOperationAdd(path_in_repo=mask_path, path_or_fileobj=mask_bytes)
                )

        # ── Split into batches of MAX_FILES_PER_COMMIT ─────────
        total_ops  = len(all_operations)
        batch_size = MAX_FILES_PER_COMMIT
        all_success = True

        for i in range(0, total_ops, batch_size):
            batch    = all_operations[i : i + batch_size]
            batch_no = i // batch_size + 1
            total_batches = (total_ops + batch_size - 1) // batch_size

            commit_msg = (
                f"[video/{language}/frames/{video_id}] "
                f"Batch {batch_no}/{total_batches} — {len(batch)} files"
            )
            success = self._commit_with_throttle(batch, commit_msg, f"{video_id}-batch{batch_no}")
            if not success:
                all_success = False
                log.error(f"Frame batch {batch_no} failed for video {video_id}")

        return all_success

    # ── Push image files (batched) ────────────────────────────
    def push_images(
        self,
        language: str,
        images: list[dict],   # [{ "image_id": str, "image_bytes": bytes, "face_bytes": list[bytes] }]
    ) -> bool:
        """Push image files and face crops in batched commits."""
        all_operations = []
        lang = language if language in SUPPORTED_LANGUAGES else "unknown"

        for img in images:
            img_id = img["image_id"]

            # Full image
            ext = img.get("ext", "jpg")
            path = f"image/{lang}/frames/{img_id}.{ext}"
            all_operations.append(
                CommitOperationAdd(path_in_repo=path, path_or_fileobj=img["image_bytes"])
            )

            # Face crops
            for fi, face_bytes in enumerate(img.get("face_bytes", [])):
                face_path = f"image/{lang}/faces/{img_id}_face_{fi:02d}.jpg"
                all_operations.append(
                    CommitOperationAdd(path_in_repo=face_path, path_or_fileobj=face_bytes)
                )

        # Batch push
        all_success = True
        for i in range(0, len(all_operations), MAX_FILES_PER_COMMIT):
            batch    = all_operations[i : i + MAX_FILES_PER_COMMIT]
            batch_no = i // MAX_FILES_PER_COMMIT + 1
            success  = self._commit_with_throttle(
                batch,
                f"[image/{lang}] Image batch {batch_no} — {len(batch)} files",
                f"img-batch-{lang}-{batch_no}"
            )
            if not success:
                all_success = False
        return all_success

    # ── Push shard registry + schema metadata ────────────────
    def push_metadata(self, shard_registry: list[dict]) -> bool:
        """Push _metadata/ folder — shard registry + schema."""
        registry_bytes = json.dumps(shard_registry, indent=2).encode()
        schema = {
            "version": SCHEMA_VERSION,
            "fields": {
                "sample_id":       "UUID v4 string",
                "source_id":       "Source identifier string",
                "source_url":      "Original URL string",
                "content_type":    "text | image | video | audio",
                "language":        "ISO-639-1 language code",
                "raw_content":     "Text content or storage path string",
                "label":           "AI_GENERATED | HUMAN | UNCERTAIN",
                "final_confidence":"Float 0.0–1.0",
                "model_scores":    "Dict of {model_name: score}",
                "verified":        "Boolean",
                "scraped_at":      "ISO-8601 timestamp",
                "labeled_at":      "ISO-8601 timestamp",
            },
            "thresholds": {
                "AI_GENERATED": ">= 0.75",
                "HUMAN":        "<= 0.35",
                "UNCERTAIN":    "0.35 < x < 0.75",
            }
        }
        schema_bytes = json.dumps(schema, indent=2).encode()

        operations = [
            CommitOperationAdd("_metadata/shard_registry.json", registry_bytes),
            CommitOperationAdd("_metadata/schema_v1.json", schema_bytes),
        ]

        return self._commit_with_throttle(
            operations,
            f"[_metadata] Update shard registry — {len(shard_registry)} shards",
            "metadata-update"
        )

    # ── Push README / dataset card ────────────────────────────
    def push_readme(self) -> bool:
        readme = f"""---
license: other
task_categories:
- text-classification
- image-classification
- video-classification
language: [{", ".join(SUPPORTED_LANGUAGES[:20])}]
size_categories:
- 1B<n<10B
tags:
- ai-detection
- deepfake
- multi-modal
- multi-language
pretty_name: DETECT-AI Dataset
---

# DETECT-AI Dataset

**1B+ samples/month** — Multi-modal, multi-language AI content detection dataset.

## Folder Structure

```
{HF_REPO_ID}/
├── text/{{lang}}/part-{{NNNN}}.parquet
├── image/{{lang}}/metadata/part-{{NNNN}}.parquet
├── image/{{lang}}/frames/{{image_id}}.jpg
├── image/{{lang}}/faces/{{image_id}}_face_NN.jpg
├── video/{{lang}}/metadata/part-{{NNNN}}.parquet
├── video/{{lang}}/frames/{{video_id}}/full/frame_{{NNNNN}}.png
├── video/{{lang}}/frames/{{video_id}}/faces/face_{{NNNNN}}.jpg
├── video/{{lang}}/frames/{{video_id}}/textures/mask_{{NNNNN}}.png
├── audio/{{lang}}/metadata/part-{{NNNN}}.parquet
└── _metadata/
    ├── shard_registry.json
    └── schema_v1.json
```

## Labels
| Label | Threshold |
|---|---|
| `AI_GENERATED` | ensemble confidence ≥ 0.75 |
| `HUMAN` | ensemble confidence ≤ 0.35 |
| `UNCERTAIN` | 0.35 – 0.75 (flagged for human review) |

## Schema Version: {SCHEMA_VERSION}
Last updated: {datetime.now(timezone.utc).strftime("%Y-%m-%d")}
"""
        return self._commit_with_throttle(
            [CommitOperationAdd("README.md", readme.encode())],
            "Update dataset card",
            "readme"
        )

    # ── Internal push logger ──────────────────────────────────
    def _log_push(self, shard: ShardRecord, success: bool):
        entry = {
            "shard_id":      shard.shard_id,
            "content_type":  shard.content_type,
            "language":      shard.language,
            "sample_count":  shard.sample_count,
            "size_bytes":    shard.size_bytes,
            "sha256":        shard.sha256_hash,
            "hf_path":       shard.hf_path,
            "status":        shard.push_status,
            "attempts":      shard.push_attempts,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "success":       success,
        }
        self.push_log.append(entry)
        status_icon = "✅" if success else "❌"
        log.info(f"{status_icon} [{shard.content_type}/{shard.language}] {shard.shard_id} "
                 f"— {shard.sample_count:,} rows, {shard.size_bytes/1024/1024:.1f}MB")

    # ── Full push pipeline: Supabase → Parquet → HF ──────────
    def run_shard_export(self, content_type: str, language: str, shard_number: int) -> bool:
        """
        Pull processed samples from Supabase → serialize to Parquet → push to HF.
        Called by the Python labeling worker when threshold (200k) is reached.
        """
        import requests

        shard_id = f"part-{str(shard_number).zfill(4)}"
        log.info(f"Starting shard export: {content_type}/{language}/{shard_id}")

        # Query Supabase for unshareded processed samples
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/samples_processed",
            params={
                "content_type": f"eq.{content_type}",
                "language":     f"eq.{language}",
                "shard_id":     "is.null",
                "select":       "sample_id,source_id,source_url,content_type,language,"
                                "raw_content,metadata,scraped_at,label,final_confidence,"
                                "model_scores,verified,labeled_at",
                "limit":        str(SHARD_ROW_LIMIT),
                "order":        "labeled_at.asc",
            },
            headers={
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
            },
            timeout=120,
        )

        if not response.ok:
            log.error(f"Supabase query failed: {response.status_code} {response.text}")
            return False

        rows = response.json()
        if not rows:
            log.info(f"No unshareded samples for {content_type}/{language}")
            return True

        df = pd.DataFrame(rows)
        # Flatten metadata JSONB column
        if "metadata" in df.columns:
            df["metadata"] = df["metadata"].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )
        if "model_scores" in df.columns:
            df["model_scores"] = df["model_scores"].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )

        shard = ShardRecord(
            shard_id=shard_id,
            content_type=content_type,
            language=language,
            sample_count=len(df),
            size_bytes=0,
            sha256_hash="",
            created_at=datetime.now(timezone.utc).isoformat(),
            source_distribution=df["source_id"].value_counts().to_dict(),
        )

        success = self.push_shard(df, shard)

        if success:
            # Update shard_id in Supabase for all exported samples
            sample_ids = df["sample_id"].tolist()
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/samples_processed",
                params={"sample_id": f"in.({','.join(sample_ids)})"},
                json={"shard_id": shard_id, "exported_at": datetime.now(timezone.utc).isoformat()},
                headers={
                    "apikey":        SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type":  "application/json",
                },
                timeout=60,
            )

            # Register shard in Supabase shard_registry
            requests.post(
                f"{SUPABASE_URL}/rest/v1/shard_registry",
                json={
                    "shard_id":            shard.shard_id,
                    "content_type":        shard.content_type,
                    "language":            shard.language,
                    "sample_count":        shard.sample_count,
                    "size_bytes":          shard.size_bytes,
                    "sha256_hash":         shard.sha256_hash,
                    "created_at":          shard.created_at,
                    "schema_version":      SCHEMA_VERSION,
                    "hf_url":              f"https://huggingface.co/datasets/{HF_REPO_ID}/blob/main/{shard.hf_path}",
                    "push_status":         "pushed",
                    "source_distribution": shard.source_distribution,
                },
                headers={
                    "apikey":        SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type":  "application/json",
                    "Prefer":        "resolution=merge-duplicates",
                },
                timeout=30,
            )

        return success


# ── CLI Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    import sys

    manager = HFPushManager()

    cmd = sys.argv[1] if len(sys.argv) > 1 else "setup"

    if cmd == "setup":
        # Initialize repo + push README + schema
        print("Setting up DETECT-AI HF dataset repo...")
        manager.ensure_repo()
        manager.push_readme()
        manager.push_metadata([])
        print(f"\n✅ Done! View at: https://huggingface.co/datasets/{HF_REPO_ID}")

    elif cmd == "export":
        # Export specific shard: python hf_push_manager.py export text en 0
        content_type = sys.argv[2]
        language     = sys.argv[3]
        shard_no     = int(sys.argv[4])
        manager.ensure_repo()
        manager.run_shard_export(content_type, language, shard_no)

    elif cmd == "test-throttle":
        # Test the rate limiter
        print("Testing token bucket limiter (will do 5 fake commits)...")
        for i in range(5):
            manager.limiter.acquire()
            print(f"Token {i+1} acquired. Tokens remaining: {manager.limiter.tokens:.1f}")
            time.sleep(1)
        print("Throttle test complete.")
