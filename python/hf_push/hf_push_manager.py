"""
DETECT-AI — HuggingFace Push Manager v2
========================================
Handles pushing Parquet shards to anas775/DETECT-AI-Dataset
with correct rate limiting for 2M+ samples/day scale.

HF Rate Limits (from official docs):
  - Max ~50 commits/hour per user (practical limit)
  - No hard limit on file size per commit
  - Recommended: large files via LFS, batch ≤100 files/commit

Our strategy for 2M samples/day:
  - Buffer samples in Supabase until 200k rows (1 shard)
  - Push 1 Parquet shard = 1 commit
  - 2M samples/day = 10 shards/day = 10 commits/day → 120x UNDER limit
  - Push frames in batches of 50 files per commit (well under limit)
  - Minimum 5 seconds between commits
  - Exponential backoff on 429/500 errors
"""

import os
import io
import json
import time
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from huggingface_hub import HfApi, CommitOperationAdd

log = logging.getLogger("detect-ai.hf-push")

HF_TOKEN     = os.environ.get("HF_TOKEN")
REPO_ID      = os.environ.get("HF_DATASET_REPO", "anas775/DETECT-AI-Dataset")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

# ── Rate limit config (safe for 2M/day) ─────────────────────────
SHARD_SIZE          = 200_000   # rows per Parquet shard
MAX_COMMITS_PER_HR  = 40        # Well under 50/hr limit (20% safety margin)
MIN_SECONDS_BETWEEN = 5         # Min gap between commits
MAX_FILES_PER_COMMIT= 50        # Well under HF's ~100 file recommendation
BACKOFF_BASE        = 30        # Seconds for first retry
BACKOFF_MAX         = 300       # Max 5 min backoff

SB_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=minimal",
}


class HFPushManager:
    def __init__(self, token: str = HF_TOKEN, repo_id: str = REPO_ID):
        self.api     = HfApi(token=token)
        self.repo_id = repo_id
        self._last_commit_time = 0.0
        self._commits_this_hour = 0
        self._hour_start = time.time()

    # ── Rate limiter ──────────────────────────────────────────────
    def _wait_if_needed(self):
        """Enforce rate limits before any commit."""
        now = time.time()

        # Reset hourly counter
        if now - self._hour_start >= 3600:
            self._commits_this_hour = 0
            self._hour_start = now

        # Wait if at hourly limit
        if self._commits_this_hour >= MAX_COMMITS_PER_HR:
            wait = 3600 - (now - self._hour_start) + 10
            log.warning(f"  HF hourly limit ({MAX_COMMITS_PER_HR}) reached — waiting {wait:.0f}s")
            time.sleep(wait)
            self._commits_this_hour = 0
            self._hour_start = time.time()

        # Minimum gap between commits
        elapsed = time.time() - self._last_commit_time
        if elapsed < MIN_SECONDS_BETWEEN:
            time.sleep(MIN_SECONDS_BETWEEN - elapsed)

    def _safe_commit(self, message: str, operations: list, retries: int = 5) -> bool:
        """Commit with exponential backoff on errors."""
        self._wait_if_needed()

        for attempt in range(1, retries + 1):
            try:
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=HF_TOKEN,
                    commit_message=message,
                    operations=operations,
                )
                self._last_commit_time = time.time()
                self._commits_this_hour += 1
                return True
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    wait = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)
                    log.warning(f"  HF rate limited (attempt {attempt}) — waiting {wait}s")
                    time.sleep(wait)
                elif "500" in err or "502" in err or "503" in err:
                    wait = min(30 * attempt, 120)
                    log.warning(f"  HF server error (attempt {attempt}) — waiting {wait}s: {err[:100]}")
                    time.sleep(wait)
                else:
                    log.error(f"  HF commit failed (attempt {attempt}): {err[:200]}")
                    if attempt == retries:
                        return False
                    time.sleep(10)
        return False

    # ── Main: Push pending Parquet shards ─────────────────────────
    def push_pending_shards(self) -> int:
        """
        Pull ready-to-shard samples from Supabase → Parquet → HF.
        Returns number of shards pushed.
        """
        if not SUPABASE_URL or not SUPABASE_KEY:
            log.error("SUPABASE_URL / SUPABASE_SERVICE_KEY not set")
            return 0

        pushed_shards = 0

        for content_type in ["text", "image", "video", "audio"]:
            for language in self._get_active_languages(content_type):
                count = self._count_ready(content_type, language)
                if count < SHARD_SIZE:
                    continue

                # Pull up to SHARD_SIZE rows
                rows = self._fetch_rows(content_type, language, SHARD_SIZE)
                if not rows:
                    continue

                shard_num = self._next_shard_num(content_type, language)
                success = self._push_parquet_shard(rows, content_type, language, shard_num)

                if success:
                    sample_ids = [r["sample_id"] for r in rows]
                    self._mark_pushed(sample_ids, content_type, language, shard_num)
                    self._register_shard(content_type, language, shard_num, len(rows))
                    pushed_shards += 1
                    log.info(f"  ✅ Shard pushed: {content_type}/{language}/part-{shard_num:04d}.parquet ({len(rows):,} rows)")

        log.info(f"Push cycle complete: {pushed_shards} shards pushed this run")
        return pushed_shards

    def _get_active_languages(self, content_type: str) -> list[str]:
        """Get languages that have samples waiting to be pushed."""
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/samples_processed",
            headers=SB_HEADERS,
            params={
                "select": "language",
                "content_type": f"eq.{content_type}",
                "status": "eq.labeled",
                "group": "language",
            },
            timeout=15
        )
        if not r.ok:
            return ["en"]
        return list({row["language"] for row in r.json() if row.get("language")}) or ["en"]

    def _count_ready(self, content_type: str, language: str) -> int:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/samples_processed",
            headers={**SB_HEADERS, "Prefer": "count=exact", "Range": "0-0"},
            params={
                "content_type": f"eq.{content_type}",
                "language":     f"eq.{language}",
                "status":       "eq.labeled",
                "hf_pushed":    "eq.false",
            },
            timeout=15
        )
        if not r.ok:
            return 0
        return int(r.headers.get("Content-Range", "0-0/0").split("/")[-1])

    def _fetch_rows(self, content_type: str, language: str, limit: int) -> list[dict]:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/samples_processed",
            headers=SB_HEADERS,
            params={
                "select": "*",
                "content_type": f"eq.{content_type}",
                "language":     f"eq.{language}",
                "status":       "eq.labeled",
                "hf_pushed":    "eq.false",
                "order":        "labeled_at.asc",
                "limit":        limit,
            },
            timeout=30
        )
        return r.json() if r.ok else []

    def _next_shard_num(self, content_type: str, language: str) -> int:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/shard_registry",
            headers=SB_HEADERS,
            params={
                "select": "shard_index",
                "content_type": f"eq.{content_type}",
                "language":     f"eq.{language}",
                "order":        "shard_index.desc",
                "limit":        1,
            },
            timeout=10
        )
        rows = r.json() if r.ok else []
        return (rows[0]["shard_index"] + 1) if rows else 0

    def _push_parquet_shard(
        self, rows: list[dict], content_type: str, language: str, shard_num: int
    ) -> bool:
        """Convert rows to Parquet and push to HF."""
        try:
            df = pd.DataFrame(rows)
            # Keep only HF-relevant columns
            keep = [
                "sample_id", "source_id", "source_url", "content_type", "language",
                "raw_content", "label", "final_confidence", "model_scores",
                "verified", "scraped_at", "labeled_at", "metadata"
            ]
            df = df[[c for c in keep if c in df.columns]]

            # Serialize metadata + model_scores as strings
            for col in ["metadata", "model_scores"]:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x or "")
                    )

            buf = io.BytesIO()
            df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
            parquet_bytes = buf.getvalue()

            shard_path = f"{content_type}/{language}/part-{shard_num:04d}.parquet"
            sha256 = hashlib.sha256(parquet_bytes).hexdigest()

            operations = [CommitOperationAdd(shard_path, parquet_bytes)]
            message = (
                f"[shard] {content_type}/{language}/part-{shard_num:04d} "
                f"({len(rows):,} samples, {len(parquet_bytes)//1024}KB)"
            )
            return self._safe_commit(message, operations)

        except Exception as e:
            log.error(f"  Parquet build failed: {e}")
            return False

    def _mark_pushed(self, sample_ids: list[str], content_type: str, language: str, shard_num: int):
        """Mark samples as HF-pushed in batches of 500."""
        for i in range(0, len(sample_ids), 500):
            batch = sample_ids[i:i+500]
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/samples_processed",
                headers=SB_HEADERS,
                params={"sample_id": f"in.({','.join(batch)})"},
                json={"hf_pushed": True, "hf_shard": shard_num, "status": "pushed"},
                timeout=20
            )

    def _register_shard(self, content_type: str, language: str, shard_num: int, row_count: int):
        """Record shard in registry."""
        hf_path = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{content_type}/{language}/part-{shard_num:04d}.parquet"
        requests.post(
            f"{SUPABASE_URL}/rest/v1/shard_registry",
            headers=SB_HEADERS,
            json={
                "shard_id":     f"{content_type}-{language}-{shard_num:04d}",
                "content_type": content_type,
                "language":     language,
                "shard_index":  shard_num,
                "row_count":    row_count,
                "hf_url":       hf_path,
                "pushed_at":    datetime.now(timezone.utc).isoformat(),
            },
            timeout=10
        )

    # ── Push video frames ─────────────────────────────────────────
    def push_frames(self, video_id: str, language: str, frame_data: list[dict]) -> bool:
        """
        Push extracted frames/faces to HF in batches of MAX_FILES_PER_COMMIT.
        Each batch = 1 commit. 50 files/commit, 5s gap = safe for any volume.
        """
        all_ops = []
        lang = language[:2] if language else "en"

        for fd in frame_data:
            idx = str(fd["frame_index"]).zfill(5)
            if fd.get("full_bytes"):
                all_ops.append(CommitOperationAdd(
                    f"video/{lang}/frames/{video_id}/full/frame_{idx}.png",
                    fd["full_bytes"]
                ))
            for fi, fb in enumerate(fd.get("face_bytes") or []):
                all_ops.append(CommitOperationAdd(
                    f"video/{lang}/frames/{video_id}/faces/face_{idx}_{fi:02d}.jpg",
                    fb
                ))
            for mi, mb in enumerate(fd.get("mask_bytes") or []):
                all_ops.append(CommitOperationAdd(
                    f"video/{lang}/frames/{video_id}/textures/mask_{idx}_{mi:02d}.png",
                    mb
                ))

        # Push in batches of MAX_FILES_PER_COMMIT
        total_batches = (len(all_ops) + MAX_FILES_PER_COMMIT - 1) // MAX_FILES_PER_COMMIT
        success_count = 0
        for i in range(0, len(all_ops), MAX_FILES_PER_COMMIT):
            batch_num = i // MAX_FILES_PER_COMMIT + 1
            batch = all_ops[i:i + MAX_FILES_PER_COMMIT]
            message = f"[frames] {video_id} batch {batch_num}/{total_batches} ({len(batch)} files)"
            if self._safe_commit(message, batch):
                success_count += 1

        log.info(f"  Frames pushed: {success_count}/{total_batches} batches for {video_id}")
        return success_count == total_batches


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    manager = HFPushManager()
    shards = manager.push_pending_shards()
    print(f"✅ {shards} shards pushed to HF")
