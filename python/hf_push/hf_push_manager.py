"""
DETECT-AI — HuggingFace Push Manager v2 (fixed)
Fixed: backoff/retry flow, token validation, safe JSON handling,
       transactional marking & deletion, and improved logging.
"""
import os
import io
import json
import time
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import pandas as pd
import requests
from huggingface_hub import HfApi, CommitOperationAdd

log = logging.getLogger("detect-ai.hf-push")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

HF_TOKEN     = os.environ.get("HF_TOKEN")
REPO_ID      = os.environ.get("HF_DATASET_REPO", "anas775/DETECT-AI-Dataset")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

# ── Rate limit config (safe for 2M/day) ─────────────────────────
SHARD_SIZE           = 200_000
MAX_COMMITS_PER_HR   = 40
MIN_SECONDS_BETWEEN  = 5
MAX_FILES_PER_COMMIT = 50
BACKOFF_BASE         = 30
BACKOFF_MAX          = 300

SB_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=minimal",
}


class HFPushManager:
    def __init__(self, token: str = HF_TOKEN, repo_id: str = REPO_ID):
        # Validate token/repo early to avoid cryptic failures later
        if not token:
            raise ValueError("HF_TOKEN environment variable not set or empty.")
        if not repo_id:
            raise ValueError("HF_DATASET_REPO environment variable not set or empty.")

        self.api = HfApi(token=token)
        self.repo_id = repo_id
        self._last_commit_time = 0.0
        self._commits_this_hour = 0
        self._hour_start = time.time()
        log.info(f"HFPushManager initialized for {self.repo_id}")

    # ── Rate limiter ──────────────────────────────────────────────
    def _wait_if_needed(self):
        now = time.time()
        if now - self._hour_start >= 3600:
            self._commits_this_hour = 0
            self._hour_start = now

        if self._commits_this_hour >= MAX_COMMITS_PER_HR:
            wait = 3600 - (now - self._hour_start) + 10
            log.warning(f"HF hourly limit reached ({MAX_COMMITS_PER_HR}) — waiting {wait:.0f}s")
            time.sleep(wait)
            self._commits_this_hour = 0
            self._hour_start = time.time()

        elapsed = time.time() - self._last_commit_time
        if elapsed < MIN_SECONDS_BETWEEN:
            time.sleep(MIN_SECONDS_BETWEEN - elapsed)

    def _safe_commit(self, message: str, operations: List[CommitOperationAdd], retries: int = 5) -> bool:
        """
        Commit with exponential backoff on errors and proper retry handling.
        Returns True on success, False on unrecoverable error.
        """
        self._wait_if_needed()

        for attempt in range(1, retries + 1):
            try:
                # Create commit using the HfApi instance
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    commit_message=message,
                    operations=operations,
                )
                self._last_commit_time = time.time()
                self._commits_this_hour += 1
                log.info(f"HF commit succeeded (attempt {attempt}): {message}")
                return True

            except Exception as e:
                err = str(e)
                if "401" in err or "403" in err or "Invalid token" in err:
                    log.error(f"Authentication failed when committing to HF: {err}")
                    return False

                if "429" in err or "rate" in err.lower():
                    wait = min(BACKOFF_BASE * (2 ** (attempt - 1)), BACKOFF_MAX)
                    log.warning(f"HF rate limited (attempt {attempt}) — waiting {wait}s")
                    time.sleep(wait)
                    continue

                if any(code in err for code in ("500", "502", "503", "504")):
                    wait = min(30 * attempt, 120)
                    log.warning(f"HF server error (attempt {attempt}) — waiting {wait}s: {err[:200]}")
                    time.sleep(wait)
                    continue

                log.error(f"HF commit failed (attempt {attempt}): {err[:300]}")
                if attempt == retries:
                    return False
                time.sleep(10)
                continue

        log.error("HF commit failed after retries.")
        return False

    # ── Main: Push pending Parquet shards ─────────────────────────
    def push_pending_shards(self) -> int:
        if not SUPABASE_URL or not SUPABASE_KEY:
            log.error("SUPABASE_URL / SUPABASE_SERVICE_KEY not set")
            return 0

        pushed_shards = 0

        for content_type in ["text", "image", "video", "audio"]:
            for language in self._get_active_languages(content_type):
                count = self._count_ready(content_type, language)
                if count < SHARD_SIZE:
                    continue

                rows = self._fetch_rows(content_type, language, SHARD_SIZE)
                if not rows:
                    continue

                shard_num = self._next_shard_num(content_type, language)
                success, operations, message, sample_ids, row_count = self._build_shard_operations(rows, content_type, language, shard_num)
                if not success:
                    log.error("Failed to build shard operations; skipping")
                    continue

                if not self._safe_commit(message, operations):
                    log.error("Commit failed; not marking samples as pushed")
                    continue

                try:
                    self._mark_pushed(sample_ids, content_type, language, shard_num)
                except Exception as e:
                    log.error(f"Failed to mark samples as pushed after commit: {e}")
                    continue

                try:
                    self._delete_pushed_from_supabase(sample_ids)
                except Exception as e:
                    log.warning(f"Failed to delete pushed rows from Supabase (non-fatal): {e}")

                self._register_shard(content_type, language, shard_num, row_count)
                pushed_shards += 1
                log.info(f"Shard pushed: {content_type}/{language}/part-{shard_num:04d} ({row_count:,} rows)")

        log.info(f"Push cycle complete: {pushed_shards} shards pushed this run")
        return pushed_shards

    def _build_shard_operations(self, rows, content_type: str, language: str, shard_num: int):
        """
        Prepare parquet bytes and commit operations.
        Returns (success(bool), operations(list), message(str), sample_ids(list), row_count(int))
        """
        try:
            df = pd.DataFrame(rows)
            keep = [
                "sample_id", "source_id", "source_url", "content_type", "language",
                "raw_content", "label", "final_confidence", "model_scores",
                "verified", "scraped_at", "labeled_at", "metadata"
            ]
            df = df[[c for c in keep if c in df.columns]]

            for col in ["metadata", "model_scores"]:
                if col in df.columns:
                    def _safe_serialize(x):
                        if x is None:
                            return ""
                        if isinstance(x, str):
                            return x
                        if isinstance(x, (dict, list)):
                            return json.dumps(x, ensure_ascii=False)
                        return str(x)
                    df[col] = df[col].apply(_safe_serialize)

            buf = io.BytesIO()
            df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
            parquet_bytes = buf.getvalue()

            shard_path = f"{content_type}/{language}/part-{shard_num:04d}.parquet"
            message = (
                f"[shard] {content_type}/{language}/part-{shard_num:04d} "
                f"({len(df):,} samples, {len(parquet_bytes)//1024}KB)"
            )

            ops = [CommitOperationAdd(shard_path, parquet_bytes)]
            sample_ids = [r["sample_id"] for r in rows if r.get("sample_id")]
            return True, ops, message, sample_ids, len(df)
        except Exception as e:
            log.error(f"Parquet/operation build failed: {e}")
            return False, [], "", [], 0

    def _get_active_languages(self, content_type: str) -> list:
        try:
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
        except Exception as e:
            log.warning(f"_get_active_languages failed: {e}")
            return ["en"]

    def _count_ready(self, content_type: str, language: str) -> int:
        try:
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
        except Exception as e:
            log.warning(f"_count_ready failed: {e}")
            return 0

    def _fetch_rows(self, content_type: str, language: str, limit: int) -> list:
        try:
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
        except Exception as e:
            log.warning(f"_fetch_rows failed: {e}")
            return []

    def _next_shard_num(self, content_type: str, language: str) -> int:
        try:
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
        except Exception as e:
            log.warning(f"_next_shard_num failed: {e}")
            return 0

    def _mark_pushed(self, sample_ids: list, content_type: str, language: str, shard_num: int):
        """
        Mark samples as HF-pushed in batches. Raise on failure to avoid deleting rows
        if marking did not succeed.
        """
        if not sample_ids:
            return
        for i in range(0, len(sample_ids), 500):
            batch = sample_ids[i:i+500]
            try:
                params = {"sample_id": f"in.({','.join(batch)})"}
                r = requests.patch(
                    f"{SUPABASE_URL}/rest/v1/samples_processed",
                    headers=SB_HEADERS,
                    params=params,
                    json={"hf_pushed": True, "hf_shard": shard_num, "status": "pushed"},
                    timeout=20
                )
                r.raise_for_status()
                log.info(f"Marked {len(batch)} samples as pushed (shard {shard_num})")
            except Exception as e:
                log.error(f"Failed to mark batch as pushed: {e}")
                raise

    def _delete_pushed_from_supabase(self, sample_ids: list):
        """Delete rows from samples_processed after confirmed HF push. Non-fatal on failure."""
        if not sample_ids:
            return
        for i in range(0, len(sample_ids), 500):
            batch = sample_ids[i:i+500]
            try:
                id_list = ",".join(f'"{sid}"' for sid in batch)
                r = requests.delete(
                    f"{SUPABASE_URL}/rest/v1/samples_processed",
                    headers=SB_HEADERS,
                    params={"sample_id": f"in.({id_list})"},
                    timeout=20
                )
                r.raise_for_status()
                log.info(f"Deleted {len(batch)} pushed samples from Supabase")
            except Exception as e:
                log.warning(f"Cleanup delete failed for batch: {e}")
                # continue to next batch but do not raise

    def _register_shard(self, content_type: str, language: str, shard_num: int, row_count: int):
        hf_path = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{content_type}/{language}/part-{shard_num:04d}.parquet"
        try:
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
            log.info(f"Registered shard {shard_num} for {content_type}/{language}")
        except Exception as e:
            log.warning(f"Failed to register shard in Supabase: {e}")

    # ── Push video frames ─────────────────────────────────────────
    def push_frames(self, video_id: str, language: str, frame_data: list) -> bool:
        all_ops = []
        lang = language[:2] if language else "en"

        for fd in frame_data:
            idx = str(fd["frame_index"]).zfill(5)
            if fd.get("full_bytes"):
                all_ops.append(CommitOperationAdd(f"video/{lang}/frames/{video_id}/full/frame_{idx}.png", fd["full_bytes"]))
            for fi, fb in enumerate(fd.get("face_bytes") or []):
                all_ops.append(CommitOperationAdd(f"video/{lang}/frames/{video_id}/faces/face_{idx}_{fi:02d}.jpg", fb))
            for mi, mb in enumerate(fd.get("mask_bytes") or []):
                all_ops.append(CommitOperationAdd(f"video/{lang}/frames/{video_id}/textures/mask_{idx}_{mi:02d}.png", mb))

        if not all_ops:
            return True

        total_batches = (len(all_ops) + MAX_FILES_PER_COMMIT - 1) // MAX_FILES_PER_COMMIT
        success_count = 0
        for i in range(0, len(all_ops), MAX_FILES_PER_COMMIT):
            batch_num = i // MAX_FILES_PER_COMMIT + 1
            batch = all_ops[i:i + MAX_FILES_PER_COMMIT]
            message = f"[frames] {video_id} batch {batch_num}/{total_batches} ({len(batch)} files)"
            if self._safe_commit(message, batch):
                success_count += 1

        log.info(f"Frames pushed: {success_count}/{total_batches} batches for {video_id}")
        return success_count == total_batches


if __name__ == "__main__":
    manager = HFPushManager()
    shards = manager.push_pending_shards()
    print(f"✅ {shards} shards pushed to HF")
