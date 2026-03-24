"""
DETECT-AI: Frame & Face Extractor v2
=====================================
Pulls pending frame_jobs from Supabase, downloads videos,
extracts frames + faces + textures, uploads to Supabase Storage,
then pushes to HuggingFace.

Run: python3 frame_extractor.py
Cron: every 5 minutes via GitHub Actions
"""

import os
import io
import uuid
import logging
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
import mediapipe as mp
import requests

# Local analyzers (same directory)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from texture_analyzer import analyze_image
from temporal_consistency import compute_temporal_consistency, score_temporal_consistency

log = logging.getLogger("detect-ai.frame-extractor")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.INFO,
)

# ── Config ─────────────────────────────────────────────────────
SUPABASE_URL    = os.environ["SUPABASE_URL"]
SUPABASE_KEY    = os.environ["SUPABASE_SERVICE_KEY"]
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_REPO_ID      = os.environ.get("HF_DATASET_REPO", "anas775/DETECT-AI-Dataset")
STORAGE_BUCKET  = "detect-ai-frames"
JOBS_PER_CYCLE  = int(os.environ.get("FRAME_JOBS_PER_CYCLE", "2"))
MAX_VIDEO_SEC   = 120    # Max 2 min videos (GitHub Actions timeout limit)
MAX_FRAMES      = 100    # Cap frames per video
FACE_CONFIDENCE = 0.7
FACE_PADDING    = 0.15
JPEG_QUALITY    = 95
MOTION_LOW_FPS  = 1
MOTION_MID_FPS  = 2
MOTION_HIGH_FPS = 4

SB_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
}

# ── Supabase helpers ───────────────────────────────────────────
def sb_get(table: str, params: dict) -> list:
    try:
        r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}",
                         headers=SB_HEADERS, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"sb_get {table}: {e}")
        return []

def sb_post(table: str, rows: list) -> bool:
    try:
        r = requests.post(f"{SUPABASE_URL}/rest/v1/{table}",
                          headers=SB_HEADERS, json=rows, timeout=30)
        return r.ok
    except Exception as e:
        log.error(f"sb_post {table}: {e}")
        return False

def sb_patch(table: str, filters: dict, values: dict) -> bool:
    try:
        params = {k: f"eq.{v}" for k, v in filters.items()}
        r = requests.patch(f"{SUPABASE_URL}/rest/v1/{table}",
                           headers=SB_HEADERS, params=params, json=values, timeout=30)
        return r.ok
    except Exception as e:
        log.error(f"sb_patch {table}: {e}")
        return False

def sb_storage_upload(path: str, data: bytes, content_type: str) -> str:
    try:
        url = f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{path}"
        r = requests.post(url, headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": content_type,
            "x-upsert": "true",
        }, data=data, timeout=60)
        if r.ok:
            return f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{path}"
    except Exception as e:
        log.warning(f"Storage upload failed: {e}")
    return ""

# ── Video downloader ───────────────────────────────────────────
def download_video(url: str, output_path: str) -> bool:
    """Download video using yt-dlp for YouTube/TED or direct curl for others."""
    is_yt = "youtube.com" in url or "youtu.be" in url
    is_ted = "ted.com" in url

    if is_yt or is_ted:
        try:
            result = subprocess.run([
                "yt-dlp", "-x", "--audio-format", "best",
                "-f", "bestvideo[height<=480]+bestaudio/best[height<=480]",
                "--merge-output-format", "mp4",
                "--postprocessor-args", f"ffmpeg:-t {MAX_VIDEO_SEC}",
                "-o", output_path, url
            ], capture_output=True, timeout=90)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 10000
        except Exception as e:
            log.warning(f"yt-dlp failed: {e}")
            return False
    else:
        try:
            r = requests.get(url, timeout=60, stream=True)
            if not r.ok:
                return False
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            return os.path.getsize(output_path) > 10000
        except Exception as e:
            log.warning(f"Direct download failed: {e}")
            return False

# ── Face detector ──────────────────────────────────────────────
class FaceDetector:
    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=FACE_CONFIDENCE
        )

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        faces = []
        if not results.detections:
            return faces
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            pad_x = bb.width  * FACE_PADDING
            pad_y = bb.height * FACE_PADDING
            x1 = max(0, int((bb.xmin - pad_x) * w))
            y1 = max(0, int((bb.ymin - pad_y) * h))
            x2 = min(w, int((bb.xmin + bb.width  + pad_x) * w))
            y2 = min(h, int((bb.ymin + bb.height + pad_y) * h))
            crop = frame_bgr[y1:y2, x1:x2]
            faces.append({
                "x": x1, "y": y1, "w": x2-x1, "h": y2-y1,
                "confidence": round(det.score[0], 3),
                "crop": crop,
            })
        return faces

def encode_png(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return buf.tobytes() if ok else b""

def encode_jpg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""

def compute_motion(gray_prev: np.ndarray, gray_curr: np.ndarray) -> float:
    try:
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return float(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean())
    except Exception:
        return 0.3

# ── Main extraction ────────────────────────────────────────────
def extract_video(job: dict) -> dict:
    video_id  = job["video_id"]
    video_url = job["video_url"]
    language  = (job.get("language") or "en")[:2]
    fps_hint  = int(job.get("fps_target") or 2)

    result = {
        "job_id":          job["job_id"],
        "video_id":        video_id,
        "frames_extracted": 0,
        "faces_detected":  0,
        "status":          "failed",
        "error":           None,
    }

    face_det = FaceDetector()

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "video.mp4")

        log.info(f"  Downloading {video_id}...")
        if not download_video(video_url, video_path):
            result["error"] = "download_failed"
            return result

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result["error"] = "open_failed"
            return result

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / native_fps

        if duration_sec > MAX_VIDEO_SEC:
            cap.release()
            result["error"] = "too_long"
            return result

        log.info(f"  {duration_sec:.1f}s @ {native_fps:.1f}fps")

        # Sample motion
        prev_gray = None
        motion_scores = []
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if fi % int(native_fps) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    motion_scores.append(compute_motion(prev_gray, gray))
                prev_gray = gray
            fi += 1

        avg_motion = float(np.mean(motion_scores)) if motion_scores else 0.3
        target_fps = MOTION_LOW_FPS if avg_motion < 0.3 else (MOTION_MID_FPS if avg_motion < 0.7 else MOTION_HIGH_FPS)
        log.info(f"  Motion={avg_motion:.3f} → {target_fps}fps extraction")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        extract_interval = max(1, int(native_fps / target_fps))
        fi = 0
        extracted = 0
        frame_records = []
        frames_for_temporal = []
        bboxes_for_temporal = []

        while extracted < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret: break
            if fi % extract_interval != 0:
                fi += 1
                continue

            ts_ms = int((fi / native_fps) * 1000)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            idx_s = str(extracted).zfill(5)

            # Full frame
            full_bytes = encode_png(frame)
            if not full_bytes:
                fi += 1
                continue

            # Upload full frame
            full_path = f"video/{language}/frames/{video_id}/full/frame_{idx_s}.png"
            full_url  = sb_storage_upload(full_path, full_bytes, "image/png")

            # Face detection
            faces      = face_det.detect(frame)
            face_urls  = []
            first_bbox = None

            for fj, face in enumerate(faces):
                crop = face["crop"]
                if crop.size == 0: continue
                face_bytes = encode_jpg(crop)
                if face_bytes:
                    fp = f"video/{language}/frames/{video_id}/faces/face_{idx_s}_{fj:02d}.jpg"
                    fu = sb_storage_upload(fp, face_bytes, "image/jpeg")
                    if fu: face_urls.append(fu)
                if fj == 0:
                    first_bbox = {"x": face["x"], "y": face["y"], "w": face["w"], "h": face["h"]}

            # Texture analysis on frame
            texture_sigs = {}
            try:
                texture_sigs = analyze_image(frame, faces[0]["crop"] if faces else None, full_bytes)
                # Remove large array fields for DB storage
                for k in ["texture_lbp_hist", "fft_radial_profile", "color_hist_r", "color_hist_g", "color_hist_b"]:
                    texture_sigs.pop(k, None)
            except Exception:
                pass

            frames_for_temporal.append(frame)
            bboxes_for_temporal.append(first_bbox)

            frame_records.append({
                "frame_id":               str(uuid.uuid4()),
                "sample_id":              job.get("sample_id"),
                "video_id":               video_id,
                "frame_index":            extracted,
                "timestamp_ms":           ts_ms,
                "motion_score":           round(avg_motion, 3),
                "faces_detected":         len(faces),
                "bounding_boxes":         [{"x": f["x"],"y": f["y"],"w": f["w"],"h": f["h"],"confidence": f["confidence"]} for f in faces],
                "full_frame_path":        full_url,
                "face_crop_paths":        face_urls,
                "face_texture_mask_paths": [],
                "texture_signals":        texture_sigs,
                "extracted_at":           datetime.now(timezone.utc).isoformat(),
            })

            extracted += 1
            result["faces_detected"] += len(faces)
            fi += 1

        cap.release()

        # Temporal consistency
        if len(frames_for_temporal) >= 3:
            try:
                temp_sigs = compute_temporal_consistency(frames_for_temporal, bboxes_for_temporal)
                temp_verdict = score_temporal_consistency(temp_sigs)
                # Attach to all frame records
                for fr in frame_records:
                    fr["texture_signals"]["temporal_verdict"] = temp_verdict["verdict"]
                    fr["texture_signals"]["deepfake_score"]   = temp_verdict["deepfake_score"]
            except Exception as e:
                log.warning(f"  Temporal analysis failed: {e}")

        # Batch insert frame_metadata
        if frame_records:
            for i in range(0, len(frame_records), 20):
                sb_post("frame_metadata", frame_records[i:i+20])

        result["frames_extracted"] = extracted
        result["status"] = "done"
        log.info(f"  ✅ {video_id}: {extracted} frames, {result['faces_detected']} faces")

    return result

# ── Job runner ─────────────────────────────────────────────────
def run_jobs():
    jobs = sb_get("frame_jobs", {
        "status": "eq.pending",
        "order":  "created_at.asc",
        "limit":  str(JOBS_PER_CYCLE),
        "select": "job_id,sample_id,video_id,video_url,language,source_id,fps_target",
    })

    if not jobs:
        log.info("No pending frame jobs — skipping")
        return

    log.info(f"Processing {len(jobs)} frame jobs...")

    for job in jobs:
        sb_patch("frame_jobs", {"job_id": job["job_id"]}, {
            "status": "processing",
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        try:
            result = extract_video(job)
            sb_patch("frame_jobs", {"job_id": job["job_id"]}, {
                "status":           result["status"],
                "frames_extracted": result.get("frames_extracted", 0),
                "faces_detected":   result.get("faces_detected", 0),
                "error_message":    result.get("error"),
                "completed_at":     datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            log.error(f"Job {job['job_id']} crashed: {e}")
            sb_patch("frame_jobs", {"job_id": job["job_id"]}, {
                "status":        "failed",
                "error_message": str(e)[:500],
                "completed_at":  datetime.now(timezone.utc).isoformat(),
            })

if __name__ == "__main__":
    run_jobs()
