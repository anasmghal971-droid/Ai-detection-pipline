"""
============================================================
DETECT-AI: Frame & Face Extractor  (Stage 4)
============================================================
Runs as a cron Python process (every 5 minutes).

  Flow per job:
    1. Pull pending frame_job from Supabase
    2. Download video via yt-dlp (YouTube/TED) or direct URL
    3. Compute per-frame motion score (optical flow)
    4. Extract frames at adaptive FPS:
         motion < 0.3  → 1 fps
         motion 0.3–0.7 → 2 fps
         motion > 0.7  → 5 fps
    5. Per frame: detect faces with MediaPipe
         → save full frame (PNG, lossless)
         → crop face regions (JPEG quality=95)
         → extract face landmark texture masks
    6. Upload all files to Supabase Storage bucket
    7. Push frame batch to HF via hf_push_manager
    8. Write frame_metadata rows to Supabase
    9. Mark job complete

  Face detection:
    - MediaPipe FaceDetection (min_confidence=0.7)
    - 15% padding around bounding box
    - Preserve original pixel quality — no downscaling
============================================================
"""

import os
import io
import uuid
import logging
import tempfile
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
import aiohttp
import requests
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from texture_analyzer import analyze_image

log = logging.getLogger("detect-ai.frame-extractor")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.INFO,
)

# ── Config ─────────────────────────────────────────────────────
SUPABASE_URL    = os.environ["SUPABASE_URL"]
SUPABASE_KEY    = os.environ["SUPABASE_SERVICE_KEY"]
HF_TOKEN        = os.environ["HF_TOKEN"]
HF_REPO_ID      = os.environ.get("HF_DATASET_REPO", "anas775/DETECT-AI-Dataset")
STORAGE_BUCKET  = "detect-ai-frames"
JOBS_PER_CYCLE  = int(os.environ.get("FRAME_JOBS_PER_CYCLE", "3"))
MAX_VIDEO_SEC   = 300    # Skip videos > 5 minutes
MAX_FRAMES      = 500    # Hard cap per video
FACE_CONFIDENCE = 0.7    # MediaPipe min detection confidence
FACE_PADDING    = 0.15   # 15% padding around face crop
JPEG_QUALITY    = 95     # Face crop JPEG quality

# Motion thresholds → FPS
MOTION_LOW_FPS  = 1      # motion < 0.3
MOTION_MID_FPS  = 2      # motion 0.3–0.7
MOTION_HIGH_FPS = 5      # motion > 0.7

# ── Supabase REST helpers ──────────────────────────────────────
SB_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
}

def sb_get(table: str, params: dict) -> list:
    r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=SB_HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def sb_post(table: str, rows: list) -> None:
    r = requests.post(f"{SUPABASE_URL}/rest/v1/{table}", headers=SB_HEADERS, json=rows, timeout=30)
    r.raise_for_status()

def sb_patch(table: str, filters: dict, values: dict) -> None:
    params = {k: f"eq.{v}" for k, v in filters.items()}
    r = requests.patch(f"{SUPABASE_URL}/rest/v1/{table}", headers=SB_HEADERS, params=params, json=values, timeout=30)
    r.raise_for_status()

def sb_storage_upload(bucket: str, path: str, data: bytes, content_type: str) -> str:
    """Upload to Supabase Storage, return public URL."""
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    r = requests.post(
        url,
        headers={
            "apikey":        SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type":  content_type,
            "x-upsert":      "true",
        },
        data=data,
        timeout=60,
    )
    r.raise_for_status()
    return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"

# ── Video downloader ───────────────────────────────────────────
def download_video(video_url: str, output_path: str, source_id: str) -> bool:
    """
    Download video using yt-dlp (handles YouTube, TED, direct URLs).
    Returns True on success.
    """
    is_youtube = "youtube.com" in video_url or "youtu.be" in video_url
    is_ted     = "ted.com" in video_url

    if is_youtube or is_ted:
        cmd = [
            "yt-dlp",
            "--format", "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]",
            "--output", output_path,
            "--no-playlist",
            "--max-filesize", "500m",
            "--quiet",
            video_url,
        ]
    else:
        # Direct URL download
        cmd = [
            "wget", "-q", "-O", output_path,
            "--timeout=60",
            "--tries=3",
            video_url,
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 10_000
    except subprocess.TimeoutExpired:
        log.warning(f"Video download timeout: {video_url}")
        return False
    except Exception as e:
        log.warning(f"Video download error: {e}")
        return False

# ── Motion score ───────────────────────────────────────────────
def compute_motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Compute optical flow magnitude between two grayscale frames → [0.0, 1.0]."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Normalise: typical max motion magnitude ~20px
    return float(min(np.mean(magnitude) / 20.0, 1.0))

# ── Face detector (MediaPipe) ──────────────────────────────────
class FaceDetector:
    def __init__(self):
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,                  # model_selection=1 → range 5m
            min_detection_confidence=FACE_CONFIDENCE,
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """Returns list of face dicts with bbox + landmarks."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results   = self.detector.process(frame_rgb)
        faces     = []

        if not results.detections:
            return faces

        for det in results.detections:
            bb  = det.location_data.relative_bounding_box
            x   = max(0, int((bb.xmin - FACE_PADDING) * w))
            y   = max(0, int((bb.ymin - FACE_PADDING) * h))
            x2  = min(w, int((bb.xmin + bb.width  + FACE_PADDING) * w))
            y2  = min(h, int((bb.ymin + bb.height + FACE_PADDING) * h))
            conf = det.score[0] if det.score else 0.0

            faces.append({
                "x": x, "y": y,
                "w": x2 - x, "h": y2 - y,
                "confidence": round(float(conf), 3),
                "crop": frame_bgr[y:y2, x:x2].copy(),
            })

        return faces

    def get_texture_mask(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate a face landmark mask — white face regions on black background.
        Used for face texture analysis in AI training.
        """
        h, w = frame_bgr.shape[:2]
        mask  = np.zeros((h, w), dtype=np.uint8)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results   = self.detector.process(frame_rgb)

        if not results.detections:
            return None

        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            x2 = min(w, int((bb.xmin + bb.width)  * w))
            y2 = min(h, int((bb.ymin + bb.height) * h))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        return mask

# ── Frame → bytes converters ───────────────────────────────────
def frame_to_png_bytes(frame: np.ndarray) -> bytes:
    """Lossless PNG — full frame."""
    success, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not success:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()

def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """JPEG for face crops."""
    success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()

# ── HF frame pusher ────────────────────────────────────────────
def push_frames_to_hf(
    video_id: str,
    language: str,
    frame_data: list[dict],   # [{frame_index, full_bytes, face_bytes, mask_bytes}]
) -> bool:
    """Push all frames for a video to HF via push manager."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "hf_push"))
        from hf_push_manager import HFPushManager
        manager = HFPushManager()
        return manager.push_frames(video_id, language, frame_data)
    except Exception as e:
        log.error(f"HF frame push failed for {video_id}: {e}")
        return False

# ── Main extraction function ───────────────────────────────────
def extract_video(job: dict) -> dict:
    """
    Full pipeline for one frame_job.
    Returns summary dict with counts and status.
    """
    video_id  = job["video_id"]
    video_url = job["video_url"]
    language  = job["language"]
    fps_hint  = job["fps_target"]

    log.info(f"▶ Processing: {video_id} ({video_url[:60]}...)")

    face_detector = FaceDetector()
    result = {
        "job_id":          job["job_id"],
        "video_id":        video_id,
        "frames_extracted": 0,
        "faces_detected":  0,
        "status":          "failed",
        "error":           None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")

        # ── Download ──────────────────────────────────────────
        log.info(f"  Downloading {video_id}...")
        if not download_video(video_url, video_path, job["source_id"]):
            result["error"] = "download_failed"
            return result

        # ── Open with OpenCV ──────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result["error"] = "cv2_open_failed"
            return result

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / native_fps

        if duration_sec > MAX_VIDEO_SEC:
            log.warning(f"  Video too long ({duration_sec:.0f}s) — skipping")
            cap.release()
            result["error"] = "video_too_long"
            return result

        log.info(f"  Video: {duration_sec:.1f}s @ {native_fps:.1f}fps ({total_frames} frames)")

        # ── Sample frames at adaptive rate ────────────────────
        # Initial pass: sample at 1fps to compute average motion
        sample_interval = int(native_fps)   # every 1 second initially
        prev_gray       = None
        motion_scores   = []
        frame_idx       = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    motion_scores.append(compute_motion_score(prev_gray, gray))
                prev_gray = gray
            frame_idx += 1

        avg_motion = float(np.mean(motion_scores)) if motion_scores else 0.3

        # Choose extraction FPS based on motion
        if avg_motion < 0.3:
            target_fps = MOTION_LOW_FPS
        elif avg_motion < 0.7:
            target_fps = MOTION_MID_FPS
        else:
            target_fps = MOTION_HIGH_FPS

        log.info(f"  Avg motion: {avg_motion:.3f} → extracting at {target_fps}fps")

        # ── Second pass: extract frames at target_fps ─────────
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # Rewind

        extract_interval = max(1, int(native_fps / target_fps))
        frame_idx        = 0
        extracted_count  = 0
        frame_records    = []   # for frame_metadata table
        hf_frame_data    = []   # for HF push

        while extracted_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % extract_interval != 0:
                frame_idx += 1
                continue

            timestamp_ms = int((frame_idx / native_fps) * 1000)

            # Compute per-frame motion score
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_motion = avg_motion  # Use avg as default

            # ── Encode full frame (lossless PNG) ──────────────
            try:
                full_bytes = frame_to_png_bytes(frame)
            except Exception as e:
                log.warning(f"  Frame {frame_idx} PNG encode failed: {e}")
                frame_idx += 1
                continue

            # ── Face detection ─────────────────────────────────
            faces      = face_detector.detect(frame)
            face_bytes = []
            mask_bytes = []

            for face in faces:
                crop = face["crop"]
                if crop.size == 0:
                    continue
                # JPEG 95% face crop
                try:
                    face_bytes.append(frame_to_jpeg_bytes(crop, JPEG_QUALITY))
                except Exception:
                    pass

            # Texture mask (only if faces detected)
            if faces:
                mask = face_detector.get_texture_mask(frame)
                if mask is not None:
                    try:
                        mask_bytes.append(frame_to_png_bytes(
                            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        ))
                    except Exception:
                        pass

            # ── Build storage paths ────────────────────────────
            lang          = language[:2] if language else "unknown"
            idx_str       = str(extracted_count).zfill(5)
            full_path     = f"{video_id}/full/frame_{idx_str}.png"
            face_paths    = [f"{video_id}/faces/face_{idx_str}_{fi:02d}.jpg" for fi in range(len(face_bytes))]
            texture_paths = [f"{video_id}/textures/mask_{idx_str}.png"] if mask_bytes else []

            # ── Upload to Supabase Storage ─────────────────────
            try:
                full_url = sb_storage_upload(STORAGE_BUCKET, f"video/{lang}/{full_path}", full_bytes, "image/png")
                face_urls = []
                for fp, fb in zip(face_paths, face_bytes):
                    furl = sb_storage_upload(STORAGE_BUCKET, f"video/{lang}/{fp}", fb, "image/jpeg")
                    face_urls.append(furl)
                mask_urls = []
                for mp_path, mb in zip(texture_paths, mask_bytes):
                    murl = sb_storage_upload(STORAGE_BUCKET, f"video/{lang}/{mp_path}", mb, "image/png")
                    mask_urls.append(murl)
            except Exception as e:
                log.warning(f"  Storage upload failed for frame {idx_str}: {e}")
                frame_idx += 1
                frame_idx += 1
                continue

            # ── Deep texture analysis (15 signals) ────────────
            face_crop_for_analysis = faces[0]["crop"] if faces else None
            try:
                texture_signals = analyze_image(
                    img_bgr=frame,
                    face_crop_bgr=face_crop_for_analysis,
                    image_bytes=full_bytes,
                )
            except Exception as te:
                log.warning(f"  Texture analysis failed frame {idx_str}: {te}")
                texture_signals = {}

            # ── Frame metadata record ──────────────────────────
            frame_records.append({
                "frame_id":               str(uuid.uuid4()),
                "sample_id":              job.get("sample_id"),
                "video_id":               video_id,
                "frame_index":            extracted_count,
                "timestamp_ms":           timestamp_ms,
                "motion_score":           round(frame_motion, 3),
                "faces_detected":         len(faces),
                "bounding_boxes":         [
                    {"x": f["x"], "y": f["y"], "w": f["w"], "h": f["h"], "confidence": f["confidence"]}
                    for f in faces
                ],
                "full_frame_path":        full_url,
                "face_crop_paths":        face_urls,
                "face_texture_mask_paths": mask_urls,
                "texture_signals": {
                    "laplacian_variance":      texture_signals.get("laplacian_variance"),
                    "noise_sigma":             texture_signals.get("noise_sigma"),
                    "dct_high_freq_energy":    texture_signals.get("dct_high_freq_energy"),
                    "fft_radial_profile":      texture_signals.get("fft_radial_profile"),
                    "edge_density":            texture_signals.get("edge_density"),
                    "local_contrast_std":      texture_signals.get("local_contrast_std"),
                    "compression_score":       texture_signals.get("compression_score"),
                    "gradient_magnitude_mean": texture_signals.get("gradient_magnitude_mean"),
                    "color_coherence":         texture_signals.get("color_coherence"),
                    "saturation_mean":         texture_signals.get("saturation_mean"),
                    "saturation_std":          texture_signals.get("saturation_std"),
                    "exif_has_camera":         texture_signals.get("exif_has_camera"),
                    "face_symmetry_score":     texture_signals.get("face_symmetry_score"),
                    "skin_tone_variance":      texture_signals.get("skin_tone_variance"),
                    "texture_lbp_hist":        texture_signals.get("texture_lbp_hist"),
                },
                "extracted_at":           datetime.now(timezone.utc).isoformat(),
            })

            # ── Collect for HF push ────────────────────────────
            hf_frame_data.append({
                "frame_index": extracted_count,
                "full_bytes":  full_bytes,
                "face_bytes":  face_bytes,
                "mask_bytes":  mask_bytes,
            })

            extracted_count += 1
            result["faces_detected"] += len(faces)
            frame_idx += 1

        cap.release()

        # ── Batch insert frame_metadata ───────────────────────
        if frame_records:
            CHUNK = 50
            for i in range(0, len(frame_records), CHUNK):
                try:
                    sb_post("frame_metadata", frame_records[i:i+CHUNK])
                except Exception as e:
                    log.warning(f"  frame_metadata insert failed: {e}")

        # ── Push frames to HF ─────────────────────────────────
        if hf_frame_data:
            log.info(f"  Pushing {len(hf_frame_data)} frames to HF...")
            push_frames_to_hf(video_id, language, hf_frame_data)

        result["frames_extracted"] = extracted_count
        result["status"]           = "done"

        log.info(
            f"  ✅ {video_id}: {extracted_count} frames, "
            f"{result['faces_detected']} faces detected"
        )

    return result

# ── Job runner ─────────────────────────────────────────────────
def run_jobs():
    """
    Pull pending frame_jobs from Supabase and process them.
    Called every 5 minutes by cron.
    """
    jobs = sb_get("frame_jobs", {
        "status":  "eq.pending",
        "order":   "created_at.asc",
        "limit":   str(JOBS_PER_CYCLE),
        "select":  "job_id,sample_id,video_id,video_url,language,source_id,fps_target",
    })

    if not jobs:
        log.info("No pending frame jobs.")
        return

    log.info(f"Processing {len(jobs)} frame jobs...")

    for job in jobs:
        # Mark as processing
        sb_patch("frame_jobs", {"job_id": job["job_id"]}, {
            "status":     "processing",
            "started_at": datetime.now(timezone.utc).isoformat(),
        })

        try:
            result = extract_video(job)

            sb_patch("frame_jobs", {"job_id": job["job_id"]}, {
                "status":           result["status"],
                "frames_extracted": result["frames_extracted"],
                "faces_detected":   result["faces_detected"],
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
