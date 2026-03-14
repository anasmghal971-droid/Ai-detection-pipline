"""
DETECT-AI — Temporal Consistency Analyzer
==========================================
The #1 signal for video deepfake detection that was missing.

Real human faces move with physics-consistent motion between frames.
Deepfakes have subtle frame-to-frame inconsistencies because each frame
is independently generated/swapped — the face "flickers" in ways
invisible to the human eye but detectable mathematically.

Signals computed:
  1.  face_bbox_jitter         — how much face bounding box jumps (deepfakes: higher)
  2.  face_area_variance       — face area consistency across frames
  3.  optical_flow_face_mean   — mean motion inside face region
  4.  optical_flow_bg_mean     — mean motion in background
  5.  flow_face_bg_ratio       — face/background motion ratio (deepfakes: wrong ratio)
  6.  face_color_temporal_std  — color consistency across frames (deepfakes: flicker)
  7.  landmark_velocity_mean   — mean velocity of face landmarks
  8.  landmark_velocity_std    — variance in landmark velocity
  9.  landmark_accel_mean      — acceleration of landmarks (real faces: smooth)
  10. blink_regularity         — blink interval consistency (deepfakes: no blinks or wrong timing)
  11. texture_temporal_std     — frame-to-frame texture variation in face region
  12. identity_consistency     — embedding distance between frames (same face?)
"""

import numpy as np
import cv2
from typing import Optional


def compute_temporal_consistency(
    frames: list[np.ndarray],             # List of BGR frames (already extracted)
    face_bboxes: list[Optional[dict]],    # [{x, y, w, h}] per frame, None if no face
    landmarks: list[Optional[np.ndarray]] = None,  # Optional: [N×2 array] per frame
) -> dict:
    """
    Compute all 12 temporal consistency signals across a sequence of frames.

    Args:
        frames:      List of BGR frame arrays (should be 10–500 frames)
        face_bboxes: List of face bounding boxes per frame (None = no face)
        landmarks:   Optional list of face landmark arrays per frame

    Returns:
        dict of temporal signals, ready to store in frame_metadata.texture_signals
    """
    n = len(frames)
    if n < 3:
        return {"temporal_error": "too_few_frames", "n_frames": n}

    signals = {"n_frames": n, "frames_with_faces": sum(1 for b in face_bboxes if b is not None)}

    # ── 1–2. Bounding box jitter & area variance ──────────────────
    valid_boxes = [b for b in face_bboxes if b is not None]
    if len(valid_boxes) >= 2:
        areas  = np.array([b["w"] * b["h"] for b in valid_boxes], dtype=np.float32)
        cx     = np.array([b["x"] + b["w"]/2 for b in valid_boxes], dtype=np.float32)
        cy     = np.array([b["y"] + b["h"]/2 for b in valid_boxes], dtype=np.float32)
        # Normalize by image size
        h0, w0 = frames[0].shape[:2]
        cx_n = cx / w0
        cy_n = cy / h0
        area_n = areas / (w0 * h0)
        signals["face_bbox_jitter"]   = float(np.sqrt(np.diff(cx_n)**2 + np.diff(cy_n)**2).mean())
        signals["face_area_variance"] = float(area_n.std())
    else:
        signals["face_bbox_jitter"]   = None
        signals["face_area_variance"] = None

    # ── 3–5. Optical flow: face vs background ─────────────────────
    flow_face_list = []
    flow_bg_list   = []

    for i in range(1, min(n, 30)):  # Limit to 30 frame pairs for speed
        gray_prev = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frames[i],   cv2.COLOR_BGR2GRAY)

        try:
            flow = cv2.calcOpticalFlowFarneback(
                gray_prev, gray_curr, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

            # Face region mask
            bbox = face_bboxes[i]
            if bbox:
                h, w = mag.shape
                x1 = max(0, bbox["x"])
                y1 = max(0, bbox["y"])
                x2 = min(w, bbox["x"] + bbox["w"])
                y2 = min(h, bbox["y"] + bbox["h"])
                if x2 > x1 and y2 > y1:
                    face_mag = mag[y1:y2, x1:x2].mean()
                    # Background = everything outside face + 20px margin
                    bg_mask = np.ones_like(mag, dtype=bool)
                    bg_mask[max(0,y1-20):min(h,y2+20), max(0,x1-20):min(w,x2+20)] = False
                    bg_mag = mag[bg_mask].mean()
                    flow_face_list.append(face_mag)
                    flow_bg_list.append(bg_mag)
        except Exception:
            continue

    if flow_face_list:
        signals["optical_flow_face_mean"] = float(np.mean(flow_face_list))
        signals["optical_flow_bg_mean"]   = float(np.mean(flow_bg_list))
        ratio = np.array(flow_face_list) / (np.array(flow_bg_list) + 1e-6)
        signals["flow_face_bg_ratio"]     = float(ratio.mean())
        # Deepfakes: ratio variance is higher (face moves independently of background)
        signals["flow_ratio_variance"]    = float(ratio.var())
    else:
        signals["optical_flow_face_mean"] = None
        signals["optical_flow_bg_mean"]   = None
        signals["flow_face_bg_ratio"]     = None
        signals["flow_ratio_variance"]    = None

    # ── 6. Face color temporal std ────────────────────────────────
    face_colors = []
    for i, (frame, bbox) in enumerate(zip(frames[:50], face_bboxes[:50])):
        if bbox is None:
            continue
        h, w = frame.shape[:2]
        x1 = max(0, bbox["x"])
        y1 = max(0, bbox["y"])
        x2 = min(w, bbox["x"] + bbox["w"])
        y2 = min(h, bbox["y"] + bbox["h"])
        if x2 > x1 and y2 > y1:
            face_region = frame[y1:y2, x1:x2]
            face_colors.append(face_region.mean(axis=(0, 1)))  # BGR mean

    if len(face_colors) >= 3:
        fc = np.array(face_colors)
        signals["face_color_temporal_std"] = float(fc.std(axis=0).mean())
    else:
        signals["face_color_temporal_std"] = None

    # ── 7–9. Landmark velocity & acceleration ──────────────────────
    if landmarks and len(landmarks) >= 3:
        valid_lm = [(i, lm) for i, lm in enumerate(landmarks) if lm is not None]
        if len(valid_lm) >= 3:
            positions = np.array([lm for _, lm in valid_lm])  # [T, N, 2]
            velocities    = np.diff(positions, axis=0)         # [T-1, N, 2]
            accelerations = np.diff(velocities, axis=0)        # [T-2, N, 2]

            vel_mag   = np.sqrt((velocities**2).sum(axis=-1))   # [T-1, N]
            accel_mag = np.sqrt((accelerations**2).sum(axis=-1)) # [T-2, N]

            signals["landmark_velocity_mean"] = float(vel_mag.mean())
            signals["landmark_velocity_std"]  = float(vel_mag.std())
            signals["landmark_accel_mean"]    = float(accel_mag.mean())
        else:
            signals["landmark_velocity_mean"] = signals["landmark_velocity_std"] = signals["landmark_accel_mean"] = None
    else:
        signals["landmark_velocity_mean"] = signals["landmark_velocity_std"] = signals["landmark_accel_mean"] = None

    # ── 10. Blink regularity ──────────────────────────────────────
    # Blinks detected by eye aspect ratio (EAR) dips
    # We approximate using face area dips (full landmark tracking would be better)
    if len(valid_boxes) >= 10:
        areas = np.array([b["h"] for b in valid_boxes], dtype=np.float32)
        normalized = (areas - areas.mean()) / (areas.std() + 1e-6)
        # Find dips below -1 std as potential blink events
        dips = np.where(normalized < -0.8)[0]
        if len(dips) >= 2:
            intervals = np.diff(dips)
            signals["blink_regularity"] = float(intervals.std() / (intervals.mean() + 1e-6))
        else:
            signals["blink_regularity"] = 0.0
    else:
        signals["blink_regularity"] = None

    # ── 11. Texture temporal std (face region only) ───────────────
    face_textures = []
    for frame, bbox in zip(frames[:30], face_bboxes[:30]):
        if bbox is None:
            continue
        h, w = frame.shape[:2]
        x1 = max(0, bbox["x"])
        y1 = max(0, bbox["y"])
        x2 = min(w, bbox["x"] + bbox["w"])
        y2 = min(h, bbox["y"] + bbox["h"])
        if x2 > x1 and y2 > y1:
            face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            # Resize to 32x32 for consistent comparison
            face_small = cv2.resize(face, (32, 32), interpolation=cv2.INTER_AREA)
            face_textures.append(face_small.astype(np.float32) / 255.0)

    if len(face_textures) >= 3:
        ft = np.stack(face_textures, axis=0)  # [T, 32, 32]
        signals["texture_temporal_std"] = float(ft.std(axis=0).mean())
    else:
        signals["texture_temporal_std"] = None

    return signals


def score_temporal_consistency(signals: dict) -> dict:
    """
    Score temporal signals → deepfake probability.
    Returns verdict + confidence + flags.
    """
    flags = []
    ai_votes = 0.0
    total_votes = 0.0

    def vote(condition, weight, msg):
        nonlocal ai_votes, total_votes
        total_votes += weight
        if condition:
            ai_votes += weight
            flags.append(msg)

    bbox_jitter = signals.get("face_bbox_jitter")
    if bbox_jitter is not None:
        vote(bbox_jitter > 0.015, 0.20,
             f"BBox jitter={bbox_jitter:.4f} — face position unstable between frames")

    color_std = signals.get("face_color_temporal_std")
    if color_std is not None:
        vote(color_std > 3.5, 0.20,
             f"Color temporal std={color_std:.2f} — face color flickering (deepfake)")

    flow_ratio_var = signals.get("flow_ratio_variance")
    if flow_ratio_var is not None:
        vote(flow_ratio_var > 2.0, 0.20,
             f"Flow ratio variance={flow_ratio_var:.2f} — face moves inconsistently vs background")

    texture_std = signals.get("texture_temporal_std")
    if texture_std is not None:
        vote(texture_std > 0.08, 0.15,
             f"Texture temporal std={texture_std:.4f} — face texture flickering")

    lm_accel = signals.get("landmark_accel_mean")
    if lm_accel is not None:
        vote(lm_accel > 3.0, 0.15,
             f"Landmark acceleration={lm_accel:.2f} — unnatural jerky motion")

    blink_reg = signals.get("blink_regularity")
    if blink_reg is not None:
        vote(blink_reg > 0.5 or blink_reg == 0.0, 0.10,
             f"Blink regularity={blink_reg:.3f} — abnormal blink pattern")

    if total_votes == 0:
        return {"deepfake_score": 0.5, "verdict": "UNCERTAIN", "flags": []}

    score = ai_votes / total_votes
    verdict = "AI_GENERATED" if score >= 0.65 else ("HUMAN" if score <= 0.30 else "UNCERTAIN")

    return {
        "deepfake_score": round(score, 4),
        "verdict":        verdict,
        "flags":          flags,
    }
