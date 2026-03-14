"""
DETECT-AI — Real-Time Audio Analysis API
=========================================
Deployed at: https://detect-ai-nu.vercel.app/api/analyze/audio
             OR as a Supabase Edge Function

When a user uploads audio on your site:
  1. Receives audio file (WAV/MP3/OGG/M4A/FLAC, up to 50MB)
  2. Runs 20-signal analysis in ~1–3 seconds
  3. Returns verdict: AI_GENERATED | HUMAN | UNCERTAIN
  4. Returns full signal breakdown + human-readable explanation
  5. Optionally stores sample in pipeline for dataset training

Designed to power the /detect/audio page at detect-ai-nu.vercel.app
"""

import os
import json
import uuid
import base64
import io
import time
from datetime import datetime, timezone
import requests
from pathlib import Path
import sys

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "audio_analysis"))
from audio_analyzer import analyze_audio, score_audio

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

SUPPORTED_FORMATS = {"wav", "mp3", "ogg", "flac", "m4a", "aac", "webm"}

SB_HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
}


def analyze_user_upload(
    audio_bytes: bytes,
    file_format: str,
    store_in_dataset: bool = True,
    user_id: str = None,
) -> dict:
    """
    Full pipeline for user-uploaded audio.
    Called by the Vercel API route handler.

    Returns complete analysis report.
    """
    t0 = time.time()

    if len(audio_bytes) > MAX_FILE_SIZE:
        return {"error": "File too large. Maximum 50MB.", "status": 400}

    if file_format.lower().lstrip(".") not in SUPPORTED_FORMATS:
        return {"error": f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}", "status": 400}

    # ── Step 1: Extract 20 audio signals ─────────────────────────
    try:
        signals = analyze_audio(audio_bytes, file_format=file_format)
    except Exception as e:
        return {"error": f"Audio analysis failed: {str(e)}", "status": 500}

    if "error" in signals:
        return {"error": signals["error"], "status": 400}

    # ── Step 2: Score signals → verdict ──────────────────────────
    verdict = score_audio(signals)
    processing_ms = int((time.time() - t0) * 1000)

    # ── Step 3: Build human-readable report ──────────────────────
    report = _build_report(signals, verdict)

    # ── Step 4: Store in dataset pipeline (optional) ──────────────
    sample_id = str(uuid.uuid4())
    if store_in_dataset and SUPABASE_URL and SUPABASE_KEY:
        try:
            _store_sample(
                sample_id=sample_id,
                audio_bytes=audio_bytes,
                file_format=file_format,
                signals=signals,
                verdict=verdict,
                user_id=user_id,
            )
        except Exception as e:
            # Don't fail the response if storage fails
            pass

    return {
        "sample_id":       sample_id,
        "verdict":         verdict["verdict"],
        "ai_probability":  verdict["ai_score"],
        "human_probability": verdict["human_score"],
        "confidence":      max(verdict["ai_score"], verdict["human_score"]),
        "processing_ms":   processing_ms,
        "duration_sec":    signals.get("duration_sec", 0),
        "signals":         _format_signals_for_ui(signals),
        "report":          report,
        "flags":           verdict["flags"],
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }


def _build_report(signals: dict, verdict: dict) -> dict:
    """
    Build the human-readable breakdown shown on the detect-ai site.
    Maps raw signals → user-friendly labels and explanations.
    """
    ai_score = verdict["ai_score"]

    # Verdict label
    if verdict["verdict"] == "AI_GENERATED":
        summary = "🤖 This audio is very likely AI-generated or synthetic."
        color = "red"
    elif verdict["verdict"] == "HUMAN":
        summary = "✅ This audio appears to be authentic human speech."
        color = "green"
    else:
        summary = "⚠️ Inconclusive — this audio has mixed characteristics."
        color = "yellow"

    # Signal-level breakdown for UI display
    breakdown = []

    f0_std = signals.get("f0_std", 0)
    breakdown.append({
        "signal":      "Pitch Variation",
        "value":       f"{f0_std:.1f} Hz",
        "status":      "suspicious" if f0_std < 8.0 else "natural",
        "explanation": "Real voices have natural pitch variation (>10 Hz std). "
                       "TTS voices maintain unnaturally stable pitch.",
        "ai_indicator": f0_std < 8.0,
    })

    silence_ratio = signals.get("silence_ratio", 0)
    breakdown.append({
        "signal":      "Natural Pauses & Breathing",
        "value":       f"{silence_ratio:.1%} silence",
        "status":      "suspicious" if silence_ratio < 0.03 else "natural",
        "explanation": "Real speakers pause and breathe naturally. "
                       "AI voices often have no breathing gaps.",
        "ai_indicator": silence_ratio < 0.03,
    })

    hnr = signals.get("harmonic_to_noise_ratio", 0)
    breakdown.append({
        "signal":      "Voice Quality / Noise Floor",
        "value":       f"HNR {hnr:.1f}",
        "status":      "suspicious" if hnr > 15.0 else "natural",
        "explanation": "Real voices have natural background noise. "
                       "AI voices are often too acoustically perfect (HNR > 15).",
        "ai_indicator": hnr > 15.0,
    })

    phase_disc = signals.get("phase_discontinuity", 0)
    breakdown.append({
        "signal":      "Phase Coherence",
        "value":       f"{phase_disc:.3f}",
        "status":      "suspicious" if phase_disc > 1.8 else "natural",
        "explanation": "Voice cloning systems leave phase artifacts in the spectrogram "
                       "that are invisible but mathematically detectable.",
        "ai_indicator": phase_disc > 1.8,
    })

    rms_std = signals.get("rms_energy_std", 0)
    breakdown.append({
        "signal":      "Energy Dynamics",
        "value":       f"{rms_std:.4f} RMS σ",
        "status":      "suspicious" if rms_std < 0.015 else "natural",
        "explanation": "Real speakers naturally vary in loudness. "
                       "AI voices maintain unnaturally consistent volume.",
        "ai_indicator": rms_std < 0.015,
    })

    breath = signals.get("breath_event_count", 0)
    breakdown.append({
        "signal":      "Breath Detection",
        "value":       f"{breath} breath events",
        "status":      "suspicious" if breath == 0 else "natural",
        "explanation": "Real human speakers breathe between sentences. "
                       "AI voices never breathe — zero breath events is a red flag.",
        "ai_indicator": breath == 0,
    })

    mfcc_std = signals.get("mfcc_std")
    if mfcc_std:
        avg_mfcc = float(sum(mfcc_std) / len(mfcc_std))
        breakdown.append({
            "signal":      "Vocal Tract Dynamics (MFCC)",
            "value":       f"σ = {avg_mfcc:.2f}",
            "status":      "suspicious" if avg_mfcc < 8.0 else "natural",
            "explanation": "MFCCs capture the shape of the vocal tract over time. "
                           "Real speech shows natural coarticulation. TTS is too static.",
            "ai_indicator": avg_mfcc < 8.0,
        })

    return {
        "summary":   summary,
        "color":     color,
        "breakdown": breakdown,
        "n_signals_checked": len(breakdown),
        "n_signals_triggered": sum(1 for b in breakdown if b["ai_indicator"]),
    }


def _format_signals_for_ui(signals: dict) -> dict:
    """Clean signals dict for frontend display — remove large arrays."""
    ui_signals = {}
    skip_keys = {"mfcc_mean", "mfcc_std", "mfcc_delta_mean", "n_samples", "texture_lbp_hist"}
    for k, v in signals.items():
        if k in skip_keys:
            continue
        if isinstance(v, float):
            ui_signals[k] = round(v, 4)
        elif isinstance(v, list) and len(v) <= 16:
            ui_signals[k] = [round(x, 4) for x in v]
        elif isinstance(v, (int, str, bool)):
            ui_signals[k] = v
    return ui_signals


def _store_sample(
    sample_id: str,
    audio_bytes: bytes,
    file_format: str,
    signals: dict,
    verdict: dict,
    user_id: str = None,
):
    """
    Store the uploaded sample in the pipeline for dataset training.
    Samples verified by users (thumbs up/down) become high-confidence training data.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return

    # Upload audio to Supabase Storage
    storage_path = f"audio/user-uploads/{sample_id}.{file_format}"
    storage_url = ""
    try:
        content_types = {
            "wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg",
            "flac": "audio/flac", "m4a": "audio/mp4", "aac": "audio/aac"
        }
        ct = content_types.get(file_format.lower(), "audio/octet-stream")
        storage_headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": ct,
            "x-upsert": "true",
        }
        r = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/detect-ai-frames/{storage_path}",
            headers=storage_headers,
            data=audio_bytes,
            timeout=30,
        )
        if r.ok:
            storage_url = f"{SUPABASE_URL}/storage/v1/object/public/detect-ai-frames/{storage_path}"
    except Exception:
        pass

    # Insert into samples_staging for pipeline processing
    row = {
        "sample_id":        sample_id,
        "source_id":        "user-upload",
        "source_url":       f"user-upload/{sample_id}",
        "content_type":     "audio",
        "language":         "en",
        "raw_content":      storage_url or f"user-upload/{sample_id}",
        "label":            verdict["verdict"],
        "final_confidence": max(verdict["ai_score"], verdict["human_score"]),
        "model_scores":     json.dumps(verdict),
        "verified":         False,  # Will be set True when user confirms
        "metadata": json.dumps({
            "source":        "user-upload",
            "user_id":       user_id,
            "file_format":   file_format,
            "duration_sec":  signals.get("duration_sec", 0),
            "audio_signals": {k: v for k, v in signals.items()
                             if not isinstance(v, list) or len(v) <= 16},
        }),
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "worker_id":  "user-upload-api",
        "status":     "staged",
    }

    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/samples_staging",
            headers=SB_HEADERS, json=[row], timeout=10
        )
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# Vercel Edge Function handler (serverless)
# Place this in: vercel-api/pages/api/analyze/audio.ts
# and call this Python logic via a subprocess or Supabase Edge Function
# ═══════════════════════════════════════════════════════════════════
VERCEL_HANDLER_TS = '''
// vercel-api/pages/api/analyze/audio.ts
// Real-time audio analysis endpoint for detect-ai-nu.vercel.app
// Calls Supabase Edge Function which runs the Python audio_analyzer

import type { NextApiRequest, NextApiResponse } from "next";

export const config = { api: { bodyParser: { sizeLimit: "50mb" } } };

const SUPABASE_URL    = process.env.SUPABASE_URL!;
const SUPABASE_KEY    = process.env.SUPABASE_SERVICE_KEY!;
const EDGE_FN_URL     = `${SUPABASE_URL}/functions/v1/analyze-audio`;

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { audio_base64, file_format, store_in_dataset, user_id } = req.body;

  if (!audio_base64 || !file_format) {
    return res.status(400).json({ error: "audio_base64 and file_format required" });
  }

  try {
    const response = await fetch(EDGE_FN_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${SUPABASE_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ audio_base64, file_format, store_in_dataset, user_id }),
    });

    const result = await response.json();
    return res.status(200).json(result);

  } catch (error) {
    console.error("Audio analysis error:", error);
    return res.status(500).json({ error: "Analysis failed" });
  }
}
'''

if __name__ == "__main__":
    # Print the Vercel handler for reference
    print("Vercel handler (save to vercel-api/pages/api/analyze/audio.ts):")
    print(VERCEL_HANDLER_TS)
