"""
DETECT-AI — Deep Audio Signal Analyzer
=======================================
Extracts 20 signals per audio clip that are critical for AI voice vs real voice detection.
Used by the audio scraper, labeler, AND the real-time user-upload endpoint on detect-ai-nu.vercel.app

When a user uploads audio to your site, this runs in ~2 seconds and returns a verdict.

Signals:
  1.  mfcc_mean[13]           — vocal tract shape (TTS has unnaturally perfect MFCCs)
  2.  mfcc_std[13]            — MFCC variation (AI voice too consistent)
  3.  mfcc_delta_mean[13]     — MFCC dynamics (rate of change — TTS too smooth)
  4.  spectral_centroid_mean  — brightness (TTS too uniform)
  5.  spectral_centroid_std   — brightness variation
  6.  spectral_rolloff_mean   — high-freq energy dropoff
  7.  spectral_bandwidth_mean — spread of frequencies
  8.  zero_crossing_rate_mean — signal smoothness (AI voice: too low)
  9.  zero_crossing_rate_std  — ZCR variation
  10. rms_energy_mean         — loudness
  11. rms_energy_std          — loudness variation (real voices fluctuate more)
  12. f0_mean                 — fundamental pitch (Hz)
  13. f0_std                  — pitch variation (TTS: too stable = low std)
  14. f0_range                — pitch range (TTS too narrow)
  15. silence_ratio           — fraction of silence (TTS has no breath gaps)
  16. breath_event_count      — number of audible breath sounds
  17. harmonic_to_noise_ratio — voice clarity (AI: too high = too perfect)
  18. spectral_flux_mean      — frame-to-frame spectral change
  19. chroma_std              — pitch class variation
  20. phase_discontinuity     — phase coherence (voice cloning artifacts)
"""

import numpy as np
import io
import struct
import wave
import math
from typing import Optional


# ── Try to import librosa (needed for full analysis) ─────────────
try:
    import librosa
    import librosa.feature
    import librosa.effects
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# ═══════════════════════════════════════════════════════════════════
# LIGHTWEIGHT FALLBACK (no librosa) — for CF Workers / edge use
# ═══════════════════════════════════════════════════════════════════

def _read_wav_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Read WAV bytes → (float32 mono array, sample_rate)."""
    with wave.open(io.BytesIO(audio_bytes)) as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        n_ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    # Convert to float32
    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        samples = np.frombuffer(raw, dtype=np.float32)

    # Mono
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)

    return samples, sr


def _rms_energy(samples: np.ndarray, frame_len: int = 2048, hop: int = 512) -> np.ndarray:
    frames = librosa.util.frame(samples, frame_length=frame_len, hop_length=hop) if HAS_LIBROSA \
             else _manual_frame(samples, frame_len, hop)
    return np.sqrt((frames ** 2).mean(axis=0))


def _manual_frame(samples: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    n = (len(samples) - frame_len) // hop + 1
    if n <= 0:
        return samples.reshape(1, -1)
    frames = np.zeros((frame_len, n), dtype=np.float32)
    for i in range(n):
        start = i * hop
        frames[:, i] = samples[start:start + frame_len]
    return frames


def _zero_crossing_rate(samples: np.ndarray, frame_len: int = 2048, hop: int = 512) -> np.ndarray:
    frames = _manual_frame(samples, frame_len, hop)
    signs = np.sign(frames)
    zcr = (np.diff(signs, axis=0) != 0).sum(axis=0).astype(np.float32) / frame_len
    return zcr


def _basic_signals(samples: np.ndarray, sr: int) -> dict:
    """Fast signals computable without librosa."""
    frame_len = 2048
    hop = 512

    rms = _rms_energy(samples, frame_len, hop)
    zcr = _zero_crossing_rate(samples, frame_len, hop)

    # Silence: frames below -40 dBFS
    silence_thresh = 0.01
    silence_ratio = float((rms < silence_thresh).mean())

    # Simple F0 via autocorrelation on 30ms windows
    f0_list = []
    win = int(0.03 * sr)
    step = int(0.01 * sr)
    for i in range(0, len(samples) - win, step):
        chunk = samples[i:i + win]
        ac = np.correlate(chunk, chunk, mode="full")[win - 1:]
        # Find first peak after minimum lag (60Hz = sr/60 samples)
        min_lag = max(1, int(sr / 800))  # 800 Hz max pitch
        max_lag = int(sr / 60)           # 60 Hz min pitch
        if max_lag >= len(ac):
            continue
        sub = ac[min_lag:max_lag]
        if len(sub) < 2:
            continue
        peak = np.argmax(sub) + min_lag
        if ac[peak] > 0.3 * ac[0] and ac[0] > 0:
            f0_list.append(sr / peak)

    f0_arr = np.array(f0_list) if f0_list else np.array([0.0])

    return {
        "rms_energy_mean":         float(rms.mean()),
        "rms_energy_std":          float(rms.std()),
        "zero_crossing_rate_mean": float(zcr.mean()),
        "zero_crossing_rate_std":  float(zcr.std()),
        "silence_ratio":           silence_ratio,
        "f0_mean":                 float(f0_arr.mean()),
        "f0_std":                  float(f0_arr.std()),
        "f0_range":                float(f0_arr.max() - f0_arr.min()) if len(f0_arr) > 1 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════
# FULL ANALYSIS (requires librosa) — GitHub Actions / local
# ═══════════════════════════════════════════════════════════════════

def _librosa_signals(y: np.ndarray, sr: int) -> dict:
    """Full 20-signal analysis using librosa."""
    signals = {}
    hop = 512
    n_fft = 2048

    # ── 1–3. MFCC (13 coefficients + delta) ──────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop, n_fft=n_fft)
    mfcc_delta = librosa.feature.delta(mfcc)
    signals["mfcc_mean"]       = mfcc.mean(axis=1).tolist()       # [13]
    signals["mfcc_std"]        = mfcc.std(axis=1).tolist()        # [13]
    signals["mfcc_delta_mean"] = mfcc_delta.mean(axis=1).tolist() # [13]

    # ── 4–5. Spectral centroid ────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop, n_fft=n_fft)[0]
    signals["spectral_centroid_mean"] = float(centroid.mean())
    signals["spectral_centroid_std"]  = float(centroid.std())

    # ── 6. Spectral rolloff ───────────────────────────────────────
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop, n_fft=n_fft)[0]
    signals["spectral_rolloff_mean"]  = float(rolloff.mean())

    # ── 7. Spectral bandwidth ─────────────────────────────────────
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop, n_fft=n_fft)[0]
    signals["spectral_bandwidth_mean"] = float(bw.mean())

    # ── 8–9. ZCR ─────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
    signals["zero_crossing_rate_mean"] = float(zcr.mean())
    signals["zero_crossing_rate_std"]  = float(zcr.std())

    # ── 10–11. RMS energy ─────────────────────────────────────────
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    signals["rms_energy_mean"] = float(rms.mean())
    signals["rms_energy_std"]  = float(rms.std())

    # ── 12–14. F0 (pitch tracking) ────────────────────────────────
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            hop_length=hop
        )
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
        if len(f0_voiced) > 0:
            signals["f0_mean"]  = float(f0_voiced.mean())
            signals["f0_std"]   = float(f0_voiced.std())
            signals["f0_range"] = float(f0_voiced.max() - f0_voiced.min())
        else:
            signals["f0_mean"] = signals["f0_std"] = signals["f0_range"] = 0.0
    except Exception:
        signals["f0_mean"] = signals["f0_std"] = signals["f0_range"] = 0.0

    # ── 15–16. Silence + breath events ───────────────────────────
    silence_thresh = 0.01
    silence_ratio = float((rms < silence_thresh).mean())
    signals["silence_ratio"] = silence_ratio

    # Breath events: short bursts of low-level energy after silence
    in_breath = False
    breath_count = 0
    breath_thresh = 0.005
    breath_max = 0.04
    for r in rms:
        if not in_breath and breath_thresh < r < breath_max:
            in_breath = True
        elif in_breath and r > breath_max:
            breath_count += 1
            in_breath = False
        elif in_breath and r < breath_thresh:
            in_breath = False
    signals["breath_event_count"] = breath_count

    # ── 17. Harmonic-to-noise ratio ───────────────────────────────
    try:
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_rms = float(np.sqrt((harmonic ** 2).mean()))
        noise_rms = float(np.sqrt(((y - harmonic) ** 2).mean()))
        signals["harmonic_to_noise_ratio"] = harmonic_rms / (noise_rms + 1e-9)
    except Exception:
        signals["harmonic_to_noise_ratio"] = 0.0

    # ── 18. Spectral flux ─────────────────────────────────────────
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
    signals["spectral_flux_mean"] = float(flux.mean())

    # ── 19. Chroma std (pitch class variation) ────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_fft=n_fft)
    signals["chroma_std"] = float(chroma.std())

    # ── 20. Phase discontinuity (voice cloning artifact) ─────────
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    phase = np.angle(D)
    phase_diff = np.diff(phase, axis=1)
    # Wrap to [-π, π]
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    signals["phase_discontinuity"] = float(np.abs(phase_diff).mean())

    return signals


# ═══════════════════════════════════════════════════════════════════
# MASTER ANALYZE FUNCTION
# ═══════════════════════════════════════════════════════════════════

def analyze_audio(
    audio_bytes: bytes,
    file_format: str = "wav",
) -> dict:
    """
    Run all 20 signals on an audio clip.
    Returns flat dict ready for Supabase JSONB + HF Parquet.

    Args:
        audio_bytes:  Raw audio file bytes (WAV or MP3/OGG via librosa)
        file_format:  'wav', 'mp3', 'ogg', 'flac', 'm4a'
    """
    # Load audio
    try:
        if HAS_LIBROSA:
            import soundfile as sf
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        else:
            # WAV only fallback
            y, sr = _read_wav_bytes(audio_bytes)
            # Resample to 16kHz using simple linear interpolation if needed
            if sr != 16000:
                factor = 16000 / sr
                n_new = int(len(y) * factor)
                y = np.interp(
                    np.linspace(0, len(y) - 1, n_new),
                    np.arange(len(y)), y
                ).astype(np.float32)
                sr = 16000
    except Exception as e:
        return {"error": str(e), "duration_sec": 0.0, "sample_rate": 0}

    duration_sec = len(y) / sr
    signals: dict = {
        "duration_sec":  round(duration_sec, 3),
        "sample_rate":   sr,
        "n_samples":     len(y),
    }

    # Always compute basic signals
    basic = _basic_signals(y, sr)
    signals.update(basic)

    # Full signals if librosa available
    if HAS_LIBROSA:
        try:
            full = _librosa_signals(y, sr)
            signals.update(full)
        except Exception as e:
            signals["librosa_error"] = str(e)

    return signals


def score_audio(signals: dict) -> dict:
    """
    Heuristic scoring of audio signals → AI probability estimate.
    Returns:
      ai_score:    float 0–1 (1 = almost certainly AI)
      human_score: float 0–1
      verdict:     'AI_GENERATED' | 'HUMAN' | 'UNCERTAIN'
      flags:       list of triggered signals with explanation
    """
    flags = []
    ai_votes = 0
    total_votes = 0

    def vote(condition: bool, weight: float, flag_msg: str):
        nonlocal ai_votes, total_votes
        total_votes += weight
        if condition:
            ai_votes += weight
            flags.append(flag_msg)

    # ── Pitch signals ────────────────────────────────────────────
    f0_std = signals.get("f0_std", None)
    if f0_std is not None:
        vote(f0_std < 8.0, 0.20,
             f"F0 std={f0_std:.1f}Hz — unnaturally stable pitch (TTS signature)")

    f0_range = signals.get("f0_range", None)
    if f0_range is not None:
        vote(f0_range < 40.0, 0.15,
             f"F0 range={f0_range:.1f}Hz — very narrow pitch range (TTS lacks prosody)")

    # ── Energy signals ───────────────────────────────────────────
    rms_std = signals.get("rms_energy_std", None)
    if rms_std is not None:
        vote(rms_std < 0.015, 0.12,
             f"RMS std={rms_std:.4f} — too-uniform energy (real voices vary more)")

    # ── Silence / breathing ───────────────────────────────────────
    silence_ratio = signals.get("silence_ratio", None)
    if silence_ratio is not None:
        vote(silence_ratio < 0.03, 0.10,
             f"Silence ratio={silence_ratio:.3f} — no breathing gaps (TTS never breathes)")

    breath_count = signals.get("breath_event_count", None)
    if breath_count is not None:
        vote(breath_count == 0, 0.08,
             "No breath events detected — strong TTS/voice-cloning indicator")

    # ── Spectral signals ─────────────────────────────────────────
    hnr = signals.get("harmonic_to_noise_ratio", None)
    if hnr is not None:
        vote(hnr > 15.0, 0.12,
             f"HNR={hnr:.1f} — too-perfect harmonic clarity (real voices have noise)")

    zcr_std = signals.get("zero_crossing_rate_std", None)
    if zcr_std is not None:
        vote(zcr_std < 0.02, 0.08,
             f"ZCR std={zcr_std:.4f} — too-smooth signal texture (AI voice artifact)")

    phase_disc = signals.get("phase_discontinuity", None)
    if phase_disc is not None:
        vote(phase_disc > 1.8, 0.10,
             f"Phase discontinuity={phase_disc:.3f} — voice cloning phase artifacts")

    # ── MFCC signals ─────────────────────────────────────────────
    mfcc_std = signals.get("mfcc_std", None)
    if mfcc_std:
        avg_mfcc_std = np.mean(mfcc_std)
        vote(avg_mfcc_std < 8.0, 0.15,
             f"Avg MFCC std={avg_mfcc_std:.2f} — vocal tract too static (TTS has no coarticulation)")

    if total_votes == 0:
        return {"ai_score": 0.5, "human_score": 0.5, "verdict": "UNCERTAIN", "flags": []}

    ai_score = ai_votes / total_votes

    if ai_score >= 0.70:
        verdict = "AI_GENERATED"
    elif ai_score <= 0.30:
        verdict = "HUMAN"
    else:
        verdict = "UNCERTAIN"

    return {
        "ai_score":    round(ai_score, 4),
        "human_score": round(1.0 - ai_score, 4),
        "verdict":     verdict,
        "flags":       flags,
    }


# ── CLI test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python3 audio_analyzer.py <audio_file.wav>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        data = f.read()

    ext = sys.argv[1].rsplit(".", 1)[-1].lower()
    signals = analyze_audio(data, file_format=ext)
    verdict = score_audio(signals)

    print("\n═══ AUDIO ANALYSIS REPORT ═══")
    print(f"  Duration:          {signals.get('duration_sec', 0):.2f}s")
    print(f"  F0 mean:           {signals.get('f0_mean', 0):.1f} Hz")
    print(f"  F0 std:            {signals.get('f0_std', 0):.2f} Hz")
    print(f"  F0 range:          {signals.get('f0_range', 0):.1f} Hz")
    print(f"  Silence ratio:     {signals.get('silence_ratio', 0):.3f}")
    print(f"  Breath events:     {signals.get('breath_event_count', 0)}")
    print(f"  HNR:               {signals.get('harmonic_to_noise_ratio', 0):.2f}")
    print(f"  Phase disc:        {signals.get('phase_discontinuity', 0):.3f}")
    print(f"  RMS std:           {signals.get('rms_energy_std', 0):.4f}")
    print(f"\n═══ VERDICT ═══")
    print(f"  AI Score:    {verdict['ai_score']:.1%}")
    print(f"  Human Score: {verdict['human_score']:.1%}")
    print(f"  Verdict:     {verdict['verdict']}")
    if verdict["flags"]:
        print(f"\n  Triggered signals:")
        for flag in verdict["flags"]:
            print(f"    ⚑ {flag}")
