/**
 * DETECT-AI — Supabase Edge Function: analyze-audio
 * Deployed at: https://igwimowqtbgatqvdrqjf.supabase.co/functions/v1/analyze-audio
 *
 * Receives audio bytes → returns AI/human verdict in <3 seconds.
 * Called by detect-ai-nu.vercel.app when users upload audio.
 *
 * Uses lightweight JS signal analysis (no Python needed at edge).
 * For full 20-signal analysis, delegates to GitHub Actions batch.
 */

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// ── Fast audio analysis in Deno (subset of full 20 signals) ──────
// Uses Web Audio API concepts — fast enough for real-time edge use

function decodeWavHeader(buffer: ArrayBuffer): { sampleRate: number; numChannels: number; bitsPerSample: number; dataOffset: number; dataLength: number } | null {
  const view = new DataView(buffer);
  if (view.getUint32(0) !== 0x52494646) return null; // "RIFF"
  const sampleRate    = view.getUint32(24, true);
  const numChannels   = view.getUint16(22, true);
  const bitsPerSample = view.getUint16(34, true);
  const dataOffset    = 44;
  const dataLength    = view.getUint32(40, true);
  return { sampleRate, numChannels, bitsPerSample, dataOffset, dataLength };
}

function extractFloat32(buffer: ArrayBuffer, header: ReturnType<typeof decodeWavHeader>): Float32Array {
  if (!header) return new Float32Array(0);
  const { bitsPerSample, dataOffset, dataLength, numChannels } = header;
  const raw = new DataView(buffer, dataOffset, Math.min(dataLength, buffer.byteLength - dataOffset));
  const bytesPerSample = bitsPerSample / 8;
  const totalSamples = Math.floor(raw.byteLength / bytesPerSample);
  const monoSamples = Math.floor(totalSamples / numChannels);
  const out = new Float32Array(monoSamples);

  for (let i = 0; i < monoSamples; i++) {
    const offset = i * numChannels * bytesPerSample;
    let val = 0;
    if (bitsPerSample === 16) val = raw.getInt16(offset, true) / 32768;
    else if (bitsPerSample === 32) val = raw.getFloat32(offset, true);
    else if (bitsPerSample === 8) val = (raw.getUint8(offset) - 128) / 128;
    out[i] = val;
  }
  return out;
}

function computeRMSFrames(samples: Float32Array, frameSize = 2048, hop = 512): Float32Array {
  const n = Math.floor((samples.length - frameSize) / hop) + 1;
  const rms = new Float32Array(Math.max(1, n));
  for (let i = 0; i < n; i++) {
    const start = i * hop;
    let sum = 0;
    for (let j = 0; j < frameSize && start + j < samples.length; j++) {
      sum += samples[start + j] ** 2;
    }
    rms[i] = Math.sqrt(sum / frameSize);
  }
  return rms;
}

function computeZCR(samples: Float32Array, frameSize = 2048, hop = 512): Float32Array {
  const n = Math.floor((samples.length - frameSize) / hop) + 1;
  const zcr = new Float32Array(Math.max(1, n));
  for (let i = 0; i < n; i++) {
    const start = i * hop;
    let crossings = 0;
    for (let j = 1; j < frameSize && start + j < samples.length; j++) {
      if (Math.sign(samples[start + j]) !== Math.sign(samples[start + j - 1])) crossings++;
    }
    zcr[i] = crossings / frameSize;
  }
  return zcr;
}

function estimateF0(samples: Float32Array, sr: number): { mean: number; std: number; range: number } {
  const windowSamples = Math.floor(0.03 * sr);
  const stepSamples   = Math.floor(0.01 * sr);
  const f0List: number[] = [];

  for (let i = 0; i + windowSamples < samples.length; i += stepSamples) {
    const chunk = samples.slice(i, i + windowSamples);
    // Autocorrelation
    const minLag = Math.floor(sr / 800);
    const maxLag = Math.floor(sr / 60);
    let maxAC = 0, bestLag = 0;
    const ac0 = chunk.reduce((s, v) => s + v * v, 0);
    for (let lag = minLag; lag <= maxLag && lag < chunk.length; lag++) {
      let ac = 0;
      for (let j = 0; j + lag < chunk.length; j++) ac += chunk[j] * chunk[j + lag];
      if (ac > maxAC) { maxAC = ac; bestLag = lag; }
    }
    if (maxAC > 0.3 * ac0 && ac0 > 0 && bestLag > 0) {
      f0List.push(sr / bestLag);
    }
  }

  if (f0List.length === 0) return { mean: 0, std: 0, range: 0 };
  const mean = f0List.reduce((a, b) => a + b, 0) / f0List.length;
  const variance = f0List.reduce((s, v) => s + (v - mean) ** 2, 0) / f0List.length;
  return {
    mean,
    std:   Math.sqrt(variance),
    range: Math.max(...f0List) - Math.min(...f0List),
  };
}

function countBreathEvents(rms: Float32Array): number {
  let count = 0, inBreath = false;
  for (const r of rms) {
    if (!inBreath && r > 0.005 && r < 0.04) inBreath = true;
    else if (inBreath && r > 0.04)           { count++; inBreath = false; }
    else if (inBreath && r < 0.005)           inBreath = false;
  }
  return count;
}

function mean(arr: Float32Array | number[]): number {
  return Array.from(arr).reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr: Float32Array | number[]): number {
  const m = mean(arr);
  return Math.sqrt(Array.from(arr).reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
}

function analyzeAudio(samples: Float32Array, sr: number, durationSec: number) {
  const rms = computeRMSFrames(samples);
  const zcr = computeZCR(samples);
  const f0  = estimateF0(samples, sr);

  const silenceThresh = 0.01;
  const silenceFrames = Array.from(rms).filter(v => v < silenceThresh).length;
  const silenceRatio  = silenceFrames / rms.length;
  const breathCount   = countBreathEvents(rms);

  return {
    duration_sec:              durationSec,
    rms_energy_mean:           mean(rms),
    rms_energy_std:            std(rms),
    zero_crossing_rate_mean:   mean(zcr),
    zero_crossing_rate_std:    std(zcr),
    f0_mean:                   f0.mean,
    f0_std:                    f0.std,
    f0_range:                  f0.range,
    silence_ratio:             silenceRatio,
    breath_event_count:        breathCount,
  };
}

function scoreAudio(signals: Record<string, number>) {
  const flags: string[] = [];
  let aiVotes = 0, total = 0;

  const vote = (cond: boolean, w: number, msg: string) => {
    total += w;
    if (cond) { aiVotes += w; flags.push(msg); }
  };

  vote(signals.f0_std < 8  && signals.f0_std >= 0, 0.25, `Pitch too stable (std=${signals.f0_std?.toFixed(1)}Hz) — TTS signature`);
  vote(signals.f0_range < 40 && signals.f0_range >= 0, 0.15, `Narrow pitch range (${signals.f0_range?.toFixed(0)}Hz) — lacks natural prosody`);
  vote(signals.rms_energy_std < 0.015, 0.15, `Energy too uniform (σ=${signals.rms_energy_std?.toFixed(4)}) — AI voice consistency`);
  vote(signals.silence_ratio < 0.03, 0.15, `No natural pauses (${(signals.silence_ratio*100)?.toFixed(1)}%) — AI never breathes`);
  vote(signals.breath_event_count === 0, 0.15, `Zero breath events — strong TTS/voice-cloning indicator`);
  vote(signals.zero_crossing_rate_std < 0.02, 0.15, `ZCR too smooth (σ=${signals.zero_crossing_rate_std?.toFixed(4)}) — AI voice artifact`);

  const aiScore = total > 0 ? aiVotes / total : 0.5;
  const verdict = aiScore >= 0.65 ? "AI_GENERATED" : aiScore <= 0.30 ? "HUMAN" : "UNCERTAIN";

  return { ai_score: aiScore, human_score: 1 - aiScore, verdict, flags };
}

function buildReport(signals: Record<string, number>, verdict: ReturnType<typeof scoreAudio>) {
  const summary = verdict.verdict === "AI_GENERATED"
    ? "🤖 This audio is very likely AI-generated or synthetic."
    : verdict.verdict === "HUMAN"
    ? "✅ This audio appears to be authentic human speech."
    : "⚠️ Inconclusive — this audio has mixed characteristics.";

  return {
    summary,
    color: verdict.verdict === "AI_GENERATED" ? "red" : verdict.verdict === "HUMAN" ? "green" : "yellow",
    breakdown: [
      {
        signal: "Pitch Variation",
        value: `${signals.f0_std?.toFixed(1)} Hz std`,
        ai_indicator: signals.f0_std < 8,
        explanation: "Real voices have natural pitch variation (>10 Hz). TTS maintains unnaturally stable pitch.",
      },
      {
        signal: "Breathing & Pauses",
        value: `${signals.breath_event_count} breaths, ${(signals.silence_ratio * 100).toFixed(1)}% silence`,
        ai_indicator: signals.breath_event_count === 0 || signals.silence_ratio < 0.03,
        explanation: "Real speakers breathe naturally. AI voices have zero breath events.",
      },
      {
        signal: "Energy Dynamics",
        value: `σ = ${signals.rms_energy_std?.toFixed(4)}`,
        ai_indicator: signals.rms_energy_std < 0.015,
        explanation: "Real voices vary in loudness naturally. AI voices are too consistent.",
      },
      {
        signal: "Signal Smoothness (ZCR)",
        value: `σ = ${signals.zero_crossing_rate_std?.toFixed(4)}`,
        ai_indicator: signals.zero_crossing_rate_std < 0.02,
        explanation: "AI voice synthesis produces unnaturally smooth signal texture.",
      },
    ],
  };
}

// ── Main handler ──────────────────────────────────────────────────
serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const body = await req.json() as {
      audio_base64: string;
      file_format:  string;
      store_in_dataset?: boolean;
      user_id?: string;
    };

    if (!body.audio_base64 || !body.file_format) {
      return new Response(
        JSON.stringify({ error: "audio_base64 and file_format required" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Decode audio
    const audioBytes = Uint8Array.from(atob(body.audio_base64), c => c.charCodeAt(0));
    const buffer = audioBytes.buffer;

    // Only WAV supported at edge (no FFmpeg at edge)
    // MP3/OGG → convert client-side to WAV before sending
    const header = decodeWavHeader(buffer);
    if (!header) {
      return new Response(
        JSON.stringify({ error: "Please convert audio to WAV format before uploading, or use the full API." }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const samples = extractFloat32(buffer, header);
    const durationSec = samples.length / header.sampleRate;

    if (durationSec < 1) {
      return new Response(
        JSON.stringify({ error: "Audio too short. Please upload at least 1 second." }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const signals = analyzeAudio(samples, header.sampleRate, durationSec);
    const verdict = scoreAudio(signals);
    const report  = buildReport(signals, verdict);

    const result = {
      sample_id:          crypto.randomUUID(),
      verdict:            verdict.verdict,
      ai_probability:     Math.round(verdict.ai_score * 1000) / 1000,
      human_probability:  Math.round(verdict.human_score * 1000) / 1000,
      confidence:         Math.round(Math.max(verdict.ai_score, verdict.human_score) * 1000) / 1000,
      duration_sec:       Math.round(durationSec * 100) / 100,
      signals:            signals,
      report:             report,
      flags:              verdict.flags,
      analysis_engine:    "DETECT-AI Edge v1.0 (8 signals)",
      timestamp:          new Date().toISOString(),
    };

    return new Response(
      JSON.stringify(result),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );

  } catch (err) {
    return new Response(
      JSON.stringify({ error: String(err) }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
