// ============================================================
// DETECT-AI: Vercel API — /api/dataset/stats
// GET /api/dataset/stats  — pipeline health dashboard
// ============================================================
import type { NextRequest } from "next/server";
export const runtime = "edge";

const SUPABASE_URL  = process.env.SUPABASE_URL!;
const SUPABASE_KEY  = process.env.SUPABASE_SERVICE_KEY!;
const UPSTASH_URL   = process.env.UPSTASH_REDIS_REST_URL!;
const UPSTASH_TOKEN = process.env.UPSTASH_REDIS_REST_TOKEN!;

async function sbQuery(table: string, params: Record<string, string>) {
  const p = new URLSearchParams(params);
  const r = await fetch(`${SUPABASE_URL}/rest/v1/${table}?${p}`, {
    headers: {
      "apikey": SUPABASE_KEY,
      "Authorization": `Bearer ${SUPABASE_KEY}`,
      "Prefer": "count=exact",
      "Range": "0-0",
    },
    signal: AbortSignal.timeout(5000),
  });
  const range = r.headers.get("Content-Range") ?? "0/0";
  return parseInt(range.split("/")[1] ?? "0", 10);
}

async function cacheGet(key: string) {
  if (!UPSTASH_URL) return null;
  try {
    const r = await fetch(`${UPSTASH_URL}/get/${encodeURIComponent(key)}`, {
      headers: { Authorization: `Bearer ${UPSTASH_TOKEN}` },
    });
    const d = await r.json() as { result: string | null };
    return d.result;
  } catch { return null; }
}
async function cacheSet(key: string, val: string, ttl = 60) {
  if (!UPSTASH_URL) return;
  try {
    await fetch(`${UPSTASH_URL}/set/${encodeURIComponent(key)}/${encodeURIComponent(val)}/ex/${ttl}`, {
      headers: { Authorization: `Bearer ${UPSTASH_TOKEN}` },
    });
  } catch {}
}

export async function GET(_request: NextRequest) {
  const cached = await cacheGet("stats:pipeline");
  if (cached) return new Response(cached, {
    headers: { "Content-Type": "application/json", "X-Cache": "HIT" },
  });

  // Run all count queries in parallel
  const [
    totalStaged, totalLabeled, totalExported,
    aiGenerated, human, uncertain, verified,
    shardsPushed, pendingVerify, pendingFrameJobs,
    activeSources,
  ] = await Promise.all([
    sbQuery("samples_staging",    { select: "sample_id", status: "eq.staged" }),
    sbQuery("samples_processed",  { select: "sample_id" }),
    sbQuery("samples_processed",  { select: "sample_id", exported_at: "not.is.null" }),
    sbQuery("samples_processed",  { select: "sample_id", label: "eq.AI_GENERATED" }),
    sbQuery("samples_processed",  { select: "sample_id", label: "eq.HUMAN" }),
    sbQuery("samples_processed",  { select: "sample_id", label: "eq.UNCERTAIN" }),
    sbQuery("samples_processed",  { select: "sample_id", verified: "eq.true" }),
    sbQuery("shard_registry",     { select: "shard_id",  push_status: "eq.pushed" }),
    sbQuery("verification_queue", { select: "id",        status: "eq.pending" }),
    sbQuery("frame_jobs",         { select: "job_id",    status: "eq.pending" }),
    sbQuery("sources_queue",      { select: "source_id", status: "eq.active" }),
  ]);

  // Get latest metric snapshot
  const metricsR = await fetch(
    `${SUPABASE_URL}/rest/v1/pipeline_metrics?order=measured_at.desc&limit=1&select=measured_at,samples_per_min,error_rate_pct`,
    { headers: { "apikey": SUPABASE_KEY, "Authorization": `Bearer ${SUPABASE_KEY}` } }
  );
  const metricsData = await metricsR.json() as Array<{
    measured_at: string; samples_per_min: number; error_rate_pct: number;
  }>;
  const latestMetric = metricsData[0] ?? null;

  const body = JSON.stringify({
    pipeline: {
      active_sources:      activeSources,
      samples_staged:      totalStaged,
      samples_labeled:     totalLabeled,
      samples_exported:    totalExported,
      pending_frame_jobs:  pendingFrameJobs,
      pending_verification: pendingVerify,
    },
    labels: {
      ai_generated: aiGenerated,
      human,
      uncertain,
      verified,
      ai_pct: totalLabeled > 0 ? Math.round((aiGenerated / totalLabeled) * 1000) / 10 : 0,
    },
    shards: {
      pushed: shardsPushed,
      hf_dataset: "https://huggingface.co/datasets/anas775/DETECT-AI-Dataset",
    },
    throughput: {
      samples_per_min:  latestMetric?.samples_per_min ?? null,
      error_rate_pct:   latestMetric?.error_rate_pct  ?? null,
      last_measured_at: latestMetric?.measured_at ?? null,
    },
    generated_at: new Date().toISOString(),
  });

  await cacheSet("stats:pipeline", body, 60);
  return new Response(body, { headers: { "Content-Type": "application/json", "X-Cache": "MISS" } });
}
