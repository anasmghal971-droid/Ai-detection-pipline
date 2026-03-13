// ============================================================
// DETECT-AI: Vercel API — /api/dataset/shards
// GET /api/dataset/shards?content_type=text&language=en
// ============================================================
import type { NextRequest } from "next/server";
export const runtime = "edge";

const SUPABASE_URL = process.env.SUPABASE_URL!;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_KEY!;
const UPSTASH_URL  = process.env.UPSTASH_REDIS_REST_URL!;
const UPSTASH_TOKEN = process.env.UPSTASH_REDIS_REST_TOKEN!;

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
async function cacheSet(key: string, val: string, ttl = 3600) {
  if (!UPSTASH_URL) return;
  try {
    await fetch(`${UPSTASH_URL}/set/${encodeURIComponent(key)}/${encodeURIComponent(val)}/ex/${ttl}`, {
      headers: { Authorization: `Bearer ${UPSTASH_TOKEN}` },
    });
  } catch {}
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const content_type = searchParams.get("content_type") ?? "";
  const language     = searchParams.get("language") ?? "";

  const cacheKey = `shards:${content_type}:${language}`;
  const cached   = await cacheGet(cacheKey);
  if (cached) return new Response(cached, {
    headers: { "Content-Type": "application/json", "X-Cache": "HIT" },
  });

  const params = new URLSearchParams({
    select: "shard_id,content_type,language,sample_count,size_bytes,sha256_hash,created_at,hf_url,push_status,source_distribution",
    push_status: "eq.pushed",
    order: "created_at.desc",
  });
  if (content_type) params.set("content_type", `eq.${content_type}`);
  if (language)     params.set("language",     `eq.${language}`);

  const r = await fetch(`${SUPABASE_URL}/rest/v1/shard_registry?${params}`, {
    headers: { "apikey": SUPABASE_KEY, "Authorization": `Bearer ${SUPABASE_KEY}` },
    signal: AbortSignal.timeout(8000),
  });

  const shards = await r.json() as unknown[];
  const body = JSON.stringify({
    shards,
    count: shards.length,
    hf_dataset: `https://huggingface.co/datasets/anas775/DETECT-AI-Dataset`,
  });

  await cacheSet(cacheKey, body);
  return new Response(body, { headers: { "Content-Type": "application/json", "X-Cache": "MISS" } });
}
