// ============================================================
// DETECT-AI: Vercel API — /api/dataset/query
// GET /api/dataset/query?content_type=text&language=en&label=AI_GENERATED&page=1&page_size=100
// ============================================================

import type { NextRequest } from "next/server";

export const runtime = "edge";
export const revalidate = 0;

const SUPABASE_URL = process.env.SUPABASE_URL!;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_KEY!;
const UPSTASH_URL  = process.env.UPSTASH_REDIS_REST_URL!;
const UPSTASH_TOKEN = process.env.UPSTASH_REDIS_REST_TOKEN!;
const CACHE_TTL    = 300;   // 5 minutes
const MAX_PAGE_SIZE = 1000;
const DEFAULT_PAGE_SIZE = 100;

// ── Redis cache (Upstash REST API) ──────────────────────────
async function cacheGet(key: string): Promise<string | null> {
  if (!UPSTASH_URL) return null;
  try {
    const r = await fetch(`${UPSTASH_URL}/get/${encodeURIComponent(key)}`, {
      headers: { Authorization: `Bearer ${UPSTASH_TOKEN}` },
    });
    const data = await r.json() as { result: string | null };
    return data.result;
  } catch { return null; }
}

async function cacheSet(key: string, value: string, ttl = CACHE_TTL): Promise<void> {
  if (!UPSTASH_URL) return;
  try {
    await fetch(`${UPSTASH_URL}/set/${encodeURIComponent(key)}/${encodeURIComponent(value)}/ex/${ttl}`, {
      headers: { Authorization: `Bearer ${UPSTASH_TOKEN}` },
    });
  } catch { /* non-critical */ }
}

// ── Supabase query ────────────────────────────────────────────
async function querySupabase(params: Record<string, string>): Promise<{
  data: unknown[]; count: number; error?: string;
}> {
  const url    = new URL(`${SUPABASE_URL}/rest/v1/samples_processed`);
  const headers = {
    "apikey":        SUPABASE_KEY,
    "Authorization": `Bearer ${SUPABASE_KEY}`,
    "Prefer":        "count=exact",
    "Range":         `${params._range_start}-${params._range_end}`,
  };

  // Build filter params
  const filterParams = new URLSearchParams();
  filterParams.set("select", "sample_id,source_id,source_url,content_type,language,label,final_confidence,verified,scraped_at,labeled_at,metadata");

  if (params.content_type) filterParams.set("content_type", `eq.${params.content_type}`);
  if (params.language)     filterParams.set("language",     `eq.${params.language}`);
  if (params.label)        filterParams.set("label",        `eq.${params.label}`);
  if (params.source_id)    filterParams.set("source_id",    `eq.${params.source_id}`);
  if (params.verified)     filterParams.set("verified",     `eq.${params.verified}`);
  if (params.min_confidence) filterParams.set("final_confidence", `gte.${params.min_confidence}`);

  filterParams.set("order", "labeled_at.desc");

  const r = await fetch(`${url.origin}${url.pathname}?${filterParams}`, {
    headers,
    signal: AbortSignal.timeout(8000),
  });

  if (!r.ok) {
    return { data: [], count: 0, error: `Supabase ${r.status}` };
  }

  const data  = await r.json() as unknown[];
  const range = r.headers.get("Content-Range") ?? "0/0";
  const count = parseInt(range.split("/")[1] ?? "0", 10);
  return { data, count };
}

// ── Handler ───────────────────────────────────────────────────
export async function GET(request: NextRequest): Promise<Response> {
  const start = Date.now();
  const { searchParams } = new URL(request.url);

  // Parse + validate params
  const content_type = searchParams.get("content_type") ?? "";
  const language     = searchParams.get("language") ?? "";
  const label        = searchParams.get("label") ?? "";
  const source_id    = searchParams.get("source_id") ?? "";
  const verified     = searchParams.get("verified") ?? "";
  const min_confidence = searchParams.get("min_confidence") ?? "";
  const page         = Math.max(1, parseInt(searchParams.get("page") ?? "1", 10));
  const page_size    = Math.min(MAX_PAGE_SIZE,
    Math.max(1, parseInt(searchParams.get("page_size") ?? String(DEFAULT_PAGE_SIZE), 10)));

  // Validate enums
  if (content_type && !["text","image","video","audio"].includes(content_type)) {
    return Response.json({ error: "Invalid content_type" }, { status: 400 });
  }
  if (label && !["AI_GENERATED","HUMAN","UNCERTAIN"].includes(label)) {
    return Response.json({ error: "Invalid label" }, { status: 400 });
  }

  // Build cache key
  const cacheKey = `query:${content_type}:${language}:${label}:${source_id}:${verified}:${min_confidence}:${page}:${page_size}`;
  const cached = await cacheGet(cacheKey);
  if (cached) {
    return new Response(cached, {
      headers: {
        "Content-Type":  "application/json",
        "X-Cache":       "HIT",
        "X-Response-Ms": String(Date.now() - start),
      },
    });
  }

  const range_start = (page - 1) * page_size;
  const range_end   = range_start + page_size - 1;

  const { data, count, error } = await querySupabase({
    content_type, language, label, source_id, verified, min_confidence,
    _range_start: String(range_start),
    _range_end:   String(range_end),
  });

  if (error) {
    return Response.json({ error }, { status: 500 });
  }

  const total_pages = Math.ceil(count / page_size);
  const responseBody = JSON.stringify({
    data,
    pagination: {
      page,
      page_size,
      total_count: count,
      total_pages,
      has_next: page < total_pages,
      has_prev: page > 1,
    },
    filters: { content_type, language, label, source_id, verified, min_confidence },
    response_ms: Date.now() - start,
  });

  // Cache and return
  await cacheSet(cacheKey, responseBody);

  return new Response(responseBody, {
    headers: {
      "Content-Type":  "application/json",
      "X-Cache":       "MISS",
      "X-Response-Ms": String(Date.now() - start),
    },
  });
}
