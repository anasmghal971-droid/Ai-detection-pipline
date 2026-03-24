// ============================================================
// DETECT-AI: Dispatcher v3 — 20 Concurrent Workers
// 
// Architecture:
//   - 20 parallel slots per cron tick (was 3)
//   - Smart rotation: after each scrape, source drops to BOTTOM
//     of priority so all 27 sources get equal turns
//   - Hard 25s timeout per scraper call (CF worker limit = 30s)
//   - Full error isolation — one failed source never blocks others
//   - Auto-cleanup trigger after every cron cycle
// ============================================================

import type { SourceQueueItem } from "../../shared/types/index";
import { createSupabaseClient } from "./supabase";
import { PipelineLogger } from "./logger";

const WORKER_TIMEOUT_MS    = 25_000;  // 25s — under CF 30s limit
const HEARTBEAT_MS         = 8_000;
const RATE_LIMIT_BACKOFF   = 300_000; // 5 min cooldown on rate limit
const CONCURRENT_SLOTS     = 20;      // 20 workers in parallel per cron tick
const MAX_RETRY_COUNT      = 2;       // fail fast — don't waste slots on broken sources

// ── Env type ─────────────────────────────────────────────────
interface Fetcher { fetch(url: string, init?: RequestInit): Promise<Response> }
interface Env {
  SUPABASE_URL:         string;
  SUPABASE_SERVICE_KEY: string;
  HF_TOKEN?:            string;
  WORKER_ID?:           string;
  PIPELINE_ENABLED?:    string;
  // Service bindings
  SCRAPER_TEXT:         Fetcher;
  SCRAPER_IMAGE:        Fetcher;
  SCRAPER_VIDEO:        Fetcher;
  SCRAPER_TEXT_NEWS:    Fetcher;
  SCRAPER_IMAGE_OPEN:   Fetcher;
}

// ── Source → scraper binding map ─────────────────────────────
function getScraperBinding(sourceId: string, env: Env): Fetcher | null {
  const map: Record<string, Fetcher> = {
    // Text sources
    "bbc-news":       env.SCRAPER_TEXT,
    "reuters":        env.SCRAPER_TEXT,
    "guardian":       env.SCRAPER_TEXT,
    "nytimes":        env.SCRAPER_TEXT,
    "aljazeera":      env.SCRAPER_TEXT,
    "wikipedia":      env.SCRAPER_TEXT,
    "arxiv":          env.SCRAPER_TEXT,
    "npr":            env.SCRAPER_TEXT,
    "paperswithcode": env.SCRAPER_TEXT,
    "stackexchange":  env.SCRAPER_TEXT,
    "reddit":         env.SCRAPER_TEXT,
    "worldbank":      env.SCRAPER_TEXT,
    // Real image sources
    "unsplash":       env.SCRAPER_IMAGE,
    "pexels":         env.SCRAPER_IMAGE,
    "pixabay":        env.SCRAPER_IMAGE,
    "openverse":      env.SCRAPER_IMAGE,
    "wikimedia":      env.SCRAPER_IMAGE,
    "nasa":           env.SCRAPER_IMAGE,
    "met-museum":     env.SCRAPER_IMAGE,
    // AI-generated image sources
    "civitai":        env.SCRAPER_IMAGE,
    "pollinations":   env.SCRAPER_IMAGE,
    "diffusiondb":    env.SCRAPER_IMAGE,
    "lexica":         env.SCRAPER_IMAGE,
    // Video sources
    "youtube":        env.SCRAPER_VIDEO,
    "pexels-video":   env.SCRAPER_VIDEO,
    "ted-talks":      env.SCRAPER_VIDEO,
    "voxceleb":       env.SCRAPER_VIDEO,
  };
  return map[sourceId] ?? null;
}

// ── DB helpers ────────────────────────────────────────────────
function db(env: Env) {
  return createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
}

async function claimSource(env: Env, workerId: string): Promise<SourceQueueItem | null> {
  const d = db(env);
  const now = new Date().toISOString();

  // Release timed-out active sources
  await d.from("sources_queue").update({
    status: "pending", worker_id: null, heartbeat_at: null,
    error_message: "Timed out — auto-requeued"
  }).eq("status", "active").lt("heartbeat_at", new Date(Date.now() - WORKER_TIMEOUT_MS).toISOString());

  // Release rate-limited sources that have cooled down
  await d.from("sources_queue").update({ status: "pending", worker_id: null })
    .eq("status", "rate_limited")
    .lt("updated_at", new Date(Date.now() - RATE_LIMIT_BACKOFF).toISOString());

  // Claim highest-priority pending source
  const { data } = await d.from("sources_queue").select("*")
    .eq("status", "pending")
    .lt("retry_count", MAX_RETRY_COUNT)
    .order("priority_score", { ascending: false })
    .limit(1);

  if (!data || data.length === 0) return null;
  const source = data[0] as SourceQueueItem;

  // Optimistic lock
  const { error } = await d.from("sources_queue").update({
    status: "active", worker_id: workerId, heartbeat_at: now
  }).eq("source_id", source.source_id).eq("status", "pending");

  return error ? null : source;
}

async function releaseSource(
  env: Env, sourceId: string, 
  success: boolean, samplesCount: number, errorMsg?: string
): Promise<void> {
  const d = db(env);
  // After success: drop priority by 10 so other sources get turns
  // After failure: increment retry_count, keep low priority
  if (success) {
    await d.from("sources_queue").update({
      status: "pending",
      worker_id: null, heartbeat_at: null,
      last_crawled_at: new Date().toISOString(),
      retry_count: 0, error_message: null,
      priority_score: undefined, // will use SQL to decrement
    }).eq("source_id", sourceId);
    // Lower priority after successful scrape (fair rotation)
    await db(env).from("sources_queue").update({
      priority_score: undefined,
    }).eq("source_id", sourceId);
    // Use RPC to decrement
    const { data: row } = await d.from("sources_queue").select("priority_score").eq("source_id", sourceId);
    if (row && row[0]) {
      const newScore = Math.max(1, ((row[0] as any).priority_score ?? 50) - 10);
      await d.from("sources_queue").update({ priority_score: newScore }).eq("source_id", sourceId);
    }
  } else {
    await d.from("sources_queue").update({
      status: "pending",
      worker_id: null, heartbeat_at: null,
      error_message: errorMsg ?? "Unknown error",
    }).eq("source_id", sourceId);
    // Raw increment retry_count via select then update
    const { data: row } = await d.from("sources_queue").select("retry_count").eq("source_id", sourceId);
    const rc = row && row[0] ? ((row[0] as any).retry_count ?? 0) + 1 : 1;
    await d.from("sources_queue").update({ retry_count: rc }).eq("source_id", sourceId);
  }
}

async function markRateLimited(env: Env, sourceId: string): Promise<void> {
  await db(env).from("sources_queue").update({
    status: "rate_limited", worker_id: null, error_message: "Rate limited — cooling 5min"
  }).eq("source_id", sourceId);
}

async function heartbeat(env: Env, sourceId: string, workerId: string): Promise<void> {
  await db(env).from("sources_queue").update({
    heartbeat_at: new Date().toISOString()
  }).eq("source_id", sourceId).eq("worker_id", workerId);
}

// ── Call scraper via service binding ─────────────────────────
async function callScraper(
  scraper: Fetcher, source: SourceQueueItem, env: Env, workerId: string
): Promise<{ samplesScraped: number; rateLimited: boolean }> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), WORKER_TIMEOUT_MS);

  try {
    const res = await scraper.fetch("https://internal/scrape", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source,
        worker_id: workerId,
        supabase_url: env.SUPABASE_URL,
        supabase_key: env.SUPABASE_SERVICE_KEY,
      }),
      signal: ctrl.signal,
    });
    clearTimeout(timer);

    if (res.status === 429) return { samplesScraped: 0, rateLimited: true };
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text().catch(() => "")}`);

    const json = await res.json() as { samples_scraped?: number };
    return { samplesScraped: json.samples_scraped ?? 0, rateLimited: false };
  } finally {
    clearTimeout(timer);
  }
}

// ── Process one source slot ───────────────────────────────────
async function processOneSlot(env: Env, workerId: string, logger: PipelineLogger): Promise<void> {
  const source = await claimSource(env, workerId).catch(() => null);
  if (!source) return; // No more pending sources

  const scraper = getScraperBinding(source.source_id, env);
  if (!scraper) {
    logger.log("SCRAPE_ERROR", {
      source_id: source.source_id,
      error_message: `No binding for source: ${source.source_id} — check dispatcher scraperMap`,
    });
    await releaseSource(env, source.source_id, false, 0, "No scraper binding");
    return;
  }

  // Heartbeat while scraping
  const hb = setInterval(() => heartbeat(env, source.source_id, workerId), HEARTBEAT_MS);

  try {
    const result = await callScraper(scraper, source, env, workerId);
    clearInterval(hb);

    if (result.rateLimited) {
      await markRateLimited(env, source.source_id);
      logger.log("RATE_LIMIT_HIT", { source_id: source.source_id });
    } else {
      await releaseSource(env, source.source_id, true, result.samplesScraped);
      logger.log("SCRAPE_COMPLETE", {
        source_id:    source.source_id,
        sample_count: result.samplesScraped,
      });
    }
  } catch (err: unknown) {
    clearInterval(hb);
    const msg = err instanceof Error ? err.message : String(err);
    await releaseSource(env, source.source_id, false, 0, msg);
    logger.log("SCRAPE_ERROR", { source_id: source.source_id, error_message: msg });
  }
}

// ── Cleanup trigger ───────────────────────────────────────────
async function triggerCleanup(env: Env): Promise<void> {
  try {
    const d = db(env);
    // Delete old staging rows (>24hrs)
    await d.from("samples_staging").delete()
      .lt("scraped_at", new Date(Date.now() - 86_400_000).toISOString())
      .eq("status", "staged");
    // Delete old processed rows (>72hrs)
    await d.from("samples_processed").delete()
      .lt("labeled_at", new Date(Date.now() - 259_200_000).toISOString());
  } catch { /* non-critical */ }
}

// ── Main export ───────────────────────────────────────────────
export default {
  // HTTP handler for manual triggers
  async fetch(request: Request, env: Env): Promise<Response> {
    if (env.PIPELINE_ENABLED === "false") {
      return new Response(JSON.stringify({ status: "disabled" }), { status: 200 });
    }
    const logger = new PipelineLogger(env);
    const workerId = `manual-${Date.now()}`;
    await processOneSlot(env, workerId, logger);
    await logger.flush();
    return new Response(JSON.stringify({ status: "ok" }), { status: 200 });
  },

  // Cron: fires every 5 minutes — runs 20 slots in parallel
  async scheduled(_event: ScheduledEvent, env: Env, ctx: ExecutionContext): Promise<void> {
    if (env.PIPELINE_ENABLED === "false") return;

    ctx.waitUntil((async () => {
      const logger = new PipelineLogger(env);
      logger.log("CRON_TICK", { slots: CONCURRENT_SLOTS, ts: new Date().toISOString() });

      // 20 parallel slots — each independently claims + processes a source
      const slots = Array.from({ length: CONCURRENT_SLOTS }, (_, i) =>
        processOneSlot(env, `cron-slot-${i + 1}`, logger).catch(e =>
          logger.log("SLOT_ERROR", { slot: i + 1, error: String(e) })
        )
      );

      await Promise.allSettled(slots);

      // Cleanup old data after every cron cycle
      await triggerCleanup(env);

      await logger.flush();
    })());
  },
};
