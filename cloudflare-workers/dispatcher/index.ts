// ============================================================
// DETECT-AI: Dispatcher v3.3 — Atomic RPC claiming (SELECT FOR UPDATE SKIP LOCKED)
//
// Changes from v3:
//   FIX 1: releaseSource() — removed `priority_score: undefined`
//           which serialised to {} and caused PostgREST 400 errors
//   FIX 2: CRON_TICK is now immediately flushed (awaited) so it
//           always lands in worker_logs regardless of run length
//   FIX 3: Added Supabase connectivity check on startup to detect
//           missing secrets early (visible in CF dashboard logs)
// ============================================================

import type { SourceQueueItem } from "../../shared/types/index";
import { createSupabaseClient } from "./supabase";
import { PipelineLogger } from "./logger";

const WORKER_TIMEOUT_MS    = 25_000;
const HEARTBEAT_MS         = 8_000;
const RATE_LIMIT_BACKOFF   = 300_000;
const CONCURRENT_SLOTS     = 15;      // 15 = safe for Supabase free tier (was 50, caused connection pool exhaustion)
const MAX_RETRY_COUNT      = 2;

interface ScheduledEvent { cron: string; scheduledTime: number; }
interface ExecutionContext { waitUntil(promise: Promise<any>): void; passThroughOnException(): void; }
interface Fetcher { fetch(url: string, init?: RequestInit): Promise<Response> }
interface Env {
  SUPABASE_URL:         string;
  SUPABASE_SERVICE_KEY: string;
  HF_TOKEN?:            string;
  WORKER_ID?:           string;
  PIPELINE_ENABLED?:    string;
  SCRAPER_TEXT:         Fetcher;
  SCRAPER_IMAGE:        Fetcher;
  SCRAPER_VIDEO:        Fetcher;
  SCRAPER_TEXT_NEWS:    Fetcher;
  SCRAPER_IMAGE_OPEN:   Fetcher;
}

function getScraperBinding(sourceId: string, env: Env): Fetcher | null {
  const map: Record<string, Fetcher> = {
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
    "unsplash":       env.SCRAPER_IMAGE,
    "pexels":         env.SCRAPER_IMAGE,
    "pixabay":        env.SCRAPER_IMAGE,
    "openverse":      env.SCRAPER_IMAGE,
    "wikimedia":      env.SCRAPER_IMAGE,
    "nasa":           env.SCRAPER_IMAGE,
    "met-museum":     env.SCRAPER_IMAGE,
    "civitai":        env.SCRAPER_IMAGE,
    "pollinations":   env.SCRAPER_IMAGE,
    "diffusiondb":    env.SCRAPER_IMAGE,
    "lexica":         env.SCRAPER_IMAGE,
    "youtube":        env.SCRAPER_VIDEO,
    "pexels-video":   env.SCRAPER_VIDEO,
    "ted-talks":      env.SCRAPER_VIDEO,
    "voxceleb":       env.SCRAPER_VIDEO,
  };
  return map[sourceId] ?? null;
}

function db(env: Env) {
  return createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
}

// ── Diagnostic: verify Supabase is reachable ──────────────────
// Logs to CF dashboard even when worker_logs insert fails
async function checkSupabaseConnectivity(env: Env): Promise<boolean> {
  if (!env.SUPABASE_URL || !env.SUPABASE_SERVICE_KEY) {
    console.error(JSON.stringify({
      event: "STARTUP_ERROR",
      error: `Missing secrets: URL=${!!env.SUPABASE_URL} KEY=${!!env.SUPABASE_SERVICE_KEY}`
    }));
    return false;
  }
  try {
    const d = db(env);
    const { status } = await d.from("sources_queue").select("source_id").limit(1);
    const ok = status >= 200 && status < 300;
    console.log(JSON.stringify({ event: "SUPABASE_CHECK", ok, status }));
    return ok;
  } catch (e) {
    console.error(JSON.stringify({ event: "SUPABASE_CHECK_ERROR", error: String(e) }));
    return false;
  }
}

// ── Atomic source claiming via Supabase RPC ──────────────────────────────
// Uses SELECT FOR UPDATE SKIP LOCKED — each concurrent caller gets a
// DIFFERENT source, eliminating race conditions and cutting 4 HTTP calls
// down to 1 per slot.
async function claimSource(env: Env, workerId: string): Promise<SourceQueueItem | null> {
  try {
    const d = db(env);
    const { data, error } = await d.rpc("claim_next_source", {
      p_worker_id:  workerId,
      p_timeout_ms: WORKER_TIMEOUT_MS,
      p_backoff_ms: RATE_LIMIT_BACKOFF,
    });
    if (error || !data) return null;
    const rows = data as any[];
    return rows.length > 0 ? (rows[0] as SourceQueueItem) : null;
  } catch (e) {
    console.error(JSON.stringify({ event: "CLAIM_ERROR", error: String(e) }));
    return null;
  }
}

// ── FIX 1: releaseSource — no more `priority_score: undefined` ────
// The original v3 code had two PATCH calls where the body contained
// `priority_score: undefined`. JSON.stringify drops undefined values,
// producing an empty body `{}` which PostgREST rejects with 400.
// Fix: separate the status reset from the priority decrement, and
// only send the priority update after reading the current value.
async function releaseSource(
  env: Env, sourceId: string,
  success: boolean, samplesCount: number, errorMsg?: string
): Promise<void> {
  const d = db(env);

  if (success) {
    // Step 1: Reset operational fields (NO priority_score here)
    await d.from("sources_queue").update({
      status: "pending",
      worker_id: null,
      heartbeat_at: null,
      last_crawled_at: new Date().toISOString(),
      retry_count: 0,
      error_message: null,
    }).eq("source_id", sourceId);

    // Step 2: Read current score, then decrement (separate safe update)
    const { data: row } = await d.from("sources_queue")
      .select("priority_score")
      .eq("source_id", sourceId);
    const priorityRows = row as any[];
    if (priorityRows && priorityRows.length > 0) {
      const firstRow = priorityRows[0];
      const newScore = Math.max(10, (firstRow.priority_score ?? 50) - 5);  // min=10 so rotation still works; decrement=5 (was 10)
      await d.from("sources_queue")
        .update({ priority_score: newScore })
        .eq("source_id", sourceId as string);
    }
  } else {
    await d.from("sources_queue").update({
      status: "pending",
      worker_id: null,
      heartbeat_at: null,
      error_message: errorMsg ?? "Unknown error",
    }).eq("source_id", sourceId);

    const { data: rowR } = await d.from("sources_queue")
      .select("retry_count")
      .eq("source_id", sourceId);
    const retryRows = rowR as any[];
    const rc = retryRows && retryRows.length > 0 ? (retryRows[0].retry_count ?? 0) + 1 : 1;
    await d.from("sources_queue")
      .update({ retry_count: rc })
      .eq("source_id", sourceId);
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

async function processOneSlot(env: Env, workerId: string, logger: PipelineLogger): Promise<void> {
  const source = await claimSource(env, workerId).catch(() => null);
  if (!source) return;

  const scraper = getScraperBinding(source.source_id, env);
  if (!scraper) {
    logger.log("SCRAPE_ERROR", {
      source_id: source.source_id,
      error_message: `No binding for source: ${source.source_id} — check dispatcher scraperMap`,
    });
    await releaseSource(env, source.source_id, false, 0, "No scraper binding");
    return;
  }

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

async function triggerCleanup(env: Env): Promise<void> {
  try {
    const d = db(env);
    await d.from("samples_staging").delete()
      .lt("scraped_at", new Date(Date.now() - 86_400_000).toISOString())
      .eq("status", "staged");
    await d.from("samples_processed").delete()
      .lt("labeled_at", new Date(Date.now() - 259_200_000).toISOString());
  } catch { /* non-critical */ }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (env.PIPELINE_ENABLED === "false") {
      return new Response(JSON.stringify({ status: "disabled" }), { status: 200 });
    }
    await checkSupabaseConnectivity(env);
    const logger = new PipelineLogger(env as any);
    const workerId = `manual-${Date.now()}`;
    await processOneSlot(env, workerId, logger);
    await logger.flush();
    return new Response(JSON.stringify({ status: "ok" }), { status: 200 });
  },

  async scheduled(_event: ScheduledEvent, env: Env, ctx: ExecutionContext): Promise<void> {
    if (env.PIPELINE_ENABLED === "false") return;

    ctx.waitUntil((async () => {
      // FIX 2: Check Supabase connectivity first — visible in CF logs even if DB insert fails
      const dbOk = await checkSupabaseConnectivity(env);

      const logger = new PipelineLogger(env as any);
      logger.log("CRON_TICK", {
        sample_count: CONCURRENT_SLOTS,
        // note: only valid worker_logs columns — extra fields cause PostgREST 400
      });

      // FIX 2: Flush CRON_TICK immediately — guarantees it lands in DB
      // regardless of how many subsequent events are logged
      await logger.flush();

      if (!dbOk) {
        console.error(JSON.stringify({ event: "CRON_ABORT", reason: "Supabase unreachable — check SUPABASE_URL and SUPABASE_SERVICE_KEY secrets" }));
        return;
      }

      const slots = Array.from({ length: CONCURRENT_SLOTS }, (_, i) =>
        processOneSlot(env, `cron-slot-${i + 1}`, logger).catch(e =>
          logger.log("SLOT_ERROR", { error_message: `slot-${i + 1}: ${String(e)}` } as any)
        )
      );

      await Promise.allSettled(slots);
      await triggerCleanup(env);
      await logger.flush();
    })());
  },
};
