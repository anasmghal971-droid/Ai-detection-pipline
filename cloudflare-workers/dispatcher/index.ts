// ============================================================
// DETECT-AI: Worker Dispatcher (Stage 1)
// Cloudflare Worker — Source Queue Orchestration
//
// Responsibilities:
//   - Pull highest-priority pending source from Supabase queue
//   - Assign to calling worker instance
//   - Heartbeat tracking (30s timeout)
//   - Auto-requeue timed-out jobs
//   - Global kill switch enforcement
//   - Dynamic reallocation for slow/blocked sources
// ============================================================

import type { Env, SourceQueueItem } from "../../shared/types/index";
import { createSupabaseClient } from "./supabase";
import { PipelineLogger } from "./logger";

// ── Constants ─────────────────────────────────────────────────
const WORKER_TIMEOUT_MS     = 30_000;   // 30 seconds
const MAX_RETRIES            = 3;
const HEARTBEAT_INTERVAL_MS  = 10_000;  // Worker updates heartbeat every 10s
const RATE_LIMIT_BACKOFF_MS  = 60_000;  // 60s before retrying rate-limited source

// ── Retry helper with exponential backoff ─────────────────────
async function withRetry<T>(
  fn: () => Promise<T>,
  maxAttempts = MAX_RETRIES,
  baseDelayMs = 1_000
): Promise<T> {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (attempt === maxAttempts) throw err;
      const delay = baseDelayMs * Math.pow(2, attempt - 1);
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  throw new Error("Max retries exceeded");
}

// ── Dispatcher: claim next available source ───────────────────
async function claimNextSource(
  env: Env,
  logger: PipelineLogger
): Promise<SourceQueueItem | null> {
  const db = createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);

  // Step 1: Release timed-out sources back to 'pending'
  const timeoutThreshold = new Date(Date.now() - WORKER_TIMEOUT_MS).toISOString();
  await db
    .from("sources_queue")
    .update({
      status: "pending",
      worker_id: null,
      heartbeat_at: null,
      error_message: "Worker timeout — auto-requeued",
    })
    .eq("status", "active")
    .lt("heartbeat_at", timeoutThreshold);

  // Step 2: Release rate-limited sources that have cooled down
  const rateLimitThreshold = new Date(Date.now() - RATE_LIMIT_BACKOFF_MS).toISOString();
  await db
    .from("sources_queue")
    .update({ status: "pending" })
    .eq("status", "rate_limited")
    .lt("updated_at", rateLimitThreshold);

  // Step 3: Claim the highest-priority pending source
  // Uses select + update pattern for atomic-ish claim
  const { data: sources, error } = await db
    .from("sources_queue")
    .select("*")
    .eq("status", "pending")
    .lt("retry_count", MAX_RETRIES)
    .order("priority_score", { ascending: false })
    .limit(1);

  if (error || !sources || (sources as SourceQueueItem[]).length === 0) {
    return null;
  }

  const source = (sources as SourceQueueItem[])[0];

  // Mark as active + assign worker
  const { error: updateError } = await db
    .from("sources_queue")
    .update({
      status: "active",
      worker_id: env.WORKER_ID,
      heartbeat_at: new Date().toISOString(),
      error_message: null,
    })
    .eq("source_id", source.source_id)
    .eq("status", "pending"); // Optimistic lock — prevent race condition

  if (updateError) return null; // Another worker claimed it first

  logger.log("SCRAPE_START", { source_id: source.source_id });
  return source;
}

// ── Mark source complete / failed ────────────────────────────
async function completeSource(
  env: Env,
  sourceId: string,
  success: boolean,
  errorMsg?: string
): Promise<void> {
  const db = createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);

  await db.from("sources_queue").update({
    status: success ? "pending" : "failed",  // 'pending' so sources rotate every cycle
    last_crawled_at: new Date().toISOString(),
    worker_id: null,
    heartbeat_at: null,
    retry_count: success ? 0 : undefined,
    error_message: errorMsg ?? null,
  }).eq("source_id", sourceId);
}

// ── Mark source rate-limited ──────────────────────────────────
async function rateLimitSource(env: Env, sourceId: string): Promise<void> {
  const db = createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
  await db.from("sources_queue").update({
    status: "rate_limited",
    worker_id: null,
    error_message: "Rate limit hit — cooling down",
  }).eq("source_id", sourceId);
}

// ── Update heartbeat ──────────────────────────────────────────
async function updateHeartbeat(env: Env, sourceId: string): Promise<void> {
  const db = createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
  await db.from("sources_queue").update({
    heartbeat_at: new Date().toISOString(),
  }).eq("source_id", sourceId).eq("worker_id", env.WORKER_ID);
}

// ── Dispatch job to appropriate scraper ──────────────────────
async function dispatchToScraper(
  source: SourceQueueItem,
  env: Env,
  logger: PipelineLogger
): Promise<{ success: boolean; samplesScraped: number; rateLimited?: boolean }> {
  // CF docs: Worker-to-Worker communication MUST use service bindings
  // fetch() to workers.dev URLs does NOT work between workers
  // Service bindings are declared in wrangler.toml [[services]] blocks

  // Map source_id to the correct service binding on env
  type Fetcher = { fetch: (url: string, init?: RequestInit) => Promise<Response> };
  const scraperMap: Record<string, Fetcher | undefined> = {
    // Text sources — ALL mapped, NO gaps
    "bbc-news":        env.SCRAPER_TEXT,
    "reuters":         env.SCRAPER_TEXT,
    "guardian":        env.SCRAPER_TEXT,
    "nytimes":         env.SCRAPER_TEXT,
    "aljazeera":       env.SCRAPER_TEXT,
    "wikipedia":       env.SCRAPER_TEXT,
    "arxiv":           env.SCRAPER_TEXT,
    "npr":             env.SCRAPER_TEXT,
    "paperswithcode":  env.SCRAPER_TEXT,
    "stackexchange":   env.SCRAPER_TEXT,
    "reddit":          env.SCRAPER_TEXT,
    "worldbank":       env.SCRAPER_TEXT,
    // REAL image sources
    "unsplash":        env.SCRAPER_IMAGE,
    "pexels":          env.SCRAPER_IMAGE,
    "pixabay":         env.SCRAPER_IMAGE,
    "openverse":       env.SCRAPER_IMAGE,
    "wikimedia":       env.SCRAPER_IMAGE,
    "nasa":            env.SCRAPER_IMAGE,
    "met-museum":      env.SCRAPER_IMAGE,
    // AI-GENERATED image sources — THE KEY ONES FOR DETECTION TRAINING
    "lexica":          env.SCRAPER_IMAGE,
    "civitai":         env.SCRAPER_IMAGE,
    "pollinations":    env.SCRAPER_IMAGE,
    "diffusiondb":     env.SCRAPER_IMAGE,
    // Video sources
    "pexels-video":    env.SCRAPER_VIDEO,
    "youtube":         env.SCRAPER_VIDEO,
    "ted-talks":       env.SCRAPER_VIDEO,
    "voxceleb":        env.SCRAPER_VIDEO,
  };

  const scraper = scraperMap[source.source_id];
  if (!scraper) {
    logger.log("SCRAPE_ERROR", {
      source_id: source.source_id,
      error_message: `No service binding for source: ${source.source_id}`,
    });
    return { success: false, samplesScraped: 0 };
  }

  // Call via service binding — same Cloudflare network, zero latency
  const payload = JSON.stringify({
    source,
    worker_id:    env.WORKER_ID,
    supabase_url: env.SUPABASE_URL,
    supabase_key: env.SUPABASE_SERVICE_KEY,
  });

  const response = await withRetry(() =>
    scraper.fetch("https://internal/scrape", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    })
  );

  if (response.status === 429) {
    return { success: false, samplesScraped: 0, rateLimited: true };
  }
  if (!response.ok) {
    throw new Error(`Scraper ${source.source_id} returned ${response.status}: ${await response.text()}`);
  }

  const result = await response.json() as { samples_scraped: number };
  return { success: true, samplesScraped: result.samples_scraped };
}

// ── Main Worker Handler ───────────────────────────────────────
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // ── Kill Switch ──────────────────────────────────────────
    if (env.PIPELINE_ENABLED !== "true") {
      console.log(JSON.stringify({
        event: "KILL_SWITCH_ACTIVE",
        worker_id: env.WORKER_ID,
        timestamp: new Date().toISOString(),
      }));
      return new Response(JSON.stringify({ status: "pipeline_disabled" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }

    const logger = new PipelineLogger(env);
    logger.log("WORKER_START");

    const startTime = Date.now();
    let source: SourceQueueItem | null = null;

    try {
      // ── Claim a source ──────────────────────────────────────
      source = await claimNextSource(env, logger);

      if (!source) {
        return new Response(
          JSON.stringify({ status: "no_sources_available" }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }

      // ── Heartbeat: update every 10s during processing ──────
      // (In real CF Workers, use Durable Objects for long-running)
      const heartbeatInterval = setInterval(
        () => updateHeartbeat(env, source!.source_id),
        HEARTBEAT_INTERVAL_MS
      );

      try {
        const result = await dispatchToScraper(source, env, logger);

        clearInterval(heartbeatInterval);

        if (result.rateLimited) {
          await rateLimitSource(env, source.source_id);
          logger.log("RATE_LIMIT_HIT", { source_id: source.source_id });
          return new Response(
            JSON.stringify({ status: "rate_limited", source: source.source_id }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        }

        await completeSource(env, source.source_id, result.success);

        const durationMs = Date.now() - startTime;
        logger.log("SCRAPE_COMPLETE", {
          source_id: source.source_id,
          sample_count: result.samplesScraped,
          duration_ms: durationMs,
        });

        await logger.flush();

        return new Response(
          JSON.stringify({
            status: "success",
            source: source.source_id,
            samples_scraped: result.samplesScraped,
            duration_ms: durationMs,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );

      } catch (innerErr) {
        clearInterval(heartbeatInterval);
        throw innerErr;
      }

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      const durationMs = Date.now() - startTime;

      logger.log("SCRAPE_ERROR", {
        source_id: source?.source_id,
        error_message: errorMsg,
        duration_ms: durationMs,
      });

      if (source) {
        await completeSource(env, source.source_id, false, errorMsg);
      }

      await logger.flush();

      return new Response(
        JSON.stringify({ status: "error", error: errorMsg }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      );
    }
  },

  // ── Cron Trigger: runs every 5 minutes ───────────────────────
  async scheduled(_event: ScheduledEvent, env: Env, ctx: ExecutionContext): Promise<void> {
    ctx.waitUntil(
      (async () => {
        if (env.PIPELINE_ENABLED !== "true") return;

        const logger = new PipelineLogger(env);
        logger.log("CRON_TICK", { timestamp: new Date().toISOString() });

        // Process up to 10 sources per cron cycle concurrently
        // Each claimNextSource() picks the highest-priority pending source
        // 3 concurrent cycles — prevents any single API from being rate-limited
        const cycles = Array.from({ length: 3 }, async (_, i) => {
          const source = await claimNextSource(env, logger).catch(() => null);
          if (!source) return;

          const hb = setInterval(() => updateHeartbeat(env, source.source_id), HEARTBEAT_INTERVAL_MS);
          try {
            const result = await dispatchToScraper(source, env, logger);
            clearInterval(hb);
            if (result.rateLimited) {
              await rateLimitSource(env, source.source_id);
            } else {
              await completeSource(env, source.source_id, result.success);
              logger.log("SCRAPE_COMPLETE", {
                source_id: source.source_id,
                sample_count: result.samplesScraped,
              });
            }
          } catch (e) {
            clearInterval(hb);
            const msg = e instanceof Error ? e.message : String(e);
            await completeSource(env, source.source_id, false, msg);
            logger.log("SCRAPE_ERROR", { source_id: source.source_id, error_message: msg });
          }
        });

        await Promise.allSettled(cycles);
        await logger.flush();
      })()
    );
  },
};
