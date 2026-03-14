// ============================================================
// DETECT-AI: Text Scraper Worker (Stage 2)
// Cloudflare Worker — Handles ALL text-based sources
//
// Sources: BBC, Reuters, Al Jazeera, arXiv, Wikipedia,
//          NewsAPI, PapersWithCode, StackExchange, Reddit,
//          World Bank / UN / UNESCO
// ============================================================

import type { Env, DetectAISample, SourceQueueItem } from "../../shared/types/index";
import { createSupabaseClient } from "../dispatcher/supabase";
import { PipelineLogger } from "../dispatcher/logger";

// ── Language detection via HF Inference API ──────────────────
async function detectLanguage(text: string, hfToken: string): Promise<string> {
  try {
    const truncated = text.slice(0, 512);
    const response = await fetch(
      "https://api-inference.huggingface.co/models/papluca/xlm-roberta-base-language-detection",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${hfToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ inputs: truncated }),
      }
    );
    if (!response.ok) return "unknown";
    const result = await response.json() as Array<Array<{ label: string; score: number }>>;
    return result?.[0]?.[0]?.label ?? "unknown";
  } catch {
    return "unknown";
  }
}

// ── HTML Cleaner ─────────────────────────────────────────────
function stripHtml(html: string): string {
  return html
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
    .replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, "")
    .replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, "")
    .replace(/<header[^>]*>[\s\S]*?<\/header>/gi, "")
    .replace(/<aside[^>]*>[\s\S]*?<\/aside>/gi, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&nbsp;/g, " ").replace(/&#\d+;/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

// ── Retry with exponential backoff ───────────────────────────
async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  maxAttempts = 3
): Promise<Response> {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const response = await fetch(url, options);
    if (response.status === 429) throw Object.assign(new Error("Rate limited"), { rateLimited: true });
    if (response.ok) return response;
    if (attempt === maxAttempts) throw new Error(`HTTP ${response.status}: ${url}`);
    await new Promise(r => setTimeout(r, 1000 * Math.pow(2, attempt)));
  }
  throw new Error("Max retries exceeded");
}

// ── UUID generator (CF Workers compatible) ───────────────────
function uuidv4(): string {
  return crypto.randomUUID();
}

// ════════════════════════════════════════════════════════════════
//  SOURCE-SPECIFIC SCRAPERS
// ════════════════════════════════════════════════════════════════

// ── 1. RSS Scraper (BBC, Reuters, Al Jazeera) ─────────────────
async function scrapeRSS(
  source: SourceQueueItem,
  env: Env
): Promise<Partial<DetectAISample>[]> {
  const response = await fetchWithRetry(source.source_url, {
    headers: { "User-Agent": "DETECT-AI/1.0 (research dataset; contact@detect-ai.io)" },
  });
  const xml = await response.text();

  // Parse RSS items
  const itemMatches = [...xml.matchAll(/<item>([\s\S]*?)<\/item>/gi)];
  const samples: Partial<DetectAISample>[] = [];

  for (const match of itemMatches.slice(0, 100)) {
    const itemXml = match[1];
    const title     = itemXml.match(/<title><!\[CDATA\[(.*?)\]\]><\/title>|<title>(.*?)<\/title>/)?.[1] ?? "";
    const desc      = itemXml.match(/<description><!\[CDATA\[(.*?)\]\]><\/description>|<description>(.*?)<\/description>/s)?.[1] ?? "";
    const link      = itemXml.match(/<link>(.*?)<\/link>/)?.[1] ?? "";
    const pubDate   = itemXml.match(/<pubDate>(.*?)<\/pubDate>/)?.[1] ?? "";
    const combined  = stripHtml(`${title} ${desc}`);
    if (combined.length < 50) continue;

    samples.push({
      source_url:   link || source.source_url,
      raw_content:  combined,
      metadata: {
        title:        stripHtml(title),
        publish_date: pubDate,
        license:      "editorial-public",
      },
    });
  }
  return samples;
}

// ── 2. arXiv API ─────────────────────────────────────────────
async function scrapeArXiv(
  _source: SourceQueueItem
): Promise<Partial<DetectAISample>[]> {
  const url = "https://export.arxiv.org/api/query?search_query=all:ai&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending";
  const response = await fetchWithRetry(url);
  const xml = await response.text();
  const entries = [...xml.matchAll(/<entry>([\s\S]*?)<\/entry>/gi)];
  const samples: Partial<DetectAISample>[] = [];

  for (const match of entries) {
    const e     = match[1];
    const title = e.match(/<title>(.*?)<\/title>/s)?.[1]?.trim() ?? "";
    const abs   = e.match(/<summary>(.*?)<\/summary>/s)?.[1]?.trim() ?? "";
    const link  = e.match(/<id>(.*?)<\/id>/)?.[1]?.trim() ?? "";
    const auth  = e.match(/<name>(.*?)<\/name>/)?.[1]?.trim() ?? "";
    const pub   = e.match(/<published>(.*?)<\/published>/)?.[1]?.trim() ?? "";

    if (abs.length < 100) continue;
    samples.push({
      source_url:  link,
      raw_content: `${title}\n\n${abs}`,
      metadata:    { title, author: auth, publish_date: pub, license: "arxiv-open-access" },
    });
  }
  return samples;
}

// ── 3. Wikipedia API ──────────────────────────────────────────
async function scrapeWikipedia(
  source: SourceQueueItem
): Promise<Partial<DetectAISample>[]> {
  const langs = ["en", "de", "fr", "es", "zh", "ar", "hi", "pt", "ru", "ja",
                 "ko", "it", "nl", "pl", "sv", "tr", "vi", "fa", "uk", "he"];
  const samples: Partial<DetectAISample>[] = [];

  for (const lang of langs.slice(0, 5)) { // 5 langs per cycle to stay within CPU limits
    const url = `https://${lang}.wikipedia.org/w/api.php?action=query&list=random&rnlimit=20&rnnamespace=0&format=json`;
    const resp = await fetchWithRetry(url);
    const data = await resp.json() as { query: { random: Array<{ id: number; title: string }> } };

    for (const article of data.query.random) {
      const contentUrl = `https://${lang}.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&pageids=${article.id}&format=json`;
      const cResp = await fetchWithRetry(contentUrl);
      const cData = await cResp.json() as {
        query: { pages: Record<string, { extract?: string; title: string }> };
      };
      const page = Object.values(cData.query.pages)[0];
      if (!page.extract || page.extract.length < 100) continue;

      samples.push({
        source_url:  `https://${lang}.wikipedia.org/?curid=${article.id}`,
        raw_content: page.extract.slice(0, 4000),
        metadata:    { title: page.title, license: "CC-BY-SA-4.0" },
      });
    }
  }
  return samples;
}

// ── 4. NewsAPI ────────────────────────────────────────────────
async function scrapeNewsAPI(
  _source: SourceQueueItem,
  apiKey: string
): Promise<Partial<DetectAISample>[]> {
  const url = `https://newsapi.org/v2/top-headlines?pageSize=100&apiKey=${apiKey}`;
  const response = await fetchWithRetry(url);
  const data = await response.json() as {
    articles: Array<{
      title?: string;
      description?: string;
      content?: string;
      url?: string;
      author?: string;
      publishedAt?: string;
      source?: { name?: string };
    }>;
  };

  return (data.articles ?? [])
    .filter(a => a.content && a.content.length > 100)
    .map(a => ({
      source_url:  a.url ?? "",
      raw_content: `${a.title ?? ""}\n\n${a.description ?? ""}\n\n${a.content ?? ""}`.trim(),
      metadata:    {
        title:        a.title,
        author:       a.author,
        publish_date: a.publishedAt,
        license:      "news-public",
        tags:         [a.source?.name ?? "newsapi"],
      },
    }));
}

// ── 5. StackExchange API ──────────────────────────────────────
async function scrapeStackExchange(
  _source: SourceQueueItem
): Promise<Partial<DetectAISample>[]> {
  const url = "https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&site=stackoverflow&filter=withbody&pagesize=100";
  const response = await fetchWithRetry(url, {
    headers: { "Accept-Encoding": "gzip" },
  });
  const data = await response.json() as {
    items: Array<{
      body?: string;
      title?: string;
      link?: string;
      owner?: { display_name?: string };
      creation_date?: number;
      tags?: string[];
    }>;
  };

  return (data.items ?? [])
    .filter(q => q.body && q.body.length > 100)
    .map(q => ({
      source_url:  q.link ?? "",
      raw_content: stripHtml(`${q.title ?? ""}\n\n${q.body ?? ""}`),
      metadata:    {
        title:        q.title,
        author:       q.owner?.display_name,
        publish_date: q.creation_date ? new Date(q.creation_date * 1000).toISOString() : undefined,
        license:      "CC-BY-SA-4.0",
        tags:         q.tags,
      },
    }));
}

// ── 6. Reddit API ─────────────────────────────────────────────
async function scrapeReddit(
  _source: SourceQueueItem,
  clientId: string,
  clientSecret: string
): Promise<Partial<DetectAISample>[]> {
  // Get OAuth token
  const tokenResp = await fetchWithRetry("https://www.reddit.com/api/v1/access_token", {
    method: "POST",
    headers: {
      Authorization: `Basic ${btoa(`${clientId}:${clientSecret}`)}`,
      "Content-Type": "application/x-www-form-urlencoded",
      "User-Agent": "DETECT-AI/1.0",
    },
    body: "grant_type=client_credentials",
  });
  const tokenData = await tokenResp.json() as { access_token: string };

  const resp = await fetchWithRetry(
    "https://oauth.reddit.com/r/all/hot?limit=100",
    {
      headers: {
        Authorization: `Bearer ${tokenData.access_token}`,
        "User-Agent": "DETECT-AI/1.0",
      },
    }
  );
  const data = await resp.json() as {
    data: { children: Array<{ data: {
      selftext?: string; title?: string; url?: string;
      author?: string; created_utc?: number; subreddit?: string;
    } }> };
  };

  return (data.data?.children ?? [])
    .filter(c => c.data.selftext && c.data.selftext.length > 100 && c.data.selftext !== "[removed]")
    .map(c => ({
      source_url:  `https://reddit.com${c.data.url ?? ""}`,
      raw_content: `${c.data.title ?? ""}\n\n${c.data.selftext ?? ""}`.trim(),
      metadata:    {
        title:        c.data.title,
        author:       c.data.author,
        publish_date: c.data.created_utc
          ? new Date(c.data.created_utc * 1000).toISOString() : undefined,
        license:      "public-reddit",
        tags:         [c.data.subreddit ?? "reddit"],
      },
    }));
}

// ── 7. World Bank API ─────────────────────────────────────────
async function scrapeWorldBank(
  _source: SourceQueueItem
): Promise<Partial<DetectAISample>[]> {
  const url = "https://api.worldbank.org/v2/en/topic/all/indicator?format=json&per_page=50&mrv=1";
  const response = await fetchWithRetry(url);
  const data = await response.json() as [unknown, Array<{
    name?: string; sourceNote?: string; id?: string; source?: { value?: string };
  }>];

  return (data[1] ?? [])
    .filter(item => item.sourceNote && item.sourceNote.length > 50)
    .map(item => ({
      source_url:  `https://data.worldbank.org/indicator/${item.id ?? ""}`,
      raw_content: `${item.name ?? ""}\n\n${item.sourceNote ?? ""}`.trim(),
      metadata:    {
        title:   item.name,
        license: "CC-BY-4.0",
        tags:    [item.source?.value ?? "worldbank"],
      },
    }));
}

// ── 8. PapersWithCode API ─────────────────────────────────────
async function scrapePapersWithCode(
  _source: SourceQueueItem
): Promise<Partial<DetectAISample>[]> {
  const url = "https://paperswithcode.com/api/v1/papers/?format=json&page_size=50&ordering=-published";
  const response = await fetchWithRetry(url);
  const data = await response.json() as {
    results: Array<{
      title?: string; abstract?: string; url_pdf?: string;
      authors?: string[]; published?: string;
    }>;
  };

  return (data.results ?? [])
    .filter(p => p.abstract && p.abstract.length > 100)
    .map(p => ({
      source_url:  p.url_pdf ?? "https://paperswithcode.com",
      raw_content: `${p.title ?? ""}\n\n${p.abstract ?? ""}`.trim(),
      metadata:    {
        title:        p.title,
        author:       p.authors?.join(", "),
        publish_date: p.published,
        license:      "open-access",
      },
    }));
}

// ════════════════════════════════════════════════════════════════
//  BATCH PUSH TO SUPABASE
// ════════════════════════════════════════════════════════════════

async function batchPushToStaging(
  samples: DetectAISample[],
  env: Env
): Promise<void> {
  const db = createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
  const BATCH_SIZE = 100;

  for (let i = 0; i < samples.length; i += BATCH_SIZE) {
    const batch = samples.slice(i, i + BATCH_SIZE);
    const { error } = await db.from("samples_staging").insert(batch);
    if (error) {
      console.error(JSON.stringify({ event: "STAGING_INSERT_ERROR", error }));
    }
  }
}

// ════════════════════════════════════════════════════════════════
//  MAIN HANDLER
// ════════════════════════════════════════════════════════════════

interface ScraperRequest {
  source: SourceQueueItem;
  worker_id: string;
  supabase_url: string;
  supabase_key: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method !== "POST") {
      return new Response("Method not allowed", { status: 405 });
    }

    const body = await request.json() as ScraperRequest;
    const { source } = body;
    const logger = new PipelineLogger(env);
    const startTime = Date.now();

    try {
      // ── Route to source-specific scraper ──────────────────
      let rawSamples: Partial<DetectAISample>[] = [];

      switch (source.source_id) {
        case "bbc-news":
        case "reuters":
        case "aljazeera":
          rawSamples = await scrapeRSS(source, env);
          break;
        case "arxiv":
          rawSamples = await scrapeArXiv(source);
          break;
        case "wikipedia":
          rawSamples = await scrapeWikipedia(source);
          break;
        case "newsapi":
          rawSamples = await scrapeNewsAPI(source, env.NEWSAPI_KEY);
          break;
        case "stackexchange":
          rawSamples = await scrapeStackExchange(source);
          break;
        case "reddit":
          if (!env.REDDIT_CLIENT_ID || !env.REDDIT_CLIENT_SECRET) {
            console.log("Reddit credentials not set — skipping");
            break;
          }
          rawSamples = await scrapeReddit(source, env.REDDIT_CLIENT_ID, env.REDDIT_CLIENT_SECRET);
          break;
        case "worldbank":
          rawSamples = await scrapeWorldBank(source);
          break;
        case "paperswithcode":
          rawSamples = await scrapePapersWithCode(source);
          break;
        default:
          return new Response(
            JSON.stringify({ error: `Unknown text source: ${source.source_id}` }),
            { status: 400 }
          );
      }

      // ── Enrich: detect language + build full schema ────────
      const enriched: DetectAISample[] = await Promise.all(
        rawSamples
          .filter(s => s.raw_content && s.raw_content.length >= 50)
          .slice(0, 1000) // Cap per cycle
          .map(async (s) => {
            const language = await detectLanguage(s.raw_content ?? "", env.HF_TOKEN);
            return {
              sample_id:    uuidv4(),
              source_id:    source.source_id,
              source_url:   s.source_url ?? source.source_url,
              content_type: "text" as const,
              language,
              raw_content:  s.raw_content!,
              metadata:     s.metadata ?? {},
              scraped_at:   new Date().toISOString(),
              worker_id:    env.WORKER_ID,
              status:       "staged" as const,
            };
          })
      );

      // ── Batch push to Supabase ──────────────────────────────
      await batchPushToStaging(enriched, env);

      const duration = Date.now() - startTime;
      logger.log("SCRAPE_COMPLETE", {
        source_id:    source.source_id,
        sample_count: enriched.length,
        duration_ms:  duration,
      });
      await logger.flush();

      return new Response(
        JSON.stringify({ samples_scraped: enriched.length, duration_ms: duration }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      logger.log("SCRAPE_ERROR", { source_id: source.source_id, error_message: errorMsg });
      await logger.flush();

      if ((err as { rateLimited?: boolean }).rateLimited) {
        return new Response(JSON.stringify({ error: "rate_limited" }), { status: 429 });
      }

      return new Response(JSON.stringify({ error: errorMsg }), { status: 500 });
    }
  },
};
