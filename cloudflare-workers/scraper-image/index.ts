// ============================================================
// DETECT-AI: Image Scraper Worker (Stage 2)
// Cloudflare Worker — Handles ALL image-based sources
//
// Sources: Unsplash, Pexels, Pixabay, Flickr CC, Wikimedia Commons
// ============================================================

import type { Env, DetectAISample, SourceQueueItem } from "../../shared/types/index";
import { createSupabaseClient } from "../dispatcher/supabase";
import { PipelineLogger } from "../dispatcher/logger";

function uuidv4(): string { return crypto.randomUUID(); }

async function fetchWithRetry(url: string, options: RequestInit = {}, maxAttempts = 3): Promise<Response> {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const response = await fetch(url, options);
    if (response.status === 429) throw Object.assign(new Error("Rate limited"), { rateLimited: true });
    if (response.ok) return response;
    if (attempt === maxAttempts) throw new Error(`HTTP ${response.status}: ${url}`);
    await new Promise(r => setTimeout(r, 1000 * Math.pow(2, attempt)));
  }
  throw new Error("Max retries exceeded");
}

// ── 1. Unsplash ───────────────────────────────────────────────
async function scrapeUnsplash(env: Env): Promise<Partial<DetectAISample>[]> {
  const resp = await fetchWithRetry(
    "https://api.unsplash.com/photos/random?count=30&content_filter=high",
    { headers: { Authorization: `Client-ID ${env.UNSPLASH_ACCESS_KEY}` } }
  );
  const photos = await resp.json() as Array<{
    id: string; description?: string; alt_description?: string;
    urls: { raw: string; full: string };
    user: { name?: string; username?: string };
    created_at: string; links: { html: string };
    width: number; height: number;
  }>;

  return photos.map(p => ({
    source_url:  p.links.html,
    raw_content: p.urls.full, // Store full-res URL as content ref
    metadata: {
      title:        p.description ?? p.alt_description ?? `Unsplash photo ${p.id}`,
      author:       p.user.name ?? p.user.username,
      publish_date: p.created_at,
      license:      "Unsplash-License",
      dimensions:   { width: p.width, height: p.height },
      tags:         ["unsplash", "photography"],
    },
  }));
}

// ── 2. Pexels Images ─────────────────────────────────────────
async function scrapePexelsImages(env: Env): Promise<Partial<DetectAISample>[]> {
  const resp = await fetchWithRetry(
    "https://api.pexels.com/v1/curated?per_page=80",
    { headers: { Authorization: env.PEXELS_API_KEY } }
  );
  const data = await resp.json() as {
    photos: Array<{
      id: number; url: string; photographer: string;
      src: { original: string; large2x: string };
      width: number; height: number; alt?: string;
    }>;
  };

  return (data.photos ?? []).map(p => ({
    source_url:  p.url,
    raw_content: p.src.original, // Full-resolution URL
    metadata: {
      title:      p.alt ?? `Pexels photo ${p.id}`,
      author:     p.photographer,
      license:    "Pexels-License",
      dimensions: { width: p.width, height: p.height },
      tags:       ["pexels", "photography", "CC0"],
    },
  }));
}

// ── 3. Pixabay ────────────────────────────────────────────────
async function scrapePixabay(env: Env): Promise<Partial<DetectAISample>[]> {
  const langs = ["en", "de", "fr", "es", "zh", "ja", "ko", "pt", "it", "ru"];
  const samples: Partial<DetectAISample>[] = [];

  for (const lang of langs.slice(0, 3)) {
    const url = `https://pixabay.com/api/?key=${env.PIXABAY_API_KEY}&image_type=photo&per_page=50&lang=${lang}&safesearch=true`;
    const resp = await fetchWithRetry(url);
    const data = await resp.json() as {
      hits: Array<{
        id: number; pageURL: string; webformatURL: string;
        largeImageURL: string; imageWidth: number; imageHeight: number;
        tags: string; user: string;
      }>;
    };

    for (const hit of data.hits ?? []) {
      samples.push({
        source_url:  hit.pageURL,
        raw_content: hit.largeImageURL,
        metadata: {
          title:      `Pixabay ${hit.id}`,
          author:     hit.user,
          license:    "Pixabay-License",
          dimensions: { width: hit.imageWidth, height: hit.imageHeight },
          tags:       ["pixabay", "CC0", ...hit.tags.split(",").map(t => t.trim())],
        },
      });
    }
  }
  return samples;
}

// ── 4. Flickr Creative Commons ────────────────────────────────
async function scrapeFlickr(env: Env): Promise<Partial<DetectAISample>[]> {
  if (!env.FLICKR_API_KEY) return [];
  // license=4 = CC BY, 5 = CC BY-SA, 7 = No known copyright
  const url = `https://www.flickr.com/services/rest/?method=flickr.photos.search&api_key=${env.FLICKR_API_KEY}&license=4,5,7&sort=interestingness-desc&content_type=1&media=photos&format=json&nojsoncallback=1&per_page=100&extras=url_o,url_l,owner_name,date_taken,tags,description`;
  const resp = await fetchWithRetry(url);
  const data = await resp.json() as {
    photos: {
      photo: Array<{
        id: string; owner: string; title: string;
        url_o?: string; url_l?: string;
        ownername?: string; datetaken?: string;
        tags?: string; description?: { _content?: string };
      }>;
    };
  };

  return (data.photos?.photo ?? [])
    .filter(p => p.url_o || p.url_l)
    .map(p => ({
      source_url:  `https://www.flickr.com/photos/${p.owner}/${p.id}`,
      raw_content: p.url_o ?? p.url_l ?? "",
      metadata: {
        title:        p.title,
        author:       p.ownername ?? p.owner,
        publish_date: p.datetaken,
        license:      "CC-BY-SA",
        tags:         ["flickr", "cc", ...(p.tags?.split(" ") ?? [])],
        description:  p.description?._content,
      },
    }));
}

// ── 5. Wikimedia Commons ──────────────────────────────────────
async function scrapeWikimedia(_source: SourceQueueItem): Promise<Partial<DetectAISample>[]> {
  // Get random files from Wikimedia Commons
  const url = "https://commons.wikimedia.org/w/api.php?action=query&list=random&rnlimit=50&rnnamespace=6&format=json";
  const resp = await fetchWithRetry(url);
  const data = await resp.json() as {
    query: { random: Array<{ id: number; title: string }> };
  };

  const samples: Partial<DetectAISample>[] = [];

  for (const file of (data.query.random ?? []).slice(0, 20)) {
    const infoUrl = `https://commons.wikimedia.org/w/api.php?action=query&prop=imageinfo&iiprop=url|mime|size|extmetadata&pageids=${file.id}&format=json`;
    const infoResp = await fetchWithRetry(infoUrl);
    const infoData = await infoResp.json() as {
      query: { pages: Record<string, {
        imageinfo?: Array<{
          url: string; mime: string; size: number;
          extmetadata?: {
            License?: { value?: string };
            ImageDescription?: { value?: string };
            Artist?: { value?: string };
            DateTimeOriginal?: { value?: string };
          };
        }>;
      }> };
    };

    const page = Object.values(infoData.query.pages)[0];
    const info = page.imageinfo?.[0];
    if (!info) continue;

    // Only images (not video/audio — those go to video worker)
    if (!info.mime.startsWith("image/")) continue;

    const meta = info.extmetadata ?? {};
    samples.push({
      source_url:  `https://commons.wikimedia.org/?curid=${file.id}`,
      raw_content: info.url,
      metadata: {
        title:        file.title.replace("File:", ""),
        author:       meta.Artist?.value ? stripHtml(meta.Artist.value) : undefined,
        publish_date: meta.DateTimeOriginal?.value,
        license:      meta.License?.value ?? "Wikimedia-CC",
        description:  meta.ImageDescription?.value
          ? stripHtml(meta.ImageDescription.value).slice(0, 500) : undefined,
        file_size_bytes: info.size,
        tags:         ["wikimedia", "commons", "cc"],
      },
    });
  }
  return samples;
}

function stripHtml(html: string): string {
  return html.replace(/<[^>]+>/g, " ").replace(/\s{2,}/g, " ").trim();
}

// ── Batch Push ────────────────────────────────────────────────
async function batchPushToStaging(samples: DetectAISample[], env: Env): Promise<void> {
  const db = createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
  const BATCH_SIZE = 100;
  for (let i = 0; i < samples.length; i += BATCH_SIZE) {
    const { error } = await db.from("samples_staging").insert(samples.slice(i, i + BATCH_SIZE));
    if (error) console.error(JSON.stringify({ event: "STAGING_INSERT_ERROR", error }));
  }
}

// ── Main Handler ──────────────────────────────────────────────
interface ScraperRequest {
  source: SourceQueueItem;
  worker_id: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });

    const body = await request.json() as ScraperRequest;
    const { source } = body;
    const logger = new PipelineLogger(env);
    const startTime = Date.now();

    try {
      let rawSamples: Partial<DetectAISample>[] = [];

      switch (source.source_id) {
        case "unsplash":     rawSamples = await scrapeUnsplash(env);          break;
        case "pexels":       rawSamples = await scrapePexelsImages(env);      break;
        case "pixabay":      rawSamples = await scrapePixabay(env);           break;
        case "flickr-cc":    rawSamples = await scrapeFlickr(env);            break;
        case "wikimedia":    rawSamples = await scrapeWikimedia(source);      break;
        default:
          return new Response(JSON.stringify({ error: `Unknown image source: ${source.source_id}` }), { status: 400 });
      }

      const enriched: DetectAISample[] = rawSamples
        .filter(s => s.raw_content && s.raw_content.length > 0)
        .slice(0, 500)
        .map(s => ({
          sample_id:    uuidv4(),
          source_id:    source.source_id,
          source_url:   s.source_url ?? source.source_url,
          content_type: "image" as const,
          language:     "en",           // Images use metadata language
          raw_content:  s.raw_content!,
          storage_path: undefined,
          metadata:     s.metadata ?? {},
          scraped_at:   new Date().toISOString(),
          worker_id:    env.WORKER_ID,
          status:       "staged" as const,
        }));

      await batchPushToStaging(enriched, env);

      const duration = Date.now() - startTime;
      logger.log("SCRAPE_COMPLETE", {
        source_id: source.source_id,
        sample_count: enriched.length,
        duration_ms: duration,
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
