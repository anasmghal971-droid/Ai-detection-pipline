// ============================================================
// DETECT-AI: Video Scraper Worker (Stage 2)
// Cloudflare Worker — handles ALL video sources
//
// Sources:
//   • YouTube Data API v3  (captions + metadata → frame jobs queued)
//   • TED Talks            (public transcripts + video URLs)
//   • Pexels Video API     (CC0 licensed videos)
//   • Wikimedia Commons    (video files, CC licensed)
//   • VoxCeleb/AVA         (dataset download manifest)
//
// What this worker does:
//   1. Fetches video metadata + caption/transcript text
//   2. Stores video sample record in Supabase staging
//   3. Queues frame-extraction jobs → Supabase frame_jobs table
//      (actual frame extraction runs in Python worker via cron)
//   4. Respects all API rate limits with retry + backoff
// ============================================================

import type { Env, DetectAISample, SourceQueueItem } from "../../shared/types/index";
import { createSupabaseClient } from "../dispatcher/supabase";
import { PipelineLogger } from "../dispatcher/logger";

// ── Helpers ───────────────────────────────────────────────────
function uuidv4(): string { return crypto.randomUUID(); }

async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  maxAttempts = 3
): Promise<Response> {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const res = await fetch(url, options);
    if (res.status === 429) {
      throw Object.assign(new Error("Rate limited"), { rateLimited: true });
    }
    if (res.status === 403) {
      throw Object.assign(new Error("Quota exceeded or forbidden"), { quotaExceeded: true });
    }
    if (res.ok) return res;
    if (attempt === maxAttempts) throw new Error(`HTTP ${res.status}: ${url}`);
    await new Promise(r => setTimeout(r, 1000 * Math.pow(2, attempt)));
  }
  throw new Error("Max retries exceeded");
}

// ── Frame extraction job record ───────────────────────────────
interface FrameJob {
  job_id:      string;
  sample_id:   string;
  video_id:    string;
  video_url:   string;
  language:    string;
  source_id:   string;
  fps_target:  number;   // 1 | 2 | 5
  status:      "pending" | "processing" | "done" | "failed";
  created_at:  string;
}

// ════════════════════════════════════════════════════════════════
//  1. YOUTUBE DATA API v3
// ════════════════════════════════════════════════════════════════
//
// YouTube quota: 10,000 units/day (free)
//   search.list  = 100 units/call  → max 100 calls/day
//   captions.list = 50 units/call
//   videos.list   = 1 unit/call
//
// Strategy: fetch metadata + captions only (no video download)
//           Queue frame extraction jobs for Python worker.

interface YouTubeSearchItem {
  id: { videoId: string };
  snippet: {
    title: string;
    description: string;
    channelTitle: string;
    publishedAt: string;
    defaultAudioLanguage?: string;
    thumbnails: { high?: { url: string } };
  };
}

interface YouTubeCaptionTrack {
  id: string;
  snippet: {
    language: string;
    trackKind: string;
    name: string;
  };
}

async function scrapeYouTube(
  env: Env
): Promise<{ samples: Partial<DetectAISample>[]; frameJobs: Partial<FrameJob>[] }> {
  const samples: Partial<DetectAISample>[] = [];
  const frameJobs: Partial<FrameJob>[] = [];

  // Search across multiple relevant topics for diversity
  const searchTopics = [
    "documentary", "news", "lecture", "interview", "tutorial",
    "technology", "science", "history", "culture", "education",
  ];

  const languages = ["en", "fr", "de", "es", "ar", "zh", "ja", "ko", "pt", "hi"];

  for (const lang of languages.slice(0, 3)) {   // 3 langs per worker cycle
    for (const topic of searchTopics.slice(0, 3)) {  // 3 topics per lang

      const searchUrl = new URL("https://www.googleapis.com/youtube/v3/search");
      searchUrl.searchParams.set("part", "snippet");
      searchUrl.searchParams.set("q", topic);
      searchUrl.searchParams.set("type", "video");
      searchUrl.searchParams.set("relevanceLanguage", lang);
      searchUrl.searchParams.set("videoCaption", "closedCaption");  // Only captioned
      searchUrl.searchParams.set("videoLicense", "creativeCommon"); // CC-licensed
      searchUrl.searchParams.set("maxResults", "10");
      searchUrl.searchParams.set("key", env.YOUTUBE_API_KEY);

      let searchData: { items?: YouTubeSearchItem[] };
      try {
        const resp = await fetchWithRetry(searchUrl.toString());
        searchData = await resp.json();
      } catch (e) {
        console.error(`YouTube search failed (${lang}/${topic}): ${e}`);
        continue;
      }

      for (const item of searchData.items ?? []) {
        const videoId = item.id.videoId;
        if (!videoId) continue;

        // Get caption track list
        const captionUrl = `https://www.googleapis.com/youtube/v3/captions?part=snippet&videoId=${videoId}&key=${env.YOUTUBE_API_KEY}`;
        let captionLang = lang;

        try {
          const capResp = await fetchWithRetry(captionUrl);
          const capData = await capResp.json() as { items?: YouTubeCaptionTrack[] };
          const track = capData.items?.find(
            t => t.snippet.language.startsWith(lang) && t.snippet.trackKind !== "asr"
          ) ?? capData.items?.[0];
          if (track) captionLang = track.snippet.language.slice(0, 2);
        } catch {
          // Caption list unavailable — continue with language hint
        }

        const sample: Partial<DetectAISample> = {
          source_url:  `https://www.youtube.com/watch?v=${videoId}`,
          raw_content: `${item.snippet.title}\n\n${item.snippet.description}`.trim(),
          metadata: {
            title:        item.snippet.title,
            author:       item.snippet.channelTitle,
            publish_date: item.snippet.publishedAt,
            license:      "YouTube-CC",
            tags:         ["youtube", "video", topic, lang],
            description:  item.snippet.description?.slice(0, 500),
          },
        };
        samples.push(sample);

        // Queue frame extraction job for Python worker
        frameJobs.push({
          video_id:   videoId,
          video_url:  `https://www.youtube.com/watch?v=${videoId}`,
          language:   captionLang,
          source_id:  "youtube",
          fps_target: 2,   // Medium motion default for YouTube
        });
      }
    }
  }

  return { samples, frameJobs };
}

// ════════════════════════════════════════════════════════════════
//  2. TED TALKS
// ════════════════════════════════════════════════════════════════
//
// TED provides a public JSON API at ted.com
// We scrape: talk metadata + transcript text
// Transcripts available in 100+ languages

interface TEDTalk {
  id: number;
  slug: string;
  title: string;
  description: string;
  duration: number;                    // seconds
  published_at: string;
  speaker_name: string;
  topics: string[];
  languages: Array<{ languageCode: string; languageName: string }>;
}

async function scrapeTED(): Promise<{
  samples: Partial<DetectAISample>[];
  frameJobs: Partial<FrameJob>[];
}> {
  const samples: Partial<DetectAISample>[] = [];
  const frameJobs: Partial<FrameJob>[] = [];

  // TED's public API (no auth required)
  const apiUrl = "https://www.ted.com/talks?language=en&sort=newest&page=1";
  let tedData: { results?: TEDTalk[] };

  try {
    const resp = await fetchWithRetry(
      "https://www.ted.com/graphql",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: `{
            videos(first: 24, language: "en", sortField: PUBLISHED_AT) {
              nodes {
                slug
                title
                description
                duration
                publishedAt
                speakers { nodes { firstname lastname } }
                topics { nodes { name } }
                primaryLanguage { languageCode }
                videoDownloads { nativeLangSubtitledDownload }
              }
            }
          }`,
        }),
      },
      2
    );

    if (!resp.ok) throw new Error(`TED API: ${resp.status}`);
    const graphData = await resp.json() as {
      data?: {
        videos?: {
          nodes: Array<{
            slug: string; title: string; description: string;
            duration: number; publishedAt: string;
            speakers?: { nodes: Array<{ firstname: string; lastname: string }> };
            topics?: { nodes: Array<{ name: string }> };
            primaryLanguage?: { languageCode: string };
          }>;
        };
      };
    };

    for (const talk of graphData.data?.videos?.nodes ?? []) {
      const speaker = talk.speakers?.nodes
        .map(s => `${s.firstname} ${s.lastname}`)
        .join(", ") ?? "Unknown";
      const topics = talk.topics?.nodes.map(t => t.name) ?? [];
      const lang   = talk.primaryLanguage?.languageCode ?? "en";

      samples.push({
        source_url:  `https://www.ted.com/talks/${talk.slug}`,
        raw_content: `${talk.title}\n\n${talk.description ?? ""}`.trim(),
        metadata: {
          title:            talk.title,
          author:           speaker,
          publish_date:     talk.publishedAt,
          license:          "TED-CC",
          tags:             ["ted", "talk", "education", ...topics],
          duration_seconds: talk.duration,
        },
      });

      frameJobs.push({
        video_id:   `ted_${talk.slug}`,
        video_url:  `https://www.ted.com/talks/${talk.slug}`,
        language:   lang.slice(0, 2),
        source_id:  "ted-talks",
        fps_target: 1,  // TED talks are low motion — 1fps sufficient
      });
    }
  } catch (e) {
    // Fallback: scrape the HTML listing
    console.warn(`TED GraphQL failed, using RSS fallback: ${e}`);
    try {
      const rssResp = await fetchWithRetry("https://feeds.feedburner.com/tedtalks_audio");
      const xml = await rssResp.text();
      const items = [...xml.matchAll(/<item>([\s\S]*?)<\/item>/gi)];

      for (const match of items.slice(0, 20)) {
        const item = match[1];
        const title = item.match(/<title><!\[CDATA\[(.*?)\]\]>/)?.[1] ?? "";
        const desc  = item.match(/<description><!\[CDATA\[(.*?)\]\]>/)?.[1] ?? "";
        const link  = item.match(/<link>(.*?)<\/link>/)?.[1] ?? "";
        const pub   = item.match(/<pubDate>(.*?)<\/pubDate>/)?.[1] ?? "";

        if (!title) continue;
        samples.push({
          source_url:  link,
          raw_content: `${title}\n\n${desc.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim()}`,
          metadata:    { title, publish_date: pub, license: "TED-CC", tags: ["ted","talk"] },
        });
      }
    } catch (rssErr) {
      console.error(`TED RSS also failed: ${rssErr}`);
    }
  }

  return { samples, frameJobs };
}

// ════════════════════════════════════════════════════════════════
//  3. PEXELS VIDEO API
// ════════════════════════════════════════════════════════════════

interface PexelsVideoFile {
  id: number;
  width: number;
  height: number;
  link: string;
  quality: string;
}

interface PexelsVideo {
  id: number;
  url: string;
  duration: number;
  user: { name: string };
  video_files: PexelsVideoFile[];
  tags?: string[];
}

async function scrapePexelsVideo(env: Env): Promise<{
  samples: Partial<DetectAISample>[];
  frameJobs: Partial<FrameJob>[];
}> {
  const resp = await fetchWithRetry(
    "https://api.pexels.com/videos/popular?per_page=80&min_duration=5&max_duration=300",
    { headers: { Authorization: env.PEXELS_API_KEY } }
  );
  const data = await resp.json() as { videos: PexelsVideo[] };

  const samples: Partial<DetectAISample>[] = [];
  const frameJobs: Partial<FrameJob>[] = [];

  for (const vid of data.videos ?? []) {
    // Pick highest-quality video file
    const bestFile = vid.video_files
      .filter(f => f.quality === "hd" || f.quality === "sd")
      .sort((a, b) => (b.width * b.height) - (a.width * a.height))[0];

    if (!bestFile) continue;

    samples.push({
      source_url:  vid.url,
      raw_content: bestFile.link,  // Direct video URL
      metadata: {
        title:            `Pexels Video ${vid.id}`,
        author:           vid.user.name,
        license:          "Pexels-License",
        duration_seconds: vid.duration,
        dimensions:       { width: bestFile.width, height: bestFile.height },
        tags:             ["pexels", "video", "CC0"],
      },
    });

    // High motion for stock video — use 2fps
    frameJobs.push({
      video_id:   `pexels_${vid.id}`,
      video_url:  bestFile.link,
      language:   "en",
      source_id:  "pexels-video",
      fps_target: 2,
    });
  }

  return { samples, frameJobs };
}

// ════════════════════════════════════════════════════════════════
//  4. WIKIMEDIA COMMONS (Video files)
// ════════════════════════════════════════════════════════════════

async function scrapeWikimediaVideo(): Promise<{
  samples: Partial<DetectAISample>[];
  frameJobs: Partial<FrameJob>[];
}> {
  // Search for .webm and .ogv video files on Wikimedia
  const url = "https://commons.wikimedia.org/w/api.php?" +
    "action=query&list=search&srsearch=filetype:video&srnamespace=6" +
    "&srlimit=50&format=json&srprop=title|snippet";

  const resp = await fetchWithRetry(url);
  const data = await resp.json() as {
    query: { search: Array<{ title: string; snippet: string }> };
  };

  const samples: Partial<DetectAISample>[] = [];
  const frameJobs: Partial<FrameJob>[] = [];

  for (const result of (data.query?.search ?? []).slice(0, 20)) {
    // Get actual file URL
    const infoUrl = `https://commons.wikimedia.org/w/api.php?` +
      `action=query&prop=imageinfo&iiprop=url|mime|size|extmetadata` +
      `&titles=${encodeURIComponent(result.title)}&format=json`;

    let fileUrl = "";
    let lang    = "unknown";
    let license = "Wikimedia-CC";

    try {
      const infoResp = await fetchWithRetry(infoUrl);
      const infoData = await infoResp.json() as {
        query: { pages: Record<string, {
          imageinfo?: Array<{
            url: string; mime: string;
            extmetadata?: {
              License?: { value?: string };
              LanguageCode?: { value?: string };
            };
          }>;
        }> };
      };

      const page  = Object.values(infoData.query.pages)[0];
      const info  = page.imageinfo?.[0];
      if (!info || !info.mime.startsWith("video/")) continue;

      fileUrl = info.url;
      lang    = info.extmetadata?.LanguageCode?.value?.slice(0, 2) ?? "unknown";
      license = info.extmetadata?.License?.value ?? "Wikimedia-CC";
    } catch {
      continue;
    }

    if (!fileUrl) continue;

    const cleanSnippet = result.snippet.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
    samples.push({
      source_url:  `https://commons.wikimedia.org/wiki/${encodeURIComponent(result.title)}`,
      raw_content: fileUrl,
      metadata: {
        title:   result.title.replace("File:", ""),
        license,
        tags:    ["wikimedia", "commons", "cc"],
        description: cleanSnippet,
      },
    });

    frameJobs.push({
      video_id:   `wiki_${result.title.replace(/[^a-zA-Z0-9]/g, "_").slice(0, 60)}`,
      video_url:  fileUrl,
      language:   lang,
      source_id:  "wikimedia",
      fps_target: 1,
    });
  }

  return { samples, frameJobs };
}

// ════════════════════════════════════════════════════════════════
//  5. VOXCELEB / AVA (Dataset Download Manifest)
// ════════════════════════════════════════════════════════════════
//
// VoxCeleb is a one-time academic download.
// This worker creates the manifest entry — actual download
// is handled by the Python frame extraction worker separately.

async function scrapeVoxCeleb(): Promise<{
  samples: Partial<DetectAISample>[];
  frameJobs: Partial<FrameJob>[];
}> {
  // VoxCeleb1 has 1,251 speakers, 153,516 utterances
  // We generate manifest entries pointing to official dataset
  const manifest = [
    {
      dataset: "VoxCeleb1",
      url: "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html",
      description: "VoxCeleb1: 153,516 utterances from 1,251 celebrities",
      speakers: 1251,
      utterances: 153516,
      language: "en",
    },
    {
      dataset: "VoxCeleb2",
      url: "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html",
      description: "VoxCeleb2: 1,128,246 utterances from 6,112 celebrities",
      speakers: 6112,
      utterances: 1128246,
      language: "en",
    },
    {
      dataset: "AVA-ActiveSpeaker",
      url: "https://research.google.com/ava/",
      description: "AVA: 15-minute clips of 160 Hollywood films with temporal labels",
      language: "en",
    },
  ];

  const samples: Partial<DetectAISample>[] = manifest.map(m => ({
    source_url:  m.url,
    raw_content: JSON.stringify(m),
    metadata: {
      title:   m.dataset,
      license: "Academic-Research",
      tags:    ["voxceleb", "academic", "face-recognition", "speaker"],
      description: m.description,
    },
  }));

  // Frame jobs will be created by Python worker after dataset download
  const frameJobs: Partial<FrameJob>[] = [];

  return { samples, frameJobs };
}

// ════════════════════════════════════════════════════════════════
//  BATCH PUSH HELPERS
// ════════════════════════════════════════════════════════════════

async function batchPushSamples(
  rawSamples: Partial<DetectAISample>[],
  sourceId: string,
  defaultLang: string,
  env: Env,
  workerId: string
): Promise<DetectAISample[]> {
  const enriched: DetectAISample[] = rawSamples
    .filter(s => s.raw_content && s.raw_content.length > 0)
    .map(s => ({
      sample_id:    uuidv4(),
      source_id:    sourceId,
      source_url:   s.source_url ?? "",
      content_type: "video" as const,
      language:     defaultLang,
      raw_content:  s.raw_content!,
      metadata:     s.metadata ?? {},
      scraped_at:   new Date().toISOString(),
      worker_id:    workerId,
      status:       "staged" as const,
    }));

  if (enriched.length === 0) return enriched;

  const db = createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
  const BATCH = 100;
  for (let i = 0; i < enriched.length; i += BATCH) {
    const { error } = await db.from("samples_staging").insert(enriched.slice(i, i + BATCH));
    if (error) console.error(JSON.stringify({ event: "STAGING_INSERT_ERROR", error }));
  }
  return enriched;
}

async function batchPushFrameJobs(
  jobs: Partial<FrameJob>[],
  sampleMap: Map<string, string>,  // video_id → sample_id
  env: Env
): Promise<void> {
  if (jobs.length === 0) return;

  const full: FrameJob[] = jobs.map(j => ({
    job_id:     uuidv4(),
    sample_id:  sampleMap.get(j.video_id ?? "") ?? "",
    video_id:   j.video_id ?? "",
    video_url:  j.video_url ?? "",
    language:   j.language ?? "en",
    source_id:  j.source_id ?? "",
    fps_target: j.fps_target ?? 2,
    status:     "pending" as const,
    created_at: new Date().toISOString(),
  }));

  const db = createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
  const BATCH = 100;
  for (let i = 0; i < full.length; i += BATCH) {
    const { error } = await db.from("frame_jobs").insert(full.slice(i, i + BATCH));
    if (error) console.error(JSON.stringify({ event: "FRAME_JOBS_INSERT_ERROR", error }));
  }
}

// ════════════════════════════════════════════════════════════════
//  MAIN HANDLER
// ════════════════════════════════════════════════════════════════

interface VideoScraperRequest {
  source: SourceQueueItem;
  worker_id: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });

    const body      = await request.json() as VideoScraperRequest;
    const { source } = body;
    const logger    = new PipelineLogger(env);
    const startTime = Date.now();

    try {
      let rawSamples: Partial<DetectAISample>[] = [];
      let frameJobs: Partial<FrameJob>[]         = [];
      let defaultLang = "en";

      // ── Route to source-specific scraper ──────────────────
      switch (source.source_id) {
        case "youtube": {
          const result = await scrapeYouTube(env);
          rawSamples   = result.samples;
          frameJobs    = result.frameJobs;
          break;
        }
        case "ted-talks": {
          const result = await scrapeTED();
          rawSamples   = result.samples;
          frameJobs    = result.frameJobs;
          break;
        }
        case "pexels-video": {
          const result = await scrapePexelsVideo(env);
          rawSamples   = result.samples;
          frameJobs    = result.frameJobs;
          break;
        }
        case "wikimedia": {
          const result = await scrapeWikimediaVideo();
          rawSamples   = result.samples;
          frameJobs    = result.frameJobs;
          break;
        }
        case "voxceleb": {
          const result = await scrapeVoxCeleb();
          rawSamples   = result.samples;
          frameJobs    = result.frameJobs;
          break;
        }
        default:
          return new Response(
            JSON.stringify({ error: `Unknown video source: ${source.source_id}` }),
            { status: 400 }
          );
      }

      // ── Persist samples ────────────────────────────────────
      const enriched = await batchPushSamples(
        rawSamples, source.source_id, defaultLang, env, env.WORKER_ID
      );

      // ── Build video_id → sample_id map for frame jobs ──────
      const sampleMap = new Map<string, string>();
      for (const job of frameJobs) {
        // Find matching sample by source_url containing video_id
        const match = enriched.find(s =>
          job.video_url && s.source_url.includes(job.video_id ?? "")
        );
        if (match) sampleMap.set(job.video_id ?? "", match.sample_id);
      }

      // ── Queue frame extraction jobs ────────────────────────
      await batchPushFrameJobs(frameJobs, sampleMap, env);

      const duration = Date.now() - startTime;
      logger.log("SCRAPE_COMPLETE", {
        source_id:    source.source_id,
        sample_count: enriched.length,
        duration_ms:  duration,
      });
      await logger.flush();

      return new Response(
        JSON.stringify({
          samples_scraped: enriched.length,
          frame_jobs_queued: frameJobs.length,
          duration_ms: duration,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      logger.log("SCRAPE_ERROR", { source_id: source.source_id, error_message: errorMsg });
      await logger.flush();

      if ((err as { rateLimited?: boolean }).rateLimited) {
        return new Response(JSON.stringify({ error: "rate_limited" }), { status: 429 });
      }
      if ((err as { quotaExceeded?: boolean }).quotaExceeded) {
        return new Response(JSON.stringify({ error: "quota_exceeded" }), { status: 429 });
      }
      return new Response(JSON.stringify({ error: errorMsg }), { status: 500 });
    }
  },
};
