// DETECT-AI Text Scraper — ALL SOURCES, NO BROKEN APIS
// No NewsAPI (rate limited) — replaced with direct RSS feeds
// All sources work without hitting rate limits

import type { Env, DetectAISample } from "../../shared/types/index";
import { createSupabaseClient } from "../dispatcher/supabase";
import { PipelineLogger } from "../dispatcher/logger";

function uuidv4(): string { return crypto.randomUUID(); }

async function fetchText(url: string, opts: RequestInit = {}): Promise<string | null> {
  for (let i = 1; i <= 3; i++) {
    try {
      const r = await fetch(url, { ...opts, signal: AbortSignal.timeout(10000), headers: { "User-Agent": "DETECT-AI/1.0 dataset-builder", ...(opts.headers ?? {}) } });
      if (r.status === 429) throw Object.assign(new Error("Rate limited"), { rateLimited: true });
      if (r.ok) return await r.text();
      if (i === 3) return null;
      await new Promise(x => setTimeout(x, 800 * i));
    } catch (e: any) {
      if (e.rateLimited) throw e;
      if (i === 3) return null;
      await new Promise(x => setTimeout(x, 800 * i));
    }
  }
  return null;
}

function parseRSS(xml: string, lang: string, source: string): Array<{ title: string; link: string; text: string; date: string; lang: string; source: string }> {
  const items: any[] = [];
  const blocks = xml.matchAll(/<item>([\s\S]*?)<\/item>/g);
  for (const m of blocks) {
    const b = m[1];
    const title = (b.match(/<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?<\/title>/)?.[1] ?? "").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").trim();
    const link = (b.match(/<link>(.*?)<\/link>|<link[^>]*href="([^"]+)"/)?.[1] ?? "").trim();
    const desc = (b.match(/<description>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?<\/description>/)?.[1] ?? "").replace(/<[^>]+>/g, " ").replace(/\s{2,}/g, " ").trim().slice(0, 1500);
    const date = b.match(/<pubDate>(.*?)<\/pubDate>/)?.[1] ?? "";
    const text = `${title}\n\n${desc}`.trim();
    if (title && link && text.length > 80) items.push({ title, link, text, date, lang, source });
  }
  return items;
}

// Direct RSS feeds — no API key needed, never rate limited at these volumes
const RSS_FEEDS = [
  // English news
  { url: "https://feeds.bbci.co.uk/news/rss.xml",                     source: "bbc-news",    lang: "en" },
  { url: "https://feeds.bbci.co.uk/news/world/rss.xml",               source: "bbc-news",    lang: "en" },
  { url: "https://feeds.bbci.co.uk/news/technology/rss.xml",          source: "bbc-news",    lang: "en" },
  { url: "https://feeds.reuters.com/reuters/topNews",                 source: "reuters",     lang: "en" },
  { url: "https://feeds.reuters.com/reuters/worldNews",               source: "reuters",     lang: "en" },
  { url: "https://feeds.reuters.com/reuters/technologyNews",          source: "reuters",     lang: "en" },
  { url: "https://feeds.theguardian.com/theguardian/world/rss",       source: "guardian",    lang: "en" },
  { url: "https://feeds.theguardian.com/theguardian/technology/rss",  source: "guardian",    lang: "en" },
  { url: "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",    source: "nytimes",     lang: "en" },
  { url: "https://feeds.washingtonpost.com/rss/world",                source: "washpost",    lang: "en" },
  { url: "https://abcnews.go.com/abcnews/topstories",                 source: "abc-news",    lang: "en" },
  { url: "https://feeds.npr.org/1001/rss.xml",                        source: "npr",         lang: "en" },
  // Arabic
  { url: "https://www.aljazeera.net/xml/rss/all.xml",                 source: "aljazeera",   lang: "ar" },
  { url: "https://arabic.rt.com/rss/",                                source: "rt-arabic",   lang: "ar" },
  // French
  { url: "https://www.lemonde.fr/rss/une.xml",                        source: "lemonde",     lang: "fr" },
  { url: "https://rss.dw.com/rdf/rss-fre-all",                        source: "dw-french",   lang: "fr" },
  // German
  { url: "https://www.spiegel.de/schlagzeilen/tops/index.rss",        source: "spiegel",     lang: "de" },
  { url: "https://rss.dw.com/rdf/rss-de-all",                         source: "dw-german",   lang: "de" },
  // Spanish
  { url: "https://www.bbc.com/mundo/rss.xml",                         source: "bbc-espanol", lang: "es" },
  { url: "https://ep00.epimg.net/rss/elpais/portada.xml",             source: "elpais",      lang: "es" },
];

// Wikipedia random articles — 0 rate limit, unlimited
async function scrapeWikipedia(lang = "en", count = 20): Promise<any[]> {
  const d: any = await (await fetch(`https://${lang}.wikipedia.org/w/api.php?action=query&list=random&rnlimit=${count}&rnnamespace=0&format=json&origin=*`).catch(() => null))?.json().catch(() => null);
  if (!d?.query?.random) return [];
  const out: any[] = [];
  await Promise.allSettled(d.query.random.map(async (p: any) => {
    const e: any = await (await fetch(`https://${lang}.wikipedia.org/w/api.php?action=query&pageids=${p.id}&prop=extracts&exintro=true&format=json&origin=*`).catch(() => null))?.json().catch(() => null);
    const extract = Object.values(e?.query?.pages ?? {} as Record<string, any>)[0] as any;
    const clean = (extract?.extract ?? "").replace(/<[^>]+>/g, " ").replace(/\s{2,}/g, " ").trim().slice(0, 2000);
    if (clean.length > 100) out.push({ source_url: `https://${lang}.wikipedia.org/?curid=${p.id}`, raw_content: `${p.title}\n\n${clean}`, metadata: { title: p.title, source: `wikipedia-${lang}`, language: lang, license: "CC-BY-SA", tags: ["wikipedia", "encyclopedia", "human-content"] } });
  }));
  return out;
}

// arXiv abstracts — academic text, no rate limit at these volumes
async function scrapeArxiv(): Promise<any[]> {
  const cats = ["cs.CV", "cs.AI", "cs.LG", "cs.CL", "stat.ML", "eess.IV"];
  const cat = cats[Math.floor(Math.random() * cats.length)];
  const xml = await fetchText(`http://export.arxiv.org/rss/${cat}`, { headers: { Accept: "application/rss+xml" } });
  if (!xml) return [];
  return parseRSS(xml, "en", "arxiv").slice(0, 30).map(a => ({
    source_url: a.link, raw_content: a.text,
    metadata: { title: a.title, source: "arxiv", category: cat, language: "en", license: "arXiv", tags: ["arxiv", "academic", "science", "human-content"] }
  }));
}

// StackExchange — Q&A text, no key needed for this volume
async function scrapeStackExchange(): Promise<any[]> {
  const sites = ["stackoverflow", "superuser", "askubuntu", "datascience", "ai"];
  const site = sites[Math.floor(Math.random() * sites.length)];
  try {
    const r = await fetch(`https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&site=${site}&filter=withbody&pagesize=30`);
    const d = await r.json() as any;
    return (d?.items ?? []).map((q: any) => ({
      source_url: q.link, raw_content: `${q.title}\n\n${(q.body ?? "").replace(/<[^>]+>/g, " ").replace(/\s{2,}/g, " ").trim().slice(0, 1500)}`,
      metadata: { title: q.title, tags: [...(q.tags ?? []), "stackexchange", "q&a", "human-content"], score: q.score, source: `stackexchange-${site}`, license: "CC BY-SA 4.0" }
    }));
  } catch { return []; }
}

// PapersWithCode — ML papers, no rate limit
async function scrapePapersWithCode(): Promise<any[]> {
  try {
    const r = await fetch("https://paperswithcode.com/api/v1/papers/?format=json&items_per_page=30&ordering=-arxiv_id");
    const d = await r.json() as any;
    return (d?.results ?? []).filter((p: any) => p.abstract).map((p: any) => ({
      source_url: p.url_abs ?? p.url_pdf ?? `https://paperswithcode.com/paper/${p.id}`,
      raw_content: `${p.title}\n\n${p.abstract}`.trim().slice(0, 2000),
      metadata: { title: p.title, authors: p.authors?.map((a: any) => a.full_name).join(", "), source: "paperswithcode", license: "Public", tags: ["paperswithcode", "ml-research", "academic", "human-content"] }
    }));
  } catch { return []; }
}

// Reddit — via public JSON API, no auth needed for this volume
async function scrapeReddit(env: Env): Promise<any[]> {
  if (!env.REDDIT_CLIENT_ID) return [];
  const subs = ["worldnews", "science", "technology", "MachineLearning", "artificial", "datascience"];
  const sub = subs[Math.floor(Math.random() * subs.length)];
  try {
    const r = await fetch(`https://www.reddit.com/r/${sub}/hot.json?limit=25`, { headers: { "User-Agent": "DETECT-AI/1.0" } });
    const d = await r.json() as any;
    return (d?.data?.children ?? [])
      .filter((p: any) => p.data?.selftext?.length > 50)
      .map((p: any) => ({
        source_url: `https://reddit.com${p.data.permalink}`,
        raw_content: `${p.data.title}\n\n${p.data.selftext}`.trim().slice(0, 2000),
        metadata: { title: p.data.title, subreddit: p.data.subreddit, upvotes: p.data.score, license: "CC BY-SA 4.0", tags: ["reddit", sub, "human-content"] }
      }));
  } catch { return []; }
}

// ── MAIN ─────────────────────────────────────────────────────
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
    const t0 = Date.now();
    const { source } = await request.json() as { source: { source_id: string; source_url: string } };
    const logger = new PipelineLogger(env);

    try {
      let raw: any[] = [];

      if (["bbc-news","reuters","aljazeera","guardian","nytimes","washpost","abc-news","npr","lemonde","spiegel","elpais","dw-german","dw-french","bbc-espanol","rt-arabic"].includes(source.source_id)) {
        // RSS source — fetch all matching feeds in parallel
        const feeds = RSS_FEEDS.filter(f => f.source === source.source_id);
        const results = await Promise.allSettled(feeds.map(f => fetchText(f.url).then(xml => xml ? parseRSS(xml, f.lang, f.source) : [])));
        for (const r of results) if (r.status === "fulfilled") raw.push(...r.value);
      } else {
        switch (source.source_id) {
          case "wikipedia":       raw = [...await scrapeWikipedia("en", 15), ...await scrapeWikipedia("ar", 8), ...await scrapeWikipedia("fr", 8), ...await scrapeWikipedia("de", 8), ...await scrapeWikipedia("es", 8)]; break;
          case "arxiv":           raw = await scrapeArxiv(); break;
          case "stackexchange":   raw = await scrapeStackExchange(); break;
          case "paperswithcode":  raw = await scrapePapersWithCode(); break;
          case "reddit":          raw = await scrapeReddit(env); break;
          case "newsapi":         raw = []; break; // disabled — exhausted free tier
          default:
            return new Response(JSON.stringify({ error: `Unknown text source: ${source.source_id}` }), { status: 400 });
        }
      }

      const db = createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
      const enriched: DetectAISample[] = raw
        .filter((s: any) => (s.raw_content ?? s.text ?? "").length > 80)
        .slice(0, 500)
        .map((s: any) => ({
          sample_id: uuidv4(), source_id: source.source_id,
          source_url: s.source_url ?? s.link ?? source.source_url,
          content_type: "text" as const,
          language: (s.metadata?.language ?? s.lang ?? "en") as string,
          raw_content: (s.raw_content ?? s.text ?? "").slice(0, 5000),
          label: "HUMAN" as const, final_confidence: 0.95, verified: false,
          metadata: s.metadata ?? { title: s.title, source: source.source_id, tags: ["human-content"] },
          scraped_at: new Date().toISOString(), worker_id: env.WORKER_ID ?? "scraper-text-01", status: "staged" as const,
        }));

      let pushed = 0;
      for (let i = 0; i < enriched.length; i += 100) {
        const { error } = await db.from("samples_staging").insert(enriched.slice(i, i + 100));
        if (!error) pushed += Math.min(100, enriched.length - i);
        else logger.log("INSERT_ERROR", { source_id: source.source_id, error_message: JSON.stringify(error) });
      }

      logger.log("SCRAPE_COMPLETE", { source_id: source.source_id, sample_count: pushed, duration_ms: Date.now() - t0 });
      await logger.flush();
      return new Response(JSON.stringify({ samples_scraped: pushed, duration_ms: Date.now() - t0 }), { status: 200, headers: { "Content-Type": "application/json" } });
    } catch (err: any) {
      const msg = err?.message ?? String(err);
      logger.log("SCRAPE_ERROR", { source_id: source.source_id, error_message: msg });
      await logger.flush();
      if (err.rateLimited) return new Response(JSON.stringify({ error: "rate_limited" }), { status: 429 });
      return new Response(JSON.stringify({ error: msg }), { status: 500 });
    }
  },
};
