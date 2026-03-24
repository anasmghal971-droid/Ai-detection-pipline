// DETECT-AI: Text Scraper v2 — Bulletproof
// Sources: 12 RSS feeds (no API key) + Wikipedia + arXiv + StackExchange + PapersWithCode
// All sources handle empty/error responses gracefully

import type { Env, DetectAISample } from "../../shared/types/index";
import { createSupabaseClient } from "../dispatcher/supabase";
import { PipelineLogger } from "../dispatcher/logger";

function uuidv4() { return crypto.randomUUID(); }

async function safeFetch(url: string, opts: RequestInit = {}): Promise<string | null> {
  try {
    const r = await fetch(url, {
      ...opts,
      signal: AbortSignal.timeout(10000),
      headers: { "User-Agent": "DETECT-AI/1.0 dataset-builder", ...opts.headers }
    });
    if (!r.ok) return null;
    return await r.text();
  } catch { return null; }
}

async function safeJSON(url: string, opts: RequestInit = {}): Promise<any> {
  const t = await safeFetch(url, opts);
  if (!t) return null;
  try { return JSON.parse(t); } catch { return null; }
}

function parseRSS(xml: string, lang: string, sourceId: string): any[] {
  const items: any[] = [];
  for (const m of xml.matchAll(/<item>([\s\S]*?)<\/item>/g)) {
    const b = m[1];
    const title = (b.match(/<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?<\/title>/s)?.[1] ?? "")
      .replace(/&amp;/g,"&").replace(/&lt;/g,"<").replace(/&gt;/g,">").replace(/<[^>]+>/g,"").trim();
    const link  = (b.match(/<link>(.*?)<\/link>/s)?.[1] ?? "").trim();
    const desc  = (b.match(/<description>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?<\/description>/s)?.[1] ?? "")
      .replace(/<[^>]+>/g," ").replace(/\s{2,}/g," ").trim().slice(0,1500);
    const date  = b.match(/<pubDate>(.*?)<\/pubDate>/)?.[1] ?? "";
    const text  = `${title}\n\n${desc}`.trim();
    if (title && link && text.length > 80) {
      items.push({ source_url: link, raw_content: text,
        metadata: { title, publish_date: date, source: sourceId, language: lang,
          license: "News", tags: [sourceId, "news", "human-content"], is_ai_generated: false }
      });
    }
  }
  return items;
}

// RSS feeds per source_id
const RSS_MAP: Record<string, Array<{ url: string; lang: string }>> = {
  "bbc-news":   [
    { url: "https://feeds.bbci.co.uk/news/rss.xml",           lang: "en" },
    { url: "https://feeds.bbci.co.uk/news/world/rss.xml",     lang: "en" },
    { url: "https://feeds.bbci.co.uk/news/technology/rss.xml",lang: "en" },
  ],
  "reuters":    [
    { url: "https://feeds.reuters.com/reuters/topNews",        lang: "en" },
    { url: "https://feeds.reuters.com/reuters/worldNews",      lang: "en" },
  ],
  "guardian":   [
    { url: "https://feeds.theguardian.com/theguardian/world/rss",      lang: "en" },
    { url: "https://feeds.theguardian.com/theguardian/technology/rss", lang: "en" },
  ],
  "nytimes":    [
    { url: "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",   lang: "en" },
    { url: "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml", lang: "en" },
  ],
  "aljazeera":  [
    { url: "https://www.aljazeera.net/xml/rss/all.xml",  lang: "ar" },
    { url: "https://www.aljazeera.com/xml/rss/all.xml",  lang: "en" },
  ],
  "npr":        [
    { url: "https://feeds.npr.org/1001/rss.xml",         lang: "en" },
    { url: "https://feeds.npr.org/1025/rss.xml",         lang: "en" },
  ],
};

async function scrapeRSS(sourceId: string): Promise<any[]> {
  const feeds = RSS_MAP[sourceId];
  if (!feeds) return [];
  const results = await Promise.allSettled(
    feeds.map(f => safeFetch(f.url).then(xml => xml ? parseRSS(xml, f.lang, sourceId) : []))
  );
  const out: any[] = [];
  for (const r of results) if (r.status === "fulfilled") out.push(...r.value);
  return out;
}

async function scrapeWikipedia(): Promise<any[]> {
  const langs = ["en","ar","fr","de","es","zh","ja","ko","pt","hi"];
  const out: any[] = [];
  await Promise.allSettled(langs.map(async lang => {
    const d = await safeJSON(`https://${lang}.wikipedia.org/w/api.php?action=query&list=random&rnlimit=15&rnnamespace=0&format=json&origin=*`);
    if (!d?.query?.random) return;
    await Promise.allSettled(d.query.random.map(async (p: any) => {
      const e = await safeJSON(`https://${lang}.wikipedia.org/w/api.php?action=query&pageids=${p.id}&prop=extracts&exintro=true&format=json&origin=*`);
      const page = Object.values(e?.query?.pages ?? {} as any)[0] as any;
      const clean = (page?.extract ?? "").replace(/<[^>]+>/g," ").replace(/\s{2,}/g," ").trim().slice(0,2000);
      if (clean.length > 100) out.push({
        source_url: `https://${lang}.wikipedia.org/?curid=${p.id}`,
        raw_content: `${p.title}\n\n${clean}`,
        metadata: { title: p.title, language: lang, source: "wikipedia", license: "CC-BY-SA", tags: ["wikipedia","encyclopedia","human-content"], is_ai_generated: false }
      });
    }));
  }));
  return out;
}

async function scrapeArxiv(): Promise<any[]> {
  const cats = ["cs.CV","cs.AI","cs.LG","cs.CL","stat.ML","eess.IV"];
  const cat  = cats[Math.floor(Math.random() * cats.length)];
  const xml  = await safeFetch(`https://export.arxiv.org/rss/${cat}`);
  if (!xml) return [];
  return parseRSS(xml, "en", "arxiv").slice(0, 40).map(a => ({
    ...a, metadata: { ...a.metadata, category: cat, tags: ["arxiv","academic","research","human-content"] }
  }));
}

async function scrapeStackExchange(): Promise<any[]> {
  const sites = ["stackoverflow","superuser","datascience","ai","stats"];
  const site  = sites[Math.floor(Math.random() * sites.length)];
  const d     = await safeJSON(`https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&site=${site}&filter=withbody&pagesize=30`);
  if (!d?.items) return [];
  return d.items.map((q: any) => ({
    source_url:  q.link,
    raw_content: `${q.title}\n\n${(q.body ?? "").replace(/<[^>]+>/g," ").replace(/\s{2,}/g," ").trim().slice(0,1500)}`,
    metadata: { title: q.title, site, score: q.score, tags: [...(q.tags??[]),"stackexchange","q&a","human-content"], license: "CC BY-SA 4.0", is_ai_generated: false }
  }));
}

async function scrapePapersWithCode(): Promise<any[]> {
  const d = await safeJSON("https://paperswithcode.com/api/v1/papers/?format=json&items_per_page=30&ordering=-arxiv_id");
  if (!d?.results) return [];
  return d.results.filter((p: any) => p.abstract?.length > 50).map((p: any) => ({
    source_url:  p.url_abs ?? `https://paperswithcode.com/paper/${p.id}`,
    raw_content: `${p.title}\n\n${p.abstract}`.trim().slice(0,2000),
    metadata: { title: p.title, source: "paperswithcode", tags: ["paperswithcode","ml","academic","human-content"], license: "Public", is_ai_generated: false }
  }));
}

async function scrapeReddit(env: Env): Promise<any[]> {
  const subs = ["worldnews","science","technology","MachineLearning","datascience"];
  const sub  = subs[Math.floor(Math.random() * subs.length)];
  const d    = await safeJSON(`https://www.reddit.com/r/${sub}/hot.json?limit=25`, { headers: { "User-Agent": "DETECT-AI/1.0" } });
  if (!d?.data?.children) return [];
  return d.data.children
    .filter((p: any) => p.data?.selftext?.length > 50)
    .map((p: any) => ({
      source_url:  `https://reddit.com${p.data.permalink}`,
      raw_content: `${p.data.title}\n\n${p.data.selftext}`.trim().slice(0,2000),
      metadata: { title: p.data.title, subreddit: sub, tags: ["reddit",sub,"human-content"], license: "CC BY-SA 4.0", is_ai_generated: false }
    }));
}

async function scrapeWorldBank(): Promise<any[]> {
  const d = await safeJSON("https://search.worldbank.org/api/v3/wds?format=json&rows=30&os=0&fl=id,title,docdt,abstract&srt=docdt&order=desc");
  if (!d?.documents) return [];
  return Object.values(d.documents as Record<string,any>)
    .filter((doc: any) => doc.abstract?.length > 50)
    .map((doc: any) => ({
      source_url:  `https://documents.worldbank.org/en/publication/documents-reports/documentdetail/${doc.id}`,
      raw_content: `${doc.title}\n\n${doc.abstract}`.trim().slice(0,2000),
      metadata: { title: doc.title, source: "worldbank", tags: ["worldbank","policy","research","human-content"], license: "CC BY 4.0", is_ai_generated: false }
    }));
}

// ── Main handler ──────────────────────────────────────────────
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
    const t0 = Date.now();
    const { source } = await request.json() as { source: { source_id: string; source_url: string } };
    const logger = new PipelineLogger(env);

    try {
      let raw: any[] = [];

      const rssSourceIds = Object.keys(RSS_MAP);
      if (rssSourceIds.includes(source.source_id)) {
        raw = await scrapeRSS(source.source_id);
      } else {
        switch (source.source_id) {
          case "wikipedia":      raw = await scrapeWikipedia();      break;
          case "arxiv":          raw = await scrapeArxiv();          break;
          case "stackexchange":  raw = await scrapeStackExchange();  break;
          case "paperswithcode": raw = await scrapePapersWithCode(); break;
          case "reddit":         raw = await scrapeReddit(env);      break;
          case "worldbank":      raw = await scrapeWorldBank();       break;
          default:
            return new Response(JSON.stringify({ error: `Unknown text source: ${source.source_id}` }), { status: 400 });
        }
      }

      if (raw.length === 0) {
        logger.log("SCRAPE_COMPLETE", { source_id: source.source_id, sample_count: 0, duration_ms: Date.now()-t0 });
        await logger.flush();
        return new Response(JSON.stringify({ samples_scraped: 0 }), { status: 200 });
      }

      const d = createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
      const enriched: DetectAISample[] = raw
        .filter((s: any) => (s.raw_content ?? "").length > 80)
        .slice(0, 500)
        .map((s: any) => ({
          sample_id:        uuidv4(),
          source_id:        source.source_id,
          source_url:       s.source_url ?? source.source_url,
          content_type:     "text" as const,
          language:         (s.metadata?.language ?? "en") as string,
          raw_content:      s.raw_content.slice(0, 5000),
          label:            "HUMAN" as const,
          final_confidence: 0.95,
          verified:         false,
          metadata:         s.metadata ?? {},
          scraped_at:       new Date().toISOString(),
          worker_id:        env.WORKER_ID ?? "scraper-text-01",
          status:           "staged" as const,
        }));

      let pushed = 0;
      for (let i = 0; i < enriched.length; i += 100) {
        const { error } = await d.from("samples_staging").insert(enriched.slice(i, i+100));
        if (!error) pushed += Math.min(100, enriched.length - i);
        else logger.log("INSERT_ERROR", { source_id: source.source_id, error_message: JSON.stringify(error) });
      }

      logger.log("SCRAPE_COMPLETE", { source_id: source.source_id, sample_count: pushed, duration_ms: Date.now()-t0 });
      await logger.flush();
      return new Response(JSON.stringify({ samples_scraped: pushed }), { status: 200, headers: { "Content-Type": "application/json" } });
    } catch (err: any) {
      const msg = err?.message ?? String(err);
      logger.log("SCRAPE_ERROR", { source_id: source.source_id, error_message: msg });
      await logger.flush();
      return new Response(JSON.stringify({ error: msg }), { status: 500 });
    }
  }
};
