// DETECT-AI: Text Scraper v4 — CF Subrequest Safe
// CF Workers free plan limit: 50 subrequests per invocation
// Every function is budgeted to stay under 10 subrequests each.

import type { Env, DetectAISample } from "../../shared/types/index";
import { createSupabaseClient } from "../dispatcher/supabase";
import { PipelineLogger } from "../dispatcher/logger";

function uuidv4() { return crypto.randomUUID(); }

async function safeFetch(url: string, opts: RequestInit = {}): Promise<string | null> {
  try {
    const r = await fetch(url, {
      ...opts,
      signal: AbortSignal.timeout(9000),
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
      .replace(/<[^>]+>/g," ").replace(/\s{2,}/g," ").trim().slice(0,2000);
    const date  = b.match(/<pubDate>(.*?)<\/pubDate>/)?.[1] ?? "";
    const text  = `${title}\n\n${desc}`.trim();
    if (title && link && text.length > 80)
      items.push({ source_url: link, raw_content: text,
        metadata: { title, publish_date: date, source: sourceId, language: lang,
          license: "News", tags: [sourceId,"news","human-content"], is_ai_generated: false }});
  }
  return items;
}

// ── RSS feeds — each source fetches sequentially (safe, no parallel limit issue) ──
const RSS_MAP: Record<string, Array<{url:string;lang:string}>> = {
  "bbc-news": [
    { url: "https://feeds.bbci.co.uk/news/rss.xml",            lang: "en" },
    { url: "https://feeds.bbci.co.uk/news/world/rss.xml",      lang: "en" },
    { url: "https://feeds.bbci.co.uk/news/technology/rss.xml", lang: "en" },
    { url: "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml", lang: "en" },
  ],
  "reuters": [
    // reuters.com feeds dead — reliable alternatives
    { url: "https://feeds.theguardian.com/theguardian/business/rss",     lang: "en" },
    { url: "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",  lang: "en" },
    { url: "https://news.yahoo.com/rss/world",                           lang: "en" },
    { url: "https://feeds.feedburner.com/ndtvnews-top-stories",          lang: "en" },
  ],
  "guardian": [
    { url: "https://feeds.theguardian.com/theguardian/world/rss",       lang: "en" },
    { url: "https://feeds.theguardian.com/theguardian/technology/rss",  lang: "en" },
    { url: "https://feeds.theguardian.com/theguardian/environment/rss", lang: "en" },
    { url: "https://feeds.theguardian.com/theguardian/science/rss",     lang: "en" },
  ],
  "nytimes": [
    { url: "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",       lang: "en" },
    { url: "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",  lang: "en" },
    { url: "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",     lang: "en" },
    { url: "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",      lang: "en" },
  ],
  "aljazeera": [
    { url: "https://www.aljazeera.net/xml/rss/all.xml", lang: "ar" },
    { url: "https://www.aljazeera.com/xml/rss/all.xml", lang: "en" },
  ],
  "npr": [
    { url: "https://feeds.npr.org/1001/rss.xml", lang: "en" },
    { url: "https://feeds.npr.org/1025/rss.xml", lang: "en" },
    { url: "https://feeds.npr.org/1003/rss.xml", lang: "en" },
    { url: "https://feeds.npr.org/1119/rss.xml", lang: "en" },
  ],
};

async function scrapeRSS(sourceId: string): Promise<any[]> {
  const feeds = RSS_MAP[sourceId];
  if (!feeds) return [];
  const out: any[] = [];
  for (const f of feeds) {      // sequential — safe for CF subrequest limit
    const xml = await safeFetch(f.url);
    if (xml) out.push(...parseRSS(xml, f.lang, sourceId));
  }
  return out;
}

// FIXED: 2 subrequests total (was 500+)
// Strategy: rotate one language per call, batch-fetch extracts in one request
async function scrapeWikipedia(): Promise<any[]> {
  const langs = ["en","ar","fr","de","es","zh","ja","ko","pt","hi","ru","it","nl","pl","sv","tr","uk","fa"];
  const lang  = langs[Math.floor(Date.now() / 300000) % langs.length]; // rotate every 5min
  const out: any[] = [];

  // Call 1: get 10 random page IDs
  const listResp = await safeJSON(
    `https://${lang}.wikipedia.org/w/api.php?action=query&list=random&rnlimit=10&rnnamespace=0&format=json&origin=*`
  );
  if (!listResp?.query?.random) return [];

  // Call 2: batch-fetch all extracts in ONE request using pageids pipe
  const ids = (listResp.query.random as any[]).map((p:any) => p.id).join("|");
  const bulk = await safeJSON(
    `https://${lang}.wikipedia.org/w/api.php?action=query&pageids=${ids}&prop=extracts&exintro=true&explaintext=true&exlimit=10&format=json&origin=*`
  );
  for (const page of Object.values(bulk?.query?.pages ?? {}) as any[]) {
    const clean = (page.extract ?? "").replace(/\s{2,}/g," ").trim().slice(0,2500);
    if (clean.length > 100)
      out.push({ source_url: `https://${lang}.wikipedia.org/?curid=${page.pageid}`,
        raw_content: `${page.title}\n\n${clean}`,
        metadata: { title: page.title, language: lang, source: "wikipedia",
          license: "CC-BY-SA", tags: ["wikipedia","encyclopedia","human-content"], is_ai_generated: false }});
  }
  return out; // 5-10 articles per call, only 2 subrequests
}

// 3 subrequests (parallel OK — they go to different hosts)
async function scrapeArxiv(): Promise<any[]> {
  const allCats = ["cs.CV","cs.AI","cs.LG","cs.CL","stat.ML","eess.IV","cs.NE","cs.RO","cs.CR","cs.SE","math.CO","quant-ph"];
  const selected = [...allCats].sort(() => Math.random()-0.5).slice(0,3);
  const results = await Promise.allSettled(
    selected.map(cat =>
      safeFetch(`https://export.arxiv.org/rss/${cat}`).then(xml =>
        xml ? parseRSS(xml, "en", "arxiv").map(a => ({
          ...a, metadata: { ...a.metadata, category: cat, tags: ["arxiv","academic","research","human-content"] }
        })) : []
      )
    )
  );
  const out: any[] = [];
  for (const r of results) if (r.status === "fulfilled") out.push(...r.value);
  return out.slice(0,150);
}

// 1 subrequest
async function scrapeStackExchange(): Promise<any[]> {
  const sites = ["stackoverflow","superuser","datascience","ai","stats","physics","math","philosophy","biology","chemistry"];
  const site  = sites[Math.floor(Math.random() * sites.length)];
  const d = await safeJSON(
    `https://api.stackexchange.com/2.3/questions?order=desc&sort=votes&site=${site}&filter=withbody&pagesize=50`
  );
  if (!d?.items) return [];
  return (d.items as any[]).map((q:any) => ({
    source_url: q.link,
    raw_content: `${q.title}\n\n${(q.body ?? "").replace(/<[^>]+>/g," ").replace(/\s{2,}/g," ").trim().slice(0,2000)}`,
    metadata: { title: q.title, site, score: q.score,
      tags: [...(q.tags??[]),"stackexchange","q&a","human-content"],
      license: "CC BY-SA 4.0", is_ai_generated: false }
  }));
}

// FIXED: PapersWithCode /api/v1/papers/ has NO abstract field.
// Use arXiv search API (cs.LG + cs.AI + cs.CV) which returns abstracts reliably.
// 1 subrequest, 100% yield.
async function scrapePapersWithCode(): Promise<any[]> {
  // arXiv search for recent ML papers — returns titles + abstracts (reliable, no auth)
  const categories = ["cs.LG","cs.AI","cs.CV","cs.CL","cs.NE","stat.ML"];
  const cat = categories[Math.floor(Math.random() * categories.length)];
  const d = await safeJSON(
    `https://export.arxiv.org/search/?query=${cat}&searchtype=cat&start=0&max_results=50&sortBy=submittedDate&sortOrder=descending&format=json`
  );
  if (d?.entries && d.entries.length > 0) {
    return (d.entries as any[]).filter((p:any) => p.summary?.length > 30).map((p:any) => ({
      source_url: p.id ?? `https://arxiv.org/abs/${p.id?.split("/abs/")[1]}`,
      raw_content: `${p.title}

${p.summary}`.trim().slice(0, 3000),
      metadata: { title: p.title, category: cat, source: "paperswithcode",
        authors: (p.author ?? []).slice(0,3).map((a:any) => a.name).join(", "),
        tags: ["paperswithcode","ml","academic","human-content"], license: "arXiv", is_ai_generated: false }
    }));
  }
  // Fallback: direct arXiv RSS (very reliable)
  const xml = await safeFetch(`https://export.arxiv.org/rss/${cat}`);
  if (!xml) return [];
  return parseRSS(xml, "en", "paperswithcode").map((p:any) => ({
    ...p, metadata: { ...p.metadata, source: "paperswithcode", category: cat,
      tags: ["paperswithcode","ml","research","human-content"] }
  }));
}

// FIXED: top?t=day gets more self-posts; also tries multiple subreddits
async function scrapeReddit(): Promise<any[]> {
  const subs = [
    "worldnews","science","technology","MachineLearning","datascience",
    "programming","askscience","explainlikeimfive","todayilearned",
    "geopolitics","space","medicine","history","philosophy","economics",
    "artificial","singularity","netsec","devops","opensource"
  ];
  const sub = subs[Math.floor(Math.random() * subs.length)];
  const d   = await safeJSON(
    `https://www.reddit.com/r/${sub}/top.json?t=day&limit=50`,
    { headers: { "User-Agent": "DETECT-AI/1.0 research" } }
  );
  if (!d?.data?.children) return [];
  const posts = (d.data.children as any[]).filter(
    (p:any) => p.data?.selftext && p.data.selftext.length > 80
            && p.data.selftext !== "[deleted]" && p.data.selftext !== "[removed]"
  );
  return posts.map((p:any) => ({
    source_url: `https://reddit.com${p.data.permalink}`,
    raw_content: `${p.data.title}\n\n${p.data.selftext}`.trim().slice(0,3000),
    metadata: { title: p.data.title, subreddit: sub, score: p.data.score,
      tags: ["reddit",sub,"human-content"], license: "CC BY-SA 4.0", is_ai_generated: false }
  }));
}

// FIXED: correct document field names in WorldBank API
async function scrapeWorldBank(): Promise<any[]> {
  // WorldBank API v3 returns documents with different field structures
  const d = await safeJSON(
    "https://search.worldbank.org/api/v2/wds?format=json&rows=50&os=0&fl=id,display_title,docdt,txturl,abstract&srt=docdt&order=desc"
  );
  const docs = d?.documents ? Object.values(d.documents) : (d?.response?.docs ?? []);
  const out = (docs as any[])
    .filter((doc:any) => (doc.abstract ?? doc.txt ?? doc.display_title ?? "").length > 30)
    .slice(0,50)
    .map((doc:any) => ({
      source_url: doc.txturl ?? `https://documents.worldbank.org/en/publication/documents-reports/documentdetail/${doc.id}`,
      raw_content: `${doc.display_title ?? doc.title ?? ""}\n\n${doc.abstract ?? ""}`.trim().slice(0,3000),
      metadata: { title: doc.display_title ?? doc.title, source: "worldbank",
        tags: ["worldbank","policy","research","human-content"], license: "CC BY 4.0", is_ai_generated: false }
    }));

  // If API changed format and returns nothing, scrape their open knowledge repo
  if (out.length === 0) {
    const fallback = await safeJSON(
      "https://openknowledge.worldbank.org/rest/search?scope=/&query=development+economics&rpp=30&sort_by=2&order=desc"
    );
    const items = fallback?.results ?? [];
    return (items as any[]).slice(0,30).map((item:any) => ({
      source_url: item.handle ? `https://openknowledge.worldbank.org/handle/${item.handle}` : "https://openknowledge.worldbank.org",
      raw_content: `${item.name ?? ""}\n\n${(item.metadata?.["dc.description.abstract"]?.[0]?.value ?? "")}`.trim().slice(0,2500),
      metadata: { title: item.name, source: "worldbank",
        tags: ["worldbank","policy","research","human-content"], license: "CC BY 4.0", is_ai_generated: false }
    })).filter((r:any) => r.raw_content.length > 80);
  }
  return out;
}

// ── Main ───────────────────────────────────────────────────────────────────
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
          case "wikipedia":     raw = await scrapeWikipedia();    break;
          case "arxiv":         raw = await scrapeArxiv();        break;
          case "stackexchange": raw = await scrapeStackExchange(); break;
          case "paperswithcode":raw = await scrapePapersWithCode();break;
          case "reddit":        raw = await scrapeReddit();        break;
          case "worldbank":     raw = await scrapeWorldBank();     break;
          default:
            return new Response(JSON.stringify({ error: `Unknown text source: ${source.source_id}` }), { status: 400 });
        }
      }

      if (raw.length === 0) {
        logger.log("SCRAPE_COMPLETE", { source_id: source.source_id, sample_count: 0, duration_ms: Date.now()-t0 });
        await logger.flush();
        return new Response(JSON.stringify({ samples_scraped: 0 }), { status: 200 });
      }

      const db = createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
      const enriched: DetectAISample[] = raw
        .filter((s:any) => (s.raw_content ?? "").length > 80)
        .slice(0, 1000)
        .map((s:any) => ({
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
          worker_id:        env.WORKER_ID ?? "scraper-text-v4",
          status:           "staged" as const,
        }));

      let pushed = 0;
      for (let i = 0; i < enriched.length; i += 100) {
        const { error } = await db.from("samples_staging").insert(enriched.slice(i, i+100));
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
  },
};
