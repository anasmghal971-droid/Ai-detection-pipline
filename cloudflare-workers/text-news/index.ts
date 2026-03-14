// detect-ai-text-news worker
// Scrapes news in 5 languages: EN, AR, FR, DE, ES
// Sources: BBC, Reuters, AlJazeera, Guardian, AP, DW, Le Monde, El Pais
// Target: ~1100 samples/cycle (every 5 min)

import type { Env, DetectAISample } from "../../shared/types/index";
import { createSupabaseClient } from "../dispatcher/supabase";

function uuidv4() { return crypto.randomUUID(); }

async function fetchRSS(url: string): Promise<Array<{title:string; link:string; description:string; pubDate:string; lang:string}>> {
  const resp = await fetch(url, { headers: { "User-Agent": "DETECT-AI/1.0 dataset-builder" } });
  if (!resp.ok) return [];
  const xml = await resp.text();
  const items: Array<{title:string; link:string; description:string; pubDate:string; lang:string}> = [];
  const itemMatches = xml.matchAll(/<item>([\s\S]*?)<\/item>/g);
  for (const m of itemMatches) {
    const block = m[1];
    const title       = block.match(/<title><!\[CDATA\[(.*?)\]\]><\/title>|<title>(.*?)<\/title>/)?.[1] ?? "";
    const link        = block.match(/<link>(.*?)<\/link>/)?.[1]?.trim() ?? "";
    const description = block.match(/<description><!\[CDATA\[(.*?)\]\]><\/description>|<description>(.*?)<\/description>/)?.[1]?.replace(/<[^>]+>/g,"") ?? "";
    const pubDate     = block.match(/<pubDate>(.*?)<\/pubDate>/)?.[1] ?? "";
    if (title && link) items.push({ title, link, description: description.slice(0,1000), pubDate, lang: "en" });
  }
  return items;
}

// ── NEWS SOURCES by language ──────────────────────────────────────
const FEEDS: Array<{url:string; source:string; lang:string}> = [
  // English
  { url:"https://feeds.bbci.co.uk/news/rss.xml",                         source:"bbc",       lang:"en" },
  { url:"https://feeds.bbci.co.uk/news/world/rss.xml",                   source:"bbc",       lang:"en" },
  { url:"https://feeds.reuters.com/reuters/topNews",                     source:"reuters",   lang:"en" },
  { url:"https://feeds.reuters.com/reuters/worldNews",                   source:"reuters",   lang:"en" },
  { url:"https://feeds.theguardian.com/theguardian/world/rss",           source:"guardian",  lang:"en" },
  { url:"https://feeds.theguardian.com/theguardian/technology/rss",      source:"guardian",  lang:"en" },
  { url:"https://rss.nytimes.com/services/xml/rss/nyt/World.xml",        source:"nytimes",   lang:"en" },
  { url:"https://feeds.washingtonpost.com/rss/world",                    source:"washpost",  lang:"en" },
  // Arabic
  { url:"https://www.aljazeera.net/xml/rss/all.xml",                     source:"aljazeera", lang:"ar" },
  { url:"https://arabic.rt.com/rss/",                                    source:"rt-arabic", lang:"ar" },
  { url:"https://www.alarabiya.net/rss.xml",                             source:"alarabiya", lang:"ar" },
  // French
  { url:"https://www.lemonde.fr/rss/une.xml",                            source:"lemonde",   lang:"fr" },
  { url:"https://rss.rfi.fr/rfi/fr/podcasts/63/rss.xml",                 source:"rfi",       lang:"fr" },
  { url:"https://france24.com/fr/rss",                                   source:"france24",  lang:"fr" },
  // German
  { url:"https://www.spiegel.de/schlagzeilen/tops/index.rss",            source:"spiegel",   lang:"de" },
  { url:"https://rss.dw.com/rdf/rss-de-all",                             source:"dw",        lang:"de" },
  { url:"https://www.zeit.de/news/index?output=rss",                     source:"zeit",      lang:"de" },
  // Spanish
  { url:"https://ep00.epimg.net/rss/elpais/portada.xml",                 source:"elpais",    lang:"es" },
  { url:"https://www.bbc.com/mundo/rss.xml",                             source:"bbc-es",    lang:"es" },
  { url:"https://rss.univision.com/noticias/es/rss",                     source:"univision", lang:"es" },
];

// ── Wikipedia multi-language random articles ──────────────────────
const WIKI_LANGS = ["en","ar","fr","de","es","zh","ja","ko","pt","ru","hi","it","nl","pl","sv","tr","vi"];

async function scrapeWikiMulti(maxPerLang = 15): Promise<Partial<DetectAISample>[]> {
  const samples: Partial<DetectAISample>[] = [];
  for (const lang of WIKI_LANGS.slice(0, 8)) {  // 8 langs per cycle
    try {
      const url = `https://${lang}.wikipedia.org/w/api.php?action=query&list=random&rnlimit=${maxPerLang}&rnnamespace=0&prop=extracts&exintro=true&format=json&origin=*`;
      const r = await fetch(url);
      if (!r.ok) continue;
      const d: { query: { random: Array<{id:number; title:string}> } } = await r.json();
      for (const page of d.query.random) {
        // Get extract
        const er = await fetch(
          `https://${lang}.wikipedia.org/w/api.php?action=query&pageids=${page.id}&prop=extracts&exintro=true&format=json&origin=*`
        );
        if (!er.ok) continue;
        const ed: { query: { pages: Record<string, { extract?: string }> } } = await er.json();
        const extract = Object.values(ed.query.pages)[0]?.extract ?? "";
        const clean = extract.replace(/<[^>]+>/g, " ").replace(/\s{2,}/g, " ").trim().slice(0, 2000);
        if (clean.length < 100) continue;
        samples.push({
          source_url:  `https://${lang}.wikipedia.org/?curid=${page.id}`,
          raw_content: clean,
          metadata: {
            title: page.title, source: `wikipedia-${lang}`,
            language: lang, license: "CC-BY-SA",
          },
        });
      }
    } catch { continue; }
  }
  return samples;
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const t0 = Date.now();
    const db = createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
    const samples: DetectAISample[] = [];

    // ── Scrape all news RSS feeds in parallel ─────────────────────
    const feedResults = await Promise.allSettled(
      FEEDS.map(feed => fetchRSS(feed.url).then(items => ({ items, feed })))
    );

    for (const result of feedResults) {
      if (result.status !== "fulfilled") continue;
      const { items, feed } = result.value;
      for (const item of items.slice(0, 20)) {
        const text = `${item.title}\n\n${item.description}`.trim();
        if (text.length < 50) continue;
        samples.push({
          sample_id:    uuidv4(),
          source_id:    feed.source,
          source_url:   item.link,
          content_type: "text",
          language:     feed.lang,
          raw_content:  text,
          metadata: {
            title:        item.title,
            publish_date: item.pubDate,
            source:       feed.source,
            language:     feed.lang,
          },
          scraped_at: new Date().toISOString(),
          worker_id:  env.WORKER_ID ?? "text-news-01",
          status:     "staged",
        });
      }
    }

    // ── Wikipedia multi-language ──────────────────────────────────
    const wikiSamples = await scrapeWikiMulti(12);
    for (const s of wikiSamples) {
      samples.push({
        sample_id:    uuidv4(),
        source_id:    `wikipedia`,
        source_url:   s.source_url!,
        content_type: "text",
        language:     (s.metadata as {language?:string})?.language ?? "en",
        raw_content:  s.raw_content!,
        metadata:     s.metadata ?? {},
        scraped_at:   new Date().toISOString(),
        worker_id:    env.WORKER_ID ?? "text-news-01",
        status:       "staged",
      });
    }

    // ── Push to Supabase in 100-row batches ───────────────────────
    const BATCH = 100;
    let pushed = 0;
    for (let i = 0; i < samples.length; i += BATCH) {
      const { error } = await db.from("samples_staging").insert(samples.slice(i, i + BATCH));
      if (!error) pushed += Math.min(BATCH, samples.length - i);
    }

    return new Response(JSON.stringify({
      samples_scraped: samples.length,
      pushed, duration_ms: Date.now() - t0
    }), { status: 200, headers: { "Content-Type": "application/json" } });
  }
};
