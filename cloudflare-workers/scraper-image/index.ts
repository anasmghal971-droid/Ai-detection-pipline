// DETECT-AI Image Scraper — COMPLETE REWRITE
// TWO categories: REAL (HUMAN label) + AI-GENERATED (AI_GENERATED label)
// AI sources: Lexica, Civitai, Pollinations, DiffusionDB
// Real sources: Unsplash, Pexels, Pixabay, Openverse, NASA, Met, Wikimedia, LAION

import type { Env, DetectAISample } from "../../shared/types/index";
import { createSupabaseClient } from "../dispatcher/supabase";
import { PipelineLogger } from "../dispatcher/logger";

function uuidv4(): string { return crypto.randomUUID(); }

async function fetchJ(url: string, opts: RequestInit = {}): Promise<any> {
  for (let i = 1; i <= 3; i++) {
    try {
      const r = await fetch(url, { ...opts, signal: AbortSignal.timeout(12000) });
      if (r.status === 429) throw Object.assign(new Error("Rate limited"), { rateLimited: true });
      if (r.ok) return await r.json();
      if (i === 3) return null;
      await new Promise(x => setTimeout(x, 1000 * i));
    } catch (e: any) {
      if (e.rateLimited) throw e;
      if (i === 3) return null;
      await new Promise(x => setTimeout(x, 1000 * i));
    }
  }
  return null;
}

// ── REAL PHOTOS ───────────────────────────────────────────────
async function scrapeUnsplash(env: Env) {
  const d = await fetchJ("https://api.unsplash.com/photos/random?count=30&content_filter=high", { headers: { Authorization: `Client-ID ${env.UNSPLASH_ACCESS_KEY}` } });
  if (!d) return [];
  return d.map((p: any) => ({ source_url: p.links.html, raw_content: p.urls.full, metadata: { title: p.description ?? p.alt_description, author: p.user?.name, license: "Unsplash", tags: ["unsplash","real-photo","human"], is_ai_generated: false } }));
}

async function scrapePexels(env: Env) {
  const d = await fetchJ("https://api.pexels.com/v1/curated?per_page=80", { headers: { Authorization: env.PEXELS_API_KEY } });
  if (!d?.photos) return [];
  return d.photos.map((p: any) => ({ source_url: p.url, raw_content: p.src.original, metadata: { title: p.alt ?? `Pexels ${p.id}`, author: p.photographer, license: "Pexels", tags: ["pexels","real-photo","cc0","human"], is_ai_generated: false } }));
}

async function scrapePixabay(env: Env) {
  const out: any[] = [];
  for (const lang of ["en","de","es"]) {
    const d = await fetchJ(`https://pixabay.com/api/?key=${env.PIXABAY_API_KEY}&image_type=photo&per_page=50&lang=${lang}&safesearch=true`);
    if (d?.hits) for (const h of d.hits) out.push({ source_url: h.pageURL, raw_content: h.largeImageURL, metadata: { title: `Pixabay ${h.id}`, author: h.user, license: "Pixabay", tags: ["pixabay","cc0","real-photo","human"], is_ai_generated: false } });
  }
  return out;
}

async function scrapeOpenverse() {
  const topics = ["nature","people","city","animals","food","architecture"];
  const q = topics[Math.floor(Math.random() * topics.length)];
  const d = await fetchJ(`https://api.openverse.org/v1/images/?q=${q}&license_type=commercial,modification&page_size=40`, { headers: { "User-Agent": "DETECT-AI/1.0" } });
  if (!d?.results) return [];
  return d.results.map((img: any) => ({ source_url: img.foreign_landing_url, raw_content: img.url, metadata: { title: img.title, author: img.creator, license: img.license, tags: ["openverse","cc","real-photo","human"], is_ai_generated: false } }));
}

async function scrapeNASA() {
  const qs = ["earth","space","galaxy","astronaut","moon","mars","nebula","satellite"];
  const q = qs[Math.floor(Math.random() * qs.length)];
  const d = await fetchJ(`https://images-api.nasa.gov/search?q=${q}&media_type=image&page_size=20`);
  if (!d?.collection?.items) return [];
  return d.collection.items.filter((x: any) => x.links?.some((l: any) => l.rel === "preview")).map((item: any) => {
    const m = item.data[0] ?? {};
    return { source_url: `https://images.nasa.gov/details/${m.nasa_id}`, raw_content: item.links.find((l: any) => l.rel === "preview").href, metadata: { title: m.title, license: "Public Domain (US Gov)", tags: ["nasa","space","real-photo","human"], is_ai_generated: false } };
  });
}

async function scrapeMetMuseum() {
  const deptId = [11,12,13,14,15,16,17,18,19,21][Math.floor(Math.random() * 10)];
  const list = await fetchJ(`https://collectionapi.metmuseum.org/public/collection/v1/objects?departmentIds=${deptId}&isHighlight=true`);
  const ids = (list?.objectIDs ?? []).slice(0, 20);
  const out: any[] = [];
  await Promise.allSettled(ids.slice(0, 15).map(async (id: number) => {
    const obj = await fetchJ(`https://collectionapi.metmuseum.org/public/collection/v1/objects/${id}`);
    if (obj?.isPublicDomain && obj?.primaryImage) out.push({ source_url: obj.objectURL ?? `https://www.metmuseum.org/art/collection/search/${id}`, raw_content: obj.primaryImage, metadata: { title: obj.title, artist: obj.artistDisplayName, license: "CC0 1.0", tags: ["met-museum","art","real-photo","human"], is_ai_generated: false } });
  }));
  return out;
}

async function scrapeWikimedia() {
  const d = await fetchJ("https://commons.wikimedia.org/w/api.php?action=query&list=random&rnlimit=30&rnnamespace=6&format=json");
  const out: any[] = [];
  await Promise.allSettled((d?.query?.random ?? []).slice(0, 15).map(async (f: any) => {
    const info = await fetchJ(`https://commons.wikimedia.org/w/api.php?action=query&prop=imageinfo&iiprop=url|mime|extmetadata&pageids=${f.id}&format=json`);
    const page = Object.values((info?.query?.pages ?? {}) as Record<string, any>)[0];
    const i = page?.imageinfo?.[0];
    if (i?.mime?.startsWith("image/")) out.push({ source_url: `https://commons.wikimedia.org/?curid=${f.id}`, raw_content: i.url, metadata: { title: f.title, license: i.extmetadata?.License?.value ?? "Wikimedia-CC", tags: ["wikimedia","cc","real-photo","human"], is_ai_generated: false } });
  }));
  return out;
}

// ── AI-GENERATED IMAGES ───────────────────────────────────────

// Lexica.art: 10M+ Stable Diffusion images — NO API KEY
async function scrapeLexica() {
  const prompts = ["portrait photo realistic person","landscape photography natural","street photography candid","product photography studio","wildlife photography animal","architectural photography building","food photography restaurant"];
  const q = prompts[Math.floor(Math.random() * prompts.length)];
  const d = await fetchJ(`https://lexica.art/api/v1/search?q=${encodeURIComponent(q)}&n=50`, { headers: { "User-Agent": "DETECT-AI/1.0" } });
  if (!d?.images) return [];
  return d.images.filter((img: any) => !img.nsfw && img.src).slice(0, 30).map((img: any) => ({
    source_url: `https://lexica.art/prompt/${img.id}`, raw_content: img.src,
    metadata: { prompt: img.prompt?.slice(0, 300), model: img.model ?? "stable-diffusion", dimensions: { width: img.width, height: img.height }, license: "Lexica Public", tags: ["lexica","stable-diffusion","ai-generated","synthetic"], is_ai_generated: true, generation_source: "Lexica.art / Stable Diffusion" }
  }));
}

// Civitai: massive SD/SDXL community gallery — NO API KEY
async function scrapeCivitai() {
  const d = await fetchJ("https://civitai.com/api/v1/images?limit=50&sort=Newest&nsfw=false&period=Day", { headers: { "User-Agent": "DETECT-AI/1.0 (research)" } });
  if (!d?.items) return [];
  return d.items.filter((img: any) => img.nsfw === "None" && img.url).slice(0, 30).map((img: any) => ({
    source_url: `https://civitai.com/images/${img.id}`, raw_content: img.url,
    metadata: { prompt: img.meta?.prompt?.slice(0, 300), model: img.meta?.Model ?? "stable-diffusion", sampler: img.meta?.sampler, steps: img.meta?.steps, dimensions: { width: img.width, height: img.height }, license: "Civitai Public", tags: ["civitai","ai-generated","sdxl","synthetic"], is_ai_generated: true, generation_source: "Civitai / Stable Diffusion / SDXL" }
  }));
}

// Pollinations.ai: FLUX/SDXL generation — NO API KEY, always new unique images
async function scrapePollinationsAI() {
  const prompts = ["professional portrait photo person 8k photorealistic","landscape photography golden hour photorealistic","candid street photography urban","studio product photography white background","wildlife animal in natural habitat detailed photograph","modern building architectural photography exterior","food photography restaurant table detailed"];
  const out: any[] = [];
  for (const prompt of prompts.slice(0, 5)) {
    const imageUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}?width=1024&height=1024&model=flux&seed=${Math.floor(Math.random()*999999)}&nologo=true`;
    out.push({ source_url: "https://pollinations.ai", raw_content: imageUrl, metadata: { prompt, model: "FLUX", width: 1024, height: 1024, tags: ["pollinations","flux","ai-generated","synthetic"], is_ai_generated: true, generation_source: "Pollinations.ai / FLUX model" } });
  }
  return out;
}

// DiffusionDB: 14M labeled SD images via HF Datasets — NO API KEY
async function scrapeDiffusionDB() {
  const offset = Math.floor(Math.random() * 50000);
  const d = await fetchJ(`https://datasets-server.huggingface.co/rows?dataset=poloclub/diffusiondb&config=2m_first_1k&split=train&offset=${offset}&length=30`);
  if (!d?.rows) return [];
  return d.rows.filter((r: any) => r.row?.image?.src).map((r: any) => ({
    source_url: "https://huggingface.co/datasets/poloclub/diffusiondb", raw_content: r.row.image.src,
    metadata: { prompt: r.row.prompt?.slice(0, 300), seed: r.row.seed, steps: r.row.step, sampler: r.row.sampler_name, model: "stable-diffusion", license: "CC BY 4.0", tags: ["diffusiondb","stable-diffusion","ai-generated","synthetic"], is_ai_generated: true, generation_source: "DiffusionDB / Stable Diffusion" }
  }));
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
      let label: "HUMAN" | "AI_GENERATED" = "HUMAN";
      let confidence = 0.95;

      switch (source.source_id) {
        case "unsplash":     raw = await scrapeUnsplash(env);       label = "HUMAN"; break;
        case "pexels":       raw = await scrapePexels(env);         label = "HUMAN"; break;
        case "pixabay":      raw = await scrapePixabay(env);        label = "HUMAN"; break;
        case "openverse":    raw = await scrapeOpenverse();         label = "HUMAN"; break;
        case "nasa":         raw = await scrapeNASA();              label = "HUMAN"; break;
        case "met-museum":   raw = await scrapeMetMuseum();         label = "HUMAN"; break;
        case "wikimedia":    raw = await scrapeWikimedia();         label = "HUMAN"; break;
        case "lexica":       raw = await scrapeLexica();            label = "AI_GENERATED"; confidence = 0.99; break;
        case "civitai":      raw = await scrapeCivitai();           label = "AI_GENERATED"; confidence = 0.99; break;
        case "pollinations": raw = await scrapePollinationsAI();    label = "AI_GENERATED"; confidence = 0.99; break;
        case "diffusiondb":  raw = await scrapeDiffusionDB();       label = "AI_GENERATED"; confidence = 0.99; break;
        default:
          return new Response(JSON.stringify({ error: `Unknown source: ${source.source_id}` }), { status: 400 });
      }

      const enriched: DetectAISample[] = raw
        .filter((s: any) => s?.raw_content && s.raw_content.length > 5)
        .slice(0, 300)
        .map((s: any) => ({
          sample_id: uuidv4(), source_id: source.source_id, source_url: s.source_url ?? source.source_url,
          content_type: "image" as const, language: "en", raw_content: s.raw_content,
          label, final_confidence: confidence, verified: label === "AI_GENERATED",
          metadata: s.metadata ?? {}, scraped_at: new Date().toISOString(),
          worker_id: env.WORKER_ID ?? "scraper-image-01", status: "staged" as const,
        }));

      const db = createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);
      let pushed = 0;
      for (let i = 0; i < enriched.length; i += 100) {
        const { error } = await db.from("samples_staging").insert(enriched.slice(i, i + 100));
        if (!error) pushed += Math.min(100, enriched.length - i);
        else logger.log("INSERT_ERROR", { source_id: source.source_id, error_message: JSON.stringify(error) });
      }

      logger.log("SCRAPE_COMPLETE", { source_id: source.source_id, sample_count: pushed, duration_ms: Date.now() - t0 });
      await logger.flush();
      return new Response(JSON.stringify({ samples_scraped: pushed, label, duration_ms: Date.now() - t0 }), { status: 200, headers: { "Content-Type": "application/json" } });
    } catch (err: any) {
      const msg = err?.message ?? String(err);
      logger.log("SCRAPE_ERROR", { source_id: source.source_id, error_message: msg });
      await logger.flush();
      if (err.rateLimited) return new Response(JSON.stringify({ error: "rate_limited" }), { status: 429 });
      return new Response(JSON.stringify({ error: msg }), { status: 500 });
    }
  },
};
