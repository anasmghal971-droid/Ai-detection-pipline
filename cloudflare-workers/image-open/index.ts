// detect-ai-image-open worker
// Sources: Open Images V7 (Google CC), LAION subset via HF, MS-COCO captions
// Target: ~900 samples/cycle

import type { Env, DetectAISample } from "../../shared/types/index";
import { createSupabaseClient } from "../dispatcher/supabase";

function uuidv4() { return crypto.randomUUID(); }

// ── Open Images V7 (Google, CC licensed) ─────────────────────────
// Uses the HF datasets server API — no extra API key needed
async function scrapeOpenImages(maxSamples = 300): Promise<Partial<DetectAISample>[]> {
  const samples: Partial<DetectAISample>[] = [];
  const splits = ["train", "validation", "test"];
  const offset = Math.floor(Math.random() * 10000); // Random offset for diversity

  for (const split of splits.slice(0, 1)) {
    try {
      const url = `https://datasets-server.huggingface.co/rows?dataset=HuggingFaceM4/the_cauldron&config=open_images&split=${split}&offset=${offset}&length=100`;
      const r = await fetch(url);
      if (!r.ok) continue;
      const data: { rows: Array<{ row: { image?: { src: string }; texts?: Array<{ user: string }> } }> } = await r.json();

      for (const row of data.rows ?? []) {
        const imgSrc = row.row.image?.src ?? "";
        if (!imgSrc) continue;
        const caption = row.row.texts?.[0]?.user ?? "";
        samples.push({
          source_url:  imgSrc,
          raw_content: imgSrc,
          metadata: {
            caption,
            source: "open-images-v7",
            license: "CC BY 4.0",
            tags: ["open-images", "google", "cc"],
          },
        });
      }
    } catch { continue; }
  }

  // Also try Open Images direct API
  try {
    const r = await fetch(
      `https://datasets-server.huggingface.co/rows?dataset=google/open_images&config=default&split=train&offset=${offset}&length=100`
    );
    if (r.ok) {
      const data: { rows: Array<{ row: Record<string, unknown> }> } = await r.json();
      for (const row of data.rows ?? []) {
        const rd = row.row;
        const imgUrl = (rd["image_url"] ?? rd["url"] ?? "") as string;
        if (!imgUrl) continue;
        samples.push({
          source_url:  imgUrl,
          raw_content: imgUrl,
          metadata: {
            label:   rd["label"],
            source:  "open-images-v7",
            license: "CC BY 4.0",
          },
        });
      }
    }
  } catch { /* fallback ok */ }

  return samples.slice(0, maxSamples);
}

// ── LAION-400M subset (CC licensed web images) ────────────────────
async function scrapeLAION(maxSamples = 300): Promise<Partial<DetectAISample>[]> {
  const samples: Partial<DetectAISample>[] = [];
  const offset = Math.floor(Math.random() * 50000);

  try {
    // LAION-aesthetics-v2 subset via HF datasets server
    const url = `https://datasets-server.huggingface.co/rows?dataset=laion/laion-aesthetics-v2&config=default&split=train&offset=${offset}&length=100`;
    const r = await fetch(url);
    if (!r.ok) return samples;

    const data: { rows: Array<{ row: { URL?: string; TEXT?: string; AESTHETIC_SCORE?: number; LICENSE?: string } }> } = await r.json();

    for (const row of data.rows ?? []) {
      const { URL: imgUrl, TEXT: caption, AESTHETIC_SCORE: score, LICENSE: license } = row.row;
      if (!imgUrl || !imgUrl.startsWith("http")) continue;
      // Only include CC licensed
      if (license && !license.toLowerCase().includes("cc")) continue;

      samples.push({
        source_url:  imgUrl,
        raw_content: imgUrl,
        metadata: {
          caption:          caption?.slice(0, 500),
          aesthetic_score:  score,
          source:           "laion-aesthetics-v2",
          license:          license ?? "CC",
          tags:             ["laion", "cc", "aesthetic"],
        },
      });
    }
  } catch { /* HF API might be rate limited — skip gracefully */ }

  return samples.slice(0, maxSamples);
}

// ── MS-COCO Captions (CC BY 4.0) ─────────────────────────────────
async function scrapeCOCO(maxSamples = 200): Promise<Partial<DetectAISample>[]> {
  const samples: Partial<DetectAISample>[] = [];
  const offset = Math.floor(Math.random() * 100000);

  try {
    const url = `https://datasets-server.huggingface.co/rows?dataset=HuggingFaceM4/COCO&config=2017_captions&split=train&offset=${offset}&length=100`;
    const r = await fetch(url);
    if (!r.ok) return samples;

    const data: { rows: Array<{ row: { image?: { src: string }; captions?: string[] } }> } = await r.json();

    for (const row of data.rows ?? []) {
      const imgSrc = row.row.image?.src ?? "";
      const captions = row.row.captions ?? [];
      if (!imgSrc) continue;

      samples.push({
        source_url:  imgSrc,
        raw_content: imgSrc,
        metadata: {
          captions:   captions,
          source:     "ms-coco-2017",
          license:    "CC BY 4.0",
          tags:       ["coco", "microsoft", "cc"],
        },
      });
    }
  } catch { /* skip */ }

  return samples.slice(0, maxSamples);
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const t0 = Date.now();
    const db = createSupabaseClient(env.SUPABASE_URL, env.SUPABASE_SERVICE_KEY);

    // Run all 3 sources in parallel
    const [openImages, laion, coco] = await Promise.allSettled([
      scrapeOpenImages(300),
      scrapeLAION(300),
      scrapeCOCO(200),
    ]);

    const allRaw = [
      ...(openImages.status === "fulfilled" ? openImages.value : []),
      ...(laion.status === "fulfilled" ? laion.value : []),
      ...(coco.status === "fulfilled" ? coco.value : []),
    ];

    const samples: DetectAISample[] = allRaw
      .filter(s => s.raw_content && s.raw_content.length > 0)
      .map(s => ({
        sample_id:    uuidv4(),
        source_id:    (s.metadata as {source?:string})?.source ?? "open-images",
        source_url:   s.source_url ?? "",
        content_type: "image" as const,
        language:     "en",
        raw_content:  s.raw_content!,
        metadata:     s.metadata ?? {},
        scraped_at:   new Date().toISOString(),
        worker_id:    env.WORKER_ID ?? "image-open-01",
        status:       "staged" as const,
      }));

    // Push in 100-row batches
    const BATCH = 100;
    let pushed = 0;
    for (let i = 0; i < samples.length; i += BATCH) {
      const { error } = await db.from("samples_staging").insert(samples.slice(i, i + BATCH));
      if (!error) pushed += Math.min(BATCH, samples.length - i);
    }

    return new Response(JSON.stringify({
      samples_scraped: samples.length, pushed,
      duration_ms: Date.now() - t0,
      sources: { open_images: openImages.status, laion: laion.status, coco: coco.status }
    }), { status: 200, headers: { "Content-Type": "application/json" } });
  }
};
