import { NextRequest, NextResponse } from "next/server";
import { generateLandscape, InpaintError, isInpaintConfigured } from "@/lib/inpaint";
import { uploadDataUrl, uploadFromUrl } from "@/lib/cloudinary";

const MAX_FILE_BYTES = 25 * 1024 * 1024;
const MAX_VARIATIONS = 4;
const ALLOWED_MIME = new Set(["image/jpeg", "image/png"]);

// Shared preservation constraint appended to every style. Keeps the house, hardscape,
// camera angle, and surrounding property intact — the model should ONLY alter
// landscaping, not invent buildings/fences/pools.
const PRESERVE_CONSTRAINT = " IMPORTANT: preserve the house, driveway, walkways, grade, trees, "
  + "existing structures, camera angle, and property boundaries exactly as shown. Do not add "
  + "buildings, additions, pools, fences not present in the original, neighbors' properties, "
  + "vehicles, people, or animals.";

// Variation hints make each generated image distinct from the others.
const VARIATION_HINTS = [
  " Variation: restrained planting density, lower-maintenance.",
  " Variation: fuller planting density, generous flowering.",
  " Variation: balanced mid-density, mixed textures and seasonal interest.",
  " Variation: minimalist composition with negative space.",
];

const STYLE_PROMPTS: Record<string, { positive: string; negative?: string }> = {
  natural_meadow: {
    positive: "Transform the landscaping into a natural meadow with wildflower fields, native grasses, "
            + "and casual stone borders. Soft natural lighting, photorealistic, summer afternoon.",
    negative: "manicured, geometric, formal, oversaturated, cartoon, illustration",
  },
  classic_lawn: {
    positive: "Transform the landscaping into a classic manicured green lawn with crisp edging, "
            + "neatly defined flower beds with seasonal plantings, and healthy turf grass. "
            + "Photorealistic, late morning light.",
    negative: "wild, overgrown, dead grass, cartoon, illustration",
  },
  coastal_garden: {
    positive: "Transform the landscaping into a coastal Martha's Vineyard garden: beach grasses, "
            + "lavender, low rugosa roses, dune planting with native sedges, muted natural tones. "
            + "Photorealistic, soft afternoon coastal light.",
    negative: "tropical, palm trees, mowed lawn, formal, added structures",
  },
  formal_garden: {
    positive: "Transform the landscaping into a formal garden: geometric flower beds, low boxwood "
            + "hedges, symmetric layout with subtle pathways, neatly trimmed topiary. Photorealistic, "
            + "balanced lighting.",
    negative: "wild, asymmetric, casual, cartoon, urns, statues",
  },
  shade_garden: {
    positive: "Transform the landscaping into a shade garden: lush hostas, ferns of varied textures, "
            + "moss-covered stones, dappled light through tree canopy, naturalistic groundcover. "
            + "Photorealistic, soft dappled lighting.",
    negative: "bright sun, lawn, formal, cartoon",
  },
  pollinator: {
    positive: "Transform the landscaping into a pollinator garden with native flowering plants: "
            + "coneflowers, bee balm, milkweed, black-eyed susans, asters. Photorealistic, "
            + "vibrant but natural color palette.",
    negative: "manicured, formal, monoculture, cartoon, visible bees, visible butterflies, insects",
  },
};


export async function POST(req: NextRequest) {
  if (!isInpaintConfigured()) {
    return NextResponse.json(
      { error: "Inpaint not configured. Set HIGGSFIELD_API_KEY and HIGGSFIELD_SCHEMA_VERIFIED=true after a smoke test." },
      { status: 503 }
    );
  }

  let formData: FormData;
  try {
    formData = await req.formData();
  } catch {
    return NextResponse.json({ error: "Invalid form data" }, { status: 400 });
  }

  const image = formData.get("image") as File | null;
  const styleId = (formData.get("styleId") as string) || "natural_meadow";
  const extraPrompt = ((formData.get("extraPrompt") as string) || "").trim().slice(0, 500);
  const variationsRaw = Number(formData.get("variations") ?? 3);
  const variations = Number.isFinite(variationsRaw)
    ? Math.max(1, Math.min(MAX_VARIATIONS, Math.floor(variationsRaw)))
    : 3;

  if (!image) return NextResponse.json({ error: "image is required" }, { status: 400 });
  if (image.size > MAX_FILE_BYTES) {
    return NextResponse.json({ error: `image exceeds ${MAX_FILE_BYTES} bytes` }, { status: 413 });
  }
  if (!ALLOWED_MIME.has(image.type)) {
    return NextResponse.json({ error: "image must be JPEG or PNG" }, { status: 400 });
  }
  const recipe = STYLE_PROMPTS[styleId];
  if (!recipe) {
    return NextResponse.json({ error: `unknown styleId: ${styleId}` }, { status: 400 });
  }

  // Upload the source image to Cloudinary so Higgsfield can fetch a public URL.
  let sourceUrl: string;
  try {
    const bytes = await image.arrayBuffer();
    const dataUrl = `data:${image.type};base64,${Buffer.from(bytes).toString("base64")}`;
    sourceUrl = await uploadDataUrl(dataUrl, "vg-landscape/inputs");
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ error: `Cloudinary upload failed: ${msg}` }, { status: 502 });
  }

  // Compose the prompt: style + preservation constraint + user instructions (last so they're
  // bounded by the constraint context).
  const basePrompt = recipe.positive + PRESERVE_CONSTRAINT + (extraPrompt ? ` Additional: ${extraPrompt}` : "");

  // Generate N variations in parallel, each with its own variation hint for diversity.
  const tasks = Array.from({ length: variations }, (_, i) => {
    const variationPrompt = basePrompt + (VARIATION_HINTS[i] ?? "");
    return generateLandscape(sourceUrl, { prompt: variationPrompt, negativePrompt: recipe.negative })
      .then(async (higgsfieldUrl) => {
        try {
          const ownedUrl = await uploadFromUrl(higgsfieldUrl, "vg-landscape/outputs");
          return { url: ownedUrl, style: styleId, prompt: variationPrompt };
        } catch {
          // If mirror-to-Cloudinary fails, fall back to the upstream Higgsfield URL.
          return { url: higgsfieldUrl, style: styleId, prompt: variationPrompt };
        }
      });
  });

  const settled = await Promise.allSettled(tasks);
  const results = settled
    .filter((r): r is PromiseFulfilledResult<{ url: string; style: string; prompt: string }> => r.status === "fulfilled")
    .map((r) => r.value);
  const failures = settled
    .filter((r): r is PromiseRejectedResult => r.status === "rejected")
    .map((r) => (r.reason instanceof InpaintError ? r.reason.message : String(r.reason)));

  if (results.length === 0) {
    return NextResponse.json(
      { error: `all ${variations} variations failed: ${failures.join("; ")}` },
      { status: 502 }
    );
  }

  return NextResponse.json({ results, failures });
}
