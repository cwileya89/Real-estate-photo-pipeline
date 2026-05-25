/**
 * Higgsfield Nano Banana Pro Inpaint client (vg-landscape-visualizer).
 *
 * Duplicated from photo-editor/lib/inpaint.ts; the two projects are intentionally
 * separate per v3 Phase 4 spec. Keep behavior in sync manually until a shared
 * package is justified.
 *
 * Schema-verified flag (HIGGSFIELD_SCHEMA_VERIFIED=true) gates live calls until
 * a one-time smoke test confirms field names match.
 */

const HIGGSFIELD_API_KEY = process.env.HIGGSFIELD_API_KEY;
const HIGGSFIELD_BASE_URL = process.env.HIGGSFIELD_BASE_URL?.replace(/\/$/, "") ?? "https://api.higgsfield.ai/v1";
const HIGGSFIELD_MODEL = process.env.HIGGSFIELD_INPAINT_MODEL ?? "nano-banana-pro-inpaint";
const SCHEMA_VERIFIED = process.env.HIGGSFIELD_SCHEMA_VERIFIED === "true";

function clampedInt(envName: string, defaultMs: number, minMs: number): number {
  const raw = process.env[envName];
  if (raw === undefined) return defaultMs;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < minMs) return defaultMs;
  return parsed;
}

const POLL_INTERVAL_MS = clampedInt("HIGGSFIELD_POLL_INTERVAL_MS", 2_000, 100);
const POLL_TIMEOUT_MS = clampedInt("HIGGSFIELD_POLL_TIMEOUT_MS", 180_000, 1_000);

const SUCCESS_STATUSES = new Set(["complete", "completed", "succeeded", "success", "ok"]);
const FAILURE_STATUSES = new Set(["failed", "error", "cancelled", "canceled"]);

export class InpaintError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = "InpaintError";
  }
}

export function isInpaintConfigured(): boolean {
  return !!HIGGSFIELD_API_KEY && SCHEMA_VERIFIED;
}

interface GenerationResponse {
  id: string;
  status?: string;
  output_url?: string;
  output?: string | string[];
  error?: string;
}

function authHeaders(): Record<string, string> {
  if (!HIGGSFIELD_API_KEY) throw new InpaintError("HIGGSFIELD_API_KEY is not set");
  return {
    Authorization: `Bearer ${HIGGSFIELD_API_KEY}`,
    "Content-Type": "application/json",
  };
}

function extractOutputUrl(r: GenerationResponse): string | null {
  if (r.output_url) return r.output_url;
  if (typeof r.output === "string" && r.output.startsWith("http")) return r.output;
  if (Array.isArray(r.output) && r.output.length > 0 && typeof r.output[0] === "string") return r.output[0];
  return null;
}

function abortableDelay(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(new Error("aborted"));
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new Error("aborted"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

async function pollUntilComplete(generationId: string, signal: AbortSignal): Promise<string> {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (signal.aborted) throw new InpaintError("inpaint poll aborted");
    const res = await fetch(`${HIGGSFIELD_BASE_URL}/generations/${generationId}`, {
      method: "GET",
      headers: authHeaders(),
      signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new InpaintError(`poll failed: ${res.status} ${text.slice(0, 200)}`, res.status);
    }
    const body = (await res.json()) as GenerationResponse;
    const status = (body.status ?? "").toLowerCase();
    if (SUCCESS_STATUSES.has(status)) {
      const url = extractOutputUrl(body);
      if (!url) throw new InpaintError(`generation ${generationId} ${status} but no output URL`);
      return url;
    }
    if (FAILURE_STATUSES.has(status)) {
      throw new InpaintError(`generation ${generationId} ${status}: ${body.error ?? "unknown"}`);
    }
    try {
      await abortableDelay(POLL_INTERVAL_MS, signal);
    } catch {
      throw new InpaintError("inpaint poll aborted");
    }
  }
  throw new InpaintError(`generation ${generationId} timed out after ${POLL_TIMEOUT_MS}ms`);
}

export interface GenerateOptions {
  prompt: string;
  negativePrompt?: string;
  /** Optional reference photo URL the model should match in style. */
  referenceUrl?: string;
}

/**
 * Whole-image generative transform (landscape restyling). Uses a full-white mask so
 * the entire image is in-paint scope.
 */
const FULL_WHITE_MASK_DATA_URL =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=";

export async function generateLandscape(imageUrl: string, opts: GenerateOptions): Promise<string> {
  if (!opts.prompt?.trim()) throw new InpaintError("prompt is required");
  if (!isInpaintConfigured()) {
    throw new InpaintError(
      "inpaint not configured. Set HIGGSFIELD_API_KEY and HIGGSFIELD_SCHEMA_VERIFIED=true after a smoke test."
    );
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), POLL_TIMEOUT_MS);

  try {
    const submitRes = await fetch(`${HIGGSFIELD_BASE_URL}/generations`, {
      method: "POST",
      headers: authHeaders(),
      signal: controller.signal,
      body: JSON.stringify({
        task: "inpaint",
        model: HIGGSFIELD_MODEL,
        image_url: imageUrl,
        mask_url: FULL_WHITE_MASK_DATA_URL,
        prompt: opts.prompt,
        ...(opts.negativePrompt ? { negative_prompt: opts.negativePrompt } : {}),
        ...(opts.referenceUrl ? { reference_url: opts.referenceUrl } : {}),
      }),
    });
    if (!submitRes.ok) {
      const text = await submitRes.text().catch(() => "");
      throw new InpaintError(`submit failed: ${submitRes.status} ${text.slice(0, 200)}`, submitRes.status);
    }
    const submitBody = (await submitRes.json()) as GenerationResponse;
    if (!submitBody.id) throw new InpaintError(`submit returned no generation id`);
    return await pollUntilComplete(submitBody.id, controller.signal);
  } catch (e) {
    if (e instanceof InpaintError) throw e;
    if (e instanceof Error && e.name === "AbortError") {
      throw new InpaintError(`generate timed out after ${POLL_TIMEOUT_MS}ms`);
    }
    throw new InpaintError(`generate failed: ${e instanceof Error ? e.message : String(e)}`);
  } finally {
    clearTimeout(timeout);
  }
}
