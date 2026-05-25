# V3 Spec Verification Report

Run date: 2026-05-24
Spec under review: `vg_photo_editor_master_prompt.md` (Part H tasks 1-8)

Legend: ✓ matches spec — ✗ spec is wrong — ⚠ partial/unverifiable — ✱ needs follow-up by Alan

---

## 1. AutoHDR pricing ✗ (spec is out of date)

| Plan | Spec | Live (autohdr.com/pricing) |
|---|---|---|
| Free | 10/mo, $0 | **10/mo, $0** ✓ |
| Mid tier name | "Professional" | **"Standard"** |
| Mid tier volume | 1,000/mo | **500/mo** |
| Mid tier price | $440/mo | **$265/mo** ($225.25/mo yearly) |
| Mid tier per-photo | $0.44 | **$0.53** ($0.45 yearly) |
| Enterprise volume | 5,000/mo | **5,000/mo** ✓ |
| Enterprise price | $2,025/mo | **$2,250/mo** ($2,025/mo yearly) |
| Enterprise per-photo | $0.41 | **$0.45** monthly / **$0.41** yearly |
| Yearly discount | 20% | **~15% Standard / ~10% Enterprise** |
| Enterprise features | Auto TV Blackout, Auto Add Fire, Walkthrough Re-ordering, Dedicated Slack Channel | **same** ✓ |
| Volume slider | yes | yes ✓ |

**Impact on break-even math:** Per-photo cost moved from $0.44 → $0.53 (mid tier), so the "AutoHDR is $0.44/photo" baseline in the project memory should be **$0.53/photo** for mid-tier comparison or $0.45 if Alan would buy yearly. The 1,563-photo/month break-even at $500/mo project cost still holds for the per-photo-fee-avoidance argument, but the *gap* is smaller than the spec implies.

---

## 2. AutoHDR upload-modal models ✓ (Classic / Lisa / Twilight confirmed)

Could not load the upload modal directly (auth-gated). Independent confirmation from a YouTube short ("AutoHDR: The Lisa and Twilight") and third-party docs: **Classic**, **Lisa**, **Twilight** are the three options. Spec accurate. No new models surfaced in search.

---

## 3. AutoHDR Style Settings page (account) ✱ — auth-gated, deferred

`autohdr.com/account` returns 401 unauthenticated. Cannot verify Window Pull / Cloud Style / Twilight Style / TV Screen Replacement dropdown options without a logged-in session. **Action for Alan:** when next logged into AutoHDR free tier, screenshot the Style Settings page so v3 Section A7 can be corrected.

---

## 4. Replicate pricing for the 3 models

| Model | Status | Hash | Notes |
|---|---|---|---|
| `idea-research/grounded-sam-2` | **✗ does not exist** | — | Spec is wrong. See task 8. |
| `nightmareai/real-esrgan` | ✓ exists | spec-pinned `f121d640…46aa` (versions page 403; hash not re-verified live this run) | Pricing surfaced as "varies based on inputs" — ~$0.034/run reference, T4 GPU |
| `megvii-research/nafnet` | ✓ exists | `018241a6c880319404eaa2714b764313e27e11f950a7ff0a7b5b37b27b74dcf7` | Latest version published ~4 years ago (stale model, but functional). Per-second billing. |

Real-ESRGAN hash needs a live re-pin before Phase 1 ships (the existing spec hash is plausible but blocked behind 403/auth in this run). Easy to confirm at runtime by calling `replicate.models.get("nightmareai/real-esrgan")` from the worker.

---

## 5. AutoHDR Flambient support ⚠ (likely yes, not definitive from public docs)

AutoHDR's own blog post ("Real Estate Photo Editing Showdown: HDR vs Flambient", 2025-10-29) says their **"software can process both HDR and Flambient sets in minutes"** — suggests they accept Flambient bundles as input, but this is marketing copy in a comparison post, not product documentation. Treat as a soft yes. Confirms the v3 framing that PhotoHDR MV should also support Flambient is competitively warranted.

---

## 6. Higgsfield Nano Banana Pro Inpaint MCP ✓ (with caveat)

- Hosted MCP endpoint: `https://mcp.higgsfield.ai/mcp` (HTTP transport, MCP protocol).
- Auth: existing Higgsfield account, plan credits transfer — **no API key, no developer-app registration**.
- This Claude session **already has the Higgsfield MCP server loaded** as `mcp__claude_ai_Higgsfield_Design__*` — relevant tools include `generate_image`, `generate_video`, `media_upload`, `media_confirm`, `reveal_generation`, `show_generations`, `job_status`, `personal_clipper_*`. There is **no dedicated `inpaint` tool name surfaced in the tool list** — masking is likely done via `generate_image` with a reference image + mask parameter, but exact parameter schema needs a live `tools/list` inspection at integration time.
- Capability confirmed: mask + text prompt + optional reference images; 4-step reasoning (shape → edges → geometry → texture).

**Action for Phase 3:** when wiring inpaint, first call the loaded `mcp__claude_ai_Higgsfield_Design__generate_image` tool to introspect its actual parameter schema before writing the worker integration. Don't trust the spec on parameter names.

---

## 7. Adobe Firefly Services API ✗ (definitively out for internal tool)

- **Enterprise-only.** No individual developer access.
- Min commitment: ~$1,000/month, 3-year contract, 50 seats.
- Restructured 2025 — Firefly is now standalone, no longer bundled with Creative Cloud.

**Recommendation:** drop Firefly from the v3 spec entirely. It is not procurable for VG/WTPS at any reasonable cost.

---

## 8. Grounded SAM 2 current model ID ✗ (spec is wrong)

The slug `idea-research/grounded-sam-2` **does not exist on Replicate** (404).

Closest available models:

| Slug | Pinned hash | Age | Notes |
|---|---|---|---|
| `idea-research/ram-grounded-sam` | `80a2aede4cf8e3c9f26e96c308d45b23c350dd36f1c381de790715007f1ac0ad` | 2y 11mo | RAM (Recognize Anything) + Grounding DINO + SAM. Auto-tags then segments. |
| `schananas/grounded_sam` | (not retrieved this run) | unknown | Plain Grounding DINO + SAM. Prompt-driven. |

Upstream Grounded SAM 2 (IDEA-Research GitHub repo) exists but **has no published Replicate port** under the IDEA-Research namespace. Options:

- **Pick `schananas/grounded_sam`** for prompt-driven masking (window pull, sky, etc.) — this is the closest match to the v3 spec intent.
- **Self-host** Grounded SAM 2 on Fly.io with GPU, using the IDEA-Research repo directly — adds infra burden but matches the spec.
- **Pick `idea-research/ram-grounded-sam`** if auto-tagging is wanted (helpful if Alan wants the pipeline to detect "window, sky, lamp" without being told).

**Recommendation for Phase 2:** start with `schananas/grounded_sam`, escape to self-hosted Grounded SAM 2 only if quality is insufficient. Update the v3 spec to drop `idea-research/grounded-sam-2`.

---

## Spec corrections to make before Phase 1 build

1. Update AutoHDR pricing table to: Free $0 (10), Standard $265/mo (500, $0.53/photo), Enterprise $2,250/mo (5,000, $0.45/photo). Yearly discounts ~15% / ~10%.
2. Replace `idea-research/grounded-sam-2` with `schananas/grounded_sam` everywhere it appears, with a TODO to evaluate self-hosted Grounded SAM 2 if quality lags.
3. Re-pin `nightmareai/real-esrgan` hash at integration time via `replicate.models.get`.
4. Drop Adobe Firefly Services entirely (replace any references with "Higgsfield Nano Banana Pro Inpaint" or note "no enterprise-pricing tool acceptable for internal use").
5. Note Higgsfield MCP integration is **already loaded** in this Claude environment — Phase 3 should call `tools/list` on the live server to confirm exact `generate_image` parameter schema rather than relying on the spec.
6. Soft-confirm AutoHDR supports Flambient input bundles. PhotoHDR MV should also support Flambient → corroborates v3 Section B (image grouping must distinguish HDR brackets from Flambient pairs).
7. Action for Alan: log into AutoHDR free tier and screenshot `autohdr.com/account` Style Settings so Section A7 dropdown options can be corrected.

---

## Items not blocking Phase 0a → Phase 1

- Style Settings dropdown values (A7) — only relevant if PhotoHDR MV mirrors AutoHDR's style-preset UI. Defer.
- Exact Real-ESRGAN hash re-verification — easy at runtime, not a blocker.
- Higgsfield inpaint parameter schema — required before Phase 3, not Phase 1.
