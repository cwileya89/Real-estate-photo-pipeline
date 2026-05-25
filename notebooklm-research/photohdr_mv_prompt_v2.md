ROLE
You are a senior ML engineer and product strategist. You are helping build a
proprietary AI-powered real estate photo editing platform that replicates and
improves on AutoHDR (autohdr.com). The business owner is on Martha's Vineyard
serving high-net-worth estate clients through Vineyard Gardens and WTPS.

TASK
Do not start building yet. First verify every claim marked [VERIFY] below
against live sources. Then produce a structured research report and updated
build spec. Flag anything that has changed, is wrong, or is better than what
is described.

---

SECTION 1: WHAT AUTOHDR PUBLICLY CONFIRMS (verified from live site + YouTube)

These facts were captured by navigating autohdr.com directly and watching
official walkthroughs. Verify each one is still current.

1.1 PRODUCT DESCRIPTION
AutoHDR is not just an HDR merge tool. It is a full real estate photo editing
platform that handles:
- HDR bracket editing (3 or 5 exposures at +/-2 EV)
- Flambient editing (flash + ambient pairs) [VERIFY still supported]
- Automated delivery in under 5 minutes [VERIFY]
- Workflow automation via Dropbox, Spiro, HDPhotoHub, Tonomo, Aryeo [VERIFY]

1.2 V3 PRICING (captured from YouTube walkthrough 2 months ago)
[VERIFY these are still current on autohdr.com/pricing]
- Free: 10 photos/month, $0
- Professional (most popular): 1,000 photos/month, $440/month, $0.44/photo
- Enterprise: 5,000 photos/month, $2,025/month, $0.41/photo
- Enterprise includes: credits rollover, Auto TV Blackout, Auto Add Fire,
  Walkthrough Re-ordering, Dedicated Slack Channel
- Yearly billing saves 20%
- Volume slider on paid plans to adjust photo count
- Top Up credits available at current per-photo price

1.3 AI MODELS (confirmed from upload flow)
Three named models in the project setup modal:
- Classic: "Natural, beautiful look for interiors" (default)
- Lisa: "Extra pop and deep window pulls"
- Twilight, sub-model Twilight Pink: "For photos shot at dusk"

1.4 EDITING SETTINGS (gear icon in project modal)
Four toggles:
- Sky Replacement (on by default)
- Perspective Correction (on by default)
- Grass Replacement (off by default)
- Declutter (off by default)

1.5 PER-PHOTO EDITING PANEL (V3 interface, two tabs)
[VERIFY this two-tab layout is current]

AI Tools tab (2x4 grid of icon buttons):
- Human Edit (person icon)
- Twilight (moon icon)
- AutoFill (pencil icon)
- AutoRemove (trash icon)
- Grass (sprout icon)
- Staging (sofa icon)
- Lot Lines (lines icon)
- Blur (droplet icon, teal when active)

Basic Adjustments tab:
- Tool buttons: Crop & Tilt, Pin, Lot Lines, Blur
- Blur active state: Size slider + L/M/H strength buttons + Clear + Apply
- Seven sliders with live numeric values:
  Brightness, Contrast, White Balance, Vibrance, Sharpness, Shadows, Highlights
- Reset (dark) + Apply (teal) buttons

Bottom: horizontal filmstrip of all project photos, teal border on active photo
Top right: x close button, flag/report button, Show Video toggle

1.6 LISTINGS DASHBOARD (autohdr.com/listings)
Left sidebar:
- Search listings input
- Project list: active projects = dark/black card, deleted = gray with
  "Deleted" badge
- Each project has pencil rename icon

Top toolbar:
- Copy Shareable Link, Rate, Select All (N), Download dropdown
- When photos selected: Supercharge and Delete buttons appear
- Supercharge = re-process at higher quality settings

Balance shown: "Balance: N Edits"
Main area: 4-column photo grid, checkboxes top-left on each photo

1.7 STYLE SETTINGS PAGE (autohdr.com/account)
[VERIFY still accessible at this URL]
Interior / Exterior tab toggle (teal active state)
Per-tab sliders (-2 to +2): Brightness, Contrast, Vibrance, White Balance

Right column dropdowns and toggles:
- Window Pull Intensity dropdown: Natural (default), others TBD [VERIFY options]
- Cloud Style dropdown: "Fluffy + Whispy" (confirmed text) [VERIFY]
- Twilight Style dropdown: "Pink" (confirmed) [VERIFY other options]
- TV Screen Replacement dropdown: "None" default, options unknown [VERIFY]
- Toggles: TV Screen Blackout, Fire in Fireplace, Interior Clouds,
  Exterior Clouds, Deduplication, Walkthrough Reorder (with gear sub-icon)

Preview image above settings panel showing live style example.

1.8 HUMAN-IN-THE-LOOP PANEL (on account/settings page)
- "Add Human-in-the-loop" section
- "Special requests? Integrate an editor in a click!"
- "Go to queue" button (teal)
- Email input: "Add editor's email address"
- "Add Editor" button (teal)
- "No editors added yet" shown when empty

1.9 WALKTHROUGH REORDER
Compiles still photos into a short video walkthrough (confirmed: 0:12 example
seen). This is the "Walkthrough Re-ordering" Enterprise feature.

1.10 UPLOAD FLOW
- Drag and drop zone on homepage
- Supports: JPEG, PNG, RAW, DNG, NEF, ARW, CR3, CR2, and many others
  (confirmed from HTML file input accept attribute)
- Dropbox import icon (SDK loaded via script tag)
- Google Drive import icon
- After upload: "Name Your Project" modal appears
- Modal shows: project name input, model selector, gear settings, Continue button

---

SECTION 2: WHAT IS INFERRED (plausible engineering, NOT confirmed by AutoHDR)

AutoHDR does not publicly disclose its technical architecture. The following
are reasonable implementation approaches based on industry literature and
competitor disclosures. Label them as INFERRED in your report.

2.1 BRACKET MERGING (INFERRED)
Classical method: Mertens Exposure Fusion (Mertens-Kautz-Van Reeth 2007)
- Weights pixels by contrast, saturation, and well-exposedness
- OpenCV MergeMertens implements this
- Enfuse (Hugin) is the command-line equivalent
- Neither creates a true HDR file; both produce an LDR fused result

Modern CNN method (likely used by AutoHDR):
- U-Net style fusion model OR NAFNet-style restoration on top of classical merge
- Reduces halos, ghosting, color cast vs classical Mertens
- Requires paired training data (input brackets + final edited output)

2.2 FLAMBIENT PROCESSING (INFERRED, but AutoHDR publicly supports it)
Flambient = Flash + Ambient. Two separate image types:
- Ambient brackets: natural light exposures for depth and tone
- Flash shot: off-camera bounced flash for accurate color and window control

Flambient blend process (differs from HDR):
- Layer ambient base image
- Use flash shot for color accuracy layer
- Use luminosity/color blending modes
- Mask window regions using Darken blend mode
- This is NOT exposure fusion; it is layer-based compositing

Critical implication for the build:
The upload flow must detect or ask whether incoming images are HDR brackets
OR Flambient pairs because the merging logic is completely different.

2.3 WINDOW PULL (INFERRED implementation, CONFIRMED feature)
AutoHDR confirms window pulls as a core feature.
Autoenhance.ai V4 publicly describes their implementation:
"We built a custom neural network trained on a vast dataset of real estate
images including window pulls. By analyzing visual patterns associated with
window pulls, the AI has developed a sophisticated understanding of where
window pulls need to be applied."

Likely stack (inferred):
- GroundingDINO: text-prompted detection of "window", "glass door"
- SAM 2: convert detected boxes into pixel masks
- Feathered mask edges via morphology
- Blend darkest bracket (or flash shot for flambient) into masked window area
- Apply separate white balance to window region vs room

2.4 WHITE BALANCE PROCESSING (INFERRED)
Interior: approximately 4500K-5500K (warm room tones)
Exterior/windows: approximately 6000K-7000K (daylight)
Critical: keep windows natural, not cyan-blue. Common AI failure mode.

2.5 PERSPECTIVE CORRECTION (INFERRED, CONFIRMED feature)
Vertical line correction:
- Detect near-vertical line segments (LSD algorithm or Hough transform)
- Estimate vanishing point
- Compute correction homography
- Warp image so walls and door frames stand upright
- Crop/expand edges after warp

Lens distortion:
- Read EXIF camera/lens metadata
- Apply Lensfun database corrections if available
- Or apply barrel distortion correction via OpenCV

2.6 GENERATIVE FILLS (INFERRED models, CONFIRMED features)
Sky replacement: segmentation + stock sky library (classical, reliable) OR
  diffusion inpainting for failed cases
Grass greening: HSL shift over lawn segmentation mask (classical first) OR
  generative fill
TV blackout: detect screen rectangle + fill dark (classical)
Fire: fire asset paste OR generative fill into fireplace mask
Object removal: diffusion inpainting (Stable Diffusion or Flux)
Virtual staging: generative fill with furniture prompt

NOTE: Do NOT use diffusion for structural architecture. Only use it in
clearly masked non-factual regions (sky, grass, TV, fire, staging).

---

SECTION 3: COMPETITOR LANDSCAPE (verify pricing and features are current)

For each competitor below, [VERIFY] current pricing, current features, and
any technical disclosures about their AI pipeline.

3.1 Autoenhance.ai [VERIFY pricing and V4 features]
- Web-based AI real estate photo editor
- Confirmed features: HDR merge, window pulls (V4 custom neural net),
  sky replacement, perspective correction, privacy blur
- Their V4 blog post on window pull neural networks is a key reference
- Target: verify current per-photo pricing

3.2 Fotello [VERIFY]
- AI real estate media platform
- Claims: editorial-quality photos, virtual staging, twilight shots
- Claims: unlimited human revisions, RAW support
- Confirmed competitor to AutoHDR and Imagen

3.3 BoxBrownie [VERIFY pricing]
- Hybrid human + AI editing service
- Last known pricing: $1.60/image enhancement, $24-30/virtual staging,
  $5/day-to-dusk, $5-10/item removal
- 17-step enhancement process (use as QA checklist)
- 24-hour turnaround
- Does NOT license its before/after archive for ML training

3.4 Imagen AI [VERIFY]
- Lightroom-integrated, personal AI profile trained on your own edits
- Has dedicated Window Pull AI tool with segmentation/masking
- Their support docs describe window detection and masking process
- Strong for photographers who want their own style replicated

3.5 Photomatix Pro [VERIFY pricing]
- Classical HDR software: $99 one-time
- Lightroom plugin: $69 one-time
- API/CLI subscription: ~$19/month
- Proves classical HDR alone is commercially viable

3.6 Phixer [VERIFY]
- Human real estate photo editing outsourcing
- Same-day delivery
- ~$2-5/photo depending on service level

3.7 Styldod [VERIFY]
- AI real estate editing + virtual staging
- Enhancement: ~$1.50/photo, staging: ~$16/photo

---

SECTION 4: TRAINING DATA SOURCES

4.1 FREE DATASETS (verify availability and usefulness)
[VERIFY each URL is still active and confirm what the dataset contains]

HDR-specific:
- S2R-HDR: https://huggingface.co/datasets/iimmortall/S2R-HDR
  Synthetic HDR fusion dataset, Unreal Engine 5, 1,000 sequences, 24,000
  images at 1920x1080 with HDR/SDR pairs. Best for pretraining fusion logic.
- HDRT Dataset: https://huggingface.co/datasets/jingchao-peng/HDRTDataset
  Infrared-guided HDR, 40K+ HDR/SDR pairs. Not real estate but useful for
  HDR research.
- MIT-Adobe FiveK: 5,000 RAW photos retouched by experts. Good for
  general photo enhancement pretraining.
- RAISE: 8,156 high-res RAW images, unprocessed. Good for RAW processing.

Real estate classification (not HDR):
- https://huggingface.co/datasets/jashu66/realestate-image-dataset
- https://huggingface.co/datasets/andupets/real-estate-image-classification
- https://huggingface.co/datasets/Binaryy/multimodal-real-estate-search
  [VERIFY what these contain - useful for room type routing, not HDR]

4.2 DATASET STRUCTURE FOR SELF-GENERATED DATA
Use this exact folder structure for all collected training data:

dataset/
  property_0001/
    room_001/
      shoot_type: "hdr" | "flambient"
      brackets/
        IMG_0001_-2.CR3     (dark)
        IMG_0002_0.CR3      (normal/ambient base)
        IMG_0003_+2.CR3     (bright)
        IMG_0004_flash.CR3  (flash only, for flambient)
      final/
        IMG_0002_final.jpg
      masks/
        window.png
        sky.png
        grass.png
        tv.png
        fireplace.png
      meta.json

meta.json schema:
{
  "property_id": "property_0001",
  "room_type": "living_room",
  "shoot_type": "hdr",
  "bracket_count": 3,
  "ev_spacing": 2,
  "camera": "Canon R6",
  "lens": "RF 15-35mm",
  "editor": "human_editor_01",
  "style_profile": "natural_real_estate",
  "notes": "large window, ocean view, Martha's Vineyard"
}

Target dataset sizes:
- Week 2: 500 bracket/flambient sets from own archive
- Month 2: 2,000-5,000 before/after pairs
- Month 6: 10,000-20,000 pairs for fine-tuning

4.3 MINIMUM VIABLE DATASET COST
Budget under $5,000 initially:
- Own archive: $0 (use existing RAW brackets + Lightroom exports)
- Human editing for before/after pairs: $0.60-1.60/image
  For 2,000 images: $1,200-3,200
- Window mask annotation: can use GroundingDINO + SAM 2 to auto-generate
  then manually correct
- Do NOT buy stock photos for training without confirmed ML training rights
- Do NOT scrape MLS photos (licensing risk, legal exposure)

---

SECTION 5: AI MODELS TO USE OR FINE-TUNE

Do not build from scratch. Use or fine-tune existing models.

5.1 EXPOSURE FUSION / HDR MERGE
Start with classical, add ML later:

Enfuse/OpenCV MergeMertens (classical baseline):
- Fast, explainable, cheap, good benchmark baseline
- Use Hugin align_image_stack + enfuse CLI
- Or OpenCV createMergeMertens()
- Expected quality: 60-75% of AutoHDR on normal interiors

LightFuse (first ML upgrade for HDR):
- GitHub: https://github.com/Taichi-Pink/LightFuse-Lightweight-CNN-based-Dual-exposure-Fusion
- Lightweight CNN dual-exposure fusion: GlobalNet (illumination) +
  DetailNet (local details)
- Fast inference, can run on RTX 5070 Ti
- Limitation: designed for 2 exposures, real estate uses 3-5
  Workaround: preselect dark + bright bracket as input pair

NAFNet (restoration after fusion):
- GitHub: https://github.com/megvii-research/NAFNet
- Replicate: https://replicate.com/megvii-research/nafnet
- Use for: denoise, deblur, artifact cleanup after fusion
- Do NOT use as primary HDR merger (not trained for this out of the box)
- Fine-tune later on real estate pairs for style consistency

5.2 WINDOW SEGMENTATION / MASKING
Best stack (no training required to start):

Step 1: GroundingDINO
- Text prompt: "window, glass door, sliding door, skylight"
- Returns bounding boxes
- Available on Replicate and Hugging Face

Step 2: SAM 2 (Meta)
- Replicate: https://replicate.com/meta/sam-2
- Input: image + point/box prompts from GroundingDINO
- Output: pixel-perfect masks
- Cost: verify current Replicate pricing

Step 3: OpenCV post-processing
- Morphological operations to clean mask edges
- Gaussian feathering for smooth blend

Fine-tune later with 1,000+ annotated window masks using:
- DeepLabV3+ or SegFormer on Hugging Face

5.3 SKY REPLACEMENT
Priority order (cheapest/safest first):
1. Sky segmentation + stock sky library (classical, fast, free)
2. SkyAR: https://github.com/jiupinjia/SkyAR (open source)
3. SDXL inpainting on Replicate (generative fallback)
4. Adobe Firefly Generative Fill API (commercial/licensed fallback)
5. GPT Image 2 API (highest quality, most expensive)

Known AutoHDR weakness: reviewer noted sky replacement can look artificial.
Your system should let users toggle sky replacement off per-photo.

5.4 GENERATIVE FILLS

GPT Image 2 (OpenAI):
Model: gpt-image-2
API endpoint: https://api.openai.com/v1/images/edits
Use for: object cleanup, TV screen fill, fireplace fire, virtual staging
[VERIFY current pricing per image and API limits]

Node.js pattern:
import OpenAI from "openai";
import fs from "node:fs";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const result = await openai.images.edit({
  model: "gpt-image-2",
  image: fs.createReadStream("input.jpg"),
  mask: fs.createReadStream("mask.png"),
  prompt: "Replace only the masked TV screen with a realistic dark black
    screen. Preserve all architecture, lighting, camera angle.",
  size: "1536x1024"
});

NOTE: White pixels in mask = area to edit. Verify mask convention in
current OpenAI docs before production.

Adobe Firefly Generative Fill API:
[VERIFY API access process - as of early 2025, moved to independent product]
[VERIFY cost: ~$0.02-0.10/image depending on plan]
Use for: commercially safer fills where IP licensing matters
Requires: access token + API key from Adobe Developer Console

Pseudo-pattern:
const response = await fetch(
  "https://firefly-api.adobe.io/v3/images/fill",
  {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${process.env.ADOBE_ACCESS_TOKEN}`,
      "x-api-key": process.env.ADOBE_API_KEY,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      image: { source: { url: inputImageUrl } },
      mask: { source: { url: maskPngUrl } },
      prompt: "Replace the masked sky with natural clear blue sky. Keep
        house, trees, roofline, and lighting realistic."
    })
  }
);

Nano Banana 2 and Nano Banana Pro (Higgsfield/Google):
[VERIFY these model identifications are current]
Nano Banana 2 = Gemini 3.1 Flash Image Preview (speed/volume)
Nano Banana Pro = Gemini 3 Pro Image Preview (high-fidelity/professional)
Available via: Higgsfield Design MCP (already connected) OR Gemini API directly

Use Nano Banana Pro for: high-end hero images, complex edits, virtual staging
Use Nano Banana 2 for: batch variations, fast prototypes, less critical edits

Recommendation: Use Gemini API directly for production (stable billing, logs).
Use Higgsfield MCP for creative prototyping.
Do NOT rely only on third-party MCP wrapper for production until contractually
clear.

5.5 IMAGE ENHANCEMENT / FINAL POLISH

Real-ESRGAN (Replicate):
- Model: nightmareai/real-esrgan
- Use for: 2x final upscale, mild detail recovery
- Cost: ~$0.005/run on Replicate [VERIFY]
- Pattern:
  const output = await replicate.run("nightmareai/real-esrgan", {
    input: { image: finalImageUrl, scale: 2, face_enhance: false }
  });
- Keep face_enhance: false for real estate
- Do NOT overuse: causes plastic/oversharpened look

Cloudinary:
- Use for: delivery, resizing, web optimization, CDN
- Use e_improve:indoor and e_improve:outdoor for base enhancement
- Not sufficient alone for AutoHDR-quality editing

5.6 PERSPECTIVE CORRECTION

Python microservice using OpenCV:
import cv2
import numpy as np

# Detect line segments
lsd = cv2.createLineSegmentDetector()
lines, _, _, _ = lsd.detect(gray_image)

# Filter near-vertical lines
# Estimate vanishing point
# Compute correction homography
# cv2.warpPerspective(image, H, (w, h))

Also use:
- ExifTool to read camera/lens metadata
- Lensfun database for lens profile corrections
- Hugin CLI: align_image_stack for bracket alignment

---

SECTION 6: PHASED BUILD PLAN

PHASE 0 (Week 1-2): Dataset and benchmark
- Collect 500+ bracket sets from own archive
- Set up meta.json schema above for every property
- Run 50 test properties through: AutoHDR, Fotello, human editor
- Score each output: brightness, window pull quality, color accuracy,
  perspective, artifacts. Build your benchmark baseline.
- Do NOT start coding until benchmark is done.

PHASE 1 (Week 3-6): Classical pipeline
Architecture:
Next.js upload
  -> Cloudinary original storage
  -> Job queue (PostgreSQL/Prisma)
  -> Python image worker (Fly.io or local RTX 5070 Ti box)
    -> Detect shoot type: HDR or Flambient
    -> IF HDR: align_image_stack + enfuse/MergeMertens
    -> IF Flambient: layer-based composite (ambient base + flash color)
    -> Window pull: GroundingDINO + SAM 2 + feathered mask blend
    -> Perspective correction: OpenCV vertical line correction
    -> White balance: separate interior + window regions
    -> Tone curves and color profile
    -> QA scoring
    -> Final JPEG export
  -> Cloudinary processed storage
  -> Job status update

Expected quality at Phase 1 end:
- 60-75% of AutoHDR quality on normal interiors
- 40-60% on difficult high-contrast windows
- 80%+ on exterior sky/grass with simple masking

PHASE 2 (Week 7-10): Generative fills
Add generative API calls only for specific failed masks:
- Sky: Firefly or GPT Image 2 for bad skies
- Grass: HSL shift first, generative only if patchy
- TV: classical blackout first
- Fire: fire asset paste first, generative if needed
- Object removal: Replicate zylim0702/remove-anything
- Virtual staging: separate paid workflow, adirik/interior-design on Replicate

Expected per-image cost at Phase 2:
- Classical base pipeline: $0.005-0.03
- SAM 2 / GroundingDINO on Replicate: $0.005-0.05
- Real-ESRGAN: ~$0.005
- Generative fill (if needed): $0.02-0.20 per fill
- Blended cost assuming 30% need generative: $0.08-0.25/image

PHASE 3 (Month 3+): Fine-tune
First fine-tune order:
1. LightFuse for learned exposure fusion (if classical merge quality insufficient)
2. SegFormer/DeepLabV3+ for window masks (if SAM 2 failing on unusual windows)
3. NAFNet-style restoration model for final polish and style consistency

RTX 5070 Ti (16GB VRAM, Blackwell sm_120) can handle:
- LoRA-style fine-tuning on image models
- Small segmentation model fine-tuning
- NAFNet/restoration model fine-tuning on cropped patches
- Batch size 1-4 for large image restoration with mixed precision + tiling
- NOT comfortable for full-resolution end-to-end HDR model training without
  tiling or cloud GPU (use Runpod/Lambda for that)

COST MODEL
Phase 1 at 500 images/month:
- Compute: $25-150/month
- Storage (Cloudinary): depends on plan
- Effective: ~$0.05-0.30/image

Phase 2 at 5,000 images/month:
- Classical base + segmentation: $50-250/month
- Generative fills (30% of images): $150-500/month
- Total: $200-750/month
- Effective: $0.04-0.15/image

Break-even vs AutoHDR Professional ($440/month for 1,000 images):
- If in-house costs $200/month: break-even at ~440 images/month
- At 1,000+ images/month: you're ahead financially
- Engineering time excluded from this math

---

SECTION 7: WORKFLOW AUTOMATION (AutoHDR's hidden moat)

This section is as important as image quality. AutoHDR's integrations:
[VERIFY each integration is still active and document the workflow]

7.1 DROPBOX AUTOMATION
AutoHDR watches a Dropbox folder, processes uploads, returns edited files.
Note: The "01-RAW-Photos" folder naming convention is from Spiro's Dropbox
integration configuration (verified), not AutoHDR's own convention.

For your build, implement:
- Watched folder polling via Dropbox API
- Auto-detect bracket groups by timestamp proximity (within 5 seconds)
  OR by sequential filename suffix (_1, _2, _3)
- Auto-detect shoot type (HDR vs Flambient) by bracket count and/or
  file naming convention or EXIF metadata

7.2 SPIRO INTEGRATION [VERIFY integration still active]
7.3 ARYEO INTEGRATION [VERIFY]
7.4 HDPHOTOHUB INTEGRATION [VERIFY]
7.5 TONOMO INTEGRATION [VERIFY]

For Phase 1, build Dropbox automation first. Add others by demand.

---

SECTION 8: LEGAL AND LICENSING RISKS

8.1 MLS PHOTOS
Do not scrape. Do not train without written rights. MLS image rights may
belong to: photographer, brokerage, MLS board, seller, or platform.
RESO provides data standards but NOT automatic bulk photo ML training rights.

8.2 CLIENT PROPERTY PHOTOS
Add language to client contracts: "Images may be used internally to improve
editing quality. Client can opt out." Get explicit opt-in or anonymize.

8.3 STOCK PHOTOS
Standard Adobe Stock, Getty, Shutterstock licenses do NOT automatically
include ML training rights. Requires explicit ML/data licensing agreement.
Budget $1,000+ for any licensed commercial training data acquisition.

8.4 GENERATIVE EDIT DISCLOSURE
Real estate regulations vary by state. Sky replacement, grass greening,
virtual staging, day-to-dusk, TV/fire fills may require disclosure.
MLS rules increasingly require marking AI-altered images. Build a
per-image "edits applied" metadata flag from day one.

8.5 SYNTHETIC TRAINING DATA
Firefly, GPT Image, Nano Banana can generate synthetic interiors for training.
But synthetic interiors can teach unrealistic geometry/lighting. Use for
edge case augmentation only, not as core dataset.

---

SECTION 9: THE AUTOHDR.COM BUILD SPEC

This is the actual application to build. Everything in this section was
verified from live navigation of autohdr.com + YouTube walkthroughs.

App name: VG Photo Editor
Stack: Next.js 14 App Router, TypeScript, vanilla CSS, PostgreSQL/Prisma v5,
  Better-Auth PIN, Cloudinary, Replicate.com, Fly.io backend, Vercel frontend
Repo subfolder: vineyard-gardens/vg-photo-editor/
PowerShell note: use semicolons not &&

9.1 NAVIGATION
Top nav: Logo | Pricing | Studio | Listings | Contact | Upload | [Avatar] [N Photos]

9.2 PAGES

/ - Upload homepage
- Giant ghost wordmark "VG PHOTO EDITOR" behind centered upload zone
- Upload zone: drag & drop, Browse Files button, Dropbox icon, Google Drive icon
- Dropbox SDK loaded via script tag (app key: roi9y5o4gysmxev)
- Book a Demo button + "Get up to a month of free credits" subtext
- On file select -> "Name Your Project" modal:
  - Project name input
  - Shoot type selector: HDR Brackets | Flambient (NEW - required)
  - Model selector dropdown: "VGEditor Classic >"
    Options: Classic (natural/interiors), Lisa (extra pop/window pulls),
    Twilight Pink (dusk shots)
  - Gear icon -> Settings dropdown: Sky Replacement on, Perspective Correction on,
    Grass Replacement, Declutter
  - Continue button (teal)

/listings - Main dashboard
- Left sidebar: search + project list (dark=active, gray+badge=deleted)
  Each project has pencil rename icon
- Balance: "Balance: N Edits"
- Top toolbar (default): Copy Shareable Link | Rate | Select All (N) | Download dropdown
- Top toolbar (with selections): Supercharge | Delete | Copy Shareable Link |
  Rate | Deselect All (N) | Download dropdown
- Main area: 4-column photo grid, checkboxes on each photo

/listings/[projectId] - Photo editor
Left panel TWO TABS:

AI Tools tab (2x4 icon grid):
  Human Edit | Twilight
  AutoFill | AutoRemove
  Grass | Staging
  Lot Lines | Blur (teal when active)

Basic Adjustments tab:
  Crop & Tilt | Pin
  Lot Lines | Blur
  [Blur active state: Size slider + L M H buttons + Clear + Apply]
  Brightness (numeric display)
  Contrast
  White Balance
  Vibrance
  Sharpness
  Shadows
  Highlights
  [Reset (dark)] [Apply (teal)]

Main area: large photo preview
Top right: x close, flag icon
Bottom: horizontal filmstrip (80px thumbnails, teal border active, touch-scrollable)

/account - Style preferences
- Interior / Exterior tab toggle
- Per-tab sliders (-2 to +2): Brightness, Contrast, Vibrance, White Balance
- Right column:
  - Window Pull Intensity dropdown (Natural, Strong, Subtle)
  - Cloud Style dropdown (Fluffy + Whispy, Dramatic, Clear)
  - Twilight Style dropdown (Pink, Orange, Blue)
  - TV Screen Replacement dropdown (None, Fireplace, Sports, News, Custom Upload)
  - Toggles: TV Screen Blackout on, Fire in Fireplace on, Interior Clouds,
    Exterior Clouds on, Deduplication, Walkthrough Reorder [gear sub-icon]
- Preview image above settings panel
- Add Human-in-the-loop section:
  - "Go to queue" button (teal)
  - "Add editor's email address" input
  - "Add Editor" button (teal)
  - "No editors added yet" state

/pricing - Three plans
Free: 10/mo, $0 | Professional (most popular): 1,000/mo, $440/mo ($0.44),
  slider for more, credits rollover | Enterprise: 5,000/mo, $2,025/mo ($0.41),
  slider for more, credits rollover + Auto TV Blackout + Auto Add Fire +
  Walkthrough Re-ordering + Dedicated Slack
Monthly/Yearly toggle (20% off badge)
Top Up Credits button

9.3 DATABASE SCHEMA

model Project {
  id          String   @id @default(cuid())
  name        String
  address     String?
  modelId     String   @default("classic")
  shootType   String   @default("hdr")  // "hdr" | "flambient"
  status      String   @default("active")  // active | deleted
  settings    Json     // sky, perspective, grass, declutter toggles
  userId      String
  createdAt   DateTime @default(now())
  photos      Photo[]
}

model Photo {
  id              String   @id @default(cuid())
  projectId       String
  project         Project  @relation(fields: [projectId], references: [id])
  originalUrl     String
  processedUrl    String?
  thumbnailUrl    String?
  status          String   @default("pending")
  bracketGroup    String?  // groups bracketed shots together
  shootType       String?  // override per-photo if needed
  editingJobs     EditingJob[]
  createdAt       DateTime @default(now())
}

model EditingJob {
  id              String   @id @default(cuid())
  photoId         String
  photo           Photo    @relation(fields: [photoId], references: [id])
  toolName        String
  replicateJobId  String?
  params          Json?
  status          String   @default("pending")
  startedAt       DateTime?
  completedAt     DateTime?
  error           String?
}

model StylePreference {
  id        String @id @default(cuid())
  userId    String @unique
  interior  Json   // brightness, contrast, vibrance, whiteBalance defaults
  exterior  Json
  toggles   Json   // tvBlackout, fireFill, exteriorClouds, etc.
}

9.4 PROCESSING PIPELINE

HDR path:
1. align_image_stack (Hugin CLI)
2. enfuse OR OpenCV MergeMertens
3. GroundingDINO + SAM 2 for window mask
4. Blend dark bracket into window area with feathered mask
5. Apply interior vs exterior white balance
6. Perspective correction (OpenCV vertical line correction)
7. Lens profile correction if EXIF available
8. Tone curves + color profile for selected model (Classic/Lisa/Twilight Pink)
9. Optional generative fills (sky, grass, TV, fire) - async, not in critical path
10. Real-ESRGAN 2x final polish (optional, not on every image)
11. QA score (brightness distribution, contrast, vertical line check)
12. Export JPEG, upload to Cloudinary

Flambient path (NEW):
1. Separate ambient base from flash shot(s)
2. Layer ambient as base
3. Use flash layer for color accuracy (luminosity/color blend mode)
4. Apply Darken blend for window regions (flash window shot)
5. Mask blend transitions with feathered brushes
6. Then continue from step 4 of HDR path above

Bracket detection:
- Group by EXIF timestamp (within 5 seconds = same scene)
- OR group by filename suffix (_1, _2, _3 or -2, 0, +2 in name)
- Detect flambient: look for EXIF flash=fired on one bracket
  while others have flash=off

9.5 UI DESIGN SYSTEM
Background: #000000
Cards: #111111 / #1a1a1a
Accent/CTA: #00d4b4 (teal)
Text primary: #ffffff, secondary: #888888
Sliders: dark track, teal active fill, white thumb dot
Selected photos: teal checkbox + teal border overlay
Deleted sidebar items: #333 gray background, gray "Deleted" badge
Active sidebar item: #000 black with white bold text
Filmstrip: 80px thumbnails, 4px teal border on active photo
All buttons: rounded corners
Apply button: teal #00d4b4
Reset button: dark gray

9.6 ENVIRONMENT VARIABLES
DATABASE_URL=
CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=
REPLICATE_API_TOKEN=
BETTER_AUTH_SECRET=
ADMIN_PIN=
DROPBOX_APP_KEY=roi9y5o4gysmxev
OPENAI_API_KEY=
ADOBE_API_KEY=
ADOBE_ACCESS_TOKEN=
GEMINI_API_KEY=

---

VERIFICATION TASKS FOR CLAUDE CODE

Before generating any code, verify these live:

1. Navigate to autohdr.com/pricing - confirm current plan names, prices,
   photo counts. Report any differences from Section 1.2 above.

2. Navigate to autohdr.com - confirm the upload modal still shows Classic,
   Lisa, and Twilight Pink models. Report any new models.

3. Search for AutoHDR flambient support - confirm it is still a publicly
   supported workflow (check autohdr.com blog).

4. Search for current Replicate pricing for:
   meta/sam-2, nightmareai/real-esrgan, megvii-research/nafnet

5. Search for current Autoenhance.ai V4 blog post on window pull neural
   network - get the URL and confirm it is still live.

6. Search for current Fotello pricing and feature set.

7. Verify Nano Banana 2 = Gemini 3.1 Flash Image Preview and
   Nano Banana Pro = Gemini 3 Pro Image Preview on Higgsfield's site
   or Google's model naming.

8. Verify OpenAI GPT Image 2 API pricing and current limits.

9. Verify Adobe Firefly Generative Fill API is accessible without
   enterprise sales cycle for small developers.

10. Check Hugging Face for any new real estate + HDR datasets uploaded
    since August 2025.

Report all verification results BEFORE generating any code.
After verification, output the corrected spec, then ask: "Ready to build Phase 1?"

Show diffs before deploying.
