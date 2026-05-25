# VG Photo Editor — Master Claude Code Prompt
# Internal tool for Vineyard Gardens / WTPS, Martha's Vineyard
# Generated: 2026-05-24 | Synthesized from: live AutoHDR navigation,
# YouTube walkthroughs, ChatGPT research, NotebookLM (570 sources)

---

## CONTEXT

This is a personal internal tool for Vineyard Gardens and WTPS — not a
commercial SaaS product. It processes photos from Alan's own client shoots.
No resale. No distribution. This context matters for tool and dataset choices.

---

## TASK FOR CLAUDE CODE

Do not start building yet. First verify every item marked [VERIFY] against
live sources. Produce a brief verification report. Then ask: "Ready to build
Phase 1?"

---

## PART A: AUTOHDR PRODUCT SPEC
### (Verified from live navigation of autohdr.com + YouTube walkthroughs)

### A1. What AutoHDR is

Not just an HDR merger. A full real estate photo editing platform.
Supports both HDR (bracketed exposures) and Flambient (flash + ambient).
Delivers results under 5 minutes.
Integrates with Dropbox, Spiro, HDPhotoHub, Tonomo, Aryeo.

### A2. Pricing [VERIFY against autohdr.com/pricing]

- Free: 10 photos/month, $0
- Professional (most popular): 1,000 photos/month, $440/month, $0.44/photo
- Enterprise: 5,000 photos/month, $2,025/month, $0.41/photo
- Enterprise adds: credits rollover, Auto TV Blackout, Auto Add Fire,
  Walkthrough Re-ordering, Dedicated Slack Channel
- Yearly billing: 20% discount
- Volume slider on paid plans to adjust photo count

### A3. AI Models (confirmed from upload modal)

- Classic: "Natural, beautiful look for interiors" (default)
- Lisa: "Extra pop and deep window pulls"
- Twilight Pink: "For photos shot at dusk"

### A4. Editing Settings (gear icon in project modal)

- Sky Replacement — on by default
- Perspective Correction — on by default
- Grass Replacement — off by default
- Declutter — off by default

### A5. V3 Per-Photo Editing Panel (two tabs) [VERIFY still current]

**AI Tools tab** (2x4 icon grid):

```
Human Edit    |  Twilight
AutoFill      |  AutoRemove
Grass         |  Staging
Lot Lines     |  Blur (teal when active)
```

**Basic Adjustments tab:**

```
Crop & Tilt   |  Pin
Lot Lines     |  Blur
```

Blur active state: Size slider + L / M / H strength buttons + Clear + Apply

Sliders (all show live numeric values):
- Brightness
- Contrast
- White Balance
- Vibrance
- Sharpness
- Shadows
- Highlights

Buttons: Reset (dark gray) + Apply (teal)

Bottom: horizontal filmstrip of all project photos
- 80px thumbnails
- 4px teal border on active photo
- Touch-scrollable
- Keyboard arrow navigation

Top right: x close, flag/report icon, Show Video toggle

### A6. Listings Dashboard (autohdr.com/listings)

Left sidebar:
- "Search listings..." input
- Project list: active = dark/black card, deleted = gray + "Deleted" badge
- Each project has pencil rename icon

Top toolbar (default):
- Copy Shareable Link | Rate | Select All (N) | Download dropdown

Top toolbar (with photos selected):
- Supercharge | Delete | Copy Shareable Link | Rate | Deselect All (N) | Download

Supercharge = re-process at higher quality settings (double Replicate steps)

Balance display: "Balance: N Edits"
Main area: 4-column photo grid, checkboxes top-left on each photo

### A7. Style Settings Page (autohdr.com/account) [VERIFY still current]

Interior / Exterior tab toggle (teal = active)

Per-tab sliders (-2 to +2):
- Brightness
- Contrast
- Vibrance
- White Balance

Right column:
- Window Pull Intensity dropdown: Natural / Strong / Subtle [VERIFY all options]
- Cloud Style dropdown: "Fluffy + Whispy" / Dramatic / Clear [VERIFY]
- Twilight Style dropdown: Pink [VERIFY other options]
- TV Screen Replacement dropdown: None / others unknown [VERIFY]
- TV Screen Blackout toggle — on by default
- Fire in Fireplace toggle — on by default
- Interior Clouds toggle — off by default
- Exterior Clouds toggle — on by default
- Deduplication toggle — off by default
- Walkthrough Reorder toggle — with gear sub-icon

Preview image above settings panel showing live style example.

### A8. Human-in-the-Loop Panel

- "Add Human-in-the-loop" heading
- "Special requests? Integrate an editor in a click!"
- "Go to queue" button (teal)
- "Add editor's email address" input
- "Add Editor" button (teal)
- "No editors added yet" empty state

### A9. Walkthrough Reorder

Compiles project photos into a short video walkthrough (~12 seconds).
Enterprise-only feature. Useful reference for Phase 4.

### A10. Upload Flow

Supports JPEG, PNG, RAW, DNG, NEF, ARW, CR3, CR2, and many others
(confirmed from HTML file input accept attribute on live site).

Dropbox SDK loaded via script tag (app key: roi9y5o4gysmxev).
Google Drive import icon also present.

After upload: "Name Your Project" modal appears with:
- Project name input
- Shoot type (our addition — see Section C2)
- Model selector: "VGEditor Classic >"
- Gear icon -> Settings dropdown
- Continue button (teal)

---

## PART B: TECHNICAL PIPELINE
### (Verified product behavior + labeled inferences)

### B1. AutoHDR Image Grouping [VERIFIED — AutoHDR Hackathon, Codabench May 2026]

AutoHDR processes uploads with randomized UUID filenames. A public competition
at codabench.org/competitions/15365/ confirmed this is a hard problem they
crowdsourced: grouping randomized bracket uploads into sets by camera angle
using structural visual feature clustering.

Your bracket detection logic:
1. Parse EXIF from all uploaded files
2. Group by capture timestamp (within 5 seconds = same scene/bracket set)
3. Within each group, varying shutter speeds + same focal length = HDR set
4. One file with EXIF flash=fired + others with flash=off = Flambient set
5. Sort by EV value: dark (-2), base (0), bright (+2)

### B2. HDR Exposure Fusion [INFERRED — Mertens is the standard basis]

Classical Mertens Exposure Fusion (Mertens-Kautz-Van Reeth 2007):
- Does NOT create a 32-bit HDR file
- Directly blends best-exposed pixels from each bracket
- Weight per pixel = Contrast^wc * Saturation^ws * Exposedness^we
- Exposedness: exp(-((I - 0.5)^2) / (2 * 0.04))  [sigma = 0.2]
- Multi-resolution Laplacian pyramid prevents seam artifacts

OpenCV MergeMertens implements this. Enfuse CLI is equivalent.

Modern CNN improvement: U-Net or NAFNet trained on before/after pairs
learns better local contrast, fewer halos, cleaner window regions.

### B3. Flambient Processing [CONFIRMED AutoHDR supports it; INFERRED implementation]

Flambient = Flash + Ambient. This is a DIFFERENT process from HDR.
Computational complexity: EXTREMELY HIGH vs standard HDR.

Shoot components:
- Ambient brackets: 3+ natural light exposures for depth and tone
- Flash shot: off-camera bounced flash for accurate colors and window control

Blend steps (different from exposure fusion):
1. Layer ambient base exposure
2. Apply flash frame with luminosity/color blend mode (color accuracy layer)
3. Use Darken blend mode for window regions from flash shot (50/50 opacity)
4. Feathered mask transitions between layers
5. Then proceed to perspective + white balance + color profile steps

IMPORTANT: The upload flow must ask users to specify HDR or Flambient.
The merging code path is completely different for each.

### B4. Window Pull [CONFIRMED feature; INFERRED implementation]

Autoenhance.ai V4 blog describes their approach: "custom neural network
trained on vast dataset of real estate images including window pulls."

DOMAIN SHIFT WARNING: DeepLabV3+ models pre-trained on Cityscapes
(street-view, outside looking at windows) achieve only 0.53 mIoU on indoor
windows. Cannot use street-view models without fine-tuning on interior data.

Best starting stack (no training required):

Step 1 — GroundingDINO
- Text prompt: "window pane, glass door, sliding door, skylight"
- Returns bounding boxes

Step 2 — SAM 2
- Replicate model: "idea-research/grounded-sam-2"
- Input: image URL + bounding boxes from GroundingDINO
- Output: pixel-perfect binary mask

Step 3 — OpenCV post-processing
- Morphological operations to clean mask edges
- Gaussian feathering for smooth blend

Step 4 — Blend
- Blend darkest bracket (HDR) or flash frame (Flambient) into masked window area

### B5. White Balance [INFERRED from product behavior]

AutoHDR does NOT use a global temperature slider.
Uses spatially localized color corrections via neural white balance networks.
Interior warm sources: 2700K-3000K
Exterior daylight: 5500K-6500K
AI handles mixed-illuminant dynamics per region to prevent yellow ceilings
and blue walls.

### B6. Perspective Correction [INFERRED; confirmed as product feature]

Python OpenCV pipeline:
1. LSD or Hough transform to detect near-vertical line segments
2. Estimate vanishing point
3. Compute correction homography
4. cv2.warpPerspective(image, H, (w, h))
5. Crop or expand edges

Lens distortion:
- Read EXIF camera/lens metadata via ExifTool
- Apply Lensfun database corrections if camera/lens found

### B7. Generative Fills [CONFIRMED features; INFERRED models]

Classical fills (do first — fast, cheap, no AI required):
- TV blackout: detect screen rectangle, fill dark gray with subtle reflection
- Grass greening: HSL shift over lawn segmentation mask
- Sky replacement: sky segmentation + stock sky library

Generative fills (for hard cases only — not every image):
- Sky: Nano Banana Pro Inpaint with sky mask + prompt
- Grass: generative only if lawn is patchy after HSL shift
- Fire: fire asset paste first, generative if needed
- TV: classical blackout first, generative only for complex scenarios
- Object removal: generative inpainting
- Virtual staging: separate workflow

DO NOT use generative fills in the critical processing path for every image.
Merge first, run QA, call generative only on flagged photos.

---

## PART C: THE APP TO BUILD

**App name:** VG Photo Editor
**Stack:** Next.js 14 App Router, TypeScript, vanilla CSS, PostgreSQL/Prisma v5,
Better-Auth PIN login, Cloudinary, Replicate.com, Fly.io (backend), Vercel (frontend)
**Repo subfolder:** vineyard-gardens/vg-photo-editor/
**PowerShell note:** semicolons not &&
**Deploy:** npx vercel --prod (frontend); npx prisma generate then flyctl deploy --remote-only (backend)

### C1. Navigation

```
Logo | Pricing | Studio | Listings | Contact | Upload | [Avatar] [N Photos]
```

### C2. Pages

---

**`/` — Upload homepage**

Giant ghost wordmark "VG PHOTO EDITOR" behind centered upload zone.

Upload zone:
- Drag and drop target
- "Browse Files" button
- Dropbox icon (SDK via script tag, app key: roi9y5o4gysmxev)
- Google Drive icon

Below upload zone: "Book a Demo" button + "Get up to a month of free credits"

After file selection: "Name Your Project" modal with:
- Project name text input
- **Shoot type selector: HDR Brackets | Flambient** (new — required)
- Model selector showing current: "VGEditor Classic >"
  - Classic (natural, beautiful interiors)
  - Lisa (extra pop, deep window pulls)
  - Twilight Pink (dusk shots)
- Gear icon -> Settings dropdown:
  - Sky Replacement — toggle, on by default
  - Perspective Correction — toggle, on by default
  - Grass Replacement — toggle, off by default
  - Declutter — toggle, off by default
- Continue button (teal)

---

**`/listings` — Main dashboard**

Left sidebar:
- "Search listings..." input at top
- Project list:
  - Active projects: dark/black card, white bold text, pencil icon
  - Deleted projects: gray #333 card, "Deleted" badge, pencil icon

Balance display: "Balance: N Edits"

Top toolbar (default state):
- Copy Shareable Link | Rate | Select All (N) | Download dropdown

Top toolbar (photos selected state):
- Supercharge | Delete | Copy Shareable Link | Rate | Deselect All (N) | Download

Main area: responsive 4-column photo grid
- Each photo: checkbox top-left, checkmark overlay when selected
- Teal checkbox + teal border when selected

---

**`/listings/[projectId]` — Photo editor**

Left panel — TWO TABS:

Tab 1: AI Tools
```
[Human Edit]  [Twilight]
[AutoFill]    [AutoRemove]
[Grass]       [Staging]
[Lot Lines]   [Blur]
```
Blur button: teal highlight when active
Blur active state shows: Size slider + L M H strength buttons + Clear + Apply

Tab 2: Basic Adjustments
```
[Crop & Tilt] [Pin]
[Lot Lines]   [Blur]
```

Seven sliders (all with live numeric values):
- Brightness
- Contrast
- White Balance
- Vibrance
- Sharpness
- Shadows
- Highlights

Buttons: Reset (dark gray) | Apply (teal)

Main area: large photo preview
Top right: x close | flag icon
Bottom: horizontal filmstrip
- 80px thumbnails
- 4px teal border on active photo
- Touch-scrollable
- Keyboard arrow navigation

---

**`/account` — Style preferences**

Interior / Exterior tab toggle (teal = active)

Per-tab sliders (-2 to +2):
- Brightness
- Contrast
- Vibrance
- White Balance

Right column:
- Window Pull Intensity dropdown (Natural / Strong / Subtle)
- Cloud Style dropdown (Fluffy + Whispy / Dramatic / Clear)
- Twilight Style dropdown (Pink / Orange / Blue)
- TV Screen Replacement dropdown (None / Fireplace / Sports / News / Custom Upload)
- TV Screen Blackout toggle — on by default
- Fire in Fireplace toggle — on by default
- Interior Clouds toggle — off by default
- Exterior Clouds toggle — on by default
- Deduplication toggle — off by default
- Walkthrough Reorder toggle — with gear sub-icon

Preview image above settings showing live style example.

Add Human-in-the-loop section:
- "Go to queue" button (teal, full width)
- "Add editor's email address" input
- "Add Editor" button (teal)
- "No editors added yet" empty state

---

**`/pricing` — Three plans**

Monthly / Yearly toggle (20% off badge on yearly)

| Plan | Photos/mo | Price | Per photo |
|------|-----------|-------|-----------|
| Free | 10 | $0 | — |
| Professional (badge: Most popular) | 1,000 | $440/mo | $0.44 |
| Enterprise | 5,000 | $2,025/mo | $0.41 |

Paid plans: volume slider ("Slide for more photos")
Professional features: credits rollover
Enterprise features: credits rollover + Auto TV Blackout + Auto Add Fire +
  Walkthrough Re-ordering + Dedicated Slack Channel

Top Up Credits button (teal, top right area)

---

### C3. Database Schema

```prisma
model Project {
  id          String   @id @default(cuid())
  name        String
  address     String?
  modelId     String   @default("classic")
  shootType   String   @default("hdr")    // "hdr" | "flambient"
  status      String   @default("active") // "active" | "deleted"
  settings    Json     // { sky, perspective, grass, declutter booleans }
  userId      String
  createdAt   DateTime @default(now())
  photos      Photo[]
}

model Photo {
  id              String      @id @default(cuid())
  projectId       String
  project         Project     @relation(fields: [projectId], references: [id])
  originalUrl     String
  processedUrl    String?
  thumbnailUrl    String?
  status          String      @default("pending")
  bracketGroup    String?     // groups bracketed shots by scene
  shootType       String?     // per-photo override if needed
  editsApplied    Json?       // audit trail: which tools ran on this photo
  editingJobs     EditingJob[]
  createdAt       DateTime    @default(now())
}

model EditingJob {
  id              String    @id @default(cuid())
  photoId         String
  photo           Photo     @relation(fields: [photoId], references: [id])
  toolName        String
  replicateJobId  String?
  params          Json?
  status          String    @default("pending")
  startedAt       DateTime?
  completedAt     DateTime?
  error           String?
}

model StylePreference {
  id        String @id @default(cuid())
  userId    String @unique
  interior  Json   // { brightness, contrast, vibrance, whiteBalance }
  exterior  Json   // { brightness, contrast, vibrance, whiteBalance }
  toggles   Json   // { tvBlackout, fireFill, exteriorClouds, deduplication, etc. }
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
}
```

### C4. Processing Pipeline

**HDR path:**
1. align_image_stack (Hugin CLI)
2. enfuse OR OpenCV createMergeMertens()
3. Grounded SAM 2 for window mask
4. Blend darkest bracket into window area with feathered mask
5. Interior vs exterior white balance (separate regions)
6. Perspective correction (OpenCV Python microservice)
7. Lens profile correction via Lensfun if EXIF available
8. Tone curves + color profile (Classic / Lisa / Twilight Pink)
9. QA score (brightness distribution, contrast, vertical line check)
10. Optional generative fills (sky/grass/TV/fire) — async, NOT in critical path
11. Optional Real-ESRGAN 2x final polish — NOT on every image
12. JPEG export -> Cloudinary processed storage
13. Job status: complete

**Flambient path:**
1. Separate ambient base frames from flash frame (by EXIF flash field)
2. Layer ambient base exposure
3. Apply flash frame with luminosity/color blend mode (color accuracy)
4. Use Darken blend for window regions from flash shot
5. Feathered mask transitions
6. Continue from step 5 of HDR path above

**Bracket auto-detection:**
- Group by EXIF timestamp (within 5 seconds)
- OR by filename suffix pattern (-2, 0, +2 or _1, _2, _3)
- Flambient detection: one frame with flash=fired, others flash=off

### C5. UI Design System

```
Background:       #000000
Cards:            #111111 / #1a1a1a
Accent/CTA:       #00d4b4 (teal)
Text primary:     #ffffff
Text secondary:   #888888
Slider track:     dark gray
Slider fill:      teal #00d4b4
Slider thumb:     white dot
Selected photos:  teal checkbox + teal border overlay
Active sidebar:   #000 background, white bold text
Deleted sidebar:  #333 gray, gray "Deleted" badge
Filmstrip active: 4px teal border
All buttons:      rounded corners
Apply button:     #00d4b4 teal
Reset button:     dark gray
```

### C6. Environment Variables

```
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
HIGGSFIELD_API_KEY=
```

---

## PART D: AI MODELS AND APIs

### D1. Exposure Fusion — Classical First

**Enfuse (free, run this first):**
```javascript
// Node.js worker — Enfuse binary via child_process
import { spawn } from 'child_process';
import path from 'path';

async function runEnfuse(bracketPaths, outputFilename) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join('/tmp', outputFilename);
    const args = ['-o', outputPath, ...bracketPaths];
    const proc = spawn('enfuse', args);
    proc.on('close', (code) => {
      if (code === 0) resolve(outputPath);
      else reject(new Error(`Enfuse failed: exit ${code}`));
    });
  });
}

// Fly.io Dockerfile: RUN apt-get update && apt-get install -y enfuse hugin
// Sharp.js does NOT do exposure fusion — must use Enfuse binary
```

**OpenCV MergeMertens (Python microservice alternative):**
```python
import cv2
import numpy as np

def merge_brackets(image_paths):
    images = [cv2.imread(p) for p in image_paths]
    merge_mertens = cv2.createMergeMertens()
    fusion = merge_mertens.process(images)
    fusion_8bit = np.clip(fusion * 255, 0, 255).astype('uint8')
    return fusion_8bit
```

### D2. Bracket Alignment

```javascript
// align_image_stack from Hugin before Enfuse
import { spawn } from 'child_process';

async function alignBrackets(inputPaths, outputPrefix) {
  return new Promise((resolve, reject) => {
    const args = ['-a', outputPrefix, ...inputPaths];
    const proc = spawn('align_image_stack', args);
    proc.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`align_image_stack failed: exit ${code}`));
    });
  });
}
```

### D3. Grounded SAM 2 — Window Segmentation

Replicate model: `idea-research/grounded-sam-2`
Cost: ~$0.034/call on T4

```javascript
import Replicate from "replicate";

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

async function getWindowMask(imageUrl) {
  const output = await replicate.run(
    "idea-research/grounded-sam-2:latest",
    {
      input: {
        image: imageUrl,
        text_prompt: "window pane",
        box_threshold: 0.25,
        text_threshold: 0.25
      }
    }
  );
  // Returns annotated image + binary mask URL
  return output.mask_url;
}
```

### D4. Nano Banana Pro Inpaint — Primary Generative Fill

Access via Higgsfield Design MCP (already connected to this account).
This is the preferred tool for all generative fills.

Models:
- Nano Banana Pro = Gemini 3 Pro Image (quality, precision)
- Nano Banana 2 = Gemini 3.1 Flash Image (speed, volume)

Nano Banana Pro Inpaint: mask + prompt editing system
- Protects everything outside mask at pixel level
- 4-step reasoning: shape, edges, geometry, texture
- Structure-preserving: architecture is not distorted
- Cost: ~$0.05-0.06/generation on Creator plan

Use for:
- Sky replacement: mask sky, prompt "natural clear blue sky, keep roofline exact"
- Grass: mask lawn, prompt "lush green grass, same lighting as scene"
- TV blackout: mask screen, prompt "dark black screen with subtle reflection"
- Fire: mask fireplace, prompt "warm realistic fire burning in fireplace"
- Object removal: mask object, prompt "clean floor/wall matching surroundings"
- Staging: mask empty room, prompt "tastefully furnished living room, natural light"

[VERIFY exact MCP tool parameter names for inpainting mask input via Higgsfield
Design MCP generate_image tool]

### D5. GPT Image 2 — Backup Generative Fill

Endpoint: `https://api.openai.com/v1/images/edits`
Model: `gpt-image-2`
Cost: ~$0.04 per 1024x1024

```javascript
import FormData from 'form-data';
import fs from 'fs';
import fetch from 'node-fetch';

async function runGPTInpainting(imagePath, maskPath, prompt) {
  // Image and mask: square PNG, same dimensions, under 4MB
  // Mask format: PNG with alpha channel
  //   - Transparent (alpha=0) = edit this area
  //   - Opaque (alpha=255) = protect this area
  const formData = new FormData();
  formData.append('image', fs.createReadStream(imagePath));
  formData.append('mask', fs.createReadStream(maskPath));
  formData.append('prompt', prompt);
  formData.append('n', 1);
  formData.append('size', '1024x1024');
  formData.append('model', 'gpt-image-2');

  const response = await fetch('https://api.openai.com/v1/images/edits', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
      ...formData.getHeaders()
    },
    body: formData
  });

  const data = await response.json();
  return data.data[0].url;
}
```

### D6. Adobe Tools (Personal CC Subscription)

Since this is an internal tool, Adobe CC tools are available freely:
- Lightroom HDR Merge (Photo Merge to HDR DNG) — use for testing/benchmarking
- Photoshop Generative Fill — use for complex manual fills
- Firefly in-app — available within Photoshop/Lightroom

For programmatic/API use, Adobe Firefly Services requires separate enterprise
access. [VERIFY whether individual developer API access is available]

### D7. Real-ESRGAN — Final Polish (optional)

Model: `nightmareai/real-esrgan`
Hash: `42fd3caa8788a20c7edab74eb74718301540f2824ecfce8276f756ea3c4ba7e4`
Cost: ~$0.034/call

```javascript
import Replicate from "replicate";

const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

async function upscaleImage(imageUrl) {
  const output = await replicate.run(
    "nightmareai/real-esrgan:42fd3caa8788a20c7edab74eb74718301540f2824ecfce8276f756ea3c4ba7e4",
    {
      input: {
        image: imageUrl,
        scale: 2,
        face_enhance: false  // Always false for real estate
      }
    }
  );
  return output;
}
// Use sparingly — overuse causes plastic/oversharpened look
```

### D8. NAFNet — Restoration/Polish Upgrade

Replicate: `replicate.com/megvii-research/nafnet`
GitHub: `github.com/megvii-research/NAFNet`
Use for: denoise, deblur, artifact cleanup after fusion
NOT a primary HDR merger — needs fine-tuning for real estate use

### D9. LightFuse — CNN Fusion Upgrade (Phase 3)

GitHub: `github.com/Taichi-Pink/LightFuse-Lightweight-CNN-based-Dual-exposure-Fusion`
GlobalNet (illumination) + DetailNet (local texture)
Designed for 2 exposures — preselect dark + bright bracket as input pair
Fine-tune on your own bracket archive for real estate style

---

## PART E: TRAINING DATA

### E1. For Internal Personal Use (not for sale)

Since this is personal internal use, you can freely use:

**Good starting datasets:**
- Your own archive: RAW brackets from Martha's Vineyard shoots + Lightroom finals
  (best training data — already your style, legally clean)
- MIT-Adobe FiveK: 5,000 RAW/TIFF pairs — fine for internal experiments
  URL: data.csail.mit.edu/graphics/fivek/
- SICE: 589 multi-exposure sequences — fine for local testing
  URL: github.com/csjcai/SICE
- S2R-HDR: synthetic HDR dataset, permissive
  URL: huggingface.co/datasets/iimmortall/S2R-HDR
- BracketIRE: bracketed exposure research dataset
  URL: arxiv.org/html/2401.00766v2

**Commercial-licensed options (if you ever want to go further):**
- Chimnie.com (chimnie.com/ai): UK property images licensed for AI training
- Shutterstock bulk ML licensing

**Always avoid:**
- MLS listing photos you don't own (VHT v. Zillow — copyright regardless of use)
- Scraping Zillow, Redfin, or any listing portal

### E2. Dataset Folder Structure

```
dataset/
  property_0001/
    room_001/
      shoot_type: "hdr" | "flambient"
      brackets/
        IMG_0001_-2.CR3     <- dark exposure
        IMG_0002_0.CR3      <- base/ambient
        IMG_0003_+2.CR3     <- bright exposure
        IMG_0004_flash.CR3  <- flash only (flambient)
      final/
        IMG_0002_final.jpg
      masks/
        window.png
        sky.png
        grass.png
        tv.png
        fireplace.png
      meta.json
```

```json
{
  "property_id": "property_0001",
  "room_type": "living_room",
  "shoot_type": "hdr",
  "bracket_count": 3,
  "ev_spacing": 2,
  "camera": "Canon R6",
  "lens": "RF 15-35mm",
  "flash_used": false,
  "editor": "alan",
  "style": "natural_real_estate",
  "notes": "ocean view, Edgartown, Martha's Vineyard"
}
```

### E3. Target Dataset Sizes

- Phase 1: 0 training pairs needed (off-the-shelf APIs only)
- Phase 2: 400-500 pairs (validate generative prompt effectiveness)
- Phase 3: 20,000 pairs minimum for fine-tuned NAFNet

---

## PART F: PHASED BUILD PLAN

### Phase 0 — Week 1-2: Data and Benchmark

1. Export 500+ bracket sets from own MV archive to dataset/ structure above
2. Apply meta.json to every property
3. Run 50 test properties through AutoHDR, Fotello, and Lightroom HDR Merge
4. Score each on: brightness, window pull quality, color accuracy,
   perspective, sky, grass, artifacts
5. Establish benchmark baseline before writing a single line of app code

Do NOT start Phase 1 until benchmark scores exist.

### Phase 1 — Week 3-6: Classical Pipeline

Target: ~$0.05/image

Infrastructure:
```
Next.js upload
  -> Cloudinary original storage
  -> Job created in PostgreSQL (status: pending)
  -> Fly.io Python worker polls queue
    -> EXIF parse + bracket grouping
    -> Shoot type detection (HDR or Flambient)
    -> IF HDR: align_image_stack + enfuse/MergeMertens
    -> IF Flambient: layer-based composite
    -> Grounded SAM 2 window mask
    -> Blend darkest bracket into window area
    -> Perspective correction (OpenCV)
    -> Interior vs exterior white balance
    -> Tone curves + model color profile
    -> QA scoring
    -> JPEG export -> Cloudinary processed
  -> Job status: complete
  -> Frontend polls every 10 seconds
```

Expected quality at Phase 1 end:
- 60-75% of AutoHDR quality on standard interiors
- 40-60% on difficult high-contrast windows
- 80%+ on exterior sky/grass with simple masking

### Phase 2 — Week 7-10: Generative Fills

Target: ~$0.12-0.22/image

Add generative fills only for flagged photos, not every image:
- Nano Banana Pro Inpaint as primary (via Higgsfield MCP)
- GPT Image 2 as fallback
- Sky: generative only when classical fails
- Grass: HSL shift first, generative only if patchy
- TV: classical blackout first (rectangle detection + dark fill)
- Fire: paste fire asset first, generative if needed
- Object removal: Nano Banana Pro Inpaint + "remove-anything" on Replicate

Expected quality at Phase 2 end: 85-95% of AutoHDR quality

### Phase 3 — Month 3+: Fine-Tune Your Own Model

Target: ~$0.05/image (mostly local compute)

Fine-tune order:
1. LightFuse — for learned exposure fusion (if Enfuse quality is insufficient)
2. SegFormer/DeepLabV3+ — window masks (if SAM 2 failing on unusual windows)
   Need 1,000+ annotated interior window masks
3. NAFNet — final style and quality consistency
   Need 20,000 before/after pairs

RTX 5070 Ti (16GB VRAM, CUDA 12.8, Blackwell sm_120) can handle:
- 256x256 patch training, batch size 16
- LoRA-style fine-tuning
- Small segmentation model fine-tuning
- NOT full-resolution end-to-end training (use RunPod for that)

Bootstrap Phase 3 training data:
Run your own shoots through the Phase 2 pipeline to generate polished
before/after pairs. Use those as NAFNet training inputs. This is how you
reach 20,000 pairs without spending $5k on human editors.

### Phase 4 — Month 6+: Module 2 (Landscape Visualizer)

Separate project (vineyard-gardens/vg-landscape-visualizer/)
Core tool: Nano Banana Pro Inpaint
Input: customer's existing yard photo + design style + text prompt
Output: 2-3 photorealistic landscaping variations
Use case: show WTPS/VG clients what their property could look like before
any work begins. No AutoHDR equivalent exists for this.

---

## PART G: COST AND BREAK-EVEN

### G1. Per-Image Cost by Phase

| Step | Tool | Cost |
|------|------|------|
| Bracket grouping | EXIF parse | $0.00 |
| Alignment | Hugin CLI (Fly.io) | ~$0.005 |
| Fusion | Enfuse (Fly.io CPU) | ~$0.005 |
| Window segmentation | Grounded SAM 2 | ~$0.034 |
| Generative fill (if needed) | Nano Banana Pro Inpaint | ~$0.05 |
| Final polish (optional) | Real-ESRGAN | ~$0.034 |
| Storage | Cloudinary | ~$0.002 |
| **Phase 1 total (no generative)** | | **~$0.05** |
| **Phase 2 total (with generative)** | | **~$0.12** |

### G2. Break-Even vs AutoHDR

AutoHDR actual price: $0.44/photo (Professional plan, verified live)
Your Phase 2 cost: ~$0.12/photo
Savings: $0.32/photo
Fixed monthly overhead (Fly.io + Cloudinary + DB): ~$500/month
Break-even: 500 / 0.32 = **~1,563 images/month** (~62 properties at 25 images)

At current VG/WTPS volume this is likely below break-even on pure cost.
The real case for building in-house:
1. No per-photo fees as volume scales with business growth
2. Proprietary workflow embedded in VG operations
3. Module 2 (landscape visualization) has no AutoHDR equivalent
4. Potential to white-label to other Martha's Vineyard photographers
   at $0.44/photo revenue vs $0.12/photo cost = $0.32/photo margin

---

## PART H: VERIFICATION TASKS

Run these before writing any code:

1. Navigate autohdr.com/pricing — confirm plan names, prices, photo counts.
   Report any differences from Section A2.

2. Navigate autohdr.com — confirm upload modal still shows Classic, Lisa,
   Twilight Pink models. Report any new models added.

3. Navigate autohdr.com/account — confirm Style Settings page layout matches
   Section A7. Capture all dropdown options for Cloud Style, Twilight Style,
   TV Screen Replacement.

4. [VERIFY] Replicate current pricing for:
   - idea-research/grounded-sam-2
   - nightmareai/real-esrgan
   - megvii-research/nafnet

5. [VERIFY] AutoHDR Flambient support — check autohdr.com blog or help docs.

6. [VERIFY] Higgsfield Nano Banana Pro Inpaint — confirm it supports mask +
   prompt editing and what the correct MCP tool parameters are for masking.

7. [VERIFY] Whether individual Adobe Firefly Services API access is available
   without enterprise contract for small-scale internal use.

8. [VERIFY] Current Replicate model ID for Grounded SAM 2 (hashes change
   frequently — get the current pinned version).

Report all verification results before generating any code.
After verification, output the corrected spec, then ask: "Ready to build Phase 1?"

---

## PART I: NOTES FOR CLAUDE CODE

- PowerShell uses semicolons not &&
- DB schema changes: add SQL to prisma/migrations/, run
  npx prisma generate then flyctl deploy --remote-only
- All sliders: debounced 300ms before triggering Apply
- Filmstrip: touch-scrollable, keyboard arrow navigation
- Processing is async: never block the UI on image jobs
- Supercharge = re-process at double quality (higher Replicate steps)
- Per-photo editsApplied JSON field tracks which tools ran (audit trail for MLS)
- The shoot type field (hdr/flambient) must propagate from Project to Photo
  to EditingJob so the correct pipeline is always selected

Show diffs before deploying.
```
