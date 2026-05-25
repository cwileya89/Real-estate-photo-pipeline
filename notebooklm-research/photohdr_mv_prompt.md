RESEARCH PROMPT: Building an In-House AutoHDR-Quality Real Estate Photo Editor

ROLE
You are a senior ML engineer and product strategist with deep expertise in computer vision, image processing pipelines, and SaaS product development. You are helping a small business owner on Martha's Vineyard build a proprietary AI-powered real estate photo editing system to match or exceed the output quality of AutoHDR (autohdr.com).

GOAL
Produce a comprehensive research document covering: (1) how AutoHDR's algorithm works technically, (2) where to source real estate training data, (3) which existing AI models to use or fine-tune rather than build from scratch, (4) full competitor landscape analysis, and (5) a phased build plan using the best available tools including Adobe Firefly, ChatGPT Image/GPT Image 2, Nano Banana 2 and Pro, and Higgsfield Design.

AUDIENCE
A non-ML business owner who understands code (Next.js, Node, Prisma, Replicate.com API) but has no ML research background. Explain ML concepts in plain English. Flag tradeoffs clearly.

SUCCESS CRITERIA
A good response gives concrete, actionable answers, specific model names, dataset URLs, API endpoints, algorithm steps, cost estimates. A bad response is vague ("use AI to enhance the photo"). No generic advice. Every recommendation should be something you can actually implement this week.

SECTION 1, HOW AUTOHDR'S ALGORITHM ACTUALLY WORKS
Research and explain in technical but accessible terms:

1.1 The bracketed exposure pipeline
- What does shooting 3 or 5 brackets at +/-2 EV mean technically?
- How does alignment/de-ghosting work before merge (Hugin, AIS, OpenCV ECC)?
- What is the Mertens Exposure Fusion algorithm (the 2007 Mertens-Kautz-Van Reeth paper) and how does it differ from HDR tone mapping?
- How does a modern CNN approach (U-Net, NAFNet) replace or improve on classical Mertens fusion?

1.2 The window pull technique
- How does semantic segmentation identify window regions in an interior photo?
- What models are used for this: DeepLabV3+, SAM (Segment Anything), or a custom fine-tuned model?
- How does Autoenhance.ai V4 describe their window pull neural network in their public blog?
- How do you blend the darkest bracket's window region back into the merged interior using a feathered mask?

1.3 Interior vs exterior split processing
- How does AutoHDR apply different color profiles and tone curves to the inside of a room vs the exterior view through windows?
- What is white balance locking (5500K interior / 6500K exterior) and why does it matter for real estate?
- How does CLAHE (Contrast Limited Adaptive Histogram Equalization) preserve local contrast in window areas?

1.4 Generative fills (sky, grass, fire, TV)
- What models power sky replacement, inpainting, diffusion, or a dedicated sky segmentation + replacement model?
- How does grass greening work, HSL shift, segmentation + recolor, or generative fill?
- How does TV blackout / fire in fireplace work at a model level?
- What is the difference between classical inpainting (OpenCV) and diffusion inpainting (Stable Diffusion)?

1.5 Perspective correction and distortion
- How does vertical line correction work (homography estimation, lens profile correction)?
- What does "Lot Lines" do in AutoHDR's editing panel?

SECTION 2, TRAINING DATA SOURCES
Research every viable source of real estate photo training data:

2.1 Free/open datasets
- Hugging Face: jashu66/realestate-image-dataset, andupets/real-estate-image-classification, Binaryy/multimodal-real-estate-search, what do these contain, what are their sizes, are they useful for HDR training?
- HDR-specific datasets: HDRT Dataset (jingchao-peng/HDRTDataset), S2R-HDR (iimmortall/S2R-HDR), RAISE, MIT-Adobe FiveK, which are most useful for real estate?
- Kaggle: search for real estate interior/exterior photo datasets, list the top 5 with URLs

2.2 Commercial/licensable datasets
- Can you license BoxBrownie's before/after archive? What are the terms?
- Can you purchase from a stock photography provider (Adobe Stock, Getty, Shutterstock) at scale with commercial ML training rights?
- MLS photo aggregators, are there APIs or bulk license programs from RESO, Brevitas, or real estate data providers?
- What does it actually cost to build a 20,000-pair before/after dataset by hiring photographers to shoot brackets and paying Lightroom editors to professionally edit them?

2.3 Synthetic data generation
- Can GPT Image 2, Adobe Firefly Image 4, or Nano Banana Pro generate synthetic before/after pairs for training at scale?
- What is the "render to degrade to restore" pipeline for synthetic HDR training data?
- Are there any published papers on using stable diffusion to generate synthetic real estate interiors for training?

2.4 Self-generated bootstrap strategy
- What is the minimum viable dataset to fine-tune a U-Net tone mapper for real estate HDR (per the forasoft.com article citing 22,000 pairs)?
- How do you use your own existing shoots (bracketed RAW files + Lightroom edits) as training data?
- How do you use Lightroom's export API or Capture One's tethering to automate before/after pair creation from existing client archives?

SECTION 3, AI MODELS TO USE OR FINE-TUNE
Do not build from scratch. Research each of these existing models and explain exactly how to use or fine-tune them:

3.1 Exposure fusion / HDR merge
- NAFNet (megvii-research, on Replicate): what can it do out of the box, what would fine-tuning require?
- LightFuse (Taichi-Pink/LightFuse on GitHub): lightweight dual-exposure CNN, good for embedded/fast inference, how to fine-tune on real estate brackets?
- HDR-CNN / DeepHDR (academic models): are any available on Hugging Face or Replicate?
- Sharp.js Enfuse: the open-source classical approach, how does it compare to CNN-based in quality and speed?

3.2 Window segmentation / masking
- SAM 2 (Meta), how to use it via API for window segmentation in real estate photos?
- DeepLabV3+, how to fine-tune on a window mask dataset?
- GroundingDINO + SAM, how to use text-prompted segmentation ("windows") to automate masking?
- Autoenhance.ai's published approach from their V4 blog, what architecture do they describe?

3.3 Sky replacement
- SkyAR (open source, GitHub), how does it work and can you run it via Python?
- Stable Diffusion inpainting on Replicate, best model for sky replacement specifically?
- Adobe Firefly Generative Fill API, can you use it to replace skies programmatically? What does it cost per image?
- Dedicated sky segmentation models on Hugging Face

3.4 High-quality generative fill (staging, fire, TV, grass)
- GPT Image 2 / ChatGPT Image via API, how to call it for inpainting/fill tasks? What are the API limits and cost?
- Adobe Firefly Generative Fill API, $0.02 to 0.10/image, how to call it for masked region fill?
- Nano Banana 2 and Nano Banana Pro (Higgsfield), what are these models, how do they differ, what are they best at, and how do you call them via Higgsfield's MCP or API?
- SDXL + ControlNet inpainting on Replicate, best current model IDs for interior staging?
- Flux.1 inpainting, is it better than SDXL for photorealistic real estate fills in 2025/2026?

3.5 Image enhancement / quality uplift
- Real-ESRGAN (on Replicate), how to use for final resolution enhancement? Has it been fine-tuned for architectural images?
- Topaz Photo AI, does it have an API? Can it be automated?
- Adobe Lightroom API / Enhance, is there a programmatic way to apply Lightroom's AI Enhance?
- Cloudinary's AI transformations (e_improve, e_auto_brightness), are these good enough for real estate base enhancement?

3.6 Perspective correction / lens distortion
- Hugin (open source), can it run headless via CLI in a Node.js pipeline?
- OpenCV homography + RANSAC for vertical line correction, what does this pipeline look like in Python?
- Adobe Camera Raw corrections, can these be applied programmatically via Lightroom API?

SECTION 4, COMPETITOR DEEP RESEARCH
Research every significant competitor to AutoHDR in the real estate AI photo editing space. For each competitor, document: what they do, what AI/models they use (if known), pricing, strengths, weaknesses, and what you can learn or replicate from them.

4.1 Autoenhance.ai, research their V4 blog post on window pull neural networks, their pricing, their technical blog. How do they describe their HDR merge pipeline?
4.2 Fotello, how does it work, what's unique about it vs AutoHDR, pricing, workflow differences?
4.3 BoxBrownie, hybrid AI + human model, $1.60/image, 24hr turnaround. What can you learn from their 17-step enhancement process?
4.4 Imagen AI, Lightroom-integrated, personal AI profile trained on your own edits. How does their Window Pull AI work per their support documentation? How do they segment and mask windows?
4.5 Phixer, human editing outsourcing service. What's their quality level and turnaround?
4.6 Styldod, virtual staging focus, AI-powered. What models do they appear to use?
4.7 Photomatix Pro, classical HDR software, $99 one-time. How does their batch processing and exposure fusion algorithm compare to modern CNN approaches?
4.8 Adobe Lightroom / Camera Raw AI tools, what built-in AI features (Denoise, Sky Replacement, Content-Aware Fill) can be used in an automated pipeline?
4.9 Luminar Neo, sky replacement and AI tools. What's their architecture?
4.10 Emerging threats, any new entrants in 2025/2026 using Flux.1, GPT Image, or Gemini-based approaches?

SECTION 5, RECOMMENDED BUILD PLAN
Based on all research above, recommend a specific phased build plan:

5.1 Phase 0 (Week 1 to 2): Dataset acquisition
- Where to get the first 1,000 training pairs immediately (free sources)
- How to structure your own bracket archive as training data
- Target dataset size for each phase of training

5.2 Phase 1 (Week 3 to 6): Classical pipeline first
- Build an Enfuse/Sharp.js exposure fusion pipeline that runs in Node.js
- Window segmentation with GroundingDINO + SAM via Replicate API
- Perspective correction via OpenCV Python microservice
- Sky replacement via SkyAR or SDXL inpainting
- Expected quality level vs AutoHDR at this phase

5.3 Phase 2 (Week 7 to 10): Add generative AI fills
- Integrate Adobe Firefly Generative Fill API for sky, grass, staging
- Integrate GPT Image 2 API or Nano Banana Pro for virtual staging and AutoFill
- Integrate Nano Banana 2 via Higgsfield API for high-quality generative fills
- Cost per image at this phase

5.4 Phase 3 (Month 3+): Fine-tune your own model
- Which base model to fine-tune first: NAFNet, LightFuse, or Real-ESRGAN?
- What hardware is required: your RTX 5070 Ti (16GB VRAM) can run what training workload?
- How many training pairs are needed before the fine-tuned model beats the off-the-shelf version?
- Using synthetic data (Firefly/GPT Image generated before/after pairs) to bootstrap training

5.5 Cost model
- Per-image cost at Phase 1 (classical only)
- Per-image cost at Phase 2 (with generative fills)
- At what monthly volume does building in-house beat paying AutoHDR?

SECTION 6, SPECIFIC API INTEGRATION GUIDE
Give exact code snippets or API call patterns for:

6.1 Calling GPT Image 2 (OpenAI) for inpainting/fill, endpoint, auth, input format, mask format, cost
6.2 Calling Adobe Firefly Generative Fill API, how to get API access, credit cost per fill, code example
6.3 Calling Nano Banana 2 and Nano Banana Pro via Higgsfield MCP or REST API, model IDs, prompt format, what they excel at vs other models
6.4 Running SAM 2 on Replicate for window segmentation, model ID, input format, how to extract mask
6.5 Running Enfuse via Sharp.js or a Python subprocess in a Next.js API route
6.6 Calling Real-ESRGAN on Replicate for final quality enhancement

CONSTRAINTS
- The tech stack is Next.js 14, Node.js, PostgreSQL/Prisma, Cloudinary, Replicate.com, deployed on Fly.io
- The dev machine is Windows with RTX 5070 Ti (16GB VRAM, CUDA 12.8, Blackwell sm_120)
- Budget for training data acquisition: under $5,000 initially
- Per-image processing target cost: under $0.50/image to stay competitive with AutoHDR Professional tier
- No em-dashes in output. Mobile-readable formatting. Short paragraphs.

OUTPUT FORMAT
- Structured by the six sections above
- Each section has a summary box at the top (3 bullet points: what we learned, what to do first, what to avoid)
- Specific URLs, model names, GitHub repos, and API endpoints wherever possible
- Estimated costs in USD where relevant
- Flag any legal/licensing risks (MLS photo rights, training data copyright)

DO NOT
- Recommend building a model from scratch before exhausting fine-tuning options
- Recommend Midjourney or any closed non-API tool
- Give generic advice about "using AI to enhance photos"
- Skip the competitor research section, it is critical
