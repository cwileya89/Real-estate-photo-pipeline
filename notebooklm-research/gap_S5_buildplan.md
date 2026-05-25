Produce Section 5 of a comprehensive research document on building an in-house AutoHDR-quality real estate photo editor.

Audience: A non-ML business owner on Martha's Vineyard who codes (Next.js, Node, Prisma, Replicate.com API) but has no ML background.

Constraints to honor:
- Tech stack: Next.js 14, Node.js, PostgreSQL/Prisma, Cloudinary, Replicate.com, deployed on Fly.io
- Dev machine: Windows with RTX 5070 Ti (16GB VRAM, CUDA 12.8, Blackwell sm_120)
- Budget for training data acquisition: under $5,000 initially
- Per-image processing target: under $0.50/image to stay competitive with AutoHDR Professional tier

This section is the RECOMMENDED PHASED BUILD PLAN. Be concrete and actionable. Cite imported sources with [N] inline citations.

5.1 Phase 0 (Week 1 to 2): Dataset acquisition
- Specific URLs to get the first 1,000 training pairs immediately from free sources
- How to structure the user's own bracket archive as training data (folder layout, naming convention, metadata extraction from EXIF, bracket grouping by capture time)
- Target dataset size for each phase

5.2 Phase 1 (Week 3 to 6): Classical pipeline first
- Build an Enfuse exposure fusion pipeline that runs in Node.js (Sharp.js or Python subprocess)
- Window segmentation with GroundingDINO + SAM 2 via Replicate API (specific model IDs)
- Perspective correction via OpenCV Python microservice (Fly.io worker)
- Sky replacement via SkyAR or SDXL inpainting on Replicate
- Expected quality level vs AutoHDR at this phase, what's missing

5.3 Phase 2 (Week 7 to 10): Add generative AI fills
- Integrate Adobe Firefly Generative Fill API (signup process, credit cost, code pattern)
- Integrate GPT Image 2 (OpenAI) for inpainting (cost per image, endpoint)
- Integrate Nano Banana 2 and Pro via Higgsfield Design API or MCP server (model IDs, prompt format)
- Per-image cost at this phase

5.4 Phase 3 (Month 3+): Fine-tune your own model
- Which base model to fine-tune first: NAFNet, LightFuse, or Real-ESRGAN. Justify the choice.
- What hardware is required: 16GB VRAM RTX 5070 Ti can run what training workload (batch size, image resolution, training time)
- How many training pairs needed before the fine-tuned model beats the off-the-shelf version
- Using synthetic data (Firefly/GPT Image generated before/after pairs) to bootstrap training

5.5 Cost model and break-even
- Per-image cost at Phase 1 (classical only) in USD, broken down by component
- Per-image cost at Phase 2 (with generative fills) in USD
- At what monthly image volume does building in-house beat paying AutoHDR ($1.20/image at Pro tier)
- Compute the break-even with specific numbers

Format requirements:
- Start with a 3-bullet summary box: what we learned, what to do first, what to avoid
- Short paragraphs, mobile-readable
- No em-dashes
- Specific URLs, model names, pricing in USD throughout
- End with a Gantt-style timeline table: Phase, Weeks, Key deliverable, Cost
