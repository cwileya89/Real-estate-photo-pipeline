Produce Section 6 of a comprehensive research document on building an in-house AutoHDR-quality real estate photo editor.

Audience: A non-ML business owner on Martha's Vineyard who codes (Next.js 14, Node.js, Prisma, Replicate.com API) but has no ML background.

This section is the SPECIFIC API INTEGRATION GUIDE. For each API below, give exact code snippets or API call patterns. Use Node.js / fetch / axios where appropriate, Python where the API only has Python SDK. Cite imported sources with [N] inline citations.

6.1 Calling GPT Image 2 (OpenAI) for inpainting/fill
- Exact endpoint URL
- Auth header format
- Input format: image, mask, prompt
- Mask format: PNG with alpha or RGB white/black, what is required
- Cost per image (token math)
- Working Node.js code example

6.2 Calling Adobe Firefly Generative Fill API
- How to get API access (enterprise vs creative cloud)
- Credit cost per fill
- Exact endpoint URL and request payload structure
- Working code example
- Output format and how to download the result

6.3 Calling Nano Banana 2 and Nano Banana Pro via Higgsfield Design API
- Model IDs for both
- Prompt format Higgsfield expects
- What each model excels at vs the other
- Working REST API call example
- Higgsfield MCP server option as alternative
- Cost per image

6.4 Running SAM 2 on Replicate for window segmentation
- Specific Replicate model ID for SAM 2
- Input format: image URL or base64
- How to provide a text prompt (e.g., "window") via GroundingDINO + SAM 2 combined model
- How to extract the mask from the API response
- Cost per call
- Working Node.js code example using Replicate SDK

6.5 Running Enfuse for exposure fusion in Node.js
- Sharp.js capability check: does Sharp do exposure fusion or is it just bracket merge
- Enfuse binary subprocess from Node.js (child_process.spawn) on Fly.io worker
- Where to get the Enfuse binary for Linux Docker image
- Working code example

6.6 Calling Real-ESRGAN on Replicate for final quality enhancement
- Replicate model ID
- Input parameters: scale, face_enhance, etc.
- Cost per call
- Working code example
- When to use this vs Topaz Photo AI

Format requirements:
- Start with a 3-bullet summary box: what we learned, what to do first, what to avoid
- Short paragraphs, mobile-readable
- No em-dashes
- For each API: include the actual code snippet in a fenced code block, with comments explaining inputs and outputs
- For each API: include a cost summary line at the bottom of that subsection
- End with a single cost-per-image table summing all APIs called per processed photo
