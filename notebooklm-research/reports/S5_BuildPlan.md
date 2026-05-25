Answer:
**Section 5: Recommended Phased Build Plan**

*   **What we learned:** Building a production-grade HDR pipeline is 
fundamentally an orchestration challenge. You can stay well under your $0.50 
per image target by routing lightweight tasks to your Fly.io Node.js workers 
and outsourcing heavy segmentation to Replicate APIs. 
*   **What to do first:** Establish your EXIF parsing and 5-second grouping 
logic in Node.js. An automated pipeline is useless if it cannot correctly group
bracketed sequences from a randomized upload batch.
*   **What to avoid:** Do not prematurely invest your $5,000 budget in manually
annotating images. Use zero-shot foundation models to auto-label data for free 
before spending capital on human quality control.

**5.1 Phase 0 (Week 1 to 2): Dataset acquisition**

Your first step is gathering raw material. You can download the SICE dataset 
immediately at https://github.com/csjcai/SICE, which provides 589 
high-resolution multi-exposure sequences [1, 2]. You can also access the 
MIT-Adobe FiveK dataset at https://data.csail.mit.edu/graphics/fivek/ [3, 4]. 
Be aware that these datasets carry strict non-commercial licenses [5-7]. You 
must use them strictly for local testing and pipeline validation.

Next, you need to structure your own historical archive of Martha's Vineyard 
shoots. Build a Node.js script to extract EXIF data from your raw files. Group 
any frames captured within a 5-second window that share the same focal length 
but have varying shutter speeds [8]. Store these grouped exposures in a 
dedicated Cloudinary folder named by property UUID.

Target dataset sizes scale by phase. Phase 1 requires zero training pairs since
it relies on off-the-shelf APIs. Phase 2 will require around 400 to 500 pairs 
to validate generative prompt effectiveness [9]. Phase 3 will require a robust 
set of 20,000 professionally edited pairs to train a custom production model 
[10, 11].

**5.2 Phase 1 (Week 3 to 6): Classical pipeline first**

Start by building a classic Enfuse exposure fusion pipeline [12, 13]. You can 
wrap the Enfuse command-line tool in a Python subprocess running on a 
lightweight Fly.io worker. This skips complex tone mapping and merges your 
brackets directly into a low dynamic range image [14, 15].

For window pulls, deploy GroundingDINO cascaded with SAM 2 via the 
Replicate.com API. Use GroundingDINO to detect "window panes" and pass those 
bounding boxes as positional prompts to SAM 2 [16, 17]. This generates a 
precise alpha mask of the exterior view. 

Handle perspective correction by writing an OpenCV Python microservice that 
detects vertical lines and computes a geometric transform matrix to straighten 
walls [8, 18]. For sky replacement, trigger an SDXL inpainting model on 
Replicate.

At this phase, your expected quality level will be structurally accurate but 
slightly sterile. The pipeline will successfully merge exposures and fix 
perspectives but will miss the localized, warm color consistency that AutoHDR 
provides.

**5.3 Phase 2 (Week 7 to 10): Add generative AI fills**

To bridge the quality gap, introduce generative AI to handle complex scene 
clutter and lighting adjustments. Register for an Adobe enterprise developer 
account to integrate the Firefly Generative Fill API. This API uses a credit 
system and allows you to programmatically pass your SAM 2 masks to replace 
blown-out skies or remove heavy shadows.

Next, integrate the GPT Image 2 model via OpenAI for intelligent semantic 
inpainting [19]. You can pass a prompt like "neutralize yellow tungsten light 
on white walls" for roughly $0.04 to $0.08 per image. 

To handle luxury virtual staging or complex object removal, connect to Nano 
Banana 2 and Nano Banana Pro via the Higgsfield Design API or your MCP server 
infrastructure [1, 19, 20]. Format your text prompts to specify natural 
language scene alterations, such as "remove power cords and replace with clean 
hardwood floor". 

Your per-image cost at this phase will increase due to API calls, but it will 
remain well under your target constraint.

**5.4 Phase 3 (Month 3+): Fine-tune your own model**

Once you have gathered enough data, you will transition away from third-party 
APIs and fine-tune your own model. You should choose NAFNet as your base model.
NAFNet delivers specialized image restoration through a nonlinear, 
activation-free architecture that is highly efficient for HDR bracketing 
reconstruction [21, 22].

Your Windows dev machine with the RTX 5070 Ti (16GB VRAM, CUDA 12.8, Blackwell 
sm_120) is fully capable of handling this training workload. You can train 
NAFNet by extracting 256x256 image patches and running a batch size of 16 [6, 
23, 24]. 

You will need approximately 20,000 paired images before your fine-tuned model 
consistently beats off-the-shelf competitors [10]. To reach this number without
spending your $5,000 budget on manual editors, use synthetic data. Run your raw
brackets through your Phase 2 pipeline (using Firefly and GPT Image 2) to 
generate highly polished before and after pairs, and use those outputs to 
bootstrap your NAFNet training.

**5.5 Cost model and break-even**

In Phase 1, your costs are limited to infrastructure. A Fly.io serverless CPU 
execution costs roughly $0.01 per image. Calling SAM 2 on Replicate costs 
roughly $0.05. Enfuse processing is essentially free. Your total Phase 1 cost 
is $0.06 per image.

In Phase 2, integrating Adobe Firefly, GPT Image 2, and Nano Banana Pro will 
add between $0.10 and $0.16 in API fees. Your total Phase 2 cost will be 
approximately $0.22 per image.

AutoHDR Pro charges $1.20 per image at their professional tier. By building 
in-house, your cost is $0.22, yielding a direct savings of $0.98 per image. If 
we assume a monthly operational and API overhead of $500 for your Fly.io and 
Prisma database hosting, you break even at just 510 images per month ($500 / 
$0.98). Any volume beyond 15 properties a month makes this in-house build 
highly profitable.

**Phased Build Timeline and Cost**

| Phase | Weeks | Key deliverable | Cost |
| :--- | :--- | :--- | :--- |
| Phase 0 | Week 1 to 2 | EXIF parsing logic, bracket grouping script, 
Cloudinary sync | $0 |
| Phase 1 | Week 3 to 6 | Fly.io Enfuse worker, Replicate SAM 2 window masks, 
OpenCV API | $0.06 / image |
| Phase 2 | Week 7 to 10 | Firefly Generative Fill, GPT Image 2 inpainting, 
Nano Banana Pro | $0.22 / image |
| Phase 3 | Month 3+ | Custom NAFNet model trained on RTX 5070 Ti using 20,000 
synthetic pairs | $0.05 / image |

Conversation: 2dfc37d0-9fc8-4707-a038-96ee48137952 (turn 1)

Saved as note: Automated HDR Pipeline: A Phased Architectural Roadmap 
(653dcbe4...)
