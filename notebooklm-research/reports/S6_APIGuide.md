Answer:
**Section 6: Specific API Integration Guide**

*Please note: While the provided sources discuss the conceptual use of these AI
tools [1-4], they do not contain the exact API documentation, endpoint URLs, or
Node.js code snippets requested. The code, endpoints, and exact pricing details
below are drawn from standard external developer knowledge to fulfill your 
request. You should independently verify the API documentation for each 
provider.*

*   **What we learned:** Tying these microservices together in Node.js requires
handling different mask formats (alpha transparency vs. binary black/white) and
managing asynchronous webhooks to prevent long-running requests from timing 
out.
*   **What to do first:** Build the local Enfuse pipeline. Generating exposure 
fusion locally on your Fly.io worker is nearly free and handles the heavy 
lifting before you spend API credits on generative refinements.
*   **What to avoid:** Do not run heavy generative APIs on every single 
bracketed photo. Merge them first, run a quality check, and only call Firefly 
or GPT Image 2 on the final merged output to keep your unit costs under $0.50.

**6.1 Calling GPT Image 2 (OpenAI) for inpainting/fill**
OpenAI's image editing API is useful for semantic inpainting, such as 
neutralizing strange color casts or fixing small artifacts. 
*   **Endpoint URL:** `https://api.openai.com/v1/images/edits`
*   **Auth header format:** `Authorization: Bearer YOUR_OPENAI_API_KEY`
*   **Input format:** Both the image and mask must be square, less than 4MB, 
and have the exact same dimensions. 
*   **Mask format:** A PNG with an alpha channel (transparency). The fully 
transparent areas indicate where the AI should edit, while the opaque areas are
protected.

```javascript
// Node.js implementation using standard fetch and FormData
import FormData from 'form-data';
import fs from 'fs';
import fetch from 'node-fetch';

async function runGPTInpainting(imagePath, maskPath, prompt) {
  const formData = new FormData();
  // Image and mask must be square PNGs with identical dimensions
  formData.append('image', fs.createReadStream(imagePath));
  formData.append('mask', fs.createReadStream(maskPath)); 
  formData.append('prompt', prompt);
  formData.append('n', 1);
  formData.append('size', '1024x1024');

  const response = await fetch('https://api.openai.com/v1/images/edits', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
      // formData.getHeaders() is required for multipart/form-data
      ...formData.getHeaders() 
    },
    body: formData
  });

  const data = await response.json();
  // Returns a JSON object containing the URL of the edited image
  return data.data.url; 
}
```
*Cost summary: Approximately $0.04 per 1024x1024 image.*

**6.2 Calling Adobe Firefly Generative Fill API**
Adobe Firefly is the industry standard for commercial-safe generative fill. To 
get access, you cannot use a standard Creative Cloud subscription; you must 
apply for an Adobe Developer Enterprise account to access the Firefly Services 
API.
*   **Endpoint URL:** 
`https://image.adobe.io/firefly-services/v1/generativeFill` (Requires standard 
Adobe OAuth 2.0 flow for a Bearer token).
*   **Output format:** The API typically returns a cloud-hosted URL to the 
generated asset or deposits it directly into a designated Adobe Cloud Storage 
bucket.

```javascript
// Node.js implementation using fetch
async function runFireflyGenerativeFill(uploadUrl, maskUrl, prompt, token, 
clientId) {
  const payload = {
    "input": {
      "href": uploadUrl // URL to your pre-uploaded source image
    },
    "mask": {
      "href": maskUrl // URL to your pre-uploaded binary mask (black/white)
    },
    "prompt": prompt,
    "options": {
      "shape": "square"
    }
  };

  const response = await 
fetch('https://image.adobe.io/firefly-services/v1/generativeFill', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'x-api-key': clientId,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });

  const data = await response.json();
  // The response contains an output href to download the result
  return data.output.href; 
}
```
*Cost summary: 1 Generative Credit per call (approx. $0.05 - $0.10 based on 
Enterprise contract tiers).*

**6.3 Calling Nano Banana 2 and Nano Banana Pro via Higgsfield Design API**
Nano Banana Pro is listed among advanced generative tools [3], and Gemini 2.5 
Flash Nano Banana is referenced for on-device or fast generation [5]. 
Higgsfield provides advanced video and image controls.
*   **Model IDs:** `nano-banana-2-base` (best for fast, standard object 
removal) and `nano-banana-pro` (best for high-fidelity virtual staging and 
complex texture matching).
*   **Prompt format:** Higgsfield expects natural language, affirmative action 
prompts (e.g., "replace the floor with clean oak hardwood"). 
*   **MCP Server:** You can route this through an MCP (Model Context Protocol) 
server to orchestrate the API calls locally without writing raw HTTP wrappers.

```javascript
// Node.js implementation using fetch for Higgsfield Design API
async function runNanoBanana(imageUrl, prompt) {
  const payload = {
    model: "nano-banana-pro",
    inputs: {
      image_url: imageUrl,
      prompt: prompt,
      negative_prompt: "distorted, low quality, artifacts"
    }
  };

  const response = await fetch('https://api.higgsfield.ai/v1/design/generate', 
{
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.HIGGSFIELD_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });

  const data = await response.json();
  // Returns the URL of the processed image
  return data.result_url; 
}
```
*Cost summary: Approximately $0.05 per image.*

**6.4 Running SAM 2 on Replicate for window segmentation**
Grounded SAM 2 pairs GroundingDINO (a text-to-box model) with Segment Anything 
2 to automate masking based on text prompts like "window" [2, 6]. 
*   **Model ID:** `idea-research/grounded-sam-2` (Specific Replicate hash 
strings update frequently).
*   **Input format:** Publicly accessible image URL or Base64 data URI.
*   **Extracting the mask:** The API returns an array of JSON objects or mask 
URLs. You must download the mask image from the returned URL.

```javascript
// Node.js implementation using the official Replicate SDK
import Replicate from "replicate";

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

async function getWindowMask(imageUrl) {
  const output = await replicate.run(
    "idea-research/grounded-sam-2:latest-hash-goes-here",
    {
      input: {
        image: imageUrl,
        text_prompt: "window pane", // GroundingDINO prompt
        box_threshold: 0.25,
        text_threshold: 0.25
      }
    }
  );
  
  // Output usually contains the annotated image and the isolated binary mask 
URL
  // We extract the binary mask for use in our HDR pipeline
  return output.mask_url; 
}
```
*Cost summary: Runs on Nvidia T4 hardware taking about 2 seconds, costing 
roughly $0.034 per call [7].*

**6.5 Running Enfuse for exposure fusion in Node.js**
Sharp.js is excellent for resizing and format conversion, but it does not have 
the mathematical capability to perform exposure fusion algorithms. You must use
the `enfuse` C++ binary.
*   **Binary source:** In your Fly.io `Dockerfile`, add `RUN apt-get update && 
apt-get install -y enfuse` to install the binary natively.
*   **How it works:** You write the bracketed images to local disk (`/tmp/` on 
Fly.io), run the Enfuse subprocess, and read the output back into Node.js.

```javascript
// Node.js implementation using child_process
import { spawn } from 'child_process';
import path from 'path';

async function runEnfuse(bracketPaths, outputFilename) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join('/tmp', outputFilename);
    
    // Command: enfuse -o /tmp/output.tif /tmp/dark.jpg /tmp/mid.jpg 
/tmp/bright.jpg
    const args = ['-o', outputPath, ...bracketPaths];
    const enfuseProcess = spawn('enfuse', args);

    enfuseProcess.on('close', (code) => {
      if (code === 0) {
        resolve(outputPath); // Return the path to the fused file
      } else {
        reject(new Error(`Enfuse failed with exit code ${code}`));
      }
    });
  });
}
```
*Cost summary: $0.00 (Requires only server CPU time, covered by your Fly.io 
compute costs).*

**6.6 Calling Real-ESRGAN on Replicate for final quality enhancement**
Real-ESRGAN upscales the image and removes compression artifacts, utilizing 
GFPGAN for face enhancement if needed [4]. It is best used as the final step in
your pipeline before delivering to the client. Use this instead of Topaz Photo 
AI when you need an API-first, serverless approach rather than relying on a 
heavy desktop application.
*   **Model ID:** `nightmareai/real-esrgan`
*   **Input parameters:** `image` (URL), `scale` (factor to upscale), 
`face_enhance` (boolean).

```javascript
import Replicate from "replicate";

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

async function upscaleImage(imageUrl) {
  const output = await replicate.run(
    "nightmareai/real-esrgan:42fd3caa8788a20c7edab74eb74718301540f2824ecfce8276
f756ea3c4ba7e4",
    {
      input: {
        image: imageUrl,
        scale: 2, // Upscale by 2x
        face_enhance: false // False for real estate, we don't want it altering
paintings/portraits
      }
    }
  );
  
  // Returns the URL of the upscaled image
  return output; 
}
```
*Cost summary: Runs on Nvidia T4, taking a few seconds. Approximately $0.034 
per call [7].*

**Cost-per-Image Summary**

| Pipeline Step | API / Tool | Estimated Cost per Photo |
| :--- | :--- | :--- |
| **Exposure Fusion** | Enfuse (Local CPU) | $0.01 (Fly.io compute) |
| **Window Segmentation** | Grounded SAM 2 (Replicate) | $0.03 |
| **Generative Fill (Pick One)** | GPT Image 2 or Firefly | $0.04 - $0.05 |
| **Final Polish** | Real-ESRGAN (Replicate) | $0.03 |
| **Total Blended Cost** | | **$0.11 - $0.12** |

Conversation: 2dfc37d0-9fc8-4707-a038-96ee48137952 (turn 1)

Saved as note: Node.js Image Processing and Generative AI Integration Guide 
(31f44a95...)
