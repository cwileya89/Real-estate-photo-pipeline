### Computational Photometric Reconstruction: Architectural Paradigms of Exposure Fusion, Deep Gated Networks, and Automated Real Estate Post-Processing

#### Foundations of Multi-Exposure High Dynamic Range Imaging and Exposure Fusion

The reconstruction of scenes exhibiting a high dynamic range has historically presented a severe challenge to digital imaging systems.[1, 2, 3] Because conventional imaging sensors possess a limited dynamic range, they fail to simultaneously preserve detail in extreme highlights and deep shadows.[1, 2, 3] The classical stack-based approach addresses this limitation by acquiring a sequence of low dynamic range images of the same scene under varying exposure parameters, commonly referred to as exposure bracketing.[1, 3, 4] These individual frames must subsequently be merged into a single representation that displays details across the entire luminance range.[1, 3]
Traditional high dynamic range pipelines operate by reconstructing an intermediate, physically-based 32-bit radiance map, whose pixel values are directly proportional to the true scene radiance.[1, 3, 5] This step requires precise calibration of the non-linear camera response function to linearize the input exposures using the inverse relationship [3, 6]:
$$X\_k(p) = f^{-1}(I\_k(p))$$
Once the linearized radiance map is generated, it must be compressed via a tone-mapping operator to be compatible with standard 8-bit displays.[1, 3, 6] This sequence is computationally intensive, highly sensitive to calibration errors, and susceptible to noise amplification and localized color distortions.[1, 6]
To bypass these operational overheads, Mertens, Kautz, and Van Reeth proposed exposure fusion, a non-physical blending technique that directly constructs a display-ready low dynamic range image from the input sequence without generating an intermediate high bit-depth radiance map.[1, 6, 7] This method computes a localized perceptual quality measure for each pixel in the multi-exposure stack, utilizing three primary quality metrics: contrast, saturation, and well-exposedness.[5, 6, 8]
The contrast metric ( $C$ ) is computed by applying a Laplacian filter to the grayscale representation of each image and taking the absolute value of the filter response [6, 8]:
$$C\_{ij,k} = | \nabla^2 I\_{ij,k}^{\text{gray}} |$$
The saturation metric ( $S$ ) measures the vividness of colors, which tend to desaturate as pixels approach overexposure or underexposure.[6] It is defined as the standard deviation across the Red, Green, and Blue channels at each pixel location [6, 8]:
$$S\_{ij,k} = \sqrt{\frac{1}{3} \sum\_{c \in \{R,G,B\}} (I\_{ij,k}^c - \mu\_{ij,k})^2}$$
The well-exposedness metric ( $E$ ) evaluates how close a pixel's intensity is to the center of the dynamic range to penalize clipped or noisy extremes.[6] This is modeled via a Gaussian curve applied to each color channel independently and then multiplied [6]:
$$E\_{ij,k} = \prod\_{c \in \{R,G,B\}} \exp \left( -\frac{(I\_{ij,k}^c - 0.5)^2}{2\sigma^2} \right)$$
where $\sigma$ is typically set to $0.2$ .[6]
These three metrics are combined multiplicatively to act as an "AND" gate, enforcing that a high weight is only assigned to a pixel if it satisfies all quality criteria simultaneously.[6] The final scalar weight $W\_{ij,k}$ for pixel $(i,j)$ in the $k$ -th exposure is controlled by exponents $\omega\_C$ , $\omega\_S$ , and $\omega\_E$ [6]:
$$W\_{ij,k} = (C\_{ij,k})^{\omega\_C} \times (S\_{ij,k})^{\omega\_S} \times (E\_{ij,k})^{\omega\_E}$$
To ensure consistent blending, the weights of the $N$ input images are normalized to sum to one at each pixel location [6]:
$$\hat{W}\_{ij,k} = \left( \sum\_{k'=1}^{N} W\_{ij,k'} \right)^{-1} W\_{ij,k}$$
A naive direct blending equation would sum the weighted intensities [6]:
$$R\_{ij} = \sum\_{k=1}^{N} \hat{W}\_{ij,k} I\_{ij,k}$$
However, because the input images contain vastly different absolute intensities due to exposure variations, applying this equation directly introduces sharp transitions and visible seams where the weight maps vary rapidly.[6] Attempting to smooth these weight maps with a simple spatial Gaussian filter creates highly undesirable low-frequency halo artifacts and color bleeding across structural boundaries.[6]
To resolve this spatial transition problem, the Mertens algorithm utilizes a multiresolution pyramid blending technique based on the Laplacian pyramid decomposition proposed by Burt and Adelson.[6, 8] The step-by-step implementation is structured as follows [6]:

1. Each of the $N$ input images $I\_k$ is decomposed into a Laplacian pyramid, representing band-pass filtered versions at progressive scales, denoted as $L{I\_k}^l$ at level $l$ .[5, 6]
2. Each normalized weight map $\hat{W}\_k$ is decomposed into a low-pass Gaussian pyramid, denoted as $G{\hat{W}\_k}^l$ .[5, 6]
3. Blending is executed independently at each scale level $l$ , where the coefficients of the resulting blended Laplacian pyramid $L{R}^l$ are calculated as a weighted average [6]:
4. $$L\{R\}^l\_{ij} = \sum\_{k=1}^{N} G\{\hat{W}\_k\}^l\_{ij,k} L\{I\_k\}^l\_{ij,k}$$
5. The blended Laplacian pyramid is collapsed scale-by-scale, starting from the lowest resolution layer up to the original resolution, yielding the final artifact-free fused image $R$ .[3, 6]
   Despite its computational efficiency, the classical Mertens exposure fusion method suffers from two primary limitations described by Hessel and Morel: the out-of-range artifact and the low-frequency halo.[7, 9] The out-of-range artifact arises when the input sequence contains areas that are saturated (either fully white or black) across all exposures, causing a loss of local contrast.[7, 9] The low-frequency halo occurs because the Gaussian pyramid of the weight map does not strictly align with the physical boundaries of high-contrast edges in the underlying Laplacian pyramid, causing a halo when the pyramid depth is insufficient.[7, 9]
   Extended Exposure Fusion resolves these issues through a sequence extension technique.[7, 9] Each input image's dynamic range is distributed across several newly simulated, narrower dynamic range images using a remapping function $g(t)$ governed by a restricted range parameter $\beta \in (0, 1]$ [9]:
   $$g(t) = \text{smoothdecay}\left(\frac{t - \rho(k)}{\beta}\right)$$
   where $\rho(k)$ represents the center of the $k$ -th reduced dynamic range segment.[9] By artificially expanding the sequence from $N$ to $M \times N$ images (where $M = \lceil 1/\beta \rceil$ ), Extended Exposure Fusion enhances local contrast and eliminates out-of-range artifacts, producing a highly vibrant output while preserving fine spatial structures.[7, 9]

---

#### Spatial Registration, Homography, and De-ghosting Methodologies

A fundamental assumption of stack-based multi-exposure fusion is that the camera and all elements within the scene remain completely static throughout the entire acquisition period.[10, 11] In practical settings, hand-held camera shake, wind, or physical motion in the environment (such as moving foliage, passing vehicles, or walking occupants) violates this assumption.[4, 12, 13, 14] When unaligned or dynamic frames are fused, they produce severe spatial misalignments and transparent, duplicate structures known as ghosting artifacts.[4, 13, 14] Consequently, robust spatial alignment and de-ghosting algorithms are required.[4, 10, 13]

##### Classical Spatial Alignment

For translational hand-held camera shifts, Greg Ward introduced the Median Threshold Bitmap algorithm.[15, 16, 17] The Median Threshold Bitmap method converts grayscale versions of two differently exposed images into binary bitmaps using their respective median pixel values as thresholds.[15, 16, 17] Because the median threshold partitions the pixel population into two equal halves (brighter and darker than the median), the resulting binary structure remains highly stable regardless of exposure variations, provided the camera response is monotonic.[16]
To handle noise and lighting fluctuations near the median threshold, the algorithm constructs an exclusion bitmap ( $eb$ ) that masks out pixels whose intensities fall within a tight margin ( $\pm 4$ gray levels) around the median.[15, 18] The alignment is executed recursively using a multi-scale image pyramid.[15, 16, 18] At the lowest resolution scale, the minimum difference offset is computed by testing shifts within a $\pm 1$ pixel range.[16] This offset is multiplied by two at the next resolution level, serving as the starting point for a refined localized search.[16] The optimal shift is found by minimizing the Exclusive OR difference between the aligned bitmaps, masked by their corresponding exclusion bitmaps [15, 16, 18]:
$$\text{Error} = \sum ((\text{tb}\_1 \oplus \text{tb}\_2) \wedge \text{eb}\_1 \wedge \text{eb}\_2)$$
For complex perspective shifts containing three-dimensional rotations, homography-based registration must be used.[13, 19] Advanced registration pipelines, such as those in Extended Exposure Fusion, utilize midway image equalization to match the histograms of consecutive frames before estimating the homography matrices.[19, 20] This step ensures that feature detection algorithms (such as SIFT) can identify robust matches across highly disparate exposure settings.[19, 21]

##### Classical De-ghosting Mechanisms

Once global alignment is established, localized object motion must be corrected.[10, 13] Classical de-ghosting typically relies on motion detection followed by motion rejection or patch replacement.[10, 22] The Bitmap Movement Detection algorithm detects moving clusters by summing the binary threshold bitmaps across the exposure sequence.[17] In a completely static scene, each pixel is expected to preserve its binary state across all aligned frames.[17] State variations indicate local motion, which is isolated using connected component labeling to form a labeled motion map [17]:
$$M^\* = \sum\_{k=1}^{N} \text{tb}\_{ij,k}$$
Once these moving clusters are identified, the algorithm rejects the multi-exposure blend in those areas, replacing them with pixels extracted exclusively from a single "base" or best-exposed frame, thereby rendering the moving object sharp and free of duplicate artifacts.[17, 22, 23]

##### Deep Learning and Attention-Guided De-ghosting

While classical de-ghosting techniques work well for simple motions, they fail when faced with complex, non-rigid deformations or when the motion occurs in heavily saturated regions where spatial gradients are entirely lost.[24, 25, 26] Early deep learning approaches attempted to resolve this by performing explicit optical flow warping of non-reference frames prior to convolutional neural network merging.[24, 26, 27] However, optical flow is highly error-prone in regions containing severe occlusion, motion blur, or exposure saturation, leading to significant structural distortion and warping artifacts in the final high dynamic range output.[24, 26, 28]
To address these limitations, modern architectures employ implicit feature-level alignment and attention-guided networks.[11, 24, 26] The landmark Attention-Guided HDR Network (AHDR-Net) proposed by Yan et al. completely bypasses explicit optical flow computation.[26, 28, 29] AHDR-Net designates a middle-exposure frame as the reference $I\_r$ , and feeds the encoded features of both the reference and non-reference frames ( $I\_i$ ) into convolutional attention modules.[24, 26, 29]
These modules generate continuous, soft spatial attention maps that dynamically evaluate the importance of different image regions.[26, 29] The attention weights suppress features originating from misaligned motion zones or heavily saturated regions in the non-reference frames, while accentuating complementary, high-frequency details that match the reference scene.[26, 28, 29]
To reconstruct missing details in areas where the reference frame is severely over- or underexposed, the attention-guided features are passed into Dilated Residual Dense Blocks.[26, 28, 29] By utilizing dilated convolutions, the Dilated Residual Dense Blocks significantly enlarge the network's receptive field without loss of spatial resolution, enabling the model to exploit global contextual cues to reconstruct structural details in saturated regions.[26, 28, 29]
Furthermore, advanced architectures utilize Deformable Convolutions to implement learnable, non-rigid geometric transformations.[11, 30, 31] Unlike standard convolutions with fixed sampling grids, deformable convolutions learn additional 2D offsets ( $\Delta p$ ) at each position, allowing the kernel to adaptively warp its receptive field to match the underlying motion of non-reference frames [11, 27, 31]:
$$y(p\_0) = \sum\_{n=1}^{K} w\_n \cdot x(p\_0 + p\_n + \Delta p\_n)$$
This enables high-fidelity, implicit feature-level alignment that is robust to large displacement boundaries and complex movements.[11, 27]
| Tool / Algorithm | Alignment Preset / Parameter | Functional Mechanism | Recommended Use-Case | Reference |
| ------ | ------ | ------ | ------ | ------ |
| **Adobe Lightroom Classic** | Auto Align | Automatically detects structural features across bracketed exposures and performs geometric transformations to correct translation and rotation. | Essential for handheld bracketed captures where slight hand tremors introduce inter-frame displacement. | [4, 12, 23] |
| **Adobe Lightroom Classic** | Deghosting (Low / Medium / High) | Compares pixel differences between frames, identifies localized movement, and masks out dynamic pixels, replacing them with content from the base exposure. | Cures dynamic motion (foliage, ceiling fans, moving occupants). Medium is standard; High is reserved for severe movement. | [4, 12, 32] |
| **Photomatix Pro** | On Tripod | Skips intensive spatial warping, assuming a perfectly locked camera perspective. Only corrects minimal high-frequency vibrations (e.g., wind-induced tripod shake). | Standard setup for real estate interiors captured on a heavy, professional-grade tripod. | [13] |
| **Photomatix Pro** | Hand-held (Minimal / Standard / Large Shifts) | Scale-dependent search for camera translation and rotational shifts, employing perspective and keystone correction algorithms. | "Large Shifts" is optimized for shooting without a tripod under difficult, high-vibration, or windy conditions. | [13] |
| **Greg Ward's MTB** | Exclusion Bitmap Margin | Generates a binary mask of pixels residing near the median threshold value ( $\pm 4$ gray levels) to suppress high-frequency sensor noise. | Evaluates translational alignment offsets rapidly using fast integer-based bitwise operations. | [15, 18] |

---

#### Deep Learning Architectures and the Rise of U-Net and NAFNet

The integration of convolutional neural networks has redefined high dynamic range reconstruction, moving the field away from non-parametric pixel blending toward end-to-end parametric mapping.[11, 33, 34] In this domain, U-Net-like architectures serve as a foundational design, utilizing symmetric encoder-decoder pathways coupled with lateral skip connections.[33, 35, 36] The encoder pathway progressively reduces spatial resolution via downsampling layers to build highly compressed, low-frequency semantic representations of the scene, while the decoder pathway systematically upscales these representations back to the original image dimensions.[35, 36]
Crucially, the skip connections bridge the encoder and decoder states directly, transferring high-frequency spatial details (such as edges, textures, and fine boundaries) across the network to preserve image resolution and prevent the lossy degradation typical of deep bottlenecks.[35, 36]
This encoder-decoder structure is widely employed across high dynamic range and raw enhancement pipelines.[34, 36, 37] For example, the Reversing Lossy UNet model leverages a deep U-Net framework designed specifically to reverse the degradations introduced by low dynamic range camera signal processing.[34] It incorporates specialized Image Pipeline Modules, including the Recovery of Overexposed Regions, Dequantization, Reconstruction of Linearized Mappings, and the Compression Artifact Adjustment Module.[34]
By analyzing the physical principles of digital image formation, Reversing Lossy UNet reconstructs clean, linear high dynamic range outputs from highly compressed 8-bit inputs.[34] Similarly, U-Net structures are deployed in low-light raw enhancement networks to map extremely noisy, underexposed sensor data to bright, clean, display-ready sRGB images in a single forward pass.[36]

##### NAFNet Architectural Breakthroughs

Despite the success of these deep architectures, the computational complexity of state-of-the-art models became a limiting factor for practical applications.[33, 38, 39] Traditional networks rely heavily on nested attention layers, multi-stage processing blocks, and complex nonlinear activations like GELU, ReLU, and Sigmoid.[33, 38, 40]
In 2022, Chen et al. introduced the Nonlinear Activation Free Network, proving that traditional nonlinear activation functions are entirely dispensable in low-level vision tasks.[33, 38, 39] By substituting these activations with simple linear operations and element-wise multiplications, NAFNet matches or exceeds state-of-the-art restoration performance at half the computational cost.[33, 38, 39]
The standard NAFNet block operates as a residual unit comprising [38, 40]:

1. **Layer Normalization:** Applied at the input of the block to stabilize training. This replaces Batch Normalization, which often becomes unstable when training with small batch sizes.[33, 38, 40]
2. **Channel-Mixing Convolution:** A $1 \times 1$ convolution designed for channel interactions.[38, 40]
3. **Spatial Feature Extraction:** A $3 \times 3$ depthwise convolution that captures localized spatial context.[38, 40]
4. **SimpleGate Activation:** In standard architectures, the Gated Linear Unit is defined as $GLU(X) = (X \* \mathbf{W}\_1) \odot \sigma(X \* \mathbf{W}\_2)$ , which relies on the nonlinear Sigmoid function ( $\sigma$ ).[40] NAFNet reveals that the standard GELU activation can be mathematically formulated as a special case of GLU, and proves that the nonlinearity can be completely omitted.[40] SimpleGate splits the input feature map $X \in \mathbb{R}^{H \times W \times C}$ along the channel dimension into two equal halves $X\_1, X\_2 \in \mathbb{R}^{H \times W \times \frac{C}{2}}$ , and multiplies them element-wise [38, 41, 42]:
5. $$\text{SimpleGate}(X\_1, X\_2) = X\_1 \odot X\_2$$
6. This simple operation provides a highly effective, learnable gating mechanism without explicit nonlinear functions.[38, 40, 41]
7. **Simplified Channel Attention:** Traditional channel attention blocks (such as Squeeze-and-Excitation networks) apply global average pooling, pass the pooled features through a multi-layer perceptron with a ReLU activation, and compress the output using a Sigmoid function.[38, 40, 41] NAFNet simplifies this by removing all non-linearities.[40, 41] The Simplified Channel Attention module pools spatial dimensions and applies a single linear projection matrix to generate channel weights, allowing the network to modulate channel-wise features using linear projections where weights can dynamically scale above $1$ or become negative [38, 41, 43]:
8. $$\text{SCA}(X) = X \times \mathbf{W}\_{\text{linear}}(\text{Pool}(X))$$

##### Modern Multi-Exposure Gated Pipelines

The efficiency and representing power of NAFNet have driven its integration into advanced multi-exposure and high dynamic range reconstruction networks, operating as a primary processing backbone.[37, 44, 45]

###### RawEnhancer

RawEnhancer is an attentive multi-exposure selection network designed for bracketed RAW image reconstruction.[37] To bypass explicit optical flow alignment, RawEnhancer adopts an "encoding-then-fusion" strategy.[37] It features a weight-sharing encoder built upon a novel Exposure-Aware Selective Attention block, which is derived from NAFNet.[37]
The Exposure-Aware Selective Attention block replaces the internally-learned spatial gate of standard NAFNet blocks with an externally-computed spatial gate derived from the gradient magnitude of the RAW input.[37] By using the raw gradient map, Exposure-Aware Selective Attention identifies valid, content-rich regions in under- or overexposed frames, effectively suppressing noisy or clipped channels before applying channel attention.[37] The multi-scale features are subsequently integrated within an Iterative Feature Selection decoder, which employs Gated Recurrent Units to adaptively propagate complementary details across exposure scales.[37]

###### GMODiff

GMODiff is a gain map-driven one-step latent diffusion framework for multi-exposure high dynamic range reconstruction, designed to overcome the computational and representation constraints of standard Latent Diffusion Models.[22, 45] Because Latent Diffusion Models operate in a highly compressed 8-bit latent space, they struggle to directly represent 32-bit floating-point high dynamic range radiance values, often producing severe content hallucinations and high computational latency.[45]
GMODiff resolves this by decomposing a target high dynamic range image $H$ into an 8-bit low dynamic range base layer $L$ and an 8-bit normalized gain map $G \in $ via the following formulation [45]:
$$H = (L + \alpha) \cdot \text{exp2} (1 + G \cdot Q\_{\text{max}})$$
where $Q\_{\text{max}}$ is the maximum gain value, and $\alpha = 1/64$ is a scaling constant.[45] Since both $L$ and $G$ are 8-bit representations, the model can leverage pre-trained low dynamic range diffusion priors without restructuring the latent space.[45]
GMODiff implements a two-stage pipeline [45, 46]:

1. **Degradation-aware Regressor Network Pre-training:** GMODiff trains a regression-based network built upon a NAFNet backbone to generate an initial, high-fidelity gain map estimation $\hat{G}$ and multi-scale spatial embeddings $c\_L$ .[45] Concurrently, a lightweight segmentation head predicts a reliability mask to identify motion blur and saturation boundaries.[45] The initial gain map prediction is supervised via an $L1$ loss:
2. $$\mathcal{L}\_{\text{gm}} = | \hat{G} - G |\_1$$
3. while structural consistency is enforced via a joint $L1$ and VGG perceptual loss in the differentiable $\mu$ -law domain [45]:
4. $$\mathcal{L}\_{\text{r}} = | \mu(\hat{H}) - \mu(H) |\_1 + \lambda\_{\text{per}} | \text{VGG}(\mu(\hat{H})) - \text{VGG}(\mu(H)) |\_1$$
5. **Prior-Guided One-Step Diffusion Refinement:** The initial NAFNet priors ( $\hat{G}$ and $c\_L$ ) are mapped into the latent space of a pre-trained Stable Diffusion model.[45] Rather than running multi-step denoising, GMODiff performs a single-step denoising operation to estimate the refined latent representation $\hat{Z}\_H$ [45]:
6. $$\hat{Z}\_H = Z\_L - \frac{\sqrt{1 - \bar{\alpha}\_T}}{\sqrt{\bar{\alpha}\_T}} \epsilon\_\theta(Z\_L; c, T)$$
7. where $\epsilon\_\theta$ is the denoising network, $T$ is the diffusion timestep, and the conditioning vector $c$ includes the spatial embeddings $c\_L$ from the NAFNet Degradation-aware Regressor and the encoded underexposed base image $L\_0$ .[45]
   To prevent structural distortion caused by the Latent Diffusion Model's lossy latent compression, GMODiff incorporates a Degradation-Aware Decoder.[45, 46] The Degradation-Aware Decoder bypasses standard decoding by concatenating intermediate latent features $\hat{Z}\_H^i$ with high-resolution multi-scale features $Z\_L^i$ extracted from the initial NAFNet-predicted gain map [45]:
   $$\hat{Z}^i = \text{Conv}\_{3 \times 3} [\hat{Z}\_H^i, Z\_L^i]$$
   This combined feature representation is dynamically modulated by a channel-wise vector $f\_i$ derived from the spatial embeddings $c\_L$ via an MLP and Sigmoid activation, preserving fine spatial structures [45]:
   $$\hat{Z}\_H^{i+1} = \text{Conv}\_{3 \times 3} (\hat{Z}^i \odot f\_i) + \hat{Z}\_H^i$$
   By utilizing this hybrid NAFNet-Diffusion architecture, GMODiff achieves superior perceptual realism and eliminates ghosting artifacts, while running in approximately $0.89$ seconds per megapixel—a $100\times$ speedup over previous diffusion-based high dynamic range models.[45]
   | Architectural Attribute | Mertens Exposure Fusion (MKVr) | AHDR-Net | RawEnhancer | GMODiff |
   | ------ | ------ | ------ | ------ | ------ |
   | **Primary Framework** | Non-parametric Pyramid Blending [6] | Dilated Residual Dense Blocks (DRDB) [26, 29] | Asymmetric Encoder-Decoder (EASA) [37] | Two-Stage NAFNet DaReg + Latent Diffusion [45] |
   | **Activation Functions** | None (Manual quality metrics) [6] | ReLU, Sigmoid (Standard CNN) [24, 26] | Nonlinear Activation Free (SimpleGate) [37] | SimpleGate, SCA (DaReg); Sigmoid (Diffusion Decoder) [45] |
   | **U-Net Integration** | None (Operates in pyramid domain) [6] | None (Cascaded feed-forward blocks) [26, 29] | None (Asymmetric encoder-decoder network) [37] | U-Net-like DaReg backbone with lateral skip connections [33, 45] |
   | **Alignment Mode** | Pre-aligned Stack Requirement [1, 6] | Implicit Feature-Level Soft Attention [26, 28, 29] | Implicit Exposure-Aware Spatial Attention [37] | Implicit spatial embeddings from NAFNet DaReg [45] |
   | **Processing Domain** | Gamma-corrected sRGB [1] | Concatenated LDR + Gamma-corrected sRGB [24, 47] | Linear RAW Sensor Data [37] | Gain Map ( $G$ ) + LDR Base ( $L$ ) [45] |
   | **Computational Cost** | Very Low (CPU-friendly, near interactive) [6] | Medium (GPU-accelerated feed-forward CNN) [47] | Low (Lightweight, no explicit warping) [37] | Low-Medium (One-step diffusion; $100\times$ speedup) [45] |

---

#### Applied Computational Photography in Real Estate Production

Real estate photography serves as a premier industrial application and a demanding stress-test for multi-exposure fusion algorithms.[48, 49, 50] This domain is characterized by extreme luminance contrasts.[48, 49, 50] A typical interior shot features relatively dim, shadow-filled indoor spaces juxtaposed with sun-blasted exterior views visible through windows.[48, 50] Capturing this scene with a single exposure forces an unviable compromise: one must either expose for the room, which completely blows out the windows, or expose for the windows, which plunges the interior into pitch blackness.[48, 50]
Furthermore, real estate photographers face the problem of mixed lighting.[48, 50] Standard interiors combine daylight entering through windows ( $5500\text{K}$ ), warm tungsten lighting from overhead lamps ( $3200\text{K}$ ), and green-tinted emissions from fluorescent fixtures under cabinets ( $4000\text{K}$ ).[50] This spatial variance in color temperature causes color casts that cannot be corrected by a single global white balance adjustment, requiring localized post-processing.[50]

##### Traditional Editing Workflows

To capture these environments, photographers traditionally shoot a bracketed RAW sequence, typically consisting of 3 to 5 frames spaced 1.5 to 2.0 EV apart [4, 14, 48]:

* **-2 EV Frame (Underexposed):** Preserves highlight details in bright windows, light fixtures, and outdoor scenery.[14, 48, 50]
* **0 EV Frame (Base Exposure):** Captures the natural mid-tones of the room.[14, 48, 50]
* **+2 EV Frame (Overexposed):** Lifts the deep shadows and corners, revealing texture in dark furniture and floors.[14, 48, 50]
  For rooms with extreme contrast, such as a space with direct midday sunlight blasting through floor-to-ceiling windows, photographers deploy a 5-shot bracket at -4, -2, 0, +2, and +4 EV to cover the extreme dynamic range.[48, 51]
  In aerial drone photography, which is increasingly utilized to capture properties and their surrounding neighborhoods, dynamic range is even more critical.[52] Drones operating in hover or tripod mode capture bracketed exposures.[52] However, in windy conditions, drone movement between exposures can introduce severe ghosting artifacts.[52] To resolve this, drone photographers have two main workarounds: bracketing manually using high-speed continuous drive modes, or capturing a single RAW exposure and generating several copies in post-processing—adjusting the exposure of each copy to mimic a physical bracket sequence.[52] While a single RAW exposure reduces dynamic range compared to physical bracketing, it is often the only way to produce a clean, artifact-free image when wind forces the drone to drift between frames.[52]
  In post-processing, professional editors utilize the "Flambient" technique or manual "Window Pulls".[51, 53, 54] The Flambient workflow requires layering a flash-exposed frame over an ambient-exposure frame in Adobe Photoshop.[53, 54] The flash frame provides accurate colors and crisp window details, while the ambient frame supplies natural shadows and realistic room ambiance.[53, 54]
  For manual window pulls, the editor manually paints a layer mask over the window frames to layer in a darker exposure.[50, 51] This process requires tedious brushwork to avoid over-painting the window frames, taking up to 30 minutes per image.[50, 55]
  In the standard Lightroom Classic workflow, editors often face challenges when merging exposures, as the resulting high dynamic range DNG files can appear extremely dark.[56] To address this, Adobe extended the exposure control range to $\pm 10$ full stops specifically for high dynamic range merged files, allowing editors to lift mid-tones significantly without introducing clipping.[56] Furthermore, utilizing the Adobe Profile SDK, editors can construct custom profiles (such as "Adobe Color HDR") with expanded highlights and shadows curves, providing a wider range of slider adjustments than the standard default profiles.[56]

##### Deep Learning Automation and AI Post-Processing

In recent years, the real estate imaging industry has transitioned toward automated, deep-learning-driven pipelines to handle high-volume processing.[49, 57, 58] This shift has been driven by the need for rapid turnaround times and cost-effective scalability.[55, 57, 59] AI-powered post-processing engines (such as Imagen, Autoenhance.ai, and Roomagen) automatically execute the complex editing steps that once required hours of manual labor.[55, 57, 58]
Rather than relying on generic global filters, these platforms employ local and global neural networks.[58, 60] A common architecture uses a two-stage approach [60]:

1. **Global Tone Optimization:** A 3D Lookup Table estimation network is trained to learn global parameters, including exposure, white balance, and contrast, from a database of professional before-and-after edits.[60]
2. **Local Feature Refinement:** A small, specialized U-Net is trained end-to-end with the 3D Lookup Table layer to handle localized effects, such as window highlight recovery (window pulls) and shadow lifting in dark corners, which global adjustments cannot resolve.[60]
   The AI "Window Pull" tool automatically detects the window boundaries in a scene, masks the glass panes, and blends the darker exposures into the final image to reveal the outdoor scenery clearly.[50, 54, 57] Concurrently, automated perspective correction models identify vertical lines (such as door frames and wall edges) and compute geometric transform matrices to correct slanting walls and wide-angle lens distortions automatically.[49, 50, 54]
   | Operational Parameter | Manual Post-Processing (Photoshop / Flambient) | Classical HDR Blending (Lightroom / Photomatix) | Deep Learning AI Platforms (Imagen / Autoenhance.ai) |
   | ------ | ------ | ------ | ------ |
   | **Editing Speed** | Extremely Slow (15–30 minutes per image) [55] | Moderate (2–5 minutes per image; manual slider tweaks) [51] | Ultra-Fast (5–15 seconds per image; batch processing) [54, 55] |
   | **Window Pull Method** | Manual brush masking over window panes [50, 51] | Local highlight reduction / gradient masks [23, 53, 56] | Semantic segmentation of windows + automated exposure pull [50, 57] |
   | **Mixed Lighting Correction** | Manual brush adjustments and local hue/saturation desaturation [50] | Global white balance adjustments (often leaves color casts) [50, 51] | Localized neural adjustments that selectively balance color temperatures [50, 58] |
   | **Perspective Alignment** | Manual adjustment of horizontal/vertical warp sliders [50] | Semi-automated (Lightroom Auto-Transform) [49, 61] | Fully automated line detection and geometric matrix correction [49, 50] |
   | **Production Cost** | High (Requires dedicated local workstation and manual labor) [55, 60] | Medium (Software subscription + editor's time) [51, 55] | Low (Subscription / pay-per-image pricing; $0.0225/MP) [55, 62] |
   | **Throughput Scalability** | Low (Limited by the number of active manual editors) [55] | Medium (Batch processing groups exposures based on time) [49] | High (Processes full listings of 30–40 images in minutes via cloud) [49, 55, 57] |

These automated pipelines reduce processing times by up to 90%, allowing a professional real estate photography business to process an entire property listing in minutes.[55] By automating the repetitive 80% of editing tasks, these computational systems enable photographers to scale their throughput while maintaining consistent, professional-grade visual standards.[55, 57]

---

#### Conclusions

The evolution of computational imaging has bridged the gap between classical, non-parametric exposure blending and generative, deep-learning-driven reconstruction.[1, 6, 45] The classical Mertens exposure fusion algorithm established a highly efficient, calibration-free paradigm for blending exposures using multiresolution pyramids.[1, 6] However, its sensitivity to spatial misalignments and halo artifacts highlighted the need for robust de-ghosting and structural alignment.[6, 13]
The introduction of deep learning networks—progressing from explicit, error-prone optical flow warps to implicit attention-guided mechanisms (such as AHDR-Net and Deformable Convolutions)—fundamentally changed how models handle motion in dynamic scenes.[11, 24, 26]
By eliminating traditional nonlinear activation functions, the NAFNet architecture demonstrated that neural network complexity can be significantly reduced without sacrificing restoration quality, paving the way for efficient edge and mobile deployments.[38, 39, 62] This efficiency has been successfully integrated into advanced HDR pipelines such as RawEnhancer and GMODiff, which combine NAFNet backbones with gain map formulations and generative diffusion models to reconstruct ghost-free, perceptually rich images.[37, 45]
In the commercial sphere, particularly real estate photography, these technological advancements have enabled fully automated editing pipelines.[49, 50, 54] AI-driven platforms can now perform window pulls, correct perspective geometry, and neutralize mixed lighting in seconds, transforming high-end architectural editing from a manual, labor-intensive chore into an automated, cloud-based utility.[50, 54, 55]

---

1. Exposure fusion - Wikipedia, <https://en.wikipedia.org/wiki/Exposure_fusion>
2. [2510.21815] HDR Image Reconstruction using an Unsupervised Fusion Model - arXiv, <https://arxiv.org/abs/2510.21815>
3. Overview of State-of-the-Art Algorithms for Stack-Based High-Dynamic Range (HDR) Imaging, <https://web.ece.ucsb.edu/~psen/Papers/IST18_OverviewOfHDR.pdf>
4. What Is Deghosting in Lightroom: Explained & How to Use It - Dan Kosmayer Fine Art Photography, <https://dankosmayer.com/blogs/photography/what-is-deghosting-in-lightroom>
5. Multiple-Exposure Image Fusion for HDR Image Synthesis Using Learned Analysis Transformations - PMC, <https://pmc.ncbi.nlm.nih.gov/articles/PMC8320921/>
6. Exposure Fusion - Stanford University, <https://web.stanford.edu/class/cs231m/project-1/exposure-fusion.pdf>
7. An Extended Exposure Fusion and its Application to Single Image Contrast Enhancement - CVF Open Access, <https://openaccess.thecvf.com/content_WACV_2020/papers/Hessel_An_Extended_Exposure_Fusion_and_its_Application_to_Single_Image_WACV_2020_paper.pdf>
8. Exposure fusion (HDR) visualization - Minhaz's Blog, <https://blog.minhazav.dev/research/exposure-fusion>
9. Extended Exposure Fusion, <https://www.ipol.im/pub/art/2019/278/article_lr.pdf>
10. The State of the Art in HDR Deghosting: A Survey and Evaluation, <https://web.cs.hacettepe.edu.tr/~erkut/publications/HDR-deghosting-star.pdf>
11. DBHDR: Dual branch Network Guided Multi-Exposure HDR Image Reconstruction - IEEE Xplore, <https://ieeexplore.ieee.org/iel7/9964480/9964481/09965323.pdf>
12. What Is HDR Deghost in Lightroom? - Cultured Kiwi, <https://www.culturedkiwi.com/what-is-deghost-in-lightroom/>
13. Loading and Merging Photos - Photomatix Pro documentation, <https://www.hdrsoft.com/support/doc/photomatix-pro/v7/loading-and-merging-photos.html>
14. How to Use HDR in Real Estate Photography, <https://www.gregceephoto.com/how-to-use-hdr-in-real-estate-photography/>
15. Project 1 Report: High Dynamic Range Imaging Computer Vision - CS766 by Kong Yang - cs.wisc.edu, <https://pages.cs.wisc.edu/~lizhang/courses/cs766-2008f/projects/hdr/students/yangk/index.html>
16. Fast, Robust Image Registration for Compositing High Dynamic Range Photographs from Handheld Exposures, <https://pages.cs.wisc.edu/~lizhang/courses/cs766-2007f/projects/hdr/jgtpap2.pdf>
17. Bitmap Movement Detection: HDR for Dynamic Scenes - Jan Kautz, <https://jankautz.com/publications/BMD_CVMP10.pdf>
18. Project 1: High Dynamic Range Imaging, <https://momohuang.github.io/assets/img/HDR/HDR_report.pdf>
19. Implementation of the Extended Exposure Fusion (EEF) proposed in [WACV '20 paper reference] - GitHub, <https://github.com/chlsl/extended-exposure-fusion-ipol>
20. Infrared Image Stripe Noise Removing Using Least Squares and Gradient Domain Guided Filtering | Request PDF - ResearchGate, <https://www.researchgate.net/publication/356152102_Infrared_Image_Stripe_Noise_Removing_Using_Least_Squares_and_Gradient_Domain_Guided_Filtering>
21. Driven by Deformable Convolution and Multi-Plane Scale Constraint: A Hazy Image Dehazing–Stitching System - MDPI, <https://www.mdpi.com/1424-8220/26/5/1551>
22. GMODiff: One-Step Gain Map Refinement with Diffusion Priors for Efficient HDR Reconstruction - arXiv, <https://arxiv.org/html/2512.16357v2>
23. How to Create an HDR in Lightroom Classic: An Illustrated Guide - Improve Photography, <https://improvephotography.com/49762/create-hdr-lightroom-classic-illustrated-guide/>
24. Deep HDR Deghosting by Motion-Attention Fusion Network - PMC, <https://pmc.ncbi.nlm.nih.gov/articles/PMC9610388/>
25. Generating Content for HDR Deghosting from Frequency View - arXiv, <https://arxiv.org/html/2404.00849v1>
26. Attention-Guided Network for Ghost-Free High Dynamic Range Imaging - CVF Open Access, <https://openaccess.thecvf.com/content_CVPR_2019/papers/Yan_Attention-Guided_Network_for_Ghost-Free_High_Dynamic_Range_Imaging_CVPR_2019_paper.pdf>
27. HDRTransDC: High Dynamic Range Image Reconstruction with Transformer Deformation Convolution - arXiv, <https://arxiv.org/html/2403.06831v1>
28. [1904.10293] Attention-guided Network for Ghost-free High Dynamic Range Imaging - arXiv, <https://arxiv.org/abs/1904.10293>
29. The architecture of the proposed AHDRNet. The network consists of an... | Download Scientific Diagram - ResearchGate, <https://www.researchgate.net/figure/The-architecture-of-the-proposed-AHDRNet-The-network-consists-of-an-attention-network_fig2_332603952>
30. Improving Dynamic HDR Imaging with Fusion Transformer - AAAI Publications, <https://ojs.aaai.org/index.php/AAAI/article/view/25107/24879>
31. A Deformable Convolutional Neural Network with Dual Attention for Multi-Exposure HDR Reconstruction | Request PDF - ResearchGate, <https://www.researchgate.net/publication/397519246_A_Deformable_Convolutional_Neural_Network_with_Dual_Attention_for_Multi-Exposure_HDR_Reconstruction>
32. HDR photo merge in Lightroom Classic - Adobe Help Center, <https://helpx.adobe.com/lightroom-classic/help/hdr-photo-merge.html>
33. NAFNET : A Machine Learning Model to Deblur Images | by David Cochard - Medium, <https://medium.com/axinc-ai/nafnet-a-machine-learning-model-to-deblur-images-a0a03e94feae>
34. (PDF) RLUNet: Overexposure-Content-Recovery-Based Single HDR Image Reconstruction with the Imaging Pipeline Principle - ResearchGate, <https://www.researchgate.net/publication/386406640_RLUNet_Overexposure-Content-Recovery-Based_Single_HDR_Image_Reconstruction_with_the_Imaging_Pipeline_Principle>
35. High Dynamic Range Image Reconstruction from Saturated Images of Metallic Objects - PMC, <https://pmc.ncbi.nlm.nih.gov/articles/PMC11051178/>
36. Guided Filter-Inspired Network for Low-Light RAW Image Enhancement - PMC, <https://pmc.ncbi.nlm.nih.gov/articles/PMC12074441/>
37. RawEnhancer: Attentive Multi-Exposure Selection for Bracketing Image Reconstruction, <https://openreview.net/forum?id=rAUPsRr0mO>
38. [Literature Review] A Comparative Study of NAFNet Baselines for Image Restoration, <https://www.themoonlight.io/en/review/a-comparative-study-of-nafnet-baselines-for-image-restoration>
39. megvii-research/NAFNet: The state-of-the-art image restoration model without nonlinear activation functions. - GitHub, <https://github.com/megvii-research/NAFNet>
40. Simple Baselines for Image Restoration - arXiv, <https://arxiv.org/pdf/2204.04676>
41. A Comparative Study of NAFNet Baselines for Image Restoration - arXiv, <https://arxiv.org/html/2506.19845v1>
42. NAFSSR: Stereo Image Super-Resolution Using NAFNet - CVF Open Access, <https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Chu_NAFSSR_Stereo_Image_Super-Resolution_Using_NAFNet_CVPRW_2022_paper.pdf>
43. (PDF) A Comparative Study of NAFNet Baselines for Image Restoration - ResearchGate, <https://www.researchgate.net/publication/392980489_A_Comparative_Study_of_NAFNet_Baselines_for_Image_Restoration>
44. DRIFT: Deep Restoration, ISP Fusion, and Tone-mapping - arXiv, <https://arxiv.org/html/2604.03402v2>
45. [Literature Review] GMODiff: One-Step Gain Map Refinement with ..., <https://www.themoonlight.io/en/review/gmodiff-one-step-gain-map-refinement-with-diffusion-priors-for-hdr-reconstruction>
46. GMODiff: One-Step Gain Map Refinement with Diffusion Priors for HDR Reconstruction - arXiv, <https://arxiv.org/pdf/2512.16357>
47. High dynamic range image reconstruction based on efficient multi-scale attention, <https://opg.optica.org/ao/fulltext.cfm?uri=ao-64-21-5949>
48. HDR Real Estate Photography: Pro Settings, Bracketing & Editing (2026 Guide) - Bounti.ai, <https://bounti.ai/blog/photography/hdr-photography-real-estate>
49. 10 Best HDR for Real Estate in 2026 - Imagen AI, <https://imagen-ai.com/valuable-tips/best-hdr-for-real-estate/>
50. High-Volume Real Estate Photo Editing: Workflow, Techniques, and AI Integration - Imagen, <https://imagen-ai.com/valuable-tips/real-estate-photo-editing/>
51. HDR Real Estate Photography: Complete Shooting & Editing Guide - Amplifiles, <https://www.amplifiles.ai/blog/hdr-real-estate-photography>
52. HDR Drone Photography: A Guide for Real Estate Photographers - Photomatix, <https://hdrsoft.com/learn/hdr-drone-photography-guide-for-real-estate-photographers.html>
53. Real Estate Photography Tip: How to Do a Window Pull - YouTube, <https://www.youtube.com/shorts/ECZDU73WB0I>
54. 10 Best Real Estate Photo Editing Tools in 2026 - Imagen AI, <https://imagen-ai.com/valuable-tips/best-real-estate-photo-editing-tools/>
55. AI Photo Editing vs Photoshop for Real Estate: Speed, Cost & Quality Compared, <https://roomagen.com/blog/ai-photo-editing-vs-photoshop-real-estate>
56. Real Estate Photography HDR Window Pulls - Adobe Community, <https://community.adobe.com/questions-675/real-estate-photography-hdr-window-pulls-953265>
57. Detect and mask windows to balance inside and outside exposure - Imagen, <https://support.imagen-ai.com/hc/en-us/articles/24749571325469-Detect-and-mask-windows-to-balance-inside-and-outside-exposure>
58. Introducing Autoenhance.ai 4.0 Beta - Instant photo editing powered by AI, <https://autoenhance.ai/blog/introducing-autoenhance-ai-4-0-beta>
59. Edit real estate fast & deliver by 9 AM - Imagen AI, <https://imagen-ai.com/solution/edit-a-full-listing-in-minutes/>
60. Automated Real Estate Photo Enhancement using Deep Learning · ultralytics · Discussion #23716 - GitHub, <https://github.com/orgs/ultralytics/discussions/23716>
61. HDR Photography Real Estate: Complete Guide to Stunning Listings - Pedra's AI, <https://pedra.ai/blog/hdr-photography-real-estate>
62. NAFNet: Deblur & Restore Images with AI | fal, <https://fal.ai/models/fal-ai/nafnet/deblur>