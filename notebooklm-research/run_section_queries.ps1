$ErrorActionPreference = 'Continue'

function Step($label, $block) {
    Write-Host ""
    Write-Host "===== $label ====="
    Write-Host (Get-Date -Format 'HH:mm:ss')
    & $block
    Write-Host "[done] $label at $(Get-Date -Format 'HH:mm:ss')"
}

Step 'S1: wait for in-flight AutoHDR algorithm probe' {
    notebooklm research wait --import-all --timeout 3600
}

Step 'S2: Mertens fusion, U-Net NAFNet, HDR alignment for real estate' {
    notebooklm source add-research 'Mertens exposure fusion U-Net NAFNet HDR bracketing alignment de-ghosting real estate photography' --mode deep --import-all --timeout 3600
}

Step 'S3: Window pull, SAM 2, DeepLabV3+, GroundingDINO segmentation' {
    notebooklm source add-research 'Window pull semantic segmentation SAM 2 DeepLabV3+ GroundingDINO Autoenhance.ai real estate interior' --mode deep --import-all --timeout 3600
}

Step 'S4: Real estate HDR training datasets and licensing' {
    notebooklm source add-research 'Real estate photo HDR training datasets MIT-Adobe FiveK Hugging Face Kaggle MLS licensing commercial' --mode deep --import-all --timeout 3600
}

Step 'S5: Generative fill APIs (Firefly, GPT Image, Nano Banana, SDXL, Flux)' {
    notebooklm source add-research 'Adobe Firefly Generative Fill API GPT Image 2 OpenAI Nano Banana Higgsfield Stable Diffusion SDXL Flux inpainting Replicate cost per image' --mode deep --import-all --timeout 3600
}

Step 'S6: Competitor landscape and perspective/sky/upscale pipeline' {
    notebooklm source add-research 'Autoenhance Fotello BoxBrownie Imagen AI Phixer Styldod Photomatix Luminar Neo pricing architecture, sky replacement SkyAR Real-ESRGAN perspective correction Hugin OpenCV Lightroom API' --mode deep --import-all --timeout 3600
}

Write-Host ""
Write-Host "===== ALL 6 SECTIONS COMPLETE at $(Get-Date -Format 'HH:mm:ss') ====="
notebooklm source list
