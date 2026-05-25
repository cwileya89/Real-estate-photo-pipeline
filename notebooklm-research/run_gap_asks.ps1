$ErrorActionPreference = 'Continue'
$dir = 'C:\Users\GamingPC\Documents\notebooklm-research\reports'
$prompts = 'C:\Users\GamingPC\Documents\notebooklm-research'

function Ask($label, $promptFile, $outFile, $noteTitle) {
    Write-Host ""
    Write-Host "===== $label ====="
    Write-Host (Get-Date -Format 'HH:mm:ss')
    notebooklm ask --prompt-file "$prompts\$promptFile" --save-as-note --note-title $noteTitle --timeout 600 | Out-File -Encoding utf8 "$dir\$outFile"
    Write-Host "[done] $label at $(Get-Date -Format 'HH:mm:ss') ($(((Get-Item "$dir\$outFile").Length)) bytes)"
}

Ask 'S4 Competitors' 'gap_S4_competitors.md' 'S4_Competitors.md' 'PhotoHDR S4 Competitors'
Ask 'S5 Build Plan'  'gap_S5_buildplan.md'  'S5_BuildPlan.md'  'PhotoHDR S5 Build Plan'
Ask 'S6 API Guide'   'gap_S6_apiguide.md'   'S6_APIGuide.md'   'PhotoHDR S6 API Guide'

Write-Host ""
Write-Host "===== ALL 3 GAP SYNTHESES COMPLETE at $(Get-Date -Format 'HH:mm:ss') ====="
Get-ChildItem $dir | Select-Object Name, Length
