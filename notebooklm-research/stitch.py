"""Stitch the 7 NotebookLM reports into one consolidated document."""
import re
from pathlib import Path

ROOT = Path(__file__).parent / 'reports'
OUT = Path(__file__).parent / 'PhotoHDR_MV_consolidated.md'


def read(name):
    return (ROOT / name).read_text(encoding='utf-8')


def demote(md: str, levels: int = 2) -> str:
    """Demote markdown heading levels by N (e.g. # -> ###)."""
    def fix(m):
        hashes = m.group(1)
        new = '#' * min(len(hashes) + levels, 6)
        return f"{new}{m.group(2)}"
    return re.sub(r'^(#{1,6})( +.+)$', fix, md, flags=re.MULTILINE)


def strip_ask_preamble(md: str) -> str:
    """Strip leading 'Continuing conversation...' / 'Answer:' / BOM artifacts from ask output."""
    lines = md.splitlines()
    drop_prefixes = ('continuing conversation', 'answer:', '﻿')
    while lines:
        first = lines[0].lstrip('﻿').strip().lower()
        if not first or any(first.startswith(p) for p in drop_prefixes):
            lines = lines[1:]
            continue
        break
    return '\n'.join(lines).lstrip()


pieces = []

pieces.append("""# PhotoHDR MV Real Estate, Consolidated Research Report

Generated 2026-05-24 from NotebookLM deep research grounded in 570 imported sources covering AutoHDR algorithm internals, training datasets, off-the-shelf AI models, competitor landscape, build planning, and API integration patterns.

Output sections map to the original research prompt. Sections 4, 5, 6 are inline. Sections 1, 2, 3 are heavy with technical detail and live as Appendices A through D so the build-plan content stays scannable on mobile.

---

## Table of Contents

- [Section 4, Competitor Deep Research](#section-4-competitor-deep-research)
- [Section 5, Recommended Phased Build Plan](#section-5-recommended-phased-build-plan)
- [Section 6, Specific API Integration Guide](#section-6-specific-api-integration-guide)
- [Appendix A, AutoHDR Algorithm Technical Paradigm](#appendix-a-autohdr-algorithm-technical-paradigm) (covers Section 1 algorithmic core)
- [Appendix B, Exposure Fusion, U-Net, NAFNet](#appendix-b-exposure-fusion-u-net-nafnet) (covers Section 1 detail + Section 3.1)
- [Appendix C, Window Pull, SAM 2, DeepLabV3+, GroundingDINO](#appendix-c-window-pull-sam-2-deeplabv3-groundingdino) (covers Section 1.2 + Section 3.2)
- [Appendix D, Datasets, Benchmarks, Legal Compliance](#appendix-d-datasets-benchmarks-legal-compliance) (covers Section 2)

---

""")

pieces.append("## Section 4, Competitor Deep Research\n\n")
pieces.append(strip_ask_preamble(read('S4_Competitors.md')))
pieces.append("\n\n---\n\n")

pieces.append("## Section 5, Recommended Phased Build Plan\n\n")
pieces.append(strip_ask_preamble(read('S5_BuildPlan.md')))
pieces.append("\n\n---\n\n")

pieces.append("## Section 6, Specific API Integration Guide\n\n")
pieces.append(strip_ask_preamble(read('S6_APIGuide.md')))
pieces.append("\n\n---\n\n")

pieces.append("## Appendix A, AutoHDR Algorithm Technical Paradigm\n\n")
pieces.append("Covers the original prompt Section 1 algorithmic core. Source: NotebookLM deep-research synthesis report.\n\n")
pieces.append(demote(read('S1_AutoHDR_algorithm.md'), 2))
pieces.append("\n\n---\n\n")

pieces.append("## Appendix B, Exposure Fusion, U-Net, NAFNet\n\n")
pieces.append("Covers Section 1 (Mertens fusion, alignment, deghosting) and Section 3.1 (NAFNet, LightFuse, deep gated HDR networks). Source: NotebookLM deep-research synthesis report.\n\n")
pieces.append(demote(read('S2_Exposure_fusion_NAFNet.md'), 2))
pieces.append("\n\n---\n\n")

pieces.append("## Appendix C, Window Pull, SAM 2, DeepLabV3+, GroundingDINO\n\n")
pieces.append("Covers Section 1.2 (window pull technique) and Section 3.2 (segmentation models). Source: NotebookLM deep-research synthesis report.\n\n")
pieces.append(demote(read('S3_Window_pull_SAM.md'), 2))
pieces.append("\n\n---\n\n")

pieces.append("## Appendix D, Datasets, Benchmarks, Legal Compliance\n\n")
pieces.append("Covers Section 2 (training data sources, licensing). Source: NotebookLM deep-research synthesis report.\n\n")
pieces.append(demote(read('S4_Datasets_legal.md'), 2))
pieces.append("\n")

text = ''.join(pieces)
OUT.write_text(text, encoding='utf-8')
print(f"Wrote {OUT}")
print(f"Size: {OUT.stat().st_size:,} bytes ({OUT.stat().st_size / 1024:.1f} KB)")
print(f"Lines: {text.count(chr(10)) + 1:,}")
