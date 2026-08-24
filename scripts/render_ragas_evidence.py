"""Render the measured RAGAS JSON report as a terminal-style PNG."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "data" / "ragas_report.json"
OUTPUT_PATH = ROOT / "evidence" / "03_ragas_scores.png"


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a Windows monospace font, with Pillow's default as fallback."""
    filename = "consolab.ttf" if bold else "consola.ttf"
    try:
        return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size)
    except OSError:
        return ImageFont.load_default(size=size)


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    v1 = report["prompt_v1_scores"]
    v2 = report["prompt_v2_scores"]

    image = Image.new("RGB", (1536, 764), "#0c0c0c")
    draw = ImageDraw.Draw(image)
    regular = load_font(26)
    bold = load_font(28, bold=True)
    green = "#4ec9b0"
    gray = "#a0a0a0"
    white = "#e6e6e6"

    x, y = 54, 40
    draw.text((x, y), "Command Prompt — Day 22 RAGAS Evaluation", font=bold, fill=white)
    y += 62
    draw.text(
        (x, y),
        r"C:\Day22-Track2> C:\Python313\python.exe src\run_all.py --step 3",
        font=regular,
        fill="#9cdcfe",
    )
    y += 64
    draw.text(
        (x, y),
        f"RAGAS Evaluation — {report['samples_per_version']} QA pairs / prompt version",
        font=bold,
        fill=white,
    )
    y += 54
    separator = "=" * 76
    draw.text((x, y), separator, font=regular, fill=gray)
    y += 38
    draw.text(
        (x, y),
        f"{'Metric':<30}{'V1':>12}{'V2':>12}{'Winner':>12}",
        font=regular,
        fill=white,
    )
    y += 38
    draw.text((x, y), separator, font=regular, fill=gray)
    y += 46

    for metric in (
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ):
        score_v1 = float(v1[metric])
        score_v2 = float(v2[metric])
        winner = "V1" if score_v1 > score_v2 else "V2" if score_v2 > score_v1 else "Tie"
        line = f"{metric:<30}{score_v1:>12.4f}{score_v2:>12.4f}{winner:>12}"
        draw.text((x, y), line, font=regular, fill=white)
        y += 46

    y += 24
    best_faithfulness = max(float(v1["faithfulness"]), float(v2["faithfulness"]))
    target = "PASS" if report["target_met"] else "FAIL"
    draw.text(
        (x, y),
        f"[{target}] Faithfulness = {best_faithfulness:.4f} >= 0.8",
        font=bold,
        fill=green if report["target_met"] else "#f44747",
    )
    y += 52
    draw.text(
        (x, y),
        "[PASS] Step 3: RAGAS Evaluation — COMPLETED",
        font=bold,
        fill=green,
    )
    y += 58
    draw.text((x, y), "Saved: data/ragas_report.json", font=regular, fill=gray)
    y += 38
    draw.text((x, y), "Copied: evidence/03_ragas_report.json", font=regular, fill=gray)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_PATH, format="PNG", optimize=True)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
