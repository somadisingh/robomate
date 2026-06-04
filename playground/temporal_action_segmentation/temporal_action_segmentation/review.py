from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _rel(path: str | Path, base: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_uri()


def write_review_html(records: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = output_path.parent
    rows = []
    for index, record in enumerate(records, start=1):
        sheet = record.get("contact_sheet_path")
        image_html = ""
        if sheet:
            image_html = f'<img src="{html.escape(_rel(sheet, base))}" alt="contact sheet">'
        payload = html.escape(json.dumps(record, indent=2))
        caption = html.escape(str(record.get("caption", "N/A")))
        confidence = float(record.get("confidence", 0.0) or 0.0)
        rows.append(
            f"""
            <article class="clip">
              <header>
                <strong>{index}. {html.escape(record["video_id"])} / {html.escape(record["hand"])}</strong>
                <span>{record["start_sec"]:.2f}s - {record["end_sec"]:.2f}s</span>
              </header>
              {image_html}
              <div class="label">
                <span class="caption" contenteditable="true">{caption}</span>
                <span class="confidence">{confidence:.2f}</span>
              </div>
              <details><summary>JSON</summary><pre>{payload}</pre></details>
            </article>
            """
        )

    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TAS Review</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f6f6f3; color: #20201d; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 28px auto 48px; }}
    h1 {{ font-size: 22px; margin: 0 0 4px; letter-spacing: 0; }}
    .meta {{ margin: 0 0 20px; color: #626258; font-size: 14px; }}
    .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }}
    .clip {{ background: #fff; border: 1px solid #deded6; border-radius: 8px; overflow: hidden; }}
    header {{ display: flex; justify-content: space-between; gap: 10px; padding: 10px 12px; font-size: 13px; border-bottom: 1px solid #eeeeea; }}
    header span {{ color: #626258; white-space: nowrap; }}
    img {{ display: block; width: 100%; height: auto; background: #ecece7; }}
    .label {{ display: flex; justify-content: space-between; gap: 12px; padding: 10px 12px; align-items: center; }}
    .caption {{ min-width: 0; font-size: 14px; font-weight: 650; }}
    .confidence {{ font-variant-numeric: tabular-nums; color: #626258; font-size: 13px; }}
    details {{ border-top: 1px solid #eeeeea; padding: 8px 12px 10px; }}
    summary {{ cursor: pointer; color: #626258; font-size: 13px; }}
    pre {{ overflow: auto; font-size: 12px; line-height: 1.45; }}
  </style>
</head>
<body>
  <main>
    <h1>Temporal Action Segmentation Review</h1>
    <p class="meta">{len(records)} proposed clips. Edit captions inline for triage, then copy accepted changes back into JSONL if needed.</p>
    <section class="grid">
      {''.join(rows)}
    </section>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return output_path
