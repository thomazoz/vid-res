"""Markdown -> styled HTML -> PDF (via headless Chrome).

    python3 make_pdf.py CONCLUSIONS.md [output.pdf]

No LaTeX needed; uses the Python `markdown` lib for rendering and Chrome's
print-to-PDF for a clean, paginated result with proper tables.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: Letter; margin: 18mm 16mm; }
* { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 11.5px; line-height: 1.55; color: #1b1b1b; margin: 0;
}
h1 { font-size: 23px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 22px 0 8px; padding-bottom: 4px;
     border-bottom: 2px solid #ddd; }
h3 { font-size: 13px; margin: 16px 0 6px; color: #333; }
p, li { margin: 6px 0; }
strong { color: #111; }
hr { border: none; border-top: 1px solid #e2e2e2; margin: 18px 0; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 10.5px;
       background: #f3f3f3; padding: 1px 4px; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid #e4e4e4; border-radius: 6px;
      padding: 10px 12px; overflow: auto; }
pre code { background: none; padding: 0; }
blockquote { margin: 12px 0; padding: 8px 14px; background: #f5f8ff;
             border-left: 4px solid #4a78d0; color: #234; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 10.8px; }
th, td { border: 1px solid #cfcfcf; padding: 5px 8px; text-align: left;
         vertical-align: top; }
th { background: #eef1f5; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
tr, table, pre, blockquote { page-break-inside: avoid; }
h1, h2, h3 { page-break-after: avoid; }
"""


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: make_pdf.py input.md [output.pdf]")
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".pdf")

    body = markdown.markdown(
        src.read_text(),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "nl2br"],
    )
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    html_path = src.with_suffix(".html")
    html_path.write_text(html)

    base = [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox"]
    uri = html_path.resolve().as_uri()
    # Newer Chrome supports --no-pdf-header-footer; fall back if it errors.
    for extra in (["--no-pdf-header-footer"], []):
        r = subprocess.run(
            base + extra + [f"--print-to-pdf={out}", uri],
            capture_output=True, text=True,
        )
        if out.exists() and out.stat().st_size > 0:
            break
    if not (out.exists() and out.stat().st_size > 0):
        sys.exit(f"PDF generation failed:\n{r.stderr[-800:]}")
    print(f"Wrote {out}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
