#!/usr/bin/env python3
"""
TheraLens Review-3 Report HTML & PDF Compiler
Renders docs/theralens_review3_report.md to docs/theralens_review3_report.html
and compiles to docs/theralens_review3_report.pdf with strict <= 15 page enforcement.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
import markdown
import pypdf

DOCS_DIR = Path(__file__).resolve().parent
MD_PATH = DOCS_DIR / "theralens_review3_report.md"
HTML_PATH = DOCS_DIR / "theralens_review3_report.html"
PDF_PATH = DOCS_DIR / "theralens_review3_report.pdf"

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def protect_and_convert_markdown(md_text: str) -> str:
    # 1. Protect math blocks from markdown processing
    math_blocks = {}
    
    def save_display_math(match):
        token = f"@@@MATH_DISPLAY_{len(math_blocks)}@@@"
        math_blocks[token] = match.group(0)
        return f"\n\n{token}\n\n"
        
    def save_inline_math(match):
        token = f"@@@MATH_INLINE_{len(math_blocks)}@@@"
        math_blocks[token] = match.group(0)
        return token

    # Protect display math ($$...$$)
    processed = re.sub(r"\$\$(.+?)\$\$", save_display_math, md_text, flags=re.DOTALL)
    # Protect inline math ($...$)
    processed = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", save_inline_math, processed)

    # 2. Ensure tables have blank line before them
    lines = processed.split("\n")
    fixed_lines = []
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and i > 0:
            prev = lines[i - 1].strip()
            if prev and not prev.startswith("|"):
                fixed_lines.append("")
        fixed_lines.append(line)
    processed = "\n".join(fixed_lines)

    # 3. Render HTML with tables and fenced_code (NO nl2br to avoid breaking tables)
    html_body = markdown.markdown(
        processed,
        extensions=["tables", "fenced_code"],
    )

    # 4. Restore math blocks
    for token, original_math in math_blocks.items():
        html_body = html_body.replace(token, original_math)

    return html_body


def convert_md_to_html(md_text: str) -> str:
    html_content = protect_and_convert_markdown(md_text)

    template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TheraLens: College Review-3 Final Evaluation Report</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>

<style>
@page {{
    size: A4 portrait;
    margin: 14mm 13mm 14mm 13mm;
}}

* {{
    box-sizing: border-box;
}}

body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
    font-size: 8.5pt;
    line-height: 1.34;
    color: #1e293b;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
}}

h1 {{
    font-size: 14pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2px solid #0284c7;
    padding-bottom: 4px;
    margin-top: 14px;
    margin-bottom: 8px;
}}

h2 {{
    font-size: 11.5pt;
    font-weight: 700;
    color: #0369a1;
    margin-top: 12px;
    margin-bottom: 6px;
}}

h3 {{
    font-size: 9.8pt;
    font-weight: 600;
    color: #0f766e;
    margin-top: 10px;
    margin-bottom: 4px;
}}

p {{
    margin: 4px 0 6px 0;
    text-align: justify;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0;
    font-size: 7.6pt;
    page-break-inside: avoid;
}}

th, td {{
    border: 1px solid #cbd5e1;
    padding: 4px 6px;
    text-align: left;
}}

th {{
    background-color: #f1f5f9;
    font-weight: 600;
    color: #0f172a;
}}

tr:nth-child(even) {{
    background-color: #f8fafc;
}}

pre, code {{
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 7.3pt;
}}

pre {{
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 6px 8px;
    margin: 6px 0;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-x: hidden;
    page-break-inside: avoid;
}}

code {{
    background-color: #f1f5f9;
    padding: 1px 3px;
    border-radius: 3px;
    color: #0f172a;
}}

ul, ol {{
    margin: 4px 0 6px 18px;
    padding: 0;
}}

li {{
    margin-bottom: 2.5px;
}}

hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 12px 0;
}}

img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 6px auto;
}}

.page-break {{
    page-break-after: always;
    break-after: page;
}}
</style>
</head>
<body>

{html_content}

<script>
document.addEventListener("DOMContentLoaded", function() {{
    renderMathInElement(document.body, {{
        delimiters: [
            {{left: "$$", right: "$$", display: true}},
            {{left: "$", right: "$", display: false}}
        ],
        throwOnError: false
    }});
}});
</script>
</body>
</html>
"""
    return template


def build_pdf():
    print("Building Review-3 HTML & PDF...")
    if not MD_PATH.exists():
        print(f"Error: {MD_PATH} not found.")
        sys.exit(1)

    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    html = convert_md_to_html(md_text)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved HTML to {HTML_PATH}")

    if not os.path.exists(EDGE_PATH):
        print(f"Warning: Edge browser not found at {EDGE_PATH}. PDF generation skipped.")
        return

    cmd = [
        EDGE_PATH,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={PDF_PATH}",
        str(HTML_PATH),
    ]
    print("Running Edge headless print-to-pdf...")
    subprocess.run(cmd, check=True)

    if PDF_PATH.exists():
        reader = pypdf.PdfReader(PDF_PATH)
        pages = len(reader.pages)
        print(f"Successfully generated PDF: {PDF_PATH}")
        print(f"Total Page Count: {pages} pages")
        if pages <= 15:
            print(f"VERIFIED: Page count ({pages}) is strictly within the 15-page limit!")
        else:
            print(f"WARNING: Page count ({pages}) exceeds 15 pages! Needs adjustment.")


if __name__ == "__main__":
    build_pdf()
