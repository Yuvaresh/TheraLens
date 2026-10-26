#!/usr/bin/env python3
"""
TheraLens Objective Achievement Audit HTML & PDF Compiler
Renders docs/theralens_objective_achievement_audit.md to HTML and PDF.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
import markdown
import pypdf

DOCS_DIR = Path(__file__).resolve().parent
MD_PATH = DOCS_DIR / "theralens_objective_achievement_audit.md"
HTML_PATH = DOCS_DIR / "theralens_objective_achievement_audit.html"
PDF_PATH = DOCS_DIR / "theralens_objective_achievement_audit.pdf"

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

    # 3. Render HTML with tables and fenced_code
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
<title>TheraLens: Comprehensive Objective Achievement Audit Report</title>
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
    line-height: 1.35;
    color: #1e293b;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
}}

h1 {{
    font-size: 15pt;
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
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 2px;
}}

h3 {{
    font-size: 9.8pt;
    font-weight: 600;
    color: #0f766e;
    margin-top: 10px;
    margin-bottom: 4px;
}}

h4 {{
    font-size: 8.8pt;
    font-weight: 600;
    color: #334155;
    margin-top: 8px;
    margin-bottom: 2px;
}}

p {{
    margin: 4px 0 6px 0;
    text-align: justify;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0;
    font-size: 7.5pt;
    page-break-inside: avoid;
}}

th, td {{
    border: 1px solid #cbd5e1;
    padding: 4px 6px;
    text-align: left;
    vertical-align: top;
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
    margin-bottom: 2px;
}}

blockquote {{
    border-left: 3px solid #0284c7;
    margin: 6px 0;
    padding: 4px 10px;
    background-color: #f0f9ff;
    color: #0369a1;
    font-style: italic;
}}

hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 10px 0;
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
    print("Building Objective Achievement Audit HTML & PDF...")
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

    # Close any Acrobat processes that might lock the PDF
    try:
        subprocess.run(["powershell", "-Command", "Stop-Process -Name Acrobat -ErrorAction SilentlyContinue"], timeout=5)
    except Exception:
        pass

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


if __name__ == "__main__":
    build_pdf()
