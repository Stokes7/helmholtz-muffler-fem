#!/usr/bin/env python3
"""
MSSD Helmholtz Muffler Project - Local Documentation Compiler
Translates report/report.md into public/index.html with MathJax LaTeX rendering.
"""

import os
import re
import shutil
import sys


def _compile_markdown_with_math(md_text):
    """
    Compile Markdown to HTML while shielding LaTeX math from the Markdown parser.

    Strategy: replace every $$...$$ and $...$ block with a unique placeholder
    before calling markdown(), then restore them afterwards wrapped in \\[...\\]
    and \\(...\\) for MathJax 3 to process.
    """
    try:
        import markdown
    except ImportError:
        print("Error: 'markdown' package not installed. Run: pip install markdown")
        sys.exit(1)

    store = {}
    counter = [0]

    def _key():
        k = f'XMATHX{counter[0]:04d}X'
        counter[0] += 1
        return k

    # 1. Protect display math $$...$$ first (avoids being caught by inline rule)
    def _protect_display(m):
        k = _key()
        store[k] = ('display', m.group(1))
        return f'\n\n{k}\n\n'

    md_text = re.sub(r'\$\$([\s\S]*?)\$\$', _protect_display, md_text)

    # 2. Protect inline math $...$
    def _protect_inline(m):
        k = _key()
        store[k] = ('inline', m.group(1))
        return k

    md_text = re.sub(r'\$([^\$\n]+?)\$', _protect_inline, md_text)

    # 3. Run standard Markdown (no arithmatex needed)
    html = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])

    # 4. Restore math wrapped for MathJax \(...\) / \[...\]
    for k, (kind, content) in store.items():
        if kind == 'display':
            block = f'<div class="math-display">\\[{content}\\]</div>'
            # markdown wraps lone paragraph tokens in <p>; strip that wrapper
            html = html.replace(f'<p>{k}</p>', block)
            html = html.replace(k, block)
        else:
            html = html.replace(k, f'\\({content}\\)')

    return html


def main():
    print("============================================================")
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    os.makedirs('public', exist_ok=True)

    # Copy results/figures into public/ for web access
    if os.path.exists('results'):
        if os.path.exists('public/results'):
            shutil.rmtree('public/results')
        shutil.copytree('results', 'public/results')
        print("[1/2] Copied 'results/' → 'public/results/'")
    else:
        print("[WARNING] 'results/' not found — figures will not render.")

    # Copy report/assets/ into public/assets/
    if os.path.exists('report/assets'):
        if os.path.exists('public/assets'):
            shutil.rmtree('public/assets')
        shutil.copytree('report/assets', 'public/assets')
        print("[1b/2] Copied 'report/assets/' → 'public/assets/'")

    # Compile report
    report_path = 'report/report.md'
    if not os.path.exists(report_path):
        print(f"[ERROR] '{report_path}' not found!")
        sys.exit(1)

    with open(report_path, 'r', encoding='utf-8') as f:
        report_raw = f.read()

    report_raw = report_raw.replace('../results/figures/', 'results/figures/')
    report_html = _compile_markdown_with_math(report_raw)
    print("[2/2] Compiled report/report.md successfully.")

    html_content = f'''<!DOCTYPE html>
<html data-theme="light">
<head>
    <title>Helmholtz Muffler FEM — Scientific Report</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['\\(', '\\)']],
                displayMath: [['\\[', '\\]']]
            }}
        }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" async></script>
    <style>
        body {{ padding: 2rem; font-family: system-ui, -apple-system, sans-serif; }}
        pre {{ background: #1e1e1e; padding: 1.2rem; border-radius: 6px; overflow-x: auto; border: 1px solid #333; }}
        code {{ color: #ff79c6; background: #282a36; padding: 0.2rem 0.4rem; border-radius: 4px; }}
        .header-container {{ border-bottom: 1px solid #444; padding-bottom: 1.5rem; margin-bottom: 2rem; }}
        .math-display {{ overflow-x: auto; margin: 1.5rem 0; text-align: center; }}
        img {{ border-radius: 8px; border: 1px solid #444; margin: 1.5rem 0; max-width: 100%; height: auto; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0; }}
        th, td {{ border: 1px solid #444; padding: 0.8rem; text-align: left; }}
        th {{ background-color: #222; }}
        blockquote {{ border-left: 4px solid #ff79c6; background: #202020; margin: 1rem 0; padding: 0.8rem; border-radius: 0 4px 4px 0; color: #aaa; }}
    </style>
</head>
<body class="container">
    <header class="header-container">
        <h1 style="margin:0; font-size: 1.8rem;">Helmholtz Acoustic Muffler FEM</h1>
        <p style="margin:0; color: #888;">MSSD Course Project 01 — Finite Element Method</p>
    </header>
    <main>
        {report_html}
    </main>
    <footer style="text-align: center; margin-top: 4rem; padding-top: 2rem; border-top: 1px solid #444; color: #666; font-size: 0.9rem;">
        <p>© 2026 RWTH Aachen University — Modern Simulation Software Development</p>
    </footer>
</body>
</html>'''

    with open('public/index.html', 'w', encoding='utf-8') as w:
        w.write(html_content)

    print("SUCCESS: public/index.html generated!")
    print("============================================================")


if __name__ == '__main__':
    main()
