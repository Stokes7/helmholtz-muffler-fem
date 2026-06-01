#!/usr/bin/env python3
"""
MSSD Helmholtz Muffler Project - Local Documentation Compiler
Translates README.md and report/report.md into public/index.html with MathJax LaTeX rendering.
"""

import os
import shutil
import sys

def main():
    # 1. Check if markdown package is installed
    try:
        import markdown
    except ImportError:
        print("Error: The 'markdown' package is not installed in the active environment.")
        print("Please install it using: pip install markdown")
        sys.exit(1)

    print("============================================================")
    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # 2. Ensure public/ results exist
    os.makedirs('public', exist_ok=True)
    
    # 3. Copy results/figures folder into public/ for web access
    if os.path.exists('results'):
        if os.path.exists('public/results'):
            shutil.rmtree('public/results')
        shutil.copytree('results', 'public/results')
        print("[1/3] Successfully copied 'results/' to 'public/results/'")
    else:
        print("[WARNING] 'results/' directory not found. Figures will not render locally.")

    # 3b. Copy report/assets/ into public/assets/ if it exists
    if os.path.exists('report/assets'):
        if os.path.exists('public/assets'):
            shutil.rmtree('public/assets')
        shutil.copytree('report/assets', 'public/assets')
        print("[1b/3] Successfully copied 'report/assets/' to 'public/assets/'")

    # 4. Read and compile report/report.md (replacing relative image paths)
    report_path = 'report/report.md'
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as r:
            report_raw = r.read()
            # Correct image paths from ../results/figures to results/figures for the hosted folder
            report_raw = report_raw.replace('../results/figures/', 'results/figures/')
            report_html = markdown.markdown(report_raw, extensions=['tables', 'fenced_code'])
        print("[2/2] Compiled report/report.md successfully.")
    else:
        print(f"[ERROR] '{report_path}' not found!")
        sys.exit(1)

    # 6. Write final compiled index.html with dark theme and MathJax LaTeX parser
    html_content = f'''<!DOCTYPE html>
<html data-theme="light">
<head>
    <title>Helmholtz Muffler FEM Documentation & Report</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }}
        }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" async></script>
    <style>
        body {{ padding: 2rem; font-family: system-ui, -apple-system, sans-serif; }}
        pre {{ background: #1e1e1e; padding: 1.2rem; border-radius: 6px; overflow-x: auto; border: 1px solid #333; }}
        code {{ color: #ff79c6; background: #282a36; padding: 0.2rem 0.4rem; border-radius: 4px; }}
        .header-container {{ border-bottom: 1px solid #444; padding-bottom: 1.5rem; margin-bottom: 2rem; }}
        img {{ border-radius: 8px; border: 1px solid #444; margin: 1.5rem 0; max-width: 100%; height: auto; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0; }}
        th, td {{ border: 1px solid #444; padding: 0.8rem; text-align: left; }}
        th {{ background-color: #222; }}
        blockquote {{ border-left: 4px solid #ff79c6; padding-left: 1rem; color: #aaa; background: #202020; margin: 1rem 0; padding: 0.8rem; border-radius: 0 4px 4px 0; }}
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
