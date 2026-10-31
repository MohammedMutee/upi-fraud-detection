"""Converts report/business_report.md into report/business_report.pdf (Markdown → styled HTML → wkhtmltopdf).

Usage:  .venv/bin/python scripts/build_report_pdf.py
"""
import re
import subprocess
import tempfile
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / 'report' / 'business_report.md', ROOT / 'report' / 'business_report.pdf'

CSS = """
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1f2937; }
h1 { color: #14213d; font-size: 24pt; margin: 0 0 2pt; }
h2 { color: #14213d; font-size: 14.5pt; border-bottom: 2px solid #c2571a; padding-bottom: 3pt; margin-top: 18pt;
     page-break-after: avoid; }
h3 { color: #4a5568; font-size: 12pt; margin: 0 0 6pt; font-weight: normal; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9.5pt; page-break-inside: avoid; }
th { background: #14213d; color: #fff; text-align: left; padding: 4pt 6pt; }
td { border-bottom: 1px solid #e3e6e4; padding: 4pt 6pt; vertical-align: top; }
blockquote { background: #fdebdd; border-left: 4px solid #c2571a; margin: 8pt 0; padding: 6pt 10pt; }
img { display: block; margin: 6pt auto; max-width: 100%; max-height: 62mm; page-break-inside: avoid; }
.keep { page-break-inside: avoid; }
code { background: #f1f1ee; padding: 0 3pt; font-size: 9pt; }
hr { border: none; border-top: 1px solid #e3e6e4; }
"""

html = markdown.markdown(SRC.read_text(), extensions=['tables'])
# wkhtmltopdf ignores page-break-after on headings, so glue each <h2> to the block that follows it
html = re.sub(r'(<h2>.*?</h2>\s*<(p|table|blockquote|ul|ol)\b.*?</\2>)', r'<div class="keep">\1</div>', html, flags=re.S)
html = re.sub(r'(<p><strong>[^<]*</strong></p>\s*<table\b.*?</table>)', r'<div class="keep">\1</div>', html, flags=re.S)
with tempfile.NamedTemporaryFile('w', suffix='.html', dir=SRC.parent, delete=False) as f:
    # Saved next to the .md so the ../outputs/figures/ image paths resolve the same way
    f.write(f'<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{html}</body></html>')
    tmp = Path(f.name)
try:
    subprocess.run(['wkhtmltopdf', '--quiet', '--enable-local-file-access', '--page-size', 'A4',
                    '--margin-top', '16mm', '--margin-bottom', '16mm', '--margin-left', '16mm', '--margin-right', '16mm',
                    str(tmp), str(OUT)], check=True)
finally:
    tmp.unlink()
print('wrote', OUT)
