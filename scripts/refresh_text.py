"""Apply markdown-only edits from build_notebook.py to the executed notebook, without re-running it.

Code cells whose source is unchanged keep their outputs; a changed code cell is reported, which means
a full re-run is needed.

Usage:  .venv/bin/python scripts/refresh_text.py notebooks/fraud_analysis.ipynb
"""
import subprocess
import sys
import tempfile
from pathlib import Path
import nbformat as nbf

target = Path(sys.argv[1])
with tempfile.TemporaryDirectory() as tmp:
    fresh_path = Path(tmp) / 'fresh.ipynb'
    subprocess.run([sys.executable, str(Path(__file__).with_name('build_notebook.py')), str(fresh_path)], check=True)
    fresh = nbf.read(fresh_path, as_version=4)

executed = nbf.read(target, as_version=4)
outputs = {c.source: c for c in executed.cells if c.cell_type == 'code'}
missing = 0
for cell in fresh.cells:
    if cell.cell_type != 'code':
        continue
    old = outputs.get(cell.source)
    if old is None:
        missing += 1
        print('CHANGED CODE CELL (needs a full re-run):', cell.source.splitlines()[0][:80])
    else:
        cell.outputs, cell.execution_count = old.outputs, old.execution_count

if missing:
    sys.exit(f'{missing} code cell(s) changed; run the notebook instead of refreshing text.')
nbf.write(fresh, target)
print(f'refreshed text in {target} ({len(fresh.cells)} cells, all outputs kept)')
