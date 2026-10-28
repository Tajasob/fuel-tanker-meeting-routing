#!/usr/bin/env python3
"""Check that the manuscript carries the current output of summarize.py.

    python analysis/check_inline.py MAIN.tex [SUPPLEMENT.tex ...] GENERATED_DIR

GENERATED_DIR holds macros.tex and tables/*.tex written by summarize.py. Every number
definition must appear verbatim in MAIN.tex, and every table body verbatim in one of the
given .tex files (a table may sit in the supplementary file).
"""
import sys
from pathlib import Path

*sources, generated = sys.argv[1:]
main = Path(sources[0]).read_text()
everything = '\n'.join(Path(s).read_text() for s in sources)
generated = Path(generated)
problems = []
for line in (generated / 'macros.tex').read_text().splitlines():
    if line.startswith('\\newcommand') and line not in main:
        problems.append(f'macro differs or is missing: {line}')
for table in sorted((generated / 'tables').glob('*.tex')):
    body = '\n'.join(l for l in table.read_text().splitlines() if not l.startswith('%')).strip()
    if body not in everything:
        problems.append(f'table differs or is missing: {table.name}')
if problems:
    sys.exit('\n'.join(problems))
print('the manuscript carries every generated number and table')
