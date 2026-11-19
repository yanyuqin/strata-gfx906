#!/usr/bin/env python3
"""summarize.py -- one table from every bench-*.jsonl under results/."""
import glob, json, os, sys

rows = {}
for f in glob.glob(os.path.join(os.path.dirname(__file__), "..", "results", "bench-*.jsonl")):
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rows[d["tag"]] = d

hdr = f'{"tag":<16}{"prompt_n":>9}{"prefill t/s":>12}{"decode t/s":>11}{"TTFT s":>8}{"draft_n":>8}{"accepted":>9}'
print(hdr)
print("-" * len(hdr))
for k in sorted(rows):
    d = rows[k]
    def g(x): return d.get(x)
    print(f'{k:<16}{g("prompt_n") or 0:>9}{g("prefill_tps") or 0:>12.1f}{g("decode_tps") or 0:>11.2f}'
          f'{g("ttft_s") or 0:>8.2f}{g("draft_n") or 0:>8}{g("draft_acc") or 0:>9}')
