#!/usr/bin/env python3
"""cmp_pfpk.py -- same-id comparison of an eval100 run against the opt1117 baseline.

usage: cmp_pfpk.py <new.jsonl> [baseline.jsonl]
Prints per-id score/pred, the flips, and the summary accuracy of both arms.
"""
import json
import sys

new = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "/home/yanyq/eval100-kit/results/eval100_opt_2026-11-17.jsonl"


def load(p):
    rows = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[r["id"]] = r
    return rows


def summarise(tag, rows):
    ok = [r for r in rows.values() if "error" not in r]
    mm = [r for r in ok if r.get("kind") == "mmlu"]
    dr = [r for r in ok if r.get("kind") == "drop"]
    acc = sum(r["score"] for r in mm) / max(1, len(mm))
    f1 = sum(r.get("f1", r["score"]) for r in dr) / max(1, len(dr))
    em = sum(r.get("em", r["score"]) for r in dr) / max(1, len(dr))
    sec = sum(r["sec"] for r in ok) / max(1, len(ok))
    print(f"SUMMARY {tag}: MMLU {len(mm)} acc={100*acc:.1f}% | DROP {len(dr)} F1={100*f1:.1f}% EM={100*em:.1f}% "
          f"| errors={len(rows)-len(ok)} | mean={sec:.1f}s")
    return ok


b, n = load(base), load(new)
summarise("baseline", b)
summarise("new", n)
ids = sorted(set(b) & set(n), key=lambda x: (x.split("_")[0], int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0))
print(f"common ids: {len(ids)}  (baseline {len(b)}, new {len(n)})")
flips, same_ans = [], 0
for i in ids:
    ob, on = b[i], n[i]
    sb, sn = int(ob["score"]), int(on["score"])
    pb, pn = str(ob.get("pred", ""))[:40], str(on.get("pred", ""))[:40]
    if pb == pn:
        same_ans += 1
    if sb != sn:
        flips.append((i, sb, sn))
    flag = "FLIP" if sb != sn else ("   " if pb == pn else "pred")
    print(f"{flag} {i:10s} base={sb} new={sn} | {pb!r} -> {pn!r}")
print(f"identical answers: {same_ans}/{len(ids)}  flips: {len(flips)} {flips}")
