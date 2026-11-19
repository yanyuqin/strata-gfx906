#!/usr/bin/env python3
"""Per-bucket dense-GEMM cost of the new table: sum of rocBLAS default ms vs the shipped solution's ms.

One call per shape and bucket (the tuning tool prints its own measurements in the row comment), so the sums
are the cost of one pass over the 15 prompt-path shapes, not the engine's per-chunk total (shapes like
N=1 are called once per token).  It is the honest way to compare the tail bucket: same shapes, same card,
same rocBLAS build, arithmetic-level timing.
"""
import re
import sys
from collections import defaultdict


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/yanyq/bench/gfx906-rocblas-final.txt"
    buckets = defaultdict(lambda: [0.0, 0.0, 0, 0])  # default_ms, best_ms, rows, tuned
    for line in open(path):
        if not line.startswith("f16o"):
            continue
        head, _, note = line.partition("#")
        p = head.split()
        t = int(p[4])
        m = re.search(r"([\d.]+) ms -> ([\d.]+) ms", note)
        d = float(m.group(1)) if m else float(re.search(r"([\d.]+) ms", note).group(1))
        b = float(m.group(2)) if m else d
        s = buckets[t]
        s[0] += d
        s[1] += b
        s[2] += 1
        s[3] += 1 if p[5] != "default" else 0
    print("%-6s %5s %5s %12s %12s %8s" % ("T", "rows", "tuned", "default ms", "shipped ms", "gain"))
    for t in sorted(buckets):
        d, b, rows, tuned = buckets[t]
        print("%-6d %5d %5d %12.3f %12.3f %7.2fx" % (t, rows, tuned, d, b, d / b if b else 0))


if __name__ == "__main__":
    main()
