#!/usr/bin/env python3
"""Merge the two per-card tuning outputs into one gfx906-rocblas.txt.

Rows are keyed by (N, K, ldy, T).  The two groups cover disjoint shapes, so a duplicate would mean a shape
was tuned twice; keep the row with the larger gain (comment "A ms -> B ms"), and warn if two rows disagree
on the solution index itself.  Header (arch + rocBLAS version) must match between both files.
"""
import re
import sys

HEADER_MAGIC = "STRATA_ROCBLAS_TUNING_V1"


def read(path):
    header, rows = None, {}
    for line in open(path):
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        if header is None:
            header = s
            if not header.startswith(HEADER_MAGIC):
                sys.exit("bad header in %s: %s" % (path, header))
            continue
        parts = s.split()
        if len(parts) != 6 or parts[0] != "f16o":
            sys.exit("bad row in %s: %s" % (path, s))
        n, k, ldy, t = (int(x) for x in parts[1:5])
        sol = parts[5]
        gain = 1.0
        m = re.search(r"([\d.]+) ms -> ([\d.]+) ms", line)
        if m:
            d, b = float(m.group(1)), float(m.group(2))
            if b > 0:
                gain = d / b
        key = (n, k, ldy, t)
        if key in rows:
            prev_sol, prev_gain = rows[key][0], rows[key][1]
            if prev_sol != sol:
                sys.stderr.write("WARN %s solution differs: %s vs %s\n" % (key, prev_sol, sol))
            if prev_gain >= gain:
                continue
        rows[key] = (sol, gain, line.split("#", 1)[-1].strip() if "#" in line else "")
    return header, rows


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: merge_tuning.py OUT.txt IN1.txt [IN2.txt ...]")
    out = sys.argv[1]
    headers, merged = [], {}
    for p in sys.argv[2:]:
        h, r = read(p)
        headers.append(h)
        for key, val in r.items():
            if key not in merged or val[1] > merged[key][1]:
                merged[key] = val
    if len(set(headers)) != 1:
        sys.exit("header mismatch between inputs: %r" % (headers,))
    with open(out, "w") as f:
        f.write("# rocBLAS solution indices for the prompt path's FP16-out GEMMs; valid for this architecture and rocBLAS build only\n")
        f.write("# rows: type N K ldy T solution   (type f16o: FP16 in, FP16 out, FP32 compute; Gemm::f16_inplace)\n")
        f.write("# solution: rocBLAS's solution index, or `default` where its own choice measured best at that bucket\n")
        f.write(headers[0] + "\n")
        tuned = 0
        for key in sorted(merged, key=lambda x: (x[0], x[1], x[2], x[3])):
            sol, _gain, note = merged[key]
            if sol != "default":
                tuned += 1
            f.write("f16o %d %d %d %d %s%s\n" % (key[0], key[1], key[2], key[3], sol, ("   # " + note) if note else ""))
    print("wrote %d rows (%d with a solution) to %s" % (len(merged), tuned, out))
    for t in (64, 128, 256, 270, 512, 1024, 2048, 3072):
        n = sum(1 for k in merged if k[3] == t and merged[k][0] != "default")
        print("  T=%-5d rows=%d tuned=%d" % (t, sum(1 for k in merged if k[3] == t), n))


if __name__ == "__main__":
    main()
