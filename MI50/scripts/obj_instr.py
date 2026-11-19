#!/usr/bin/env python3
# Count instructions and scratch (spill) accesses per i8_kernel instantiation in a gfx906 object dump.
import re, subprocess, sys

obj = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else "i8_kernelILi"

out = subprocess.run(["llvm-objdump", "-d", "--mcpu=gfx906", obj],
                     capture_output=True, text=True).stdout

cur, counts = None, {}
for line in out.splitlines():
    m = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
    if m:
        cur = m.group(1)
        counts[cur] = [0, 0]
        continue
    if cur and re.match(r"^\s+[0-9a-f]+:", line):
        counts[cur][0] += 1
        if "scratch_" in line:
            counts[cur][1] += 1

rows = [(k, v[0], v[1]) for k, v in counts.items() if want in k]
rows.sort(key=lambda r: r[0])
for k, n, s in rows:
    tag = re.sub(r"^.*i8_kernel", "", k)
    tag = tag.split("EEEv")[0]
    print("%-24s instr=%-6d scratch=%d" % (tag, n, s))
