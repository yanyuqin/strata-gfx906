#!/usr/bin/env python3
# Single-buffer the stage shared memory of i8_kernel (I8_SMEM_1BUF): LDS drops from 39,952 B (TM 2) to 20,496 B so that two blocks fit
# one CU (gfx906 LDS = 65,536 B); the stage then needs two barriers instead of one, which the second block is meant to cover.
import re
import sys

p = sys.argv[1]
src = open(p).read()

# 1. the macro, next to the other tuning macros
if "I8_SMEM_1BUF" not in src:
    old = "#define I8_ST_UNROLL 2"
    new = ("#ifndef I8_SMEM_1BUF\n"
           "#define I8_SMEM_1BUF 0     // 1: single-buffer the stage records/activations (LDS 20,496 B at TM 2: two blocks per CU), two barriers per stage\n"
           "#endif\n"
           "#define I8_ST_UNROLL 2")
    assert src.count(old) == 1, "I8_ST_UNROLL anchor not unique"
    src = src.replace(old, new, 1)

# 2. I8Smem: the buffer count
old = "    static constexpr int SZ_W = 64 * I8_LD"
new = "    static constexpr int NBUF = I8_SMEM_1BUF ? 1 : 2;\n    static constexpr int SZ_W = 64 * I8_LD"
assert src.count(old) == 1, "I8Smem anchor not unique"
src = src.replace(old, new, 1)

for a, b in [("o_a = o_w + 2 * SZ_W", "o_a = o_w + NBUF * SZ_W"),
             ("o_r = o_a + 2 * SZ_A", "o_r = o_a + NBUF * SZ_A"),
             ("o_d = o_r + 2 * SZ_R", "o_d = o_r + NBUF * SZ_R"),
             ("o_s = o_d + 2 * SZ_D", "o_s = o_d + NBUF * SZ_D"),
             ("o_src = o_s + 2 * SZ_S", "o_src = o_s + NBUF * SZ_S"),
             ("total = (o_s + 2 * SZ_S + 3) / 4 * 4", "total = (o_s + NBUF * SZ_S + 3) / 4 * 4")]:
    #  I8BSmem (the second struct) repeats every one of these anchors: patch the first occurrence only
    assert src.count(a) >= 1, "anchor missing: " + a
    src = src.replace(a, b, 1)

# 3. i8_kernel: the stage's buffer index and its barriers (first occurrence only: i8b_kernel keeps the old shape)
old = "        const int b = st & 1;\n"
assert src.count(old) >= 1
src = src.replace(old, "        const int b = SM::NBUF == 1 ? 0 : (st & 1);\n", 1)

old = ("        if (st + 1 < ns) { store_w(b ^ 1, st + 1, wraw[(st + 1) % PF]); store_a(b ^ 1, ar[(st + 1) % PF]); }\n"
       "        __syncthreads();\n")
new = ("        if constexpr (SM::NBUF == 1) {   // single buffer: every reader of this stage must be done before it is overwritten\n"
       "            __syncthreads();\n"
       "            if (st + 1 < ns) { store_w(0, st + 1, wraw[(st + 1) % PF]); store_a(0, ar[(st + 1) % PF]); }\n"
       "            __syncthreads();\n"
       "        } else {\n"
       "            if (st + 1 < ns) { store_w(b ^ 1, st + 1, wraw[(st + 1) % PF]); store_a(b ^ 1, ar[(st + 1) % PF]); }\n"
       "            __syncthreads();\n"
       "        }\n")
assert src.count(old) >= 1, "stage tail not found"
src = src.replace(old, new, 1)

open(p, "w").write(src)
print("patched", p, "bytes", len(src))
