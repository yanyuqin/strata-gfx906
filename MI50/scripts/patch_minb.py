#!/usr/bin/env python3
# __launch_bounds__(I8_NT, I8_MINB): the second argument tells the compiler the block count it must fit per CU, so it caps VGPR itself.
# With I8_MINB=2 the tiles above TM 2 (130-154 VGPR) are pushed toward 128 and spill instead of silently dropping to one block.
import sys

p = sys.argv[1]
src = open(p).read()

if "I8_MINB" not in src:
    old = "#define I8_ST_UNROLL 2"
    new = ("#ifndef I8_MINB\n"
           "#define I8_MINB 1     // blocks per CU the compiler must fit (2 = cap VGPR at 128, see i8_kernel's launch bounds)\n"
           "#endif\n"
           "#define I8_ST_UNROLL 2")
    assert src.count(old) == 1, "anchor not unique"
    src = src.replace(old, new, 1)

old = ("template <int T, bool GU, int TM, bool NR, int PF>\n"
       "__global__ void __launch_bounds__(I8_NT)\n"
       "i8_kernel(")
assert src.count(old) == 1, "i8_kernel launch bounds not unique"
src = src.replace(old,
                  "template <int T, bool GU, int TM, bool NR, int PF>\n"
                  "__global__ void __launch_bounds__(I8_NT, I8_MINB)\n"
                  "i8_kernel(", 1)

open(p, "w").write(src)
print("patched", p, "(i8_kernel launch bounds)")
