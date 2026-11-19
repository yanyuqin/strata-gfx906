"""Correct the measured figures in the two default-value comments (they were inherited from the TM 3 + PF 2 arm).

Measured 2026-11-17 on the same binary: tile 2 PF 2 = 863.9 tok/s, tile 2 PF 1 = 946.3 (no i8 variable set),
tile 3 PF 2 = 690.8 (the spill is heavier there).  The defaults themselves do not change.
"""
import io
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "src/kernels/cuda/iq_kernels.cu"
s = io.open(p, encoding="utf-8").read()

old = """    // prefetch depth: 2 = three stages in flight (the upstream default); on gfx906 the extra register set pushes tile 2 past
    // the 128-VGPR cap that I8_MINB=2 forces, and the spill costs more than the deeper pipeline gains (690 vs 906 tok/s on the
    // whole machine), so the gfx906 build defaults to one stage ahead.  STRATA_I8_PF overrides both."""
new = """    // prefetch depth: 2 = three stages in flight (the upstream default); on gfx906 the extra register set pushes tile 2 past
    // the 128-VGPR cap that I8_MINB=2 forces, and the spill costs more than the deeper pipeline gains: 21K prefill measured
    // 863.9 tok/s at tile 2 PF 2 against 946.3 at PF 1 (690.8 at tile 3 PF 2, where the spill is heavier).  So the gfx906
    // build defaults to one stage ahead; STRATA_I8_PF overrides both."""
assert s.count(old) == 1, "pf comment: %d" % s.count(old)
s = s.replace(old, new)

old2 = """    // this build asks for 2 blocks/CU (I8_SMEM_1BUF=1, I8_MINB=2), which caps VGPR at 128; tile 4 spills 27 under that cap
    // (783 tok/s whole machine against 941 for tile 2), so the gfx906 default is tile 2.  STRATA_I8_TM overrides it."""
new2 = """    // this build asks for 2 blocks/CU (I8_SMEM_1BUF=1, I8_MINB=2), which caps VGPR at 128; the heuristic above picks tile 4
    // at the production entry count and tile 4 spills 27 under that cap (783.0 tok/s on 21K against 946.3 for tile 2), so the
    // gfx906 default is tile 2.  STRATA_I8_TM overrides it."""
assert s.count(old2) == 1, "tm comment: %d" % s.count(old2)
s = s.replace(old2, new2)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched", p)
