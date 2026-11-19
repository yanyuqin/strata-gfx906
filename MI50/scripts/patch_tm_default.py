"""gfx906: default the i8 tile code to 2 when STRATA_I8_TM is not set.

The engine's tile choice is an occupancy judgement made for one-block-per-CU cards: at 30720 entries / 512 groups
(avg 1595) it picks tile 4 (BM 64).  With this build's 2-blocks-per-CU shape (I8_SMEM_1BUF=1, I8_MINB=2) tile 4 has
to spill 27 VGPRs under the 128 cap and measured 783 tok/s against 941 for tile 2 - a 17% silent regression for a
launch that only forgot the environment variable.  STRATA_I8_TM still overrides, and the CUDA branch keeps its
adaptive choice.
"""
import io
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "src/kernels/cuda/iq_kernels.cu"
s = io.open(p, encoding="utf-8").read()

old1 = '''    int tm = i8_env_int("STRATA_I8_TM", 0);   // research:'''
new1 = '''    const int tm_env = i8_env_int("STRATA_I8_TM", 0);   // (0 = let the heuristic choose; see the gfx906 cap below)
    int tm = tm_env;   // research:'''
assert s.count(old1) == 1, "tm decl: %d" % s.count(old1)
s = s.replace(old1, new1)

old2 = '''    tm = tm <= 2 ? 2 : tm == 3 ? 3 : tm >= 8 && b_ok && i8_tile8_type(L.gu_type) && i8_tile8_type(L.d_type) ? 8 : 4;   // (the K-quants and Q5_1 have no tile 8)
    const size_t smem_limit = i8_smem_limit();'''
new2 = '''    tm = tm <= 2 ? 2 : tm == 3 ? 3 : tm >= 8 && b_ok && i8_tile8_type(L.gu_type) && i8_tile8_type(L.d_type) ? 8 : 4;   // (the K-quants and Q5_1 have no tile 8)
#ifdef STRATA_HIP_GFX906
    // this build asks for 2 blocks/CU (I8_SMEM_1BUF=1, I8_MINB=2), which caps VGPR at 128; tile 4 spills 27 under that cap
    // (783 tok/s whole machine against 941 for tile 2), so the gfx906 default is tile 2.  STRATA_I8_TM overrides it.
    if (tm_env <= 0 && tm > 2) tm = 2;
#endif
    const size_t smem_limit = i8_smem_limit();'''
assert s.count(old2) == 1, "tm normalize: %d" % s.count(old2)
s = s.replace(old2, new2)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched", p)
