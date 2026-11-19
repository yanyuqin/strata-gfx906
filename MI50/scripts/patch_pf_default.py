"""gfx906: make STRATA_I8_PF default to 1 in the source (the launch's prefetch depth).

Upstream defaults to 2 (three stages in flight).  On gfx906 the extra register set pushes tile 2 past the
128-VGPR cap that I8_MINB=2 forces; the spill costs more than the deeper pipeline gains (690 vs 906 tok/s),
so the gfx906 build defaults to one stage ahead.  STRATA_I8_PF still overrides both, so the CUDA default stays 2.
"""
import io
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "src/kernels/cuda/iq_kernels.cu"
s = io.open(p, encoding="utf-8").read()

old = '''    I8Args a{grp_ptr, grp_start, n_groups, tile_off, ent_tok, ent_dst, (const block_q8_1*) x_q8_1, hq, out, L,
             cap_entries / bm + cap_groups, i8_env_int("STRATA_I8_BISECT", 0), i8_n8_takes(i8_switch_parse(std::getenv("STRATA_I8_N8")), tm),
             i8_env_int("STRATA_I8_PF", 2) >= 2 ? 2 : 1, nullptr, 0, 0, 0};'''

new = '''    // prefetch depth: 2 = three stages in flight (the upstream default); on gfx906 the extra register set pushes tile 2 past
    // the 128-VGPR cap that I8_MINB=2 forces, and the spill costs more than the deeper pipeline gains (690 vs 906 tok/s on the
    // whole machine), so the gfx906 build defaults to one stage ahead.  STRATA_I8_PF overrides both.
#ifdef STRATA_HIP_GFX906
    const int i8_pf = i8_env_int("STRATA_I8_PF", 1) >= 2 ? 2 : 1;
#else
    const int i8_pf = i8_env_int("STRATA_I8_PF", 2) >= 2 ? 2 : 1;
#endif
    I8Args a{grp_ptr, grp_start, n_groups, tile_off, ent_tok, ent_dst, (const block_q8_1*) x_q8_1, hq, out, L,
             cap_entries / bm + cap_groups, i8_env_int("STRATA_I8_BISECT", 0), i8_n8_takes(i8_switch_parse(std::getenv("STRATA_I8_N8")), tm),
             i8_pf, nullptr, 0, 0, 0};'''

n = s.count(old)
if n != 1:
    sys.stderr.write("expected exactly one match, found %d\n" % n)
    sys.exit(1)
s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched", p)
