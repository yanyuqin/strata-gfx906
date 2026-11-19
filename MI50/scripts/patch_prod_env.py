"""Lock the verified golden environment into the MI50 launchers.

run_mi50_prod.sh gets the three i8 knobs that used to be missing (TM/N8/PF), and run_i8tune.sh gets them as
defaults too, so a bare launch can no longer fall back to PF=2 (which spills at TM 2 and costs ~24%: 906 -> 690 tok/s).
"""
import io
import sys

B = "/home/yanyq/bench/"
PB = "/home/yanyq/bench/mi50-0214/"

# 1. production launcher: add the i8 line and a note
p = PB + "run_mi50_prod.sh"
s = io.open(p, encoding="utf-8").read()
old = """  STRATA_HIP_PROMPT_F16=1 \\
  STRATA_ROCBLAS_TUNING=/home/yanyq/bench/gfx906-rocblas.txt \\"""
new = """  STRATA_HIP_PROMPT_F16=1 \\
  STRATA_I8_TM=2 STRATA_I8_N8=0 STRATA_I8_PF=1 \\
  STRATA_ROCBLAS_TUNING=/home/yanyq/bench/gfx906-rocblas.txt \\"""
assert s.count(old) == 1, "prod launcher: %d matches" % s.count(old)
s = s.replace(old, new)
old2 = """# run_mi50_prod.sh [port] -- the MI50 production launcher (v0.2.14.1 + FP16 prompt GEMMs + rocBLAS table)"""
new2 = """# run_mi50_prod.sh [port] -- the MI50 production launcher (v0.2.14.1 + FP16 prompt GEMMs + rocBLAS table).
# Golden environment, measured 2026-11-17, 2x MI50 (ROCR 0,2) + EPYC 7F52, 21K prefill 941.8 tok/s:
#   ROCR_VISIBLE_DEVICES=0,2   keep the RX 7900 XTX (device 1) out of the HIP set
#   STRATA_HIP_PROMPT_F16=1    FP16 prompt GEMMs (+15~17% prefill)
#   STRATA_I8_TM=2             the tile code the 2-blocks/CU occupancy target is measured at
#   STRATA_I8_N8=0             16-entry slices; the 8-entry variant measured slower here
#   STRATA_I8_PF=1             one stage of prefetch: PF=2 spills at TM 2 under the 128-VGPR cap (690 vs 906 tok/s)
#   taskset -c 0-15            the 16 physical cores (SMT siblings 16-31 left to the scheduler)"""
assert s.count(old2) == 1
s = s.replace(old2, new2)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched", p)

# 2. the tuning launcher: same three knobs as defaults (I8EXTRA still overrides them, it comes last)
p = B + "run_i8tune.sh"
s = io.open(p, encoding="utf-8").read()
old = """  STRATA_I8_COUNT=1 STRATA_PREFILL_TIMING=1 ${I8EXTRA:-} \\"""
new = """  STRATA_I8_TM=2 STRATA_I8_N8=0 STRATA_I8_PF=1 \\
  STRATA_I8_COUNT=1 STRATA_PREFILL_TIMING=1 ${I8EXTRA:-} \\"""
assert s.count(old) == 1, "i8tune: %d matches" % s.count(old)
s = s.replace(old, new)
old2 = """# run_i8tune.sh [port] -- MI50 test launcher for the i8 expert-kernel tile tuning round.
# Same environment as run_mi50_prod.sh, plus the i8 diagnostics and an optional STRATA_I8_TM override."""
new2 = """# run_i8tune.sh [port] -- MI50 test launcher for the i8 expert-kernel tile tuning round.
# Same environment as run_mi50_prod.sh (including the golden STRATA_I8_TM=2 N8=0 PF=1), plus the i8 diagnostics;
# I8EXTRA comes last and overrides any of them for an arm."""
assert s.count(old2) == 1
s = s.replace(old2, new2)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched", p)
