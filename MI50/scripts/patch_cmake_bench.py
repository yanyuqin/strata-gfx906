#!/usr/bin/env python3
# Register i8_pack_bench (strata-pack against the blob) next to native_i8_parity.
import io, sys

path = sys.argv[1]
src = io.open(path, encoding="utf-8").read()
anchor = '''  add_executable(mmvf_rows_parity src/kernels/mmvf_rows_parity.cpp)'''
new = '''  # strata-pack: native_expert_i8 on the packed read layout against the blob (bit parity and time; GPU, synthetic, no model)
  add_executable(i8_pack_bench src/kernels/i8_pack_bench.cpp)
  target_include_directories(i8_pack_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(i8_pack_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  ''' + anchor
if src.count(anchor) != 1:
    sys.exit("anchor count %d" % src.count(anchor))
io.open(path, "w", encoding="utf-8").write(src.replace(anchor, new))
print("registered")
