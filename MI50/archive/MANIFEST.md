# 性能基线档案 · 2 × MI50 (gfx906) + EPYC 7F52

冻结日期：2026-11-17
软件：Strata v0.2.14.1（`3af562d5`，engine 0.1.60.1），ROCm 7.15 gfx906（`/opt/rocm715-gfx906`），rocBLAS 5.6.0.baa93758
基线：21K prefill **941.8 tok/s**（同日旧表 926.7）· TTFT 18.33 s · 5K 645.9 tok/s · 40K 942.9 tok/s · decode ~64.7 tok/s

---

## 1. 校验清单

| 文件 | 字节 | md5 | 来源 |
|---|---:|---|---|
| `gfx906-rocblas_120rows_20261117.txt` | 7 890 | `ff9cf21fe644dc996d69b524a427c787` | `/home/yanyq/bench/gfx906-rocblas.txt`（120 行，113 带解） |
| `gfx906_changes_20261117.patch` | 44 449 | `e3d95ddca2f3cf5c716d8336d46ef55f` | `git diff`（含 `i8_pack_bench.cpp` 新文件，7 文件 +496/−41） |
| `gfx906_changes_20261117.stat.txt` | — | `196a34ef79ffa734ae4462c3999ac506` | `git diff --stat` |

远端原件：表在 `/home/yanyq/bench/`，旧表备份 `gfx906-rocblas.txt.bak-20261117`（14 行）；源码在
`/home/yanyq/port/mi50-0214`，逐轮备份 `*.bak-{pf1,occ,1buf,pf,pfdefault}-20261117`。

---

## 2. 报告索引（`../results/`）

| 报告 | 内容 |
|---|---|
| `MI50_2.14.1_报告_2026-11-16.md`（`../docs/`） | 移植、prefill 2048/3072、mtp-q4、kv-resident、PROMPT_F16、rocBLAS 14 行表 |
| `mi50_i8_kernel_tile_2026-11-17.md` | i8 覆盖率 100%、tile 扫描、bisect 成本拆分、算术效率 2.5% |
| `mi50_i8_pf_repack_2026-11-17.md` | 方案 4（PF）、方案 2（packed）、VGPR/LDS 账单、2 blocks/CU 落地 |
| `mi50_bottleneck_2026-11-17.md` | 首次 kernel 级 profile（dense GEMM 51.7%）、PLE 归因作废前因 |
| `mi50_trace_21k_2026-11-17.md` | 21K 全量 trace、Kernel Top 25、类别合并 |
| `mi50_rocblas_retune_2026-11-17.md` | T 桶重校准、端到端 A/B、离 950 的账、eval100 四次 |
| `mi50_opt123_quality_2026-11-17.md` | 三项优化落地 + 质量回归 |
| `mi50_hw_probe_2026-11-17.md`、`mi50_kv32k_aff16_2026-11-17.md`、`mi50_ram_control_2026-11-17.md`、`mmq_906_2026-11-17.md` | 探针与配置 A/B |

---

## 3. 关键数字（引用时请带口径）

| 项 | 值 | 口径 |
|---|---:|---|
| 21K pp（生产档） | 941.8 | `bench40k --tokens 21000 --predict 32 --reps 3` 中位，同日旧表 A/B |
| 21K pp（零 i8 环境变量） | 946.3 | 同一二进制源码默认 TM2/PF1，941.8 之外的时段 |
| 21K 整段 GPU | 14 878 ms | `strata prefill timing: 167xx tokens`（**不是** 1272 ms 的尾块行） |
| 尾块 1360 tok | ~1 255 ms | 同上，`13xx tokens` 行 |
| i8 gate+up 单算子 | 0.79 ms | 1536 entry × 30 reps，TM2/PF1/MINB2/1BUF |
| i8 有效带宽 | 40.2 / 50.2 / 70.4 GB/s | `TCC_EA_RDREQ_sum` × 128 B / 时间 |
| rocBLAS 表 | 120 行 / 113 带解 | 15 形状 × 8 T 桶（64…3072） |

---

## 4. 复现

```bash
ssh -p 2222 yanyq@10.24.247.157
bash ~/port/build_mi50_0214.sh
bash ~/bench/mi50-0214/run_mi50_prod.sh 8013
python3 ~/bench/bench40k.py --port 8013 --tokens 21000 --predict 32 --reps 3 --label baseline
# 期望 median pp ≈ 941.8，tg ≈ 64.7
```

---

## 5. 冻结状态（2026-11-17 17:05）

- 8013 端口空，无 `serve.server` 进程；两张 MI50 显存回落到 ~10 MiB；另一张卡未使用；生产 8000 未启动。
- 源码树干净可构建（最终二进制 28 311 888 B，`strata 0.1.60.1`）。
- 归档 `gfx906_changes_20261117.patch` 对应 HEAD `3af562d5`，`git apply` 可重放到干净上游。
