# 停掉 7900 XTX 后 MI50 重测：排除"系统内存不够"（2026-11-17 08:30–08:55）

## 目的
此前所有 MI50 数据都在 7900 XTX 生产（8093）同时在线时测得。7900 占 46.84 GiB 大页 arena + 22.1 GB mlock 的
PLE 表，因此需要排除"MI50 的 prefill 天花板实际是系统内存被抢、RAM 专家缓存不够"这一可能。

## 条件
- 机：amd52（10.24.247.157:2222），247 GiB RAM，4 卡（HIP0=MI50@c5 / HIP1=7900 XTX / HIP2=MI50@03）
- 二进制：`/home/yanyq/port/mi50-0214/build-906-0214/strata`（25,409,248 B，无 MMQ，生产取向）
- 配置：`/home/yanyq/bench/cfg-mi50-final.json`（`--prefill 2048`、`--mtp-q4 all`、`--ple-io ram`、
  `--resident-budget-gib 72`、`--kv-resident 262144`）
- 环境：`ROCR_VISIBLE_DEVICES=0,2 HIP_VISIBLE_DEVICES=0,1 STRATA_GFX906_ATTN_QUERY_SWIZZLE=1
  STRATA_GFX906_ATTN_REDUCE12=1 STRATA_MMQ_KQ=1`，`taskset -c 0-7,16-23`，端口 8013
- 测试：`mi50_bench.py --case 5k --reps 3 --gen 128` / `--case 21k --reps 3 --gen 128`（非流式 timings）

## 内存账（实测）

| 时点 | MemAvailable | Mlocked | HugePages_Free | card2(7900) VRAM |
|---|---|---|---|---|
| 7900 在跑（起测前） | 116.8 GB | 22.1 GB | 24417 / 48400 | 24018 MiB |
| 停掉 8093 后 5 s | 147.1 GB | 9.4 MB | 48400 / 48400 | 26 MiB |
| MI50 就绪后 | 87.5 GB | 28.1 GB | 48400 | — |
| MI50 21K 测中 | 85.8 GB | 28.1 GB | 48400 | — |
| 收尾（MI50 停、8093 已回） | 110.8 GB | 28.1 GB | 24417 | 24010 MiB |

- 7900 停掉共释放 **+30.3 GB** 可用内存（46.84 GiB 大页 + 22.1 GB mlock 的 PLE）。
- MI50 自己要的量：28.1 GB mlock 专家 + 约 30 GB 页缓存（buff/cache），7900 在跑时也有 111 GB 可用，本来就够。
- `pswpin/pswpout` 无本次新增变化；`dmesg` 无 OOM 记录；expert cache 命中 99.6–99.8%。

## 结果（同一二进制、同一配置、同一输入）

| 臂 | 5K prefill | 5K TTFT | 21K prefill | 21K TTFT | 21K decode |
|---|---|---|---|---|---|
| A：7900 在跑（基线） | 518.5 t/s | 8.892 s | 661.8 t/s | 26.959 s | 68.5 t/s |
| B：7900 停掉 | 517.7 t/s | 8.904 s | 660.0 t/s | 27.030 s | 69.6 t/s |
| 差 | −0.15% | +0.1% | −0.27% | +0.3% | +1.6%（噪声） |

B 臂逐 rep：
- 5k：519.7 / 517.7 / 511.7 t/s，TTFT 8.871 / 8.904 / 9.009 s，decode 64.4 / 62.4 / 64.6
- 21k：655.7 / 660.0 / 661.1 t/s，TTFT 27.208 / 27.030 / 26.986 s，decode 69.6 / 69.6 / 68.0

## 结论
**不是内存不够。** 腾出 30 GB 可用内存与全部 46.84 GiB 大页后，吞吐没有任何提升（全在测量噪声内）。
MI50 的天花板仍归因于 §2.5：MoE 稀疏专家 GEMM 的 M 行数太少（M≈40~160 → 1.8~5.1 TFLOPS/卡）
+ gfx906 无 MFMA + fp32 累加实测 10.8 TFLOPS/卡；与系统内存、HBM 带宽（811 GB/s，占用 0.25%）、显存容量无关。

## 附带
- 8093 生产停机约 25 min，测完已 `run_prod_0214.sh 8093` 复原（ready 85 s，`/v1/models` 返回
  `qwen3.8-flash-next` loaded，TLS 9443 与 HTTP 8093 均通，5100 卡 VRAM 24010 MiB）。
- 复现脚本：停 8093 → 起 8013（cfg-mi50-final.json）→ `mi50_bench.py` 两 case → 停 8013 → `run_prod_0214.sh 8093`。
