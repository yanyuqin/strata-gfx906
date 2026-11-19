# MI50 prefill 瓶颈定位与可优化项（2026-11-17 10:30–11:20，内核级 profile + 选解实测）

7900 XTX 全程停机，两张 MI50 独占。二进制 `build-906-0214/strata`（25,409,248 B，无 MMQ），
配置 `cfg-mi50-final.json`（`--prefill 2048`、`--mtp-q4 all`、`--kv-resident 32768`），`taskset 0-15`。

## 一、内核级 profile（21K prompt，17,841 tok，rocprofv3 1.3.5）

`rocprofv3 --kernel-trace` 包住 serve.server，发一个 21K 请求。窗口内 **197,194 次 kernel launch**，
两块卡的 GPU 忙碌时间合计 **42.56 s**（Agent1 21.60 s / Agent3 20.96 s），wall 26.7 s → 双卡流水把 GPU 吃满。

| kernel 组 | 时间 | 占比 | 说明 |
|---|---|---|---|
| hipBLASLt/rocBLAS dense GEMM（`Cijk_*`） | **22.0 s** | **51.7%** | prompt 路径的 dense 投影：q/kv/o、shared expert、PLE |
| int8 专家内核（`i8_kernel<...>` 8 个变体） | 10.7 s | 25.1% | MoE 稀疏专家（fused int8 prompt 内核） |
| `attn_chunk` + `attn_merge` | 4.5 s | 10.5% | QSA 注意力（分块） |
| GDN 线性注意力（`gdn_rec_cols_pipe`/`gdn_out_norm`/`gr_mix_r`/`gr_write_norm_rs`） | 2.9 s | 6.8% | 12 层 SSM/GDN |
| 路由（`block_scores`/`block_topk`/`moe_combine*`） | 0.66 s | 1.6% | top-10 选专家 |
| `native_mmvq`（decode 量化 matvec） | 0.60 s | 1.4% | 8 个 decode token |

单个最大的 kernel：`Cijk_Alik_Bljk_HSS_BH_MT64x64x16_..._ISA906`，**296 次 × 12.68 ms**，
grid `[40960, 32, 1]`、workgroup 256 → 对应 `N=40960`（16 个 PLE 头 × 2560）、`K=320`、`T=2048`，
即 **PLE 头投影**：占 8.8% 时间，FLOPs 只有全程的 0.25%。

## 二、算力账：GEMM 段只有 21% 的峰值利用率

模型（`qwen4exp`）：48 层、512 专家 × 10 选、expert_ff 640、hidden 2560、head 24 / kv 2 / head_dim 256、
GDN inner 6144。按张量统计每 token 激活 FLOPs：

| 段 | 每 token FLOPs | 占比 |
|---|---|---|
| 注意力投影（q/kv/o + indexer） | 4.84 G | 44% |
| MoE 专家（10/512） | 4.72 G | 43% |
| shared expert | 1.10 G | 10% |
| router + PLE + emb | 0.25 G | 3% |
| **合计** | **10.9 G** | 100% |

17,841 token → 194 TFLOP，除以 42.56 s 的 GPU 忙碌时间 = **4.57 TFLOPS（双卡）= 2.3 TFLOPS/卡 = 10.8 峰值的 21%**。

而分段看：GEMM 承担约 55% 的 FLOPs 却用了 52% 的时间（≈2.4 TFLOPS/卡）；
int8 专家承担 43% FLOPs 只用 25% 时间（≈8.5 TFLOPS 等效，dp4a 峰值约为 fp16 两倍，即 int8 路反而更高效）。

**结论：瓶颈不在专家内核，也不在注意力，而在 dense GEMM 段。**

## 三、GEMM 段为什么慢：两个可修的根因

### 3.1 默认走了 FP32 输出的 GEMM，而不是 FP16 in/out

`gemm.cu:605 prompt_f16()`：`f16_io` 只对 **gfx1103（780M 核显）** 默认开，gfx103x 给提示、其他卡一律默认关。
关着的时候 prompt 的 16-bit GEMM 输出是 FP32（`C` 为 `HIP_R_32F`），比 FP16 in/out 多一倍的写出带宽 + 不同的 kernel 选择。

实测（同二进制同配置，只加环境变量）：

| 臂 | 5K prefill / TTFT | 21K prefill / TTFT | decode |
|---|---|---|---|
| 基线（f16_io 关） | 523.0 t/s / 8.814 s | 667.6 t/s / 26.723 s | 64~66 |
| `STRATA_HIP_PROMPT_F16=1` | **602.8 t/s / 7.648 s** | **778.3 t/s / 22.923 s** | 65.4 / 66.5 |
| 提升 | **+15.3% / −13.2%** | **+16.6% / −14.2%** | 不变 |

日志确认两块卡都进了该模式：`STRATA_HIP_PROMPT_F16=1: the prompt's 16-bit GEMMs run FP16 in and out`。
代价：每个 GEMM 输出做 FP16 舍入，**不逐位一致**；源码注释也写明"tuned only for that on gfx103x, not bit-identical"。

### 3.2 rocBLAS 默认选解远离最优，而 gfx906 构建里 tuning 表被编译掉了

仓库自带 `tools/hip/tune_rocblas.cpp`（枚举 `rocblas_gemm_ex_get_solutions`、计时、校验，写出
`STRATA_ROCBLAS_TUNING_V1` 表）。我在 MI50 上编了它，按引擎的 14 个形状 × T=2048 跑了一轮：

| 形状 (N×K, T=2048) | 默认解 | 最优解 | 倍数 |
|---|---|---|---|
| 6144×2560 | 7.120 ms | **3.812 ms** | **1.87×** |
| 2560×6144 | 7.823 ms | **4.577 ms** | **1.71×** |
| 512×2560 | 1.418 ms | **0.483 ms** | **2.93×** |
| 2560×640 | 0.624 ms | 0.412 ms | 1.51× |
| 640×2560 | 0.721 ms | 0.543 ms | 1.33× |
| 128×2560 | 0.285 ms | 0.168 ms | 1.69× |
| 320×10240 | 1.789 ms | 1.406 ms | 1.27× |
| 4×10240 | 0.880 ms | 0.370 ms | 2.38× |
| 1×2560 | 0.357 ms | 0.035 ms | 10.2× |
| 10240×2560 | 6.842 ms | 6.628 ms | 1.03× |
| 12288×2560 | 7.870 ms | 默认即最优 | 1.00× |

但**这条表在 gfx906 二进制里根本没生效**：`STRATA_ROCBLAS_AVAILABLE=1` 只在
`if(STRATA_ENABLE_HIP ...)` 分支里定义（`cmake/strata_gpu_engine.cmake:463-465`），而 gfx906 走 CUDA 分支
（`STRATA_ENABLE_HIP=OFF`）。核对：gfx906 的 `build.ninja` 里该宏出现 **0 次**、二进制里 `RocBlasState` 符号 **0 个**；
7900 的 `build-1100-mv` 里有 4 次。所以实测 E2（`STRATA_HIP_PROMPT_F16=1` + 表）与 E1 相同（5K 601.6、21K 764.7），
**不是表没用，是代码路径没编译**。

## 四、其他环节的余量

- **i8 专家内核**：10.7 s / 43% FLOPs ≈ 8.5 TFLOPS 等效，按 dp4a 峰值算约 16–20% 利用率，仍有空间但需改 kernel
  （tile/wave 调度），收益小于 GEMM 段。
- **PLE 头投影**：8.8% 时间 / 0.25% FLOPs。K=320 的 GEMM 写出 2048×40960 的 FP32 中间量（335 MB/次 × 296 次 = 99 GB），
  属明显浪费，可改 FP16 输出、或把 16 头归约融进 epilogue。
- **GDN/注意力**：合计 17%，与理论量级相符，没有明显浪费。
- **CPU 侧**：`taskset` 从 8 物理核加到 16 物理核，吞吐 0 变化（前面已测），不是瓶颈。

## 五、优化项排序（按性价比）

| 序 | 动作 | 需要改码 | 预期 | 风险 |
|---|---|---|---|---|
| 1 | `STRATA_HIP_PROMPT_F16=1` | 否（环境变量） | **已实测 +15~17% prefill** | 输出 FP16 舍入，需质量评测 |
| 2 | gfx906 打开 `STRATA_ROCBLAS_AVAILABLE` + 用它出 tuning 表并随包分发 | 一行 cmake + 一次校准 | 表上多数形状 1.3–2.9×，GEMM 占 52% → 乐观再 +10~15% | 低（同值解，有逐位校验） |
| 3 | PLE 头投影改 FP16 输出/epilogue 归约 | 是 | ~−6% 时间 | 低 |
| 4 | i8 专家 GEMM 的 tile/wave 调优 | 是 | 未知，10% 量级 | 中 |
| 5 | fp16 累加 | 是 | 理论 2× | **质量风险大，不先做** |

理论上限：GEMM 段若跑到 10.8 TFLOPS/卡、专家段跑到 int8 峰值，prefill 可到 2000+ t/s（现在是 667–778）；
现实路径是 1 → 2 → 3，合计把 prefill 推到 900–1000 t/s 量级。

## 六、环境
7900 XTX 未起（8093/9443 不监听，card2 26 MiB）；MI50 测试服务已停，两卡显存 10 MiB；MemAvailable 147 GB。
