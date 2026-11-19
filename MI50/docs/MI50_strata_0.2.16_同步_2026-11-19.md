# MI50 / gfx906：Strata v0.2.16 同步记录（2026-11-19）

**平台**：2 × Radeon Instinct MI50 32 GB（gfx906）· AMD EPYC 7F52 · 247 GB RAM · 主机 `amd52`（10.24.247.157）
**上游**：v0.2.16（`main` `58ee32277`，2026-11-18 23:40 +0100；engine `0.1.62`）
**基线**：v0.2.14.1（`3af562d5`，engine `0.1.60.1`）—— 见 `MI50_strata_项目文档_2026-11-17.md`
**范围**：只覆盖 MI50/gfx906 这条线；上游其余改动不在本文范围内。

---

## 0. 结论

| 项 | 结论 |
|---|---|
| 同步方式 | **重贴补丁，不做 merge**（上游把顶层改成 `app/ engine/ terminal/ docs/ tools/`，旧路径全部作废，merge 只会换来一堆冲突） |
| 代码改动 | 5 个文件、84 增 / 27 删（见 §2） |
| 逐位一致性 | `native_i8_parity` 全格式 × tile **all ok** |
| 21K prefill | 同夜 A/B **946.4 → 995.5 tok/s（+5.2%）** |
| 21K decode（128 口径） | 同夜 A/B **64.0 → 62.2 tok/s**（差值在 60.9~67.9 的 rep 抖动带内，不算退化） |
| eval100 | MMLU 86.7% / DROP F1 87.7% EM 80.0%，errors 0 |
| rocBLAS 表 | 复用 120 行旧表，启动日志 `120 rows, 113 with a solution` 两卡都加载 |
| 模型 pack | **不用重建**，`native_experts.txt` 解析未变，43 层 `12/7` 那套直接读 |

---

## 1. 上游 v0.2.14.1 → v0.2.16 改了什么

613 个提交，其中**引擎只有 10 个**，其余是 app/TUI/安装器/文档/测试。

### 1.1 结构（破坏性）

顶层 → `app/ engine/ terminal/ docs/ tools/`。`src/ include/ cmake/ serve/ apps/ sycl/ tests/` 全部消失：

| 旧路径（0.2.14.1） | 新路径（0.2.16） |
|---|---|
| `include/strata/platform/hip_compat/strata_hip.h` | `engine/include/strata/platform/hip_compat/strata_hip.h` |
| `cmake/backends/hip.cmake` | `engine/cmake/backends/hip.cmake` |
| `cmake/strata_gpu_engine.cmake` | `engine/cmake/strata_gpu_engine.cmake` |
| `src/kernels/cuda/i8_expert.cuh` / `iq_kernels.cu` | `engine/src/kernels/cuda/…` |
| `src/prefill/gemm.cu` | `engine/src/prefill/gemm.cu` |
| `serve/server.py` | `terminal/serve/server.py`（模块名仍是 `serve.server`，需 `PYTHONPATH=<tree>/terminal`） |
| 根 `CMakeLists.txt` | `engine/CMakeLists.txt`（`cmake -S engine`） |

### 1.2 引擎实质改动（TUNE-0216，打包在搬家提交 `4058aa78` 里）

| 文件 | 改动 | 对 gfx906 的影响 |
|---|---|---|
| `i8_expert.cuh` | +60/−20：`i8_kernel<T,GU,TM,NR,**GG**>` 新增第 5 个模板参数（大 grid 格式的 codebook 从全局内存读，省 LDS）；新增 tile 5 = Volta 张量核形态 `i8h_volta.cuh`，整段 `#if !defined(__HIPCC__)` | HIP 编译期不进；但**模板签名变了**，我们的 PF 参数要顺延成第 6 个 |
| `iq_kernels.cu` | +74/−9：`I8Volta` 探测（HIP 恒 `{}`）、`tm_set` 标志、`gg` 入 `I8Args`、`native_expert_i8_put_record` | tile 选路重写，TM/PF 默认要重贴 |
| `i8_expert_rule.hpp` | +26：Volta 两条规则 | CUDA-only |
| `hip.cmake` | **MMQ-on-HIP 编译块整段删除**（`STRATA_MMQ` / `STRATA_MMQ_KQUANTS` / `STRATA_PREFILL_MMQ` 选项消失） | 中性（906 本就无可用 MMQ 路径）；手工命令里那两个 `-D=OFF` 会变成"未使用变量"警告 |
| `gemm.cu` | +8/−7：prompt FP16 输出默认架构扩到 gfx1030/31/32 | **gfx906 仍不在默认名单** → `STRATA_HIP_PROMPT_F16=1` 继续必需 |
| 新增 `STRATA_FILE_CACHE_OLD_RULE` | 注释写明 gfx906 保留旧 file-tier cache 规则 | 默认=保旧行为，不设就不变 |
| RDNA3 trellis 对 | `#6067` 默认打开 | gfx9 明确 off，无关 |

---

## 2. 补丁重贴对照（5 个文件）

| # | 内容 | 旧位置 → 新位置 | 备注 |
|---|---|---|---|
| 1 | `cudaStreamCreateWithPriority` / `cudaDeviceGetStreamPriorityRange` 两个宏 | `include/…/strata_hip.h` → `engine/include/…/strata_hip.h` | 上游**仍然缺失**，编译 `dflash2.cpp` 会报 undeclared |
| 2 | `find_package(rocblas CONFIG QUIET)` + `STRATA_ROCBLAS_AVAILABLE` | `cmake/backends/hip.cmake` → `engine/cmake/backends/hip.cmake` | gfx906 走 CUDA 兼容分支，`hip_backend.cmake` 的探测仍不执行 |
| 3 | 守卫放宽为 `(STRATA_ENABLE_HIP OR STRATA_HIP_GFX906) AND STRATA_ROCBLAS_AVAILABLE` | `cmake/strata_gpu_engine.cmake` → `engine/cmake/strata_gpu_engine.cmake:473` | 上游**未采纳** |
| 4 | `I8_SMEM_1BUF=1`（单缓冲 LDS 20,496 B）+ `I8_MINB=2` + PF 预取环 + `I8_ST_UNROLL=2` | `src/kernels/cuda/i8_expert.cuh` → `engine/src/kernels/cuda/i8_expert.cuh` | 模板改为 `<T,GU,TM,NR,GG,PF>`；`I8Smem` 加 `NBUF` |
| 5 | gfx906 默认 `TM=2` / `PF=1` + `I8Args.pf` | `src/kernels/cuda/iq_kernels.cu` → `engine/src/kernels/cuda/iq_kernels.cu` | 上游新增 `tm_set`，我们的 cap 改用 `!tm_set` |

**未移植**：`strata-pack` packed 布局（`i8_load_pk` / `i8_pack_kernel` / `native_expert_i8_pk*`）。理由：0.2.14.1 已证逐位正确但慢 0.3~1.4%，封存不启用；在新模板上重贴它的冲突面很大、收益为零。原实现留在 `archive/gfx906_changes_20261117.patch` 里，需要时可再取。

编译：`bash MI50/scripts/build_mi50_0216.sh`（`cmake -S engine`，Release，`STRATA_BUILD_TESTS=ON` 以便跑 parity）。

---

## 3. 验证

### 3.1 逐位一致性

`native_i8_parity`（PF=1 / MINB=2 / 1BUF）：全部格式 × tile 2/3/4 → **all ok**，最差相对 RMS 2.71e-3（既定容差内）。

### 3.2 21K 同夜 A/B（同一台机、同一模型、间隔 <30 min）

| 二进制 | prefill t/s | TTFT | decode t/s（128 口径） |
|---|---:|---:|---:|
| `build-906-0214/strata`（0.1.60.1） | 946.4 | 17.7 s | **64.0** |
| `build-906-0216/strata`（0.1.62） | **995.5** | 16.8 s | **62.2** |

decode 的 rep 分布 0216 为 60.9 / 62.0 / 62.2 / 64.2 / 67.9，0214 为 62.6 / 65.5 —— 差值落在噪声带内。

### 3.3 eval100

`MMLU 60 题 acc=86.7% | DROP 40 题 F1=87.7% EM=80.0% | errors=0`（单次；eval100 本身非确定，不作涨跌结论）。

### 3.4 引擎日志确认

- `prefill gemm: rocBLAS tuning enabled (120 rows, 113 with a solution, gfx906, rocBLAS 5.6.0.baa93758)` ×2（两张卡）
- `STRATA_HIP_PROMPT_F16=1: … FP16 in and out` ×2
- `i8-count: … on native_expert_i8 (100.00 %)`
- decode expert cache 命中 **99.2~99.4%**

---

## 4. 口径台账：decode 到底是多少（2026-11-19 定案）

**结论：tg ≈ 64.7 是对的；32-token 窗口报出的 ~52 是口径假象。**

同一 0214 二进制、同一 21K prompt，只改 `--predict`：

| predict | tg | drafts 接受 | 每步 token | 每步耗时 |
|---|---:|---|---:|---:|
| 32 | 52.1 | 12/17 = 67% | 1.78（18 步） | 34 ms |
| 128 | 64.0 | 55/60 = 92% | 2.13（61 步） | 33 ms |

tg 比值 64.0/52.1 = 1.23；每步 token 比值 2.13/1.78 = 1.20 —— 对得上。**每步耗时两边一样，变的是每步吐几个 token。**

机制：`decode tok/s = (1 + 投机接受率) / 每步耗时`。短窗口里 MTP 接受率只有约 67%，生成 128 token 时升到 86~92%，于是同一个二进制在短窗口里系统性低报近 20%。

这条也解释了"昨天优化 decode 没有效果"：那次改动动的是内核算术（每步耗时），而**瓶颈在投机接受率**，不在每步耗时——改不构成瓶颈的那一项，自然不会有效果。

> **文档隐患（需修）**：`MI50_strata_项目文档_2026-11-17.md` 的 §4.3 把测量脚本写成 `--predict 32`，而 §7.1 的 decode 64.7 实际来自 128-token 的 `mi50_bench`。两处口径混用，是这次误判的根源。今后报 decode 一律注明生成长度；**≥128 token 才能当稳态值引用**。

---

## 5. 复现

```bash
ssh -p 2222 yanyq@10.24.247.157
bash ~/port/build_mi50_0216.sh          # -S engine，约 2.5 分钟
bash ~/bench/run_mi50_0216.sh 8013      # 等 ready（~160 s）
python3 ~/bench/bench40k.py --port 8013 --tokens 21000 --predict 128 --reps 5 --label x
```

## 6. 回滚

`~/port/mi50-0214` 与 `build-906-0214` 原样保留（二进制 md5 `e932050ad3422a6e260a06f9e38f8020`，与干净源码 + 原始脚本重建结果**字节相同**）。切回只需把 `cfg` 的 `exe` 指回 `build-906-0214/strata`。
