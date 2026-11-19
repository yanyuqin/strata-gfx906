# MI50 / gfx906 归档 —— 测试结果、测试脚本与启动配置

这里是 Strata 在 **2 × Radeon Instinct MI50 32 GB（gfx906）+ AMD EPYC 7F52** 上的全部实测产物，两条线：

- **v0.2.14.1 线**（`~/port/mi50-0214`，2026-11-16 ~ 11-17）：21K prefill **651.5 → 946.3 tok/s（+45%）**，TTFT **27.386 → 18.18 s（−34%）**，decode 刻意保持不变（≈64.7 tok/s）。
- **v0.2.16 同步线**（`~/port/mi50-0216`，2026-11-19）：上游目录重构（源码搬到 `engine/`）后重贴同一组补丁；同夜 A/B 21K prefill **946.4 → 995.5 tok/s（+5.2%）**、decode（**128 token 口径**）**64.0 → 62.2 tok/s**（0216 五次重复落在 60.9~67.9，差值在抖动带内，不算退化）。
  改动清单与 A/B 台账见 `docs/MI50_strata_0.2.16_同步_2026-11-19.md`。

> **口径告警**：decode 吞吐强烈依赖生成长度 —— 同一二进制、同一 21K prompt，`--predict 32` 报 52.1 tok/s，`--predict 128` 报 64.0 tok/s，每步耗时两者都是 33~34 ms，差的只是 MTP 投机接受率（短窗口里接受率还没爬到位：32 口径 12/17 = 67%，128 口径 55/60 = 92%）。**报 decode 必须注明生成长度，≥128 token 才可当稳态值**；`--predict 32` 的数字不得与 64.7 之类的稳态值混用。机制与台账见同步文档 §4。

所有数字都在该硬件上测得，不是外推。改动逐项说明见 `docs/MI50_strata_项目文档_2026-11-17.md`。

---

## 目录结构

| 路径 | 内容 |
|---|---|
| `docs/` | `MI50_strata_项目文档_2026-11-17.md`（六轮改动逐项表：位置 / 内容 / 依据 / 影响 / 验证，含作废结论、15 条坑、回滚点、距 950 的账）；`MI50_2.14.1_报告_2026-11-16.md`（移植完成时的基线） |
| `results/` | 10 份按轮次的测量报告：i8 tile 扫描、PF / strata-pack、整机 kernel trace、rocBLAS 多 T 桶重校准、KV32K、RAM 控制、MMQ、瓶颈分解、硬件探针、质量评估 |
| `results/raw/` | 13 份 `bench-*.jsonl`：`bench40k.py` 每次跑的原始逐 rep 记录（吞吐、TTFT、draft 命中、GPU 分段） |
| `scripts/` | 编译、校准（rocBLAS / i8 tune）、trace 分析、VGPR 解析、一致性比对、临时 patch 工具共 28 个 |
| `launcher/` | 8 个启动 / 测量驱动脚本 + 2 份引擎 config（生产档、调参档），启动类脚本只在这里留一份 |
| `archive/` | 120 行 rocBLAS 解表、本轮源码改动 patch 与 stat、`MANIFEST.md`（校验清单与复现口径） |

---

## 启动

最常用的一条（生产档，21K prefill ≈ 942–946 tok/s）：

```bash
bash MI50/launcher/run_mi50_prod.sh 8013
# 就绪探测
curl -s http://127.0.0.1:8013/v1/models
# 停止
for p in $(ps -eo pid=,comm=,args= | awk '$2=="python" && /serve.server --engine strata/ {print $1}'); do kill -TERM $p; done
```

`launcher/` 里各脚本的用途：

| 脚本 | 用途 |
|---|---|
| `run_mi50_prod.sh <port>` | 0.2.14.1 线生产档：黄金环境 + `cfg-mi50-final.json`，`taskset -c 0-15`，绑定 127.0.0.1 |
| `run_mi50_0216.sh <port>` | 0.2.16 线生产档：同黄金环境 + `cfg-mi50-0216.json`；服务端取自 `terminal/serve`（`PYTHONPATH=<tree>/terminal`），127.0.0.1 |
| `run_i8tune.sh <port>` | 调参档：同生产环境，加 `STRATA_I8_COUNT=1 STRATA_PREFILL_TIMING=1`，`I8EXTRA` 放最后可覆盖任一 i8 变量，日志 `log-i8tune-<port>.log` |
| `run_i8verify_nopin.sh <port>` | 验证档：**故意不带 `taskset`、也不带 i8 变量**，用来确认"零配置 = 黄金档"是否成立（实测 946.3） |
| `run_strata_iso.sh <cfg> <port> <tag> <cpuset> <memmax> [hip_devices] [host] [apikey]` | 通用隔离启动器；`apikey` 默认空，需要鉴权时从外部传入，**不要写进仓库** |
| `run_measure_21k.sh` | 起 8013 → 跑三次 21000 tok → 打印每次 rep 与引擎日志里的 GPU timeline → 收服务 |
| `run_measure_ab.sh` | 同日 A/B 对照（新旧 rocBLAS 表各跑一轮） |
| `run_mi50_ab.sh` | 两套二进制 / 两套表的整机对照驱动 |
| `run_eval_pfpk.sh` | eval100 质量对照（PF / packed 改动前后） |
| `cfg-mi50-0216.json` | 0.2.16 线的引擎配置：`--prefill 3072`、`--kv int8`、`--kv-resident 32768`、`--max-context 262144`、`--spec 2`、`--layer-split 25`、`--ple-io ram` 等，`exe` 指向 `build-906-0216/strata` |
| `cfg-mi50-final.json` / `cfg-mi50-i8tune.json` | 引擎配置：`--prefill 3072`、`--kv int8`、`--kv-resident 32768`、`--max-context 262144`、`--spec 2`、`--layer-split 25`、`--ple-io ram` 等 |

> 脚本内路径是 amd52 的绝对路径（`WD=/home/yanyq/port/mi50-0214`、`PY=~/venv-strata/bin/python`、
> `STRATA_ROCBLAS_TUNING=/home/yanyq/bench/gfx906-rocblas.txt`、模型 `/data480/models/...`）。
> 换机器时改这几处即可，环境变量的语义不变。

### 黄金环境变量（写死在 `run_mi50_prod.sh`，别漏）

| 变量 | 值 | 依据（实测） |
|---|---|---|
| `ROCR_VISIBLE_DEVICES` | `0,2` | HIP 下 device 1 是 RX 7900 XTX，必须隔离 |
| `HIP_VISIBLE_DEVICES` | `0,1` | 让引擎 `--gpu 0,1` 对上两张 MI50 |
| `STRATA_HIP_PROMPT_F16` | `1` | prompt GEMM 走 FP16，prefill +15~17% |
| `STRATA_I8_TM` | `2` | TM=4 在 `MINB=2` 下 spill 27 VGPR → 783 tok/s（−17%） |
| `STRATA_I8_N8` | `0` | 16-entry 切片；8-entry 变体在此更慢 |
| `STRATA_I8_PF` | `1` | PF=2 在 TM=2 下 863.9；TM=3+PF=2 才是 690.8 |
| `STRATA_ROCBLAS_TUNING` | `.../gfx906-rocblas.txt` | 120 行解表，启动时加载，改表必须重启 |
| `taskset -c 0-15` | — | 16 个物理核独占，SMT 兄弟 16–31 留给调度器 |

源码已把 gfx906 的 `STRATA_I8_TM` / `STRATA_I8_PF` 默认值固化为 2 / 1（`src/kernels/cuda/iq_kernels.cu`），
所以完全不带 i8 变量启动也是 946 tok/s；但生产脚本仍显式写死，防的是后人改默认值后静默回退。

---

## 测量口径（别读错行）

```bash
python3 ~/bench/bench40k.py --port 8013 --tokens 21000 --predict 32 --reps 3 --label <tag>
```

要取引擎日志里 **整段** 那一行：

```
strata prefill timing: 167xx tokens, ... GPU timeline NNNN ms, wall ...
```

`prefill timing: 1[0-9]* tokens` 命中的是**尾块**（1360 tok），不是整段，拿它算吞吐会偏低十几倍。
`rocprofv3` 采集会放大耗时约 11× 并触发引擎 native verify watchdog（5 s）→ 503，**采集轮的绝对耗时不可引用**，
只用来做占比。

---

## 基线数字

| 场景 | tok/s |
|---|---:|
| 21K prompt 整段（无 i8 变量，源码默认档） | **946.3**（935.8 / 957.0 / 946.3） |
| 21K prompt 整段（同日 A/B 新表） | **941.8**（旧表 926.7，+1.6%） |
| 21K TTFT | 18.18 s |
| 5K prompt | 645.9 |
| 40K prompt | 942.9 |
| 21K decode（128 token 口径） | ≈64.7（未动） |

质量面：`native_i8_parity` 全格式 × 全 tile 通过（最差相对 RMS 2.6e-3）；PF / 单缓冲改动逐位一致；
eval100 四次均分 0.6786 = 基线（eval 本身非确定，`mmlu_3`、`drop_96` 有单次抖动，无系统性回退）。

---

## 复现步骤

1. 装 gfx906 版 ROCm 到 `/opt/rocm715-gfx906`，`rocBLAS 5.6.0.baa93758`。
2. 建 venv：`python3 -m venv ~/venv-strata && ~/venv-strata/bin/pip install -r requirements.txt`。
3. 编译：`bash MI50/scripts/build_mi50_0214.sh`（产物 `build-906-0214/strata`，`strata --version` → `0.1.60.1`）。
4. 放解表：`cp MI50/archive/gfx906-rocblas_120rows_20261117.txt ~/bench/gfx906-rocblas.txt`（md5 `ff9cf21f…`）。
5. 起服务：`bash MI50/launcher/run_mi50_prod.sh 8013`。
6. 量：`bash MI50/launcher/run_measure_21k.sh`，按上面的口径读整段那一行。

---

## 敏感信息

仓库内不含任何密钥、token、密码。`run_strata_iso.sh` 的 `--api-key` 是第 8 个入参且默认为空；
需要对外鉴权时由调用方从环境注入，不要把值写进任何提交。

模型权重路径（`/data480/models/strata-pack-ud31-q4kxl`、`mtp-bf16/rt`）是部署路径，不是凭据；
权重本身不在本仓库，需按上游流程另行获取。
