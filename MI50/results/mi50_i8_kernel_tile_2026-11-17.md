# MI50 · i8 专家 kernel tile 调优分析（2026-11-17）

机器 amd52（10.24.247.157），2× MI50 32GB，strata 2.14.1（build-906-0214），
`cfg-mi50-final.json`（layer-split 25，prefill 3072，kv-resident 32768）。
所有 A/B 单变量，同一台机器、同一模型、同一 prompt 语料（bench40k.py 21K）。

---

## 一、现状：100% 的 prefill 专家走 native_expert_i8

`STRATA_I8_COUNT=1` 实测（每 chunk 3072 tokens）：

```
chunk of 3072 tokens: 48 layers reached the group walk, 48 on native_expert_i8;
routed entries 1474560, on native_expert_i8 1474560 (100.00 %); i8 groups 924;
smallest group average seen 1059 entries; few-rows route: 0 layers, 0 entries
```

即 UD-Q4_K_XL 的 Q4_K gate/up + Q6_K down 全部由 int8 dp4a 内核承担，没有回落 FP16/grouped。

自动 tile 选择：`avg = cap_entries/cap_groups = 30720/512 = 60` → `tm = (60+8)/16 = 4`
→ **TM=4（BM=64 entries，64 weight rows，动态 LDS ≈ 50 KB）**；
Q4_K/Q6_K 被 `i8_tile8_type` 排除，tile 8（128×128）用不上。

## 二、21K 相位分解（16718 tokens，GPU timeline 16006 ms）

| 相位 | ms | 占比 |
|---|---:|---:|
| **gemm gate/up（含 i8 的 GU + down）** | **5515** | **34.5 %** |
| qsa attn | 2084 | 13.0 % |
| gdn | 2013 | 12.6 % |
| gdn recurrence | 1014 | 6.3 % |
| hc read: down+silu | 871 | 5.4 % |
| router+shared | 766 | 4.8 % |
| qsa proj | 748 | 4.7 % |
| gdn out proj | 678 | 4.2 % |
| embed+steps | 522 | 3.3 % |
| hc read: up | 460 | 2.9 % |
| 其余（gather/dequant/combine/ple/hc 写…） | 合计 ~340 | ~2 % |

专家内核是单一最大项，也是唯一还留有大头的可优化项。

## 三、tile 参数扫描（3 组实测）

| STRATA_I8_TM | STRATA_I8_N8 | 21K pp (tok/s) | gemm gate/up（16718 tok） |
|---|---|---|---:|
| 4（自动默认） | 0 | 877.1 | 5401 ms |
| 4 | 1 | 869.1 | 5544 ms |
| 3 | 0 | 869.5 | 5515 ms |
| 2 | 0 | 未取到（前一次服务未退净，8013 被占，启动失败） | — |

**结论：tile 2/3/4 与 8-entry slices 在 gfx906 上差异 < 1.5 %，全在噪声内。**
tile 调参本身没有收益 —— 因为成本不在"每 tile 的计算量/复用率"，而在"每 stage 的权重读"。

## 四、bisect 成本分解（kernel 自带 STRATA_I8_BISECT=n，3072-token chunk）

| 停点 | gemm gate/up | 含义 |
|---|---:|---|
| 1（立即 return） | 175 ms (1.8 %) | 纯 launch + tile 前缀和二分查找（924 group × 2 launch） |
| 6（首次权重读 + 解码后） | 283 ms (2.9 %) | + 第一 stage 的 global 读 + Q4_K 解码 |
| 0（全量） | 977 ms | 完整 20 stage（GU）+ 5 stage（down） |

→ launch/首次 prologue 占 **29 %**，循环占 71 %。924 个 group launch 是串行提交的，
每次 launch 一个 `i8_tiles_kernel<<<1,256>>>` + 两个 grid（gate/up 与 down）。

## 五、算术效率：2.5 % of int8 peak

- 每 chunk 实际 MAC：1474560 对 ×（gate/up 2×2560×640 + down 640×2560）= 7.247e12 MAC（21K 全程）
- 5515 ms → **1.31 TMAC/s（双卡）**；MI50 int8 dp4a 标称 26.5 TMAC/s/卡 → **2.5 %**
- 折算每个 block 约 0.48 ms（理论 ~12 μs），每个 stage 约 6 μs（理论 ~0.75 μs），**约 8 倍差距**
- 每 stage 只读 8 KB 权重（64 行 × 128 B，行间隔 ≈1.3 KB，跨约 21 个 4 KB 页），
  有效带宽折合整卡约 41 GB/s（HBM 811 GB/s 实测）→ 典型"随机 128 B 小读"特征

## 六、根因判断（按证据强度）

**A. 权重按 GGUF 原始布局逐行小读（最强）**
每 stage 的 64×128 B 是 64 个分散行片段；L1/L2 无法合并，DRAM 侧小事务效率极低。
同类证据：gfx906 社区的 llama.cpp 分支（sixvolts/llamacpp-gfx906-furnace）用**三平面 K-quant 权重重排**，
把 decode matvec 的 HBM 利用率从 on-disk 布局的 ~58 % 提到 ~89 %，同样是 dp4a + gfx906。

**B. 占用率 1 block/CU（次强）**
动态 LDS 50 KB（TM=4），gfx906 每 CU 64 KB → 每 CU 只能驻 1 个 256 线程 block（4 个 wave64，每 SIMD 1 wave），
双缓冲只提前 1 个 stage（0.6 μs 计算 vs 1–2 μs 延迟），延迟无法被别的 block 掩盖。

**C. VGPR 溢出（未验证）**
TM=4 的 4 行 × 4 entries 寄存器块（16 累加 + 4×2 uint4 权重 + 2 uint4 激活）在高 unroll 下可能 spill；
作者注释提到 gfx1031 上曾出现 256 VGPR + 55–200 spill，默认 `I8_BLK_UNROLL=1` 压制。

## 七、怎么做（方案排序，按"收益/成本"）

| # | 方案 | 预期 | 改动量 | 风险 |
|---|---|---|---|---|
| 0 | ~~继续调 tile~~ | 0（已证无效） | — | — |
| 1 | **合并 group launch**：整层一次 launch（tile_off 只算一次），去掉 924×2 次串行提交 | gate/up −15~19 %（约 −1000 ms / 21K） | prefill.cpp 的 group 循环 + i8_group.hpp 记录合并；中 | 低（bit-identical） |
| 2 | **权重重排**：加载/填充 expert cache 时把每 expert 的权重按 (64 行块 × 128 值 stage) 重排成连续 8 KB 块；内核按阶段顺序读 | gate/up −50~70 %（约 −2700~3800 ms / 21K，pp +25~40 %） | 新增 repack（GPU 侧，随 slot 填充做，原位替换，不额外占 VRAM）+ kernel 地址改动；大 | 中（必须 bit-identical，需 parity 测试） |
| 3 | **提占用率**：把 scale record 20→4 dwords（省 8 KB）、激活单缓冲（省 9 KB）→ 50→33 KB；再压 SZ_W pad → ≤32 KB 换 2 blocks/CU | 未知，估计 −10~25 % | i8_expert.cuh 结构体 + 索引；中 | 中（bank conflict 回归） |
| 4 | **加深预取**：第 3 份 SZ_W 缓冲（用 3 省出的 8 KB），权重预取提前 2 stage | 延迟掩盖 ×2，估计 −15~25 % | 小（~20 行） | 低 |
| 5 | **验证 VGPR**：对 iq_kernels.cu 单独编译加 `-Rpass-analysis=kernel-resource-usage`，若 spill 则降 TM 或调 I8_BLK_UNROLL | 决定 3/4 是否值得 | 小 | 无 |

## 八、建议执行顺序

1. 先做 **5**（半天不到，零风险）：确认 i8_kernel<T=12,GU=true,TM=4> 的 VGPR/spill。若 spill ≥ 40，先修 spill（可能一步到位）。
2. 再做 **4**（低成本，20 行）：预取深度 2。测 21K gate/up。
3. 再做 **1**（合并 launch）：省掉 19 % 的提交开销，顺带改善 group 边界。
4. **2**（重排）单独立项：这是唯一能碰到 2.5 %→10 % 量级的改动，但必须先有 parity（bit-identical）与 VRAM 预算核算
   （2×32 GB 里 expert cache 27.44 GB，重排只能在 cache 内原位做，不能另存副本）。

## 九、未完成/待补

- TM=2 那组因端口未退净没跑成；TM=2 崩溃与否待单独复测（若崩溃，说明 tile 2 路径有边界问题，本身也是 bug）。
- bisect 4/5/7 三个停点两次启动都失败（同为端口占用），只拿到 1 和 6。
- 未有 GPU 计数器（rocprofv3 不存在；只有 rocprof-compute，gfx906 支持待确认），
  第五节的有效带宽是"MAC 数 ÷ 时间 ÷ 读字节"的折算，不是硬件计数。
