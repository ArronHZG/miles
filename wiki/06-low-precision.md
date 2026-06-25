# 06 低精度训练

Miles 的核心卖点之一：**端到端低精度**，消除训练-推理量化差异导致的 RL 崩溃。涉及 FP8 / MXFP8 / NVFP4 / INT4 QAT。

## 1. 精度管线总览

```mermaid
flowchart LR
  HF["HF BF16 检查点"] -->|"convert_hf_to_torch_dist"| TD["torch_dist<br/>(BF16 master 权重, 训练用)"]
  HF -->|"convert_hf_to_fp8"| FP8I["FP8 block-wise<br/>(128×128, FP32 scales)<br/>SGLang 推理"]
  HF -->|"convert_hf_to_mxfp8"| MXI["MXFP8 (1×32, UE8M0)<br/>Blackwell 推理"]
  HF -->|"convert_hf_to_nvfp4"| NVI["NVFP4 (E2M1, MoE 专家)<br/>实验性"]
  HF -->|"convert_hf_to_int4"| INT4C["INT4 W4A16 (GPTQ)<br/>校准 → QAT"]

  TD --> Trainer["Trainer 载入<br/>BF16 master 权重"]
  Trainer -->|"前向"| Q["动态量化到 FP8/MXFP8<br/>(TransformerEngine / 自定义 kernel)"]
  Q --> Back["反向: BF16 梯度"]
  Back --> Opt["优化器: BF16 master 更新"]
  Opt --> Sync["权重同步到 SGLang<br/>重新量化"]
  FP8I --> SGLang["SGLang rollout"]
  MXI --> SGLang
  NVI --> SGLang
  Sync --> SGLang
```

## 2. 转换工具链（tools/）

```mermaid
graph LR
  subgraph FromHF["从 HF 出发"]
    A["convert_hf_to_torch_dist.py<br/>BF16 → Megatron 分布式"]
    B["convert_hf_to_fp8.py<br/>BF16 → FP8 block (E4M3)"]
    C["convert_hf_to_mxfp8.py<br/>BF16/blockFP8 → MXFP8"]
    D["convert_hf_to_nvfp4.py<br/>BF16 → NVFP4 (MoE)"]
    E["convert_hf_to_int4.py<br/>BF16 → INT4 W4A16 (GPTQ)"]
    F["convert_hf_to_int4_direct.py<br/>直接 INT4 量化"]
    G["convert_hf_to_hf_int4.py<br/>HF 原生 INT4 格式"]
  end

  subgraph ToHF["回到 HF / 导出"]
    H["convert_torch_dist_to_hf.py"]
    I["convert_to_hf.py<br/>Megatron ckpt → HF"]
    J["convert_fsdp_to_hf.py<br/>FSDP → HF"]
    K["convert_kimi_int4_to_bf16.py"]
    L["convert_fsdp_to_hf.py"]
    M["fp8_cast_bf16.py"]
  end

  subgraph Aux["辅助"]
    N["param_name_remap.py<br/>参数名重映射"]
    O["cpu_memory_profiler.py"]
    P["tools/visualize"]
  end
```

| 工具 | 输入 → 输出 | 用途 |
| :--- | :--- | :--- |
| `convert_hf_to_torch_dist` | HF safetensors → torch_dist | 基线转换，训练加载 |
| `convert_hf_to_fp8` | BF16 → FP8 block (128×128, FP32 scale) | SGLang 推理 |
| `convert_hf_to_mxfp8` | BF16/块FP8 → MXFP8 (1×32, UE8M0) | Blackwell 推理 |
| `convert_hf_to_nvfp4` | BF16 → NVFP4 (E2M1, 仅 MoE 专家) | 实验性, Blackwell |
| `convert_hf_to_int4` | BF16 → INT4 W4A16 (GPTQ 分组) | 校准 → QAT |
| `convert_hf_to_int4_direct` | BF16 → INT4 直接量化 | 直接量化 |
| `convert_hf_to_hf_int4` | BF16 → HF INT4 格式 | HF 原生 |
| `convert_torch_dist_to_hf` | torch_dist → HF | 反向 |
| `convert_to_hf` | Megatron ckpt → HF | 训练导出 |
| `convert_fsdp_to_hf` | FSDP → HF | FSDP 导出 |
| `convert_kimi_int4_to_bf16` | Kimi INT4 → BF16 | 模型专用 upcast |

## 3. 统一 FP8：为什么重要

```mermaid
flowchart LR
  subgraph Problem["问题：非统一量化"]
    P1["SGLang 推理 FP8 (一种量化)"] --> Diff["logprob 不一致"]
    P2["Megatron 训练 FP8 (另一种量化)"] --> Diff
    Diff --> Collapse["off-policy 偏差<br/>MoE RL 崩溃"]
  end

  subgraph Solution["Miles 统一 FP8"]
    S1["推理与训练用完全相同 FP8 量化逻辑"] --> Align["bit-wise 一致 logprob"]
    Align --> Stable["训练稳定"]
  end

  Problem -.->|解决| Solution
```

- 统一 FP8 + R3 + true_on_policy 共同消除 train-inference mismatch。
- 见 `docs/advanced/fp8-low-precision.md`。

## 4. FP8 kernel

`miles/utils/fp8_kernel.py`（80 行）：

```mermaid
flowchart LR
  In["BF16 tensor"] --> Triton["_blockwise_cast_to_fp8_triton<br/>(block 默认 128×128)"]
  Triton --> Max["每块求 max"]
  Max --> Scale["缩放到 FP8 范围"]
  Scale --> Out["(FP8 量化张量,<br/>FP32 per-block scales)"]
  Out --> Use["前向调用<br/>--transformer-impl transformer_engine<br/>--fp8-format e4m3"]
```

## 5. 精度兼容性矩阵

```mermaid
flowchart TD
  M["精度组合矩阵<br/>(行=Rollout, 列=Train)"]

  M --> R1["BF16 × BF16 = ✅ 基线"]
  M --> R2["FP8 block × BF16 = ✅"]
  M --> R3["FP8 block × FP8 block = ✅ Hopper/Blackwell"]
  M --> R4["MXFP8 × MXFP8 = ✅ Blackwell"]
  M --> R5["NVFP4 × NVFP4 = 🚧 进行中"]
  M --> R6["其他交叉组合 = ✗"]
```

## 6. INT4 QAT 流程（W4A16）

受 Kimi K2-Thinking 启发，让 1TB+ 模型塞进单机（如 H200）。

```mermaid
flowchart LR
  HF["HF BF16"] --> Cal["GPTQ 校准<br/>(llmcompressor, 256 样本, seq=2048)"]
  Cal --> Quant["分组量化<br/>group_size 32-128 (典型128)"]
  Quant --> QAT["QAT 训练<br/>OPEN_TRAINING_INT4_FAKE_QAT_FLAG=1<br/>OPEN_TRAINING_INT4_GROUP_SIZE=128"]
  QAT --> W["权重 INT4 (per-group)"]
  QAT --> A["激活 BF16 (A16)"]
  QAT --> Master["master 权重 BF16<br/>(梯度稳定)"]
  W --> Comb["组合"]
  A --> Comb
  Master --> Comb
  Comb --> Benefit["显存减半<br/>rollout 效率翻倍<br/>消除跨节点瓶颈<br/>BF16 等价精度"]
```

INT4 QAT 常与 **R3 路由重放 + P2P 权重传输（权重小 4×）+ 投机解码** 组合使用。

## 7. P2P 权重传输

```mermaid
flowchart LR
  Old["标准 HTTP/RPC 权重同步"] -->|"慢"| Slow["跨节点瓶颈"]
  New["P2P 直传 + CUDA IPC 零拷贝<br/>+ 异步 gather + bucketed flatten"] -->|"快 50%"| Fast["低精度权重更受益<br/>(INT4/FP8 体积小)"]
  Slow -.->|优化| New
```

见 `docs/advanced/p2p-weight-transfer.md`。

## 8. 关键文件索引

| 文件 | 作用 |
| :--- | :--- |
| `miles/utils/fp8_kernel.py` | Triton blockwise FP8 量化 |
| `tools/convert_hf_to_*.py` | 各精度转换 |
| `miles/utils/replay_base.py` | R3 路由重放基础设施 |
| `miles/true_on_policy/` | 内核级一致性策略 |
| `docs/advanced/fp8-low-precision.md` | FP8/MXFP8/NVFP4 文档 |
| `docs/advanced/int4-qat.md` | INT4 QAT 文档 |
