# 05 插件与模型桥接

Miles 通过三层插件架构支持 DeepSeek/Qwen/Llama/GLM/Kimi/Nemotron 等多模型族。代码在 `miles_plugins/`。

## 1. 三层架构

```mermaid
flowchart TB
  HF["HuggingFace 模型<br/>(safetensors BF16)"] -->|"convert_hf_to_*"| Conv["转换工具<br/>tools/"]

  Conv --> MBridge["miles_plugins/mbridge/<br/>转换桥（HF ↔ Megatron 权重映射）<br/>@register_model 装饰器"]
  Conv --> Models["miles_plugins/models/<br/>原生层实现<br/>(GatedDeltaNet / SparseMLA / VLM mRoPE)"]
  MBridge --> MegaShim["miles_plugins/megatron_bridge/<br/>Megatron 内部 shim<br/>(PP group unwrap / VLM 补丁)"]

  MBridge -->|"AutoBridge.from_hf()"| Mega["Megatron 模型实例"]
  Models -->|"nn.Module / MegatronModule"| Mega
  MegaShim -->|"import 时 monkeypatch"| Mega

  Mega -->|"weight_sync"| SGLang["SGLang 引擎<br/>(量化推理)"]
```

### 三层职责对比

| 层 | 路径 | 职责 | 继承自 |
| :--- | :--- | :--- | :--- |
| **mbridge** | `miles_plugins/mbridge/` | 转换桥：HF↔Megatron 权重名映射、RoPE/缩放配置 | 上游 `mbridge.core.Bridge` |
| **models** | `miles_plugins/models/` | 原生层实现：自定义算子（GDN、稀疏 MLA、VLM mRoPE） | `nn.Module` + Megatron `MegatronModule` |
| **megatron_bridge** | `miles_plugins/megatron_bridge/` | Megatron shim：monkeypatch 进程组、VLM 扩展 | mbridge.bridge |

## 2. mbridge 模型注册

```mermaid
classDiagram
  class BridgeBase {
    <<mbridge.core.Bridge>>
    +_ATTENTION_MAPPING
    +_get_rope_theta()
    +_get_scaling()
  }
  class DeepseekV3Bridge
  class DeepseekV32Bridge
  class DeepseekV4Bridge
  class Qwen2Bridge
  class Qwen2MoEBridge
  class Qwen3_5Bridge
  class Qwen3NextBridge
  class GLM4Bridge
  class GLM4MoEBridge
  class MiMoBridge
  class NemotronHBridge

  BridgeBase <|-- DeepseekV3Bridge
  DeepseekV3Bridge <|-- DeepseekV32Bridge
  DeepseekV3Bridge <|-- DeepseekV4Bridge
  BridgeBase <|-- Qwen2Bridge
  Qwen2Bridge <|-- Qwen2MoEBridge
  Qwen2MoEBridge <|-- Qwen3_5Bridge
  Qwen2MoEBridge <|-- Qwen3NextBridge
  BridgeBase <|-- GLM4Bridge
  GLM4Bridge <|-- GLM4MoEBridge
  BridgeBase <|-- MiMoBridge
  BridgeBase <|-- NemotronHBridge
```

每个桥通过 `@register_model("name")` 注册，提供 `_ATTENTION_MAPPING`（HF↔Megatron 参数名映射）与架构特定配置。

## 3. 插件加载链

```mermaid
sequenceDiagram
  autonumber
  participant Trainer as Trainer
  participant MBridge as miles_plugins.mbridge
  participant MegaBridge as miles_plugins.megatron_bridge
  participant Auto as AutoBridge

  Trainer->>MBridge: import miles_plugins.mbridge
  MBridge->>Auto: @register_model 自动注册所有桥
  Trainer->>MegaBridge: import miles_plugins.megatron_bridge
  MegaBridge->>MegaBridge: 安装 PP group unwrap shim
  MegaBridge->>MegaBridge: 打 Qwen3-VL mRoPE 补丁

  Trainer->>Auto: AutoBridge.from_hf(hf_model, hf_config)
  Auto->>Auto: 按注册表选桥
  Auto->>Auto: 转换权重 (HF → Megatron)
  Auto-->>Trainer: Megatron 模型实例
```

## 4. 原生模型实现（models/）

```mermaid
graph LR
  subgraph Models["miles_plugins/models/"]
    Q35["qwen3_5.py<br/>GatedDeltaNet + CP"]
    Q3N["qwen3_next.py<br/>thinking 变体"]
    QVL["qwen3_vl.py<br/>多模态 + packed mRoPE"]
    GLM5["glm5/<br/>稀疏 MLA 算子"]
    DSV4["deepseek_v4/<br/>稀疏 MLA + TileLang kernel"]
    HFA["hf_attention.py<br/>HF 注意力 + CP/ring attention"]
    CPU["cp_utils.py<br/>上下文并行工具"]
    QGDN["qwen_gdn_backend.py<br/>Gated Delta Rule 后端"]
  end
```

- `qwen3_vl.py`：处理 packed mRoPE 位置（见近期 commit `803016a` 修复 Qwen3-VL THD packed mRoPE）。
- `deepseek_v4/`：含自定义 TileLang kernel 的稀疏 MLA。
- `hf_attention.py`：为 CP（上下文并行）+ ring attention 包装 HF 注意力。

## 5. Megatron shim（megatron_bridge/）

```mermaid
flowchart LR
  Import["import miles_plugins.megatron_bridge"] --> Shim1["PP group unwrap shim<br/>兼容 ReloadableProcessGroup"]
  Import --> Shim2["Qwen3-VL mRoPE 补丁"]
  Import --> Nemotron["nemotron_h.py<br/>Nemotron-H 桥子类"]
  Shim1 --> Mega["Megatron 内部"]
  Shim2 --> Mega
```

- `__init__.py` 在 import 时自动应用 best-effort shim。
- 解决 Megatron 与 Miles 可重载进程组（`reloadable_process_group.py`）的兼容问题。

## 6. 模型族 → 桥/原生实现 映射

| 模型族 | mbridge 桥 | 原生 models/ | 备注 |
| :--- | :--- | :--- | :--- |
| DeepSeek V3/V3.2/V4/R1 | `deepseek_v32`, `deepseekv4` | `deepseek_v4/` | DSA / 稀疏 MLA |
| Qwen 2/2.5/3 | `qwen3_5` (基类链) | `qwen3_5.py` | GatedDeltaNet |
| Qwen3-Next | `qwen3_next` | `qwen3_next.py` | thinking |
| Qwen3-VL | — | `qwen3_vl.py` | packed mRoPE |
| GLM 4/4.5/4.6/4.7/5 | `glm4`, `glm4moe`, `glm4moe_lite` | `glm5/` | 稀疏 MLA |
| Kimi K2/Moonlight | — | (走通用桥) | INT4 QAT 友好 |
| Nemotron-H | `megatron_bridge/nemotron_h` | — | |
| MiMo-7B | `mimo` | — | 继承 Qwen2Bridge |
| gpt-oss | — | — | SGLang/Megatron 通用支持 |

> 新增模型流程：写一个继承合适父桥的类 + `@register_model` → 必要时在 `models/` 加原生算子 → 用 `tools/convert_hf_to_torch_dist.py` 转检查点。
