# 08 多轮交互与 VLM

## 1. 多轮会话：TITO（Token In, Token Out）

`miles/rollout/session/` 管理多轮交互的完整状态，避免每轮重算整个上下文。

```mermaid
stateDiagram-v2
  [*] --> FirstTurn: 首轮
  FirstTurn: 渲染完整 prompt<br/>prepare_pretokenized()
  FirstTurn --> Gen: 生成 assistant 回复
  Gen --> Checkpoint: 存 accumulated_token_ids
  Checkpoint --> NextTurn: 下一轮
  NextTurn: 复用已存 tokens<br/>+ 合并新 appended 消息
  NextTurn --> Gen
  Gen --> Retry: 工具失败?
  Retry --> Rollback: 回退最多一个 assistant step
  Rollback --> Gen
  Checkpoint --> [*]: 会话结束
```

- `LinearTrajectory`（`linear_trajectory.py:20-127`）维护消息历史 `[system?, user, assistant, tool, assistant, tool, …]` 与每步的 `accumulated_token_ids` 检查点。
- `SessionRegistry`（`linear_trajectory.py`）按 ID 管理会话。
- `prepare_pretokenized()`：首轮渲染完整 prompt；后续轮复用存储 tokens + 合并新消息。
- `update_pretokenized_state()`：生成后存新检查点。

## 2. 生成 Hub 中的多轮入口

```mermaid
flowchart LR
  Entry{rollout 类型} -->|单轮| ST["single_turn.py"]
  Entry -->|多轮工具调用| MT["multi_turn.py<br/>--generate-max-turns (默认16)<br/>tool_call_parser"]
  Entry -->|agentic TITO| AGT["agentic_tool_call.py<br/>OpenAI 兼容端点追踪"]

  AGT --> Tracer["OpenAIEndpointTracer<br/>/sessions/{session_id}"]
  Tracer --> Rec["收集 ChatCompletion<br/>Request/Response"]
  Rec --> Conv["compute_samples_from_openai_records<br/>转训练样本"]
```

- `agentic_tool_call.py:35-129` 的 `OpenAIEndpointTracer` 把多轮 agent 函数调用追踪为训练样本。
- `multi_turn.py:23-77` 支持多采样跟踪与工具调用解析。

## 3. 多轮 RL 流程

```mermaid
sequenceDiagram
  autonumber
  participant DS as Data Source
  participant Gen as multi_turn generate
  participant SGLang
  participant Tool as 工具/环境

  DS->>Gen: prompt (含可用工具)
  Gen->>SGLang: 首轮采样
  SGLang-->>Gen: assistant 回复(含 tool_call)
  Gen->>Tool: 执行工具
  Tool-->>Gen: 工具结果
  Gen->>Gen: 更新 loss mask<br/>(仅 assistant 计 loss)
  Gen->>SGLang: 续轮采样(复用 KV)
  SGLang-->>Gen: assistant 回复
  Gen->>Gen: 达 max_turns 或 done
  Gen->>Gen: 生成多轮 loss mask<br/>MultiTurnLossMaskGenerator
  Gen-->>DS: 完整轨迹样本
```

## 4. VLM（视觉语言模型）采样

```mermaid
flowchart LR
  Prompt["prompt + images"] --> Proc["call_processor<br/>(sglang_rollout.py:146-151)"]
  Proc --> MM["sample.multimodal_inputs<br/>['images']"]
  Proc --> MMT["sample.multimodal_train_inputs<br/>(训练用)"]
  MM --> Payload["SGLang payload<br/>多模态采样参数"]
  Payload --> SGLang["SGLang VLM 推理"]
  SGLang --> Out["tokens + logprobs"]
  MMT --> Train["训练侧多模态输入对齐"]
```

- 处理器输出同时供给 rollout（推理）与 training（训练），保证两侧多模态输入一致。
- `miles_plugins/models/qwen3_vl.py` 处理 packed mRoPE 位置编码（THD 格式），近期 commit `803016a` 修复了 Qwen3-VL THD packed mRoPE。

## 5. VLM 多轮

```mermaid
flowchart LR
  VLM1["单轮 VLM<br/>examples/geo3k_vlm<br/>(GRPO + FSDP)"] --> VLM2["多轮 VLM<br/>examples/geo3k_vlm_multi_turn<br/>(FSDP 后端)"]
  VLM2 --> Concept["统一 VLM/LLM 多轮范式<br/>只需写自定义 rollout 函数"]
```

- README「Unified VLM/LLM Multi-Turn Training」：开发者只写自定义 `rollout` 函数即可启动 VLM 多轮 RL，与 LLM 一致。

## 6. true_on_policy 子系统

`miles/true_on_policy/` 从内核层面保证 SGLang rollout 与 Megatron/FSDP 训练的 log-prob 严格相等。

```mermaid
flowchart TB
  Enable["--true-on-policy"] --> Config["TrueOnPolicyConfig<br/>(config.py:139-237)"]
  Config --> Layout["并行布局<br/>TP/CP/PP/rollout gpus"]
  Config --> Kernel["TrueOnPolicyKernelPolicy<br/>(58-108)"]

  Kernel --> K1["deterministic_inference (SGLang)"]
  Kernel --> K2["deterministic_training (Megatron/FSDP)"]
  Kernel --> K3["sglang_attention_backend<br/>(如 fa3)"]
  Kernel --> K4["disable_rope_fusion<br/>disable_bias_swiglu_fusion"]
  Kernel --> K5["batch_invariant_mode<br/>tp_invariant_row_linear<br/>deterministic_tp_allreduce"]

  Kernel --> Env["环境变量<br/>NCCL_ALGO=Ring<br/>NVTE_ALLOW_NONDETERMINISTIC_ALGO=0<br/>CUBLAS_WORKSPACE_CONFIG"]

  Config --> Contract["TrueOnPolicyContract<br/>(contracts.py:15-86)<br/>由 model profile 签名"]
  Contract --> Schema["schema.py<br/>如 QWEN3_DENSE_TRUE_ON_POLICY_V1<br/>kernel_contract / logprob_contract<br/>attention backend / 禁用 SP"]
```

- 当 `train_tensor_parallel_size > 1` 或 `rollout_num_gpus_per_engine > 1` 时，要求 **TP-invariant rollout**。
- SGLang target：`fsdp`（无 TP）或 `fsdp_tp`（有 TP）。
- `kernel_policy_kwargs_for()` 把 contract 适配到 train backend + sglang target。
- `examples/true_on_policy/` 与 `examples/true_on_policy_vlm/` 提供演示。

## 7. 多智能体共进化（MrlX）

```mermaid
flowchart LR
  Agent1["专业 Agent A"] -.共同进化.-> Agent2["专业 Agent B"]
  Agent2 -.共同进化.-> Agent3["专业 Agent C"]
  Agent1 --> Task["复杂任务<br/>医患模拟 / DeepResearch"]
  Agent2 --> Task
  Agent3 --> Task
  Task --> MrlX["MrlX 异步共进化框架<br/>多智能体 RL"]
```

- README「Multi-Agent Co-Evolution」：支持 MrlX 异步共进化框架，多智能体共生进化。
- 见 `examples/multi_agent/`。
