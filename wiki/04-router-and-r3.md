# 04 Router 与 R3 路由重放

## 1. MilesRouter：负载均衡 + 健康检查

`miles/router/router.py` 的 `MilesRouter`（34-233 行）是位于 rollout 客户端与 SGLang worker 池之间的 FastAPI 代理。

```mermaid
flowchart LR
  Client["Rollout 引擎<br/>(sglang_rollout)"] -->|"HTTP 请求<br/>X-SMG-Routing-Key"| Router["MilesRouter"]
  Router --> Sel["_use_url()<br/>选最小活跃请求数 worker"]
  Sel --> W1["SGLang worker 1"]
  Sel --> W2["SGLang worker 2"]
  Sel --> WN["SGLang worker N"]

  subgraph HC["健康检查循环"]
    Loop["_health_check_loop<br/>每 interval 秒"]
    Loop -->|失败累计 ≥ 阈值| Dead["标记 DEAD<br/>移出路由池"]
    Dead -.->|恢复| Sel
  end

  W1 -->|"响应"| Router
  Router -->|"转发响应"| Client
```

| 能力 | 实现位置 | 说明 |
| :--- | :--- | :--- |
| 负载均衡 | `router.py:211-226` | 选活跃请求最少的 worker |
| 健康检查 | `router.py:73-127` | 后台 async loop，连续失败达阈值标记 DEAD |
| 动态加 worker | `add_worker()` 177-205 | `/add_worker?url=...` |
| 列举 worker | `list_workers()` 207-209 | |
| 通用代理 | `router.py:129-175` | 透传请求/响应体 |

参数：`sglang_router_ip:port`、`sglang_server_concurrency`（每 worker 最大并发）、`miles_router_max_connections`（httpx 连接池）。

## 2. R3（Rollout Routing Replay）解决的问题

大 MoE 模型（Qwen3、DeepSeek-V3）在 RL 中易「训练-推理 mismatch」导致崩溃：推理时 MoE router 选的专家与训练时重算的不一致，造成 off-policy 偏差。**R3 记录推理时的路由决策，训练时重放，保证 bit-wise 专家对齐。**

```mermaid
flowchart LR
  subgraph Rollout["Rollout 阶段（SGLang）"]
    RP["prompt"] --> MoEInf["MoE 前向<br/>router 选 top-k 专家"]
    MoEInf --> Rec["记录 routed_experts<br/>[tokens-1, layers, topk]"]
    Rec --> Samp["返回 tokens + logprobs<br/>+ routed_experts(base64)"]
  end

  subgraph Train["Training 阶段（Megatron）"]
    TD["训练数据含 routed_experts"] --> Rep["RoutingReplayManager"]
    Rep --> Fwd["前向时不再算 router logits<br/>直接 pop 记录的 top_indices"]
    Fwd --> Align["专家门控与推理一致<br/>true on-policy"]
  end

  Samp --> TD
```

## 3. R3 记录与重放时序

```mermaid
sequenceDiagram
  autonumber
  participant Rollout as SGLang Rollout
  participant Sample as Sample 对象
  participant Replay as RoutingReplayManager
  participant Trainer as Megatron 前向

  Note over Rollout: --use-rollout-routing-replay
  Rollout->>Rollout: payload return_routed_experts=True
  Rollout->>Rollout: MoE forward, 记录每层 top-k 专家
  Rollout-->>Sample: meta_info["routed_experts"] (base64 np)
  Sample->>Sample: 解码 np.int32<br/>reshape [tokens-1, layers, topk]
  Sample->>Sample: 存 sample.rollout_routed_experts

  Note over Trainer: 训练前
  Trainer->>Replay: 载入样本, 提取 routed_experts
  Trainer->>Replay: 构建 top_indices_list
  loop 每层 MoE 前向
    Trainer->>Replay: pop next indices
    Replay-->>Trainer: 用记录的路由(跳过 router 计算)
  end
  Note over Trainer: enable_check_replay_result 时<br/>校验重放 == 当前 router 输出
```

## 4. R3 基础设施

```mermaid
classDiagram
  class Replay {
    +name
    +data_key
    +forward/backward indices
    +circular buffer
  }
  class BaseReplayManager {
    +manage multiple Replay
    +orthogonal to backend
  }
  class RoutingReplayManager {
    name = "routing"
    data_key = "rollout_routed_experts"
  }
  class ReplayUtils {
    megatron_utils/replay_utils.py
    注入 routed_experts 到前向
  }

  BaseReplayManager <|-- RoutingReplayManager
  BaseReplayManager o-- Replay
  RoutingReplayManager ..> ReplayUtils : 集成点
```

- 基础设施在 `miles/utils/replay_base.py`（`Replay` 循环缓冲 + `BaseReplayManager`），与具体路由后端正交。
- Megatron 集成点：`backends/megatron_utils/replay_utils.py` 在 MoE 前向注入记录的路由。

## 5. Router 元数据保留

MilesRouter 不仅做负载均衡，还**保留 R3 等元数据**透传，保证 routed_experts 等信息随响应回到 rollout 引擎。这与一致哈希（`X-SMG-Routing-Key`）配合，让多轮会话与路由重放在分布式 worker 下依然正确。

## 6. 相关：true_on_policy 子系统

`miles/true_on_policy/` 进一步从内核层面保证 SGLang rollout 与 Megatron/FSDP 训练的**权重精确对齐 + 确定性**，是 R3 之外的另一条「一致性」路径（见 [08 多轮与 VLM](./08-multi-turn-and-vlm.md) 与 `true_on_policy/config.py`）。

```mermaid
flowchart LR
  Goal["训练-推理 bit-wise 一致"] --> R3["R3 路由重放<br/>MoE 专家对齐"]
  Goal --> TOP["true_on_policy<br/>内核策略对齐"]
  Goal --> FP8["统一 FP8<br/>量化逻辑对齐"]
  R3 --> Replay["replay_base + replay_utils"]
  TOP --> Contract["TrueOnPolicyContract<br/>+ KernelPolicy"]
  FP8 --> Fp8K["fp8_kernel + 统一量化"]
```
