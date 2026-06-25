# 01 系统架构总览

## 1. 进程模型

一次 Miles 运行是包裹在 **Ray 集群**中的三类进程协作：

```mermaid
flowchart TB
  Driver["Driver 进程<br/>train.py / train_async.py<br/>asyncio 训练主循环"]

  subgraph Ray["Ray 集群"]
    subgraph Trainer["Trainer（1+ Megatron/FSDP 组）"]
      TA0["Actor rank 0"]
      TA1["Actor rank 1"]
      TAN["Actor rank ..."]
      TC["Critic 组（可选）<br/>use_critic=True"]
    end

    subgraph RolloutProc["Rollout（N 个 SGLang server + MilesRouter）"]
      R1["SGLang server 1"]
      R2["SGLang server 2"]
      RN["SGLang server N"]
      MR["MilesRouter<br/>FastAPI 代理 + 负载均衡 + 健康检查"]
      R1 -. 健康/路由 .- MR
      R2 -. 健康/路由 .- MR
      RN -. 健康/路由 .- MR
    end

    DS["Data Source<br/>RolloutDataSourceWithBuffer<br/>读取 prompt JSONL + 缓冲"]
  end

  Driver -->|"create_placement_groups"| Trainer
  Driver -->|"create_rollout_manager"| RolloutProc
  Driver -->|"create_training_models"| Trainer

  TA0 <-->|"weight sync<br/>(broadcast / P2P / CUDA IPC)"| R1
  TA0 <-->|"weight sync"| R2
  DS -->|"prompts"| MR
  MR -->|"采样请求"| R1
  MR -->|"采样请求"| R2
  R1 -->|"samples + logprobs"| DS
```

- **Driver**：`train.py` 的主循环，调度 rollout 与 train，做权重同步。
- **Trainer ranks**：每个 GPU 一个 `MegatronTrainRayActor` / `FSDPTrainRayActor`，加载 `torch_dist` 检查点，执行 RL 前向/反向。
- **SGLang servers**：独立 HTTP 服务，产出 rollout（支持 FP8/MXFP8 量化推理）。
- **MilesRouter**：FastAPI 代理，分发 rollout 请求、保留 R3 元数据、执行健康检查。
- **Data Source**：Trainer 持有的 Python 对象，读取 prompt JSONL，作为 rollout 与 training 之间的缓冲。

## 2. 顶层组件依赖

```mermaid
flowchart LR
  Args["miles.utils.arguments<br/>parse_args()"] --> PG["ray.placement_group<br/>create_placement_groups"]
  Args --> RMgr["ray.rollout.rollout_manager<br/>RolloutManager (Ray Actor)"]
  Args --> TModels["ray.placement_group<br/>create_training_models"]

  PG --> RMgr
  PG --> TModels

  RMgr -->|"管理"| Servers["backends.sglang_utils<br/>SGLang 引擎 + Router"]
  RMgr -->|"调用"| RolloutFn["rollout.sglang_rollout<br/>generate_rollout()"]

  TModels --> ActorG["ray.actor_group.RayTrainGroup<br/>(Actor)"]
  TModels --> CriticG["ray.actor_group.RayTrainGroup<br/>(Critic, 可选)"]

  ActorG --> Backends["backends.megatron_utils<br/>/ experimental.fsdp_utils"]
  Backends --> TrainUtils["backends.training_utils<br/>loss / data / replay"]
```

## 3. 包布局

```mermaid
flowchart TB
  subgraph MilesPkg["miles/"]
    Backends["backends/"]
    RayPkg["ray/"]
    RolloutPkg["rollout/"]
    RouterPkg["router/"]
    TrueOnPolicy["true_on_policy/"]
    UtilsPkg["utils/"]
  end

  Backends --> MU["megatron_utils/<br/>actor, model, initialize,<br/>update_weight/, checkpoint, lora_utils"]
  Backends --> SU["sglang_utils/<br/>sglang_engine, arguments, config"]
  Backends --> TU["training_utils/<br/>loss, data, loss_hub/,<br/>replay_data, log_utils"]
  Backends --> FSDP["experimental/fsdp_utils/<br/>(in progress)"]

  RayPkg --> RA["ray_actor / train_actor<br/>actor_group / placement_group"]
  RayPkg --> RRoll["rollout/<br/>rollout_manager, rollout_server,<br/>train_data_conversion"]

  RolloutPkg --> SG["sglang_rollout / sft_rollout /<br/>sleep_rollout / on_policy_distillation"]
  RolloutPkg --> Hubs["generate_hub / filter_hub / rm_hub"]
  RolloutPkg --> Sess["session/<br/>LinearTrajectory, sessions"]
  RolloutPkg --> InfRoll["inference_rollout/ (实验性重构)"]
  RolloutPkg --> DS["data_source.py / base_types.py"]

  RouterPkg --> RT["router.py<br/>MilesRouter"]
  TrueOnPolicy --> TOP["config / contracts / schema<br/>+ model_profiles"]
  UtilsPkg --> UArgs["arguments.py (巨型参数注册)"]
  UtilsPkg --> UD["distributed / fp8_kernel /<br/>replay_base / mask / seqlen_balancing /<br/>health_monitor / memory / timer ..."]
```

## 4. 三类 GPU 放置策略

`create_placement_groups(args)` 返回 `{"actor", "critic", "rollout"}` 三组放置组：

```mermaid
flowchart LR
  subgraph NonCol["非共存（标准）"]
    A1["Actor 组<br/>actor_nodes × actor_gpus_per_node"]
    C1["Critic 组<br/>(可选)"]
    R1["Rollout 组<br/>rollout_num_gpus"]
  end

  subgraph Col["共存 colocate=True"]
    Same["同一批 GPU<br/>训练与 rollout 时间复用<br/>需 offload/onload 切换"]
  end

  subgraph Debug["调试"]
    DTO["--debug-train-only"]
    DRO["--debug-rollout-only"]
  end
```

- **策略 PACK**：bundle `[{"GPU":1,"CPU":1}]`，尽量打包到同一节点。
- 通过 `InfoActor` 内省物理 GPU ID，按 node IP + GPU ID 排序得到稳定映射。
- 共存模式要求 `offload_train` / `offload_rollout` 配合，在 rollout 期间把训练权重/显存卸载到 CPU。

## 5. Ray Actor 类层次

```mermaid
classDiagram
  class RayActor {
    <<base>>
  }
  class TrainRayActor {
    <<abstract>>
    +train(rollout_id, data_ref)
    +update_weights(info)
    +save_model(rollout_id)
    +sleep/wake_up(tags)
  }
  class MegatronTrainRayActor
  class FSDPTrainRayActor
  class RolloutManager {
    +generate(rollout_id)
    +eval(rollout_id)
    +offload/onload_weights()
    +onload_kv()
    +dispose()
  }
  class RayTrainGroup {
    +train()
    +update_weights()
    +save_model()
    +clear_memory()
  }

  RayActor <|-- TrainRayActor
  TrainRayActor <|-- MegatronTrainRayActor
  TrainRayActor <|-- FSDPTrainRayActor
  RayActor <|-- RolloutManager
  RayTrainGroup o-- TrainRayActor : 聚合多个 actor
```

- `RayTrainGroup`（`ray/actor_group.py`）管理一组训练 actor，把 `train()/update_weights()/save_model()` 广播到组内所有 rank，通过 NCCL + Gloo 协作分布式训练。
- 后端在 `actor_group.py:79-88` 按 `args.train_backend`（`megatron` / `fsdp`）选择具体实现。

## 6. 与官方 docs 的关系

| 官方文档 | 定位 | 本 Wiki 补充 |
| :--- | :--- | :--- |
| `docs/developer/architecture.md` | 30 分钟进程/包导览 | 组件依赖图、Actor 类层次、放置策略细化 |
| `docs/user-guide/concepts.md` | 四核心对象概念 | 内部数据流与调用链 |
| `docs/advanced/*` | 各高级特性怎么做 | 各特性在系统中如何串联 |
