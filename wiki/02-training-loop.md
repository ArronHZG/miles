# 02 训练主循环

## 1. 同步训练循环（train.py）

`train.py` 的主循环（`train.py:69-102`）每个 `rollout_id` 执行一次完整的 generate → train → update 周期：

```mermaid
sequenceDiagram
  autonumber
  participant Driver as Driver
  participant RM as RolloutManager
  participant ActorModel
  participant Critic as CriticModel

  Driver->>Driver: create_placement_groups<br/>create_rollout_manager<br/>create_training_models
  Driver->>RM: onload_weights (若 offload_rollout)
  Driver->>ActorModel: update_weights()  %% 首次同步给 SGLang
  Driver->>RM: onload_kv (若 offload_rollout)

  loop for rollout_id in [start, num_rollout)
    Driver->>RM: eval.remote() (周期性)
    Driver->>RM: generate.remote(rollout_id)
    RM-->>Driver: rollout_data_ref (Ray ObjectRef)

    Driver->>RM: offload(CUDA_GRAPH / KV_CACHE / WEIGHTS) (若 offload_rollout)

    alt use_critic
      Driver->>Critic: train(rollout_id, data_ref)  %% eager_create_task
      opt rollout_id >= num_critic_only_steps
        Driver->>ActorModel: train(rollout_id, data_ref)
      end
      Driver->>Critic: await critic_task
    else 无 critic
      Driver->>ActorModel: train(rollout_id, data_ref)
    end

    Driver->>ActorModel: save_model() (周期性)
    Driver->>Critic: save_model() (周期性)

    Driver->>ActorModel: offload_train / clear_memory
    Driver->>RM: onload_weights (若 offload_rollout)
    Driver->>ActorModel: update_weights()  %% 同步回 SGLang
    Driver->>RM: onload_kv (若 offload_rollout)

    Driver->>RM: eval.remote() (周期性)
  end

  Driver->>RM: dispose.remote()
```

### 步骤说明

| 步骤 | 代码位置 | 说明 |
| :--- | :--- | :--- |
| 初始化 | `train.py:23-37` | 建放置组、RolloutManager、训练模型；首次 `update_weights()` 让 SGLang 拿到训练权重 |
| 生成 rollout | `train.py:73` | `rollout_manager.generate.remote()` 返回 `ObjectRef`，内部含转换好的训练数据 |
| 卸载 rollout 显存 | `train.py:75-81` | 按级别卸载 CUDA Graph / KV Cache / 权重到 CPU |
| 训练 | `train.py:83-89` | critic 先训（warm-up 阶段仅 critic），再训 actor；critic 用 `eager_create_task` 并发 |
| 保存 | `train.py:91-92` | `should_run_periodic_action` 按 save_interval 触发 |
| 卸载训练 | `train.py:94-99` | `offload_train` 把权重移到 CPU，否则 `clear_memory` |
| 权重同步 | `train.py:97` | `actor_model.update_weights()` 广播到所有 rollout 引擎 |
| 评估 | `train.py:101` | 周期性 `eval.remote()` |

## 2. Actor 单步训练内部（Megatron）

`backends/megatron_utils/actor.py` 的 `train_actor()`（约 319-420 行）内部流程：

```mermaid
flowchart TD
  Start["train_actor(rollout_id, data_ref)"] --> Wake["wake_up (若已 offload)"]
  Wake --> Load["从 Ray Object Store 载入 rollout 数据"]
  Load --> Iter["构造 DataIterator"]
  Iter --> Replay["填充 Replay Buffer（若启用）"]
  Replay --> RefLP["计算 ref/teacher log_probs<br/>(若需要)"]
  RefLP --> Adv["compute_advantages_and_returns<br/>(KL + 优势估计 + 归一化)"]
  Adv --> Loop["训练 mini-batch 循环<br/>forward → loss → backward → optimizer step"]
  Loop --> Backup["备份更新后权重<br/>(model switching 用)"]
  Backup --> RefUpd["周期性更新参考模型"]
  RefUpd --> End["返回"]
```

- `train_critic()`（约 288-314 行）：前向取 value → 与 actor 同步（warm-up 后）→ 算优势 → 训练 value loss。
- `update_weights()`（约 465-519 行）：重载分布式组 → 连接新 rollout 引擎 → 按策略同步 → CI 校验 → 更新备份 tag。

## 3. 权重同步策略

```mermaid
flowchart LR
  ActorWeights["Actor 训练后权重"] --> Strategy{update_weight 策略}
  Strategy -->|广播| BC["UpdateWeightFromDistributed<br/>广播分片到各 SGLang"]
  Strategy -->|P2P| P2P["P2P 直传<br/>4× 更小（INT4/低精度友好）"]
  Strategy -->|Tensor| TB["基于 tensor 的 bucketed flatten"]
  BC --> ZeroCopy["CUDA IPC 零拷贝映射<br/>+ 异步 tensor gather"]
  P2P --> ZeroCopy
  TB --> ZeroCopy
  ZeroCopy --> SGLang["SGLang 引擎 refit"]
```

- 零拷贝权重同步（CUDA IPC + 异步 gather + bucketed flatten）相比 HTTP/RPC 传输减少约 50% 同步时间。
- CI 模式下 `check_weights_equal` 校验训练与推理权重逐位一致。

## 4. 同步 vs 异步循环

```mermaid
flowchart TB
  subgraph Sync["同步 (train.py)"]
    direction TB
    S1["generate (阻塞)"] --> S2["train"] --> S3["update_weights (每步)"] --> S1
    S0["显式 offload/onload 切换"]
  end

  subgraph Async["异步 (train_async.py)"]
    direction TB
    A1["generate(id) 预取"] --> A2["await generate(id-1)"]
    A2 --> A3["train"]
    A3 --> A4["update_weights<br/>(按 update_weights_interval)"]
    A4 --> A5["generate(id+1) 预取"]
    A5 --> A3
  end
```

| 维度 | 同步 `train.py` | 异步 `train_async.py` |
| :--- | :--- | :--- |
| Rollout 生成 | 阻塞 `await` | 预取下一个 rollout（`train_async.py:36,44`） |
| 权重更新时机 | 每个 rollout 后 | 可配置 `update_weights_interval` |
| offload/onload | 显式切换 CUDA Graph/KV/权重 | 移除（`assert not args.colocate`） |
| 显存压力 | 较低（有卸载点） | 较高（预取占用） |
| 适用 | 大模型需卸载、单机 | 多节点高带宽、显存充足 |
| 训练-推理重叠 | 否 | rollout 可与 training 重叠 |

## 5. 显存调度（offload 体系）

```mermaid
stateDiagram-v2
  [*] --> RolloutActive: rollout 期间
  RolloutActive --> TrainActive: offload rollout<br/>(CUDA_GRAPH/KV/WEIGHTS)
  TrainActive --> RolloutActive: onload weights + kv<br/>update_weights
  TrainActive --> TrainIdle: offload_train / clear_memory
  TrainIdle --> TrainActive: wake_up
```

- 三类显存标签：`GPU_MEMORY_TYPE_CUDA_GRAPH`、`GPU_MEMORY_TYPE_KV_CACHE`、`GPU_MEMORY_TYPE_WEIGHTS`（来自 `sglang.srt.constants`）。
- `offload_rollout_level` 控制卸载哪些层级，支持共存模式下训练与推理分时复用同一批 GPU。

## 6. 参数大类（arguments.py）

`miles.utils.arguments` 中的主要参数组：

```mermaid
mindmap
  root((Miles 参数))
    Cluster
      actor/critic nodes & gpus
      rollout_num_gpus
      colocate / offload
      distributed_backend
    Train
      train_backend (megatron/fsdp)
      qkv_format (thd/bshd)
      linear_attention_backend
      true_on_policy_mode
    Rollout
      rollout_function_path
      batch_size / max_response_length
      over_sampling
    Algo
      use_critic / num_critic_only_steps
      kl_coef / use_kl_loss
      advantage_normalization
    Fault Tolerance
      use_fault_tolerance
      health_monitor / engine_recovery
    Low Precision
      fp8 / int4 / mxfp8 / nvfp4
      routing_replay
    Tracking
      wandb / mlflow / tensorboard / prometheus
    Debug
      debug_train_only / debug_rollout_only
      check_weight_update_equal / ci_test
```
