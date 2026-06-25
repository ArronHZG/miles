# 09 工具链与基础设施

## 1. 关键 utils 子系统

```mermaid
graph LR
  subgraph Utils["miles/utils/"]
    Args["arguments.py<br/>巨型参数注册"]
    Dist["distributed_utils.py<br/>Gloo 组 / masked whiten"]
    Fp8["fp8_kernel.py<br/>Triton FP8 量化"]
    Replay["replay_base.py<br/>R3 路由重放缓冲"]
    Mask["mask_utils.py<br/>多轮 loss mask"]
    Seq["seqlen_balancing.py<br/>Karmarkar-Karp 负载均衡"]
    Health["health_monitor.py<br/>SGLang 引擎健康检查"]
    Mem["memory_utils.py<br/>GPU/Host 显存追踪"]
    TB["tensor_backper.py<br/>检查点 tensor 备份/恢复"]
    Timer["timer.py"]
    Track["tracking_utils/<br/>wandb/mlflow/tensorboard/prom"]
    Types["types.py<br/>RolloutBatch/Sample (Pydantic)"]
    Async["async_utils.py<br/>eager_create_task"]
  end
```

| 文件 | 作用 |
| :--- | :--- |
| `distributed_utils.py` | Gloo 组初始化、`distributed_masked_whiten`（跨 DP rank 优势归一化）、自定义进程组 |
| `replay_base.py` | `Replay` 循环缓冲（前向/反向索引分离）+ `BaseReplayManager`，R3 基础 |
| `seqlen_balancing.py` | Karmarkar-Karp 算法把序列均衡分配到各 worker，避免长尾 |
| `health_monitor.py` | `RolloutHealthMonitor` 线程化检查 SGLang 引擎健康 |
| `mask_utils.py` | `MultiTurnLossMaskGenerator`（Qwen chat template） |
| `tensor_backper.py` | `TensorBackuper`（CPU pinned memory 备份/恢复 tensor，用于 model switching） |
| `memory_utils.py` | GPU/host 显存追踪 + `torch.cuda.empty_cache()` 包装 |
| `types.py` | Pydantic 模型：`RolloutBatch`、`RolloutSample` 等 |
| `reloadable_process_group.py` | 可重载进程组（与 megatron_bridge shim 配合） |
| `metric_checker.py` / `metric_utils.py` | 指标校验与聚合 |

## 2. 权重/精度转换工具链

见 [06 低精度训练](./06-low-precision.md)。核心流向：

```mermaid
flowchart LR
  HF["HF safetensors BF16"] --> TD["convert_hf_to_torch_dist<br/>(训练加载)"]
  HF --> Low["convert_hf_to_fp8/mxfp8/nvfp4/int4<br/>(SGLang 推理)"]
  TD --> Train["训练"]
  Train -->|"导出"| Exp["convert_to_hf /<br/>convert_fsdp_to_hf /<br/>convert_torch_dist_to_hf"]
  Low --> SGLang["SGLang"]
```

辅助：`param_name_remap.py`（参数名重映射）、`cpu_memory_profiler.py`、`tools/visualize`（可视化）。

## 3. Docker 镜像

`docker/Dockerfile`（250+ 行），多架构（x86 / aarch64）：

```mermaid
flowchart LR
  Base["基础镜像<br/>lmsysorg/sglang:v0.5.13<br/>(SGLang + CUDA 13.2)"] --> Build["Build Args"]
  Build --> SG["SGLANG_BRANCH=sglang-miles<br/>(Miles 补丁分支)"]
  Build --> Mega["MEGATRON_REPO=radixark/Megatron-LM<br/>MEGATRON_BRANCH=miles-main"]
  Build --> Wheels["WHEELS_REPO=yueming-yuan/miles-wheels<br/>(预编译 wheel)"]
  Build --> CUDA["ENABLE_CUDA_13=1"]

  Wheels --> Arch{TARGETARCH}
  Arch -->|amd64| X86["WHEELS_TAG_X86<br/>cu130-x86_64-v0.5.12"]
  Arch -->|arm64| ARM["WHEELS_TAG_ARM64<br/>cu130-aarch64-v0.5.12"]

  X86 --> Final["最终镜像"]
  ARM --> Final
  Final --> APT["apt: nvtop, rsync, dnsutils,<br/>ethtool, nccl-tests"]
```

- 构建：`python docker/build.py --variant cu13 --image-tag dev --push`
- AMD 变体：`Dockerfile.rocm`（MI300X/MI325 ROCm 栈）
- NPU 支持：`npu_patch/`（Ascend）
- AMD 补丁：`amd_patch/`

## 4. CI 流水线

`docs/ci/` + `tests/ci/` + `.github/`：

```mermaid
flowchart LR
  PR["PR 提交"] --> Pre["pre-commit gate<br/>禁用 deprecated huggingface-cli<br/>(commit 803fa50)"]
  Pre --> Fast["tests/fast<br/>快速单测"]
  Fast --> FastGpu["tests/fast-gpu<br/>GPU 单测"]
  FastGpu --> E2E["tests/e2e<br/>端到端训练"]
  E2E --> Stage["CI 多阶段<br/>docs/ci/00-stage"]
  Stage --> Docker["Docker 构建矩阵<br/>docs/ci/02-docker-build<br/>cu12/cu13 × x86/aarch64"]
  Docker --> Label["镜像标签<br/>docs/ci/01-label"]
```

测试目录结构：

```mermaid
graph LR
  Tests["tests/"] --> Fast["fast/<br/>纯 CPU 快速单测"]
  Tests --> FastGpu["fast-gpu/<br/>GPU 单测"]
  Tests --> E2E["e2e/<br/>端到端训练验证"]
  Tests --> CI["ci/<br/>CI 专用脚本/runner"]
  Tests --> Deep["deepseekv4/<br/>DeepSeek-V4 专用"]
  Tests --> Root["根目录专题测试<br/>test_chunked_gae /<br/>test_fused_experts_backward /<br/>test_attention_output_gate_tp /<br/>test_external_rollout ..."]
```

## 5. 启动脚本（scripts/）

`scripts/` 含 40+ 各模型启动脚本，按模型族组织：

```mermaid
graph LR
  Scripts["scripts/"] --> Qwen["run-qwen3-*.sh<br/>(4B/32B/235B-A22B/<br/>next-80B-A3B/3.5-27B...)"]
  Scripts --> GLM["run-glm4*.sh<br/>(9B/4.5-355B-A32B/4.7-flash)"]
  Scripts --> Kimi["run-kimi-k2-*.sh<br/>(Instruct/Thinking/k25)"]
  Scripts --> Nemotron["run-nemotron-3-*.sh<br/>(nano-4b/30b-a3b/super-120b)"]
  Scripts --> GPTOss["run-gpt-oss-20b-*.sh<br/>(bf16/fsdp)"]
  Scripts --> Deep["run-deepseek-r1.sh"]
  Scripts --> Moon["run-moonlight-16B-A3B.sh"]
  Scripts --> Mimo["run-mimo-7B-rl-eagle.sh<br/>(投机解码)"]
  Scripts --> AMD["amd/<br/>ROCm 专用"]
  Scripts --> Models["models/<br/>模型相关"]
```

## 6. 追踪与监控

```mermaid
flowchart LR
  Run["训练运行"] --> Track["tracking_utils<br/>init_tracking / finish_tracking"]
  Track --> Wandb["wandb"]
  Track --> MLflow["mlflow"]
  Track --> TB["tensorboard"]
  Track --> Prom["prometheus"]
  Run --> Log["logging_utils<br/>configure_logger"]
  Run --> Profile["profile_utils<br/>性能剖析"]
  Run --> Dump["train_dump_utils / dumper_utils<br/>训练/rollout 数据 dump"]
  Run --> Health["health_monitor<br/>引擎健康"]
```

## 7. 容错与恢复

```mermaid
flowchart TD
  Enable["--use-fault-tolerance"] --> HM["健康监控<br/>RolloutHealthMonitor"]
  HM --> Dead{引擎死亡?}
  Dead -->|是| Recover["引擎恢复<br/>从备份权重重载"]
  Dead -->|否| Continue["继续"]
  Recover --> Continue
  Enable --> Ckpt["检查点恢复<br/>从 save_interval 检查点续训"]
  Ckpt --> Async["异步安全<br/>避免训练/rollout 状态不一致"]
```

- `--use-resilient-compression`：容错下的压缩传输。
- 见 `docs/advanced/fault-tolerance.md`。

## 8. 调试工具

```mermaid
graph LR
  Debug["调试入口"] --> DTO["--debug-train-only<br/>仅训练"]
  Debug --> DRO["--debug-rollout-only<br/>仅 rollout"]
  Debug --> CWE["--check-weight-update-equal<br/>权重同步校验"]
  Debug --> CI["--ci-test<br/>CI 故障注入"]
  Debug --> Load["--load-debug-rollout-data<br/>加载预存 rollout 数据"]
  Debug --> Env["utils/env_report.py<br/>环境报告"]
  Debug --> DU["utils/debug_utils/<br/>调试工具集"]
```
