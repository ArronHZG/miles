# 07 RL 算法与损失

算法实现在 `miles/backends/training_utils/`，核心是优势估计 + 损失分发 + 离策略校正。

## 1. 算法分发总览

```mermaid
flowchart TD
  Data["rollout 训练数据<br/>(log_probs, ref_log_probs, rewards, values, masks)"] --> Adv["compute_advantages_and_returns<br/>(loss.py)"]

  Adv --> KL["KL 散度计算<br/>(math_utils: forward/reverse/symmetric)"]
  Adv --> Est{advantage_estimator}

  Est -->|grpo| GRPO["GRPO<br/>组内相对优势"]
  Est -->|gspo| GSPO["GSPO<br/>序列级策略优化"]
  Est -->|ppo| PPO["PPO<br/>clip + value"]
  Est -->|reinforce_plus_plus| RPP["REINFORCE++<br/>+ baseline & value"]

  GRPO --> Loss["loss_function 分发"]
  GSPO --> Loss
  PPO --> Loss
  RPP --> Loss

  Loss --> Corr{离策略校正?}
  Corr -->|TIS| TIS["Truncated Importance Sampling"]
  Corr -->|MIS| MIS["Masked Importance Sampling"]
  Corr -->|否| Plain["标准 on-policy"]
  TIS --> Out["训练 loss"]
  MIS --> Out
  Plain --> Out
```

## 2. loss_hub 模块结构

```mermaid
graph LR
  subgraph Hub["backends/training_utils/loss_hub/"]
    L["losses.py<br/>GRPO / GSPO / PPO /<br/>REINFORCE++ / REINFORCE++_Baseline"]
    A["advantages.py<br/>各估计器优势计算"]
    M["math_utils.py<br/>KL: forward/reverse/symmetric"]
    LP["logit_processors.py<br/>提取 log_probs + entropy"]
    OPD["opd.py<br/>On-Policy Distillation KL 惩罚"]
    C["corrections.py<br/>TIS / MIS 校正项"]
  end

  L --> A
  L --> M
  L --> LP
  L --> C
  OPD -.->|"正交于优势估计器"| L
```

## 3. 优势估计器决策树

```mermaid
flowchart TD
  Start["选择 advantage_estimator"] --> Q1{有 critic / value?}
  Q1 -->|是| Q2{需要 clip 机制?}
  Q1 -->|否| Q3{组内可比较?<br/>(同一 prompt 多采样)}
  Q2 -->|是| PPO["ppo<br/>PPO clip + value loss"]
  Q2 -->|否, 用 baseline| RPP["reinforce_plus_plus<br/>REINFORCE + baseline & value"]
  Q3 -->|是, 组内相对| Q4{序列级 vs token 级?}
  Q3 -->|否| REINF["reinforce<br/>纯 REINFORCE"]
  Q4 -->|序列级| GSPO["gspo<br/>序列级策略优化<br/>更稳定"]
  Q4 -->|token 级| GRPO["grpo<br/>组内相对优势<br/>默认常用"]
```

> `AdvantageEstimator` 枚举定义在 `miles/utils/arguments.py`，包括 `grpo / gspo / ppo / reinforce_plus_plus / reinforce` 等。

## 4. GRPO vs GSPO vs PPO 对比

```mermaid
flowchart LR
  subgraph GRPO["GRPO"]
    G1["同一 prompt 采 G 条"]
    G1 --> G2["组内 reward 归一化<br/>得 advantage"]
    G2 --> G3["token 级策略梯度<br/>+ KL 惩罚"]
  end

  subgraph GSPO["GSPO"]
    S1["序列级 importance ratio"]
    S1 --> S2["整序列 clip<br/>而非逐 token"]
    S2 --> S3["更稳定<br/>适合长序列/MoE"]
  end

  subgraph PPO["PPO"]
    P1["critic 给 value"]
    P1 --> P2["GAE 优势"]
    P2 --> P3["token 级 clip<br/>+ value loss"]
  end
```

| 算法 | 需要 critic | 优势粒度 | clip 粒度 | 适用 |
| :--- | :--- | :--- | :--- | :--- |
| GRPO | 否 | 组内 token 级 | — | 通用默认，数学/推理 |
| GSPO | 否 | 序列级 | 序列级 | 长序列、MoE 稳定性 |
| PPO | 是 | GAE token 级 | token 级 | 有 value 函数场景 |
| REINFORCE++ | 可选 baseline | token 级 | — | 轻量基线 |

## 5. 离策略校正：TIS / MIS

当 train-inference mismatch 不可避免时（如低精度、非完全 on-policy），用算法校正防止发散：

```mermaid
flowchart LR
  Mismatch["train-inference mismatch<br/>→ importance ratio 偏差"] --> Risk["off-policy 偏差<br/>训练发散"]
  Risk --> Correct{校正方法}
  Correct -->|TIS| Trunc["Truncated Importance Sampling<br/>截断大 ratio<br/>抑制高方差样本"]
  Correct -->|MIS| Mask["Masked Importance Sampling<br/>mask 异常样本"]
  Trunc --> Safe["稳定训练"]
  Mask --> Safe
```

- 与 `examples/train_infer_mismatch_helper/` 对应，提供 TIS/MIS 的算法实现与示例。
- 与系统级方案（统一 FP8、R3、true_on_policy）互补：系统级消除 mismatch，算法级兜底。

## 6. On-Policy Distillation (OPD)

`miles/rollout/on_policy_distillation.py` — 用教师模型最小化 reverse-KL，无需显式任务奖励。

```mermaid
flowchart TD
  Rollout["rollout 采样<br/>student tokens"] --> TopK{opd-log-prob-top-k > 0?}

  TopK -->|是, Top-K 路径| TK["SGLang 返回 student top-logprobs<br/>存 sample.metadata"]
  TK --> Teacher1["查教师模型 top-k"]
  Teacher1 --> Strat["opd-top-k-strategy<br/>only-student / only-teacher /<br/>intersection / union / xor"]
  Strat --> Cross["学生交叉评估教师 top tokens"]
  Cross --> KL1["post_process_rewards<br/>逐位 reverse-KL<br/>归一化 → opd_reverse_kl"]
  KL1 --> Train1["训练: KL 惩罚"]

  TopK -->|否, 采样 token 路径| ST["存教师 log_probs<br/>sample.teacher_log_probs"]
  ST --> KL2["训练: student_logp - teacher_logp"]
  KL2 --> Reward0["reward = 0<br/>(信号全来自 KL)"]
  Reward0 --> Train2["训练: KL 惩罚"]

  Train1 --> Weight{权重模式}
  Train2 --> Weight
  Weight -->|student_p| WS["按学生概率加权<br/>(学生偏向)"]
  Weight -->|teacher_p| WT["按教师概率加权<br/>(教师保守)"]
  Weight -->|none| WU["均匀"]
```

- OPD **正交于优势估计器**：可叠加在 GRPO/GSPO/PPO 之上。
- `--opd-type`（megatron / rollout）选择 KL 在哪一侧计算。
- 见 `examples/on_policy_distillation/`。

## 7. 训练前向中的 log-prob 计算

```mermaid
flowchart LR
  Data["rollout 数据<br/>含采样 tokens"] --> Fwd["模型前向"]
  Fwd --> LP["logit_processors<br/>提取 log_probs + entropy"]
  LP --> Need{需要重算?}
  Need -->|recompute-logprobs-via-prefill| Prefill["通过 prefill 重算<br/>保持训练策略对齐"]
  Need -->|否| Direct["直接用采样 log_probs"]
  Prefill --> Compare["与 ref/teacher log_probs 比较"]
  Direct --> Compare
  Compare --> Adv["算优势 + KL"]
  Adv --> Loss["算 loss + 反向"]
```

## 8. 多轮损失掩码

`miles/utils/mask_utils.py` 的 `MultiTurnLossMaskGenerator` 按 chat template（如 Qwen）生成多轮 loss mask：仅对 assistant 回复 token 计 loss，工具调用/用户轮次 mask 掉。
