"""配置集中管理：所有超参都在这里，方便做对照实验。

L0 = 冒烟版（几分钟~半小时，单卡 T4 甚至 CPU 都能跑）
L1 = 正式学习版（2~3 小时 T4，够看出专家分工）
smoke = 纯管道验证（1 分钟，合成数据）
"""
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    # --- 基础骨架（和稠密 Transformer 完全一样）---
    vocab_size: int = 5376
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    ctx_len: int = 256
    dropout: float = 0.0
    tie_weights: bool = True          # 词嵌入与输出层共享权重

    # --- MoE 部分（学习重点）---
    use_moe: bool = True              # False = 稠密对照组
    n_experts: int = 8                # 路由专家数量
    top_k: int = 2                    # 每个 token 激活几个专家
    expert_hidden: int = 512          # 单个专家的中间维度（细粒度：2×d_model）
    n_shared: int = 1                 # 共享专家数量（DeepSeekMoE 式，所有 token 都过）
    aux_coef: float = 0.01            # 负载均衡损失系数（Switch Transformer 的 0.01）
    z_coef: float = 1e-4              # 路由 z-loss 系数（防 logits 爆炸）


@dataclass
class TrainConfig:
    batch_size: int = 32              # 每步每个 GPU 的样本数
    grad_accum: int = 4               # 梯度累积步数
    max_steps: int = 2000
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 50
    log_interval: int = 20
    save_interval: int = 200
    sample_interval: int = 500
    dtype: str = "auto"               # auto | fp16 | bf16 | fp32
    device: str = "auto"              # auto | cuda | cpu
    seed: int = 1337


# ---------------- 预设 ----------------
# L0：10M 级小 MoE，目标是"跑通 + 看懂路由"
L0_MODEL = ModelConfig(
    vocab_size=5376, d_model=256, n_layer=4, n_head=4, ctx_len=256,
    n_experts=8, top_k=2, expert_hidden=512, n_shared=1,
)
L0_TRAIN = TrainConfig(batch_size=32, grad_accum=4, max_steps=2000)

# L1：50M 级，目标是"看出专家分工 + 和稠密对照组比较"
L1_MODEL = ModelConfig(
    vocab_size=16384, d_model=384, n_layer=6, n_head=6, ctx_len=512,
    n_experts=8, top_k=2, expert_hidden=768, n_shared=1,
)
L1_TRAIN = TrainConfig(batch_size=24, grad_accum=8, max_steps=8000, lr=6e-4)

# smoke：一分钟管道验证（CPU 也能跑）
SMOKE_MODEL = ModelConfig(
    vocab_size=512, d_model=64, n_layer=2, n_head=2, ctx_len=64,
    n_experts=4, top_k=2, expert_hidden=128, n_shared=1,
)
SMOKE_TRAIN = TrainConfig(
    batch_size=8, grad_accum=1, max_steps=30, eval_interval=15, eval_iters=5,
    log_interval=5, save_interval=30, sample_interval=30, warmup_steps=5, dtype="fp32",
)

PRESETS = {
    "L0": (L0_MODEL, L0_TRAIN),
    "L1": (L1_MODEL, L1_TRAIN),
    "smoke": (SMOKE_MODEL, SMOKE_TRAIN),
}


def get_preset(name: str):
    """返回 (ModelConfig, TrainConfig)，注意返回的是新实例，避免全局被改。"""
    m, t = PRESETS[name]
    return ModelConfig(**asdict(m)), TrainConfig(**asdict(t))
