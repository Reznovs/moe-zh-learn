"""模型：一个手写的最小 MoE 语言模型（decoder-only）。

学习要点（直觉 -> 机制 -> 细节）：
  直觉：MoE 就是把 Transformer 里那个前馈层（FFN）复制成 E 份，每个 token 只走其中 top-k 份。
  机制：路由器（一个小线性层）给每个 token 打 E 个分，softmax 后取 top-k，token 的输出 = Σ 选中专家输出 × 权重。
  细节：训练时除了语言建模损失，还要加"负载均衡损失"（别让所有 token 挤同一个专家）和"z-loss"（别让路由 logits 越长越大）。
        生产环境用专家并行 + 全对全（all-to-all）通信；这里为了看懂，直接用一个 Python 循环逐专家收集 token，
        数学上完全等价，只是慢。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- 前馈层
class DenseFFN(nn.Module):
    """常规稠密 FFN，作为 MoE 的对照组。

    注意：对照组会把 hidden 放大到 (top_k + n_shared) * expert_hidden，
    这样它的"激活参数量 / 每 token 算力"和 MoE 完全一致，才是公平对照。
    """

    def __init__(self, d_model, hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Expert(nn.Module):
    """一个专家 = 一个小 FFN。细粒度专家用 2×d_model 的中间维度。"""

    def __init__(self, d_model, hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoEFFN(nn.Module):
    """MoE 前馈层：路由器 + E 个路由专家 + n_shared 个共享专家。

    返回 (输出, aux_loss, z_loss, 统计量)。
    统计量里最重要的是 per-expert 利用率 f：它一直接近 1/E 说明路由健康，
    某个专家长期接近 0 就是"路由塌缩"，是 MoE 训练最经典的失败模式。
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.aux_coef = cfg.aux_coef
        self.z_coef = cfg.z_coef
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)  # 小初始化 -> 训练初期近似均匀路由
        self.experts = nn.ModuleList(
            Expert(cfg.d_model, cfg.expert_hidden) for _ in range(cfg.n_experts)
        )
        self.shared = nn.ModuleList(
            Expert(cfg.d_model, cfg.expert_hidden) for _ in range(cfg.n_shared)
        )

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)                     # [N, D]，N = B*T
        N = xf.shape[0]

        logits = self.gate(xf).float()             # [N, E]，路由打分用 fp32，防 fp16 溢出
        probs = F.softmax(logits, dim=-1)          # [N, E]
        top_p, top_idx = probs.topk(self.top_k, dim=-1)          # [N, k]
        top_p = top_p / top_p.sum(-1, keepdim=True).clamp_min(1e-9)  # 选中权重归一化

        # ---- 负载均衡损失（Switch Transformer 版）----
        # f_i：专家 i 实际拿到的 token 份额（硬计数 / (N*k)），P_i：专家 i 的平均路由概率（软）
        onehot = F.one_hot(top_idx, self.n_experts).float()      # [N, k, E]
        f = onehot.sum(dim=1).sum(dim=0) / (N * self.top_k)      # [E]，均匀时全为 1/E
        P = probs.mean(dim=0)                                    # [E]
        aux = (self.n_experts * (f * P).sum())                   # 均匀时 = 1.0，越大越不均衡
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()         # z-loss：压住 logits 幅度

        # ---- 分发：逐专家收集 token 算完再散回（等价于 all-to-all，但可读）----
        # 丢掉梯度用的统计量随后 detach
        y = torch.zeros_like(xf)
        for e in range(self.n_experts):
            mask = top_idx == e                                  # [N, k]
            tok_idx, slot = mask.nonzero(as_tuple=True)          # 哪些 token 的第几个槽位选中了 e
            if tok_idx.numel() == 0:
                continue
            w = top_p[tok_idx, slot].unsqueeze(-1)               # [M, 1]
            y.index_add_(0, tok_idx, self.experts[e](xf[tok_idx]) * w)
        for s in self.shared:                                    # 共享专家：所有 token 都过
            y = y + s(xf)

        with torch.no_grad():
            ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
            stats = {
                "util": f.detach(),          # [E] 实际负载分布
                "prob": P.detach(),          # [E] 平均路由概率分布
                "entropy": ent,              # 平均路由熵（≈logE 表示接近均匀）
            }
        return y.reshape(B, T, D), aux, z, stats


# --------------------------------------------------------------- 骨架
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        shape = (B, T, self.n_head, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.drop(self.proj(y))


class Block(nn.Module):
    """pre-norm 结构：注意力 + 前馈（MoE 或稠密），和标准 Transformer 一致。"""

    def __init__(self, cfg, dense_hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = MoEFFN(cfg) if cfg.use_moe else DenseFFN(cfg.d_model, dense_hidden)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if isinstance(self.ffn, MoEFFN):
            out, aux, z, stats = self.ffn(self.ln2(x))
            return x + out, aux, z, stats
        return x + self.ffn(self.ln2(x)), None, None, None


class MoELM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.d_model)
        dense_hidden = (cfg.top_k + cfg.n_shared) * cfg.expert_hidden
        self.blocks = nn.ModuleList(Block(cfg, dense_hidden) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        aux_total, z_total = 0.0, 0.0
        layer_stats = []
        for blk in self.blocks:
            x, aux, z, stats = blk(x)
            if aux is not None:
                aux_total = aux_total + aux
                z_total = z_total + z
                layer_stats.append(stats)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss, aux_total, z_total, layer_stats

    # -------- 统计信息 --------
    @torch.no_grad()
    def param_report(self):
        """总参数 / 激活参数。训练显存看总参数，训练算力看激活参数。"""
        total = sum(p.numel() for p in self.parameters())
        active = total
        if self.cfg.use_moe:
            per_expert = sum(p.numel() for p in self.blocks[0].ffn.experts[0].parameters())
            active = total - (self.cfg.n_experts - self.cfg.top_k) * per_expert * self.cfg.n_layer
        return total, active

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.ctx_len:]
            logits, *_ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v = torch.topk(logits, min(top_k, logits.size(-1))).values
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def estimate_flops_per_token(active_params, ctx_len=None):
    """训练算力经验公式：每 token ≈ 6 × 激活参数量（前向 2 + 反向 4）。"""
    return 6 * active_params
