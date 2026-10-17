"""路由分析：训练完之后，回答三个问题。

  1. 负载均衡吗？    —— 每个专家分到了多少 token（有没有专家饿死）
  2. 有专业化吗？    —— 不同文体（故事/百科/代码/对话）是不是走了不同的专家
  3. 怎么演化来的？  —— 从 routing_history.jsonl 看利用率是怎么从随机走向分工的

用法：
  python analyze.py --ckpt out_L0/latest.pt --data_dir data/tinystories --mode all
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import ModelConfig
from model import MoELM, MoEFFN
from train import decode, encode_prompt

BLOCKS = " ▁▂▃▄▅▆▇█"

# 四个文体的小样本：用来观察"专家会不会按文体分工"
DOMAINS = {
    "故事": "从前有一个老木匠，他住在一座小镇上。每天清晨他都背着工具出门，为邻居修补门窗。",
    "百科": "人工智能是计算机科学的一个分支，它研究如何让机器完成需要智能的任务，例如识别图像。",
    "代码": "def quicksort(data):\n    if len(data) <= 1:\n        return data\n    pivot = data[0]\n    return quicksort",
    "对话": "你昨天说的那件事我想了想，还是觉得不太合适，要不我们明天再约个时间当面聊聊。",
}


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    raw = ck.get("cfg") or {}
    fields = {k: v for k, v in raw.items() if k in ModelConfig.__dataclass_fields__}
    cfg = ModelConfig(**fields)
    model = MoELM(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, cfg, ck.get("step", -1)


def capture(model, x):
    """跑一次前向，抓出每层的路由概率与 top-k 专家编号。"""
    records = []
    handles = []
    for i, blk in enumerate(model.blocks):
        ffn = blk.ffn
        if not isinstance(ffn, MoEFFN):
            continue

        def hook(mod, inp, out, rec=records, layer=i):
            h = inp[0].reshape(-1, mod.gate.weight.shape[1])
            logits = mod.gate(h).float()
            probs = F.softmax(logits, dim=-1)
            top_p, top_idx = probs.topk(mod.top_k, dim=-1)
            rec.append({"layer": layer, "probs": probs.detach().cpu(),
                        "top_idx": top_idx.detach().cpu(), "top_p": top_p.detach().cpu()})

        handles.append(ffn.register_forward_hook(hook))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return records


def spark(vals, uniform):
    hi = max(max(vals), uniform * 2, 1e-9)
    return "".join(BLOCKS[max(0, min(len(BLOCKS) - 1, int(round(v / hi * (len(BLOCKS) - 1)))))] for v in vals)


def mode_util(model, cfg, data_dir, n_batches, batch_size, device):
    print("\n=== ① 负载均衡：各层专家的 token 份额（越接近 1/E 越均匀）===")
    data = np.memmap(Path(data_dir) / "val.bin", dtype=np.uint16, mode="r")
    if len(data) < cfg.ctx_len + 2:
        data = np.memmap(Path(data_dir) / "train.bin", dtype=np.uint16, mode="r")
    E = cfg.n_experts
    acc = {}
    for _ in range(n_batches):
        ix = np.random.randint(0, max(1, len(data) - cfg.ctx_len - 1), size=batch_size)
        x = torch.from_numpy(np.stack([data[i:i + cfg.ctx_len] for i in ix]).astype(np.int64)).to(device)
        for rec in capture(model, x):
            onehot = F.one_hot(rec["top_idx"], E).float()
            share = onehot.sum(dim=1).sum(dim=0) / (onehot.shape[0] * cfg.top_k)
            acc.setdefault(rec["layer"], []).append(share.numpy())
    uniform = 1.0 / E
    for layer in sorted(acc):
        u = np.mean(acc[layer], axis=0)
        ent = -(u * np.log(u + 1e-9)).sum()
        dead = [i for i, v in enumerate(u) if v < 0.25 * uniform]
        print(f"  层{layer} {spark(u, uniform)}  min {u.min():.3f} max {u.max():.3f} "
              f"熵 {ent:.3f}/{math.log(E):.3f}" + (f"  <== 饿死专家 {dead}" if dead else ""))
    print("  提示：熵接近 logE=%.3f 说明路由均匀；掉到 logE 的 80%% 以下（这里是 %.3f）说明 router 已经在挑专家了。"
          % (math.log(E), 0.8 * math.log(E)))


def mode_domain(model, cfg, data_dir, device):
    print("\n=== ② 专业化：不同文体是否走了不同专家（每格=该专家在该文体上的 token 份额 %）===")
    E = cfg.n_experts
    per_layer = {}
    for name, text in DOMAINS.items():
        ids, _ = encode_prompt(text.replace("\n", " "), data_dir)
        ids = ids[:, :cfg.ctx_len]
        if ids.shape[1] < 2:
            continue
        recs = capture(model, ids.to(device))
        for rec in recs:
            onehot = F.one_hot(rec["top_idx"], E).float()
            share = (onehot.sum(dim=1).sum(dim=0) / (onehot.shape[0] * cfg.top_k)).numpy()
            per_layer.setdefault(rec["layer"], {})[name] = share
    names = [n for n in DOMAINS if n in next(iter(per_layer.values()), {})]
    uniform = 1.0 / E
    for layer in sorted(per_layer):
        print(f"  层{layer}    " + "".join(f"{n:>8}" for n in names))
        for e in range(E):
            row = "".join(f"{per_layer[layer][n][e] * 100:>7.1f}%" for n in names)
            best = max(names, key=lambda n: per_layer[layer][n][e])
            share = per_layer[layer][best][e]
            mark = "  <== " + best if share > 1.6 * uniform else ""
            print(f"    专家{e} {row}{mark}")
    print("  提示：均匀基线每格应为 %.1f%%；某一格明显高过基线（>1.6 倍，已标出）=> 那个专家在负责那个文体。" % (uniform * 100))


def mode_evolution(out_dir, cfg):
    path = Path(out_dir) / "routing_history.jsonl"
    if not path.exists():
        print(f"\n=== ③ 路由演化：找不到 {path}（还没训练或者中途没存快照）===")
        return
    snaps = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not snaps:
        print("\n=== ③ 路由演化：快照为空 ===")
        return
    E = cfg.n_experts
    uniform = 1.0 / E
    print(f"\n=== ③ 路由演化：{len(snaps)} 个快照（step {snaps[0]['step']} -> {snaps[-1]['step']}）===")
    print("  竖轴=专家 0..E-1，横轴=训练进程，字符越高表示该专家当时分到的 token 越多")
    for layer in range(len(snaps[0]["util"])):
        print(f"  层{layer}  （均匀水位 = {uniform:.3f}）")
        for e in range(E):
            series = [s["util"][layer][e] for s in snaps]
            final = series[-1]
            print(f"    专家{e} {spark(series, uniform)}  末值 {final:.3f}")
        ents = [s["entropy"][layer] for s in snaps]
        print(f"    熵     {spark(ents, math.log(E))}  末值 {ents[-1]:.3f} (上限 {math.log(E):.3f})")
    print("  常见剧本：早期全部专家接近均匀 -> 中期某几个专家变高 -> 后期稳定分工；")
    print("           如果某个专家一路往下掉到 0，就是塌缩（aux 系数该加大，或 top-1 改 top-2）。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="", help="含 routing_history.jsonl 的目录，默认取 ckpt 所在目录")
    ap.add_argument("--mode", choices=["util", "domain", "evolution", "all"], default="all")
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device != "auto" else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, step = load_model(args.ckpt, device)
    total, active = model.param_report()
    print(f"[模型] step={step} E={cfg.n_experts} top-{cfg.top_k} 共享={cfg.n_shared} "
          f"总参数={total/1e6:.2f}M 激活={active/1e6:.2f}M")

    if args.mode in ("util", "all"):
        mode_util(model, cfg, args.data_dir, args.n_batches, args.batch_size, device)
    if args.mode in ("domain", "all"):
        mode_domain(model, cfg, args.data_dir, device)
    if args.mode in ("evolution", "all"):
        mode_evolution(args.out_dir or str(Path(args.ckpt).parent), cfg)


if __name__ == "__main__":
    main()
