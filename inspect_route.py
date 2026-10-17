"""逐 token 路由追踪：看清楚"这一个字，在每一层被交给了哪个专家"。

学习价值：散点统计只能说明"大体均匀"，逐 token 追踪能让你看到分工的形状——
  比如标点/虚词是否总找同一批专家，实词是否找另一批；同一层里不同 token 的分歧有多大。

用法：
  python inspect.py --ckpt out_L0/latest.pt --data_dir data/demo --text "人工智能是计算机科学的分支"
  python inspect.py --ckpt out_L0/latest.pt --data_dir data/demo --mode chars
"""
import argparse
from collections import Counter
from pathlib import Path

import torch

from analyze import capture, load_model
from train import decode, encode_prompt

DEFAULT_TEXT = "从前有一个老木匠，他住在一座小镇上。人工智能是计算机科学的一个分支，它研究如何让机器完成需要智能的任务。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--mode", choices=["trace", "chars", "all"], default="all")
    ap.add_argument("--max_tokens", type=int, default=36)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device != "auto" else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, step = load_model(args.ckpt, device)

    if not cfg.use_moe:
        print("这个 checkpoint 是稠密对照组（--dense），没有路由可看。换一个 MoE 的 ckpt。")
        return

    ids, _ = encode_prompt(args.text, args.data_dir)
    ids = ids[:, :cfg.ctx_len]
    recs = capture(model, ids.to(device))
    n_layer = len(recs)
    toks = [decode([int(i)], args.data_dir) for i in ids[0].tolist()]
    toks = [t if t.strip() else "<空>" if t == " " else repr(t) for t in toks]

    print(f"[模型] step={step} E={cfg.n_experts} top-{cfg.top_k} 共享={cfg.n_shared} 层数={n_layer}")
    print(f"[输入] {args.text}")

    if args.mode in ("trace", "all"):
        print("\n=== 逐 token 路由表（每个格子 = 该层为该 token 选中的专家编号）===")
        head = "     " + "".join(f"层{k:<5}" for k in range(n_layer))
        print(f"  字符 {head}")
        for t in range(min(len(toks), args.max_tokens)):
            row = ""
            for rec in recs:
                e = ",".join(str(int(x)) for x in rec["top_idx"][t].tolist())
                row += f"{e:<6}"
            print(f"  {toks[t]:<4} {row}")
        # 最后一个 token 的权重明细：权重越大，说明 router 越"确定"
        last = len(toks) - 1
        print(f"\n  最后一个 token「{toks[last]}」的专家权重（softmax 后重新归一化）：")
        for k, rec in enumerate(recs):
            pairs = ", ".join(f"专家{int(i)}={float(p)*100:.0f}%" for i, p in zip(rec["top_idx"][last].tolist(), rec["top_p"][last].tolist()))
            print(f"    层{k}: {pairs}")

    if args.mode in ("chars", "all"):
        print("\n=== 每个专家最常吃掉的字符（看出专家在负责什么字形/词类）===")
        for k, rec in enumerate(recs):
            counts = {e: Counter() for e in range(cfg.n_experts)}
            idx = rec["top_idx"]                       # [T, k]
            for t in range(idx.shape[0]):
                for e in idx[t].tolist():
                    counts[e][toks[t]] += 1
            parts = []
            for e in range(cfg.n_experts):
                top = counts[e].most_common(4)
                s = "".join(f"{c}×{n} " for c, n in top) or "（没分到 token）"
                parts.append(f"    专家{e}: {s.strip()}")
            print(f"  层{k}")
            print("\n".join(parts))
        print("\n  提示：如果某专家集中在标点、数字或某个语气词上，说明它学到了浅层的字形/位置模式；")
        print("       深层专家更可能集中在语义相近的词上（可在 --text 里换成代码、对话再对比）。")


if __name__ == "__main__":
    main()
