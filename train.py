"""训练脚本：把「MoE 是怎么一训练出来的」过程尽量可视化。

看什么（这是本项目的重点，比最终效果重要）：
  1. 每层每个专家的利用率条 —— 是否均匀？有没有专家长期吃不到 token（塌缩）？
  2. aux（负载均衡损失）—— 均匀时约等于 1.0，越大越不均衡。
  3. z（路由 z-loss）—— 有没有爆炸（路由 logits 失控的早期信号）。
  4. 路由熵 —— 越接近 log(E) 越均匀；掉下去说明 router 开始强势做选择。
  5. routing_history.jsonl —— 训练全程的路由快照，用 analyze.py 可以画出演化曲线。
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import get_preset
from model import MoELM, estimate_flops_per_token

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BLOCKS = " ▁▂▃▄▅▆▇█"


# ------------------------------------------------------------------ 工具
def pick_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(name, device):
    table = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if name != "auto":
        return table[name]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def lr_at(step, base_lr, warmup, max_steps, min_ratio):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def fmt_tokens(n):
    return f"{n/1e6:.1f}M" if n < 1e9 else f"{n/1e9:.2f}B"


def spark(vals, uniform):
    """把每个专家的利用率变成 8 档字符条，一眼看出是否均匀。"""
    hi = max(max(vals), uniform * 2, 1e-9)
    out = []
    for v in vals:
        lvl = int(round(v / hi * (len(BLOCKS) - 1)))
        out.append(BLOCKS[max(0, min(len(BLOCKS) - 1, lvl))])
    return "".join(out)


class BinData:
    """读取 data_prep.py 产出的 .bin（uint16 的 token 流）。"""

    def __init__(self, data_dir, ctx_len):
        self.ctx = ctx_len
        train = np.memmap(Path(data_dir) / "train.bin", dtype=np.uint16, mode="r")
        val = np.memmap(Path(data_dir) / "val.bin", dtype=np.uint16, mode="r")
        self.train = self._pad(train)
        self.val = self._pad(val if len(val) >= ctx_len + 2 else train)
        print(f"[data] 训练 tokens={len(self.train):,} 验证 tokens={len(self.val):,} ctx={ctx_len}")
        if len(self.val) < 100:
            print("[data] 警告：验证集太小（语料本身太小），val_ce 只作管道验证用，别当效果指标")

    def _pad(self, arr):
        """语料太短就循环拼接，保证至少能切出一个完整窗口（冒烟测试用）。"""
        need = self.ctx + 2
        if len(arr) >= need or len(arr) == 0:
            return arr
        reps = int(np.ceil(need / len(arr)))
        return np.tile(np.asarray(arr), reps)

    def get_batch(self, split, batch_size, device):
        d = self.train if split == "train" else self.val
        hi = len(d) - self.ctx - 1
        if hi <= 1:  # 语料太小就整段复用，方便冒烟
            ix = np.zeros(batch_size, dtype=np.int64)
        else:
            ix = np.random.randint(0, hi, size=batch_size)
        x = np.stack([d[i:i + self.ctx] for i in ix]).astype(np.int64)
        y = np.stack([d[i + 1:i + 1 + self.ctx] for i in ix]).astype(np.int64)
        return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))


def encode_prompt(text, data_dir):
    meta = json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))
    if meta["tokenizer"] == "char":
        stoi = {c: i for i, c in enumerate(meta["itos"])}
        ids = [stoi.get(c, 0) for c in text]
    else:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(Path(data_dir) / "tokenizer.json"))
        ids = tk.encode(text).ids
    return torch.tensor([ids], dtype=torch.long), meta


def decode(ids, data_dir):
    meta = json.loads((Path(data_dir) / "meta.json").read_text(encoding="utf-8"))
    if meta["tokenizer"] == "char":
        return "".join(meta["itos"][i] for i in ids)
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(data_dir) / "tokenizer.json")).decode(ids)


def upload_gcs(local_path, prefix):
    """把 checkpoint 传到 GCS（可选）。Colab/Kaggle 上装好 google-cloud-storage 即可。"""
    if not prefix:
        return
    try:
        from google.cloud import storage
        assert prefix.startswith("gs://")
        _, _, rest = prefix.partition("gs://")
        bucket_name, _, blob_prefix = rest.partition("/")
        cli = storage.Client()
        blob = cli.bucket(bucket_name).blob(f"{blob_prefix}/{Path(local_path).name}")
        blob.upload_from_filename(str(local_path))
        print(f"[gcs] 已上传 {local_path} -> {prefix}")
    except Exception as e:  # 上传失败绝不能影响训练
        print(f"[gcs] 跳过上传：{type(e).__name__}: {e}")


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["L0", "L1", "smoke"], default="L0")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dense", action="store_true", help="对照组：等激活参数的稠密模型")
    ap.add_argument("--n_experts", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--n_shared", type=int, default=-1)
    ap.add_argument("--expert_hidden", type=int, default=0)
    ap.add_argument("--aux_coef", type=float, default=-1.0)
    ap.add_argument("--z_coef", type=float, default=-1.0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=-1.0, help="覆盖学习率（实验 E8）")
    ap.add_argument("--warmup_steps", type=int, default=0, help="覆盖 warmup 步数")
    ap.add_argument("--gcs_prefix", default="")
    args = ap.parse_args()

    mcfg, tcfg = get_preset(args.preset)
    # 命令行覆盖（方便做消融实验）
    if args.dense:
        mcfg.use_moe = False
    if args.n_experts:
        mcfg.n_experts = args.n_experts
    if args.top_k:
        mcfg.top_k = args.top_k
    if args.n_shared >= 0:
        mcfg.n_shared = args.n_shared
    if args.expert_hidden:
        mcfg.expert_hidden = args.expert_hidden
    if args.aux_coef >= 0:
        mcfg.aux_coef = args.aux_coef
    if args.z_coef >= 0:
        mcfg.z_coef = args.z_coef
    if args.max_steps:
        tcfg.max_steps = args.max_steps
    if args.dtype != "auto":
        tcfg.dtype = args.dtype
    if args.seed:
        tcfg.seed = args.seed
    if args.lr > 0:
        tcfg.lr = args.lr
    if args.warmup_steps:
        tcfg.warmup_steps = args.warmup_steps

    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    device = pick_device(args.device)
    dtype = pick_dtype(tcfg.dtype, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = BinData(args.data_dir, mcfg.ctx_len)
    meta = json.loads((Path(args.data_dir) / "meta.json").read_text(encoding="utf-8"))
    if meta["vocab_size"] > mcfg.vocab_size:
        raise SystemExit(f"[错误] 数据词表 {meta['vocab_size']} 大于模型词表 {mcfg.vocab_size}，"
                         f"请调小 data_prep 的 --vocab_size，或换更大的 preset")
    if meta["vocab_size"] < mcfg.vocab_size:
        print(f"[data] 数据词表 {meta['vocab_size']} < 预设词表 {mcfg.vocab_size}，"
              f"模型按数据词表建（预设里多出来的 embedding 行会闲置）")
        mcfg.vocab_size = meta["vocab_size"]
    model = MoELM(mcfg).to(device)
    total, active = model.param_report()
    tokens_per_step = tcfg.batch_size * tcfg.grad_accum * mcfg.ctx_len
    print(f"[model] 模式={'MoE' if mcfg.use_moe else '稠密对照'} device={device} dtype={dtype}")
    print(f"[model] 总参数={total/1e6:.2f}M  激活参数={active/1e6:.2f}M  "
          f"激活率={active/total:.1%}  每步 tokens={tokens_per_step:,}")
    if mcfg.use_moe:
        print(f"[model] 专家 E={mcfg.n_experts} top-{mcfg.top_k} 共享={mcfg.n_shared} "
              f"专家中间维={mcfg.expert_hidden} aux_coef={mcfg.aux_coef} z_coef={mcfg.z_coef}")
        print(f"[预算] 计划训 {fmt_tokens(tcfg.max_steps * tokens_per_step)} tokens，"
              f"约 {6 * active * tcfg.max_steps * tokens_per_step / 1e18:.2f}e18 FLOPs")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}], lr=tcfg.lr, betas=(0.9, 0.95)
    )
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"[resume] 从 {args.resume} 恢复，step={step}")

    csv_path = out_dir / "log.csv"
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_path.stat().st_size == 0:
        csv_writer.writerow(["step", "ce", "ppl", "aux", "z", "lr", "tok_per_s", "util_min", "util_max"])

    hist_path = out_dir / "routing_history.jsonl"
    hist_file = open(hist_path, "a", encoding="utf-8")

    def snapshot(s):
        if not mcfg.use_moe:
            return
        utils = [st["util"].tolist() for st in s["stats"]]
        ents = [float(st["entropy"]) for st in s["stats"]]
        hist_file.write(json.dumps({"step": s["step"], "util": utils, "entropy": ents}) + "\n")
        hist_file.flush()

    def save_ckpt():
        path = out_dir / "latest.pt"
        torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                    "step": step, "cfg": mcfg.__dict__, "train": tcfg.__dict__}, path)
        upload_gcs(path, args.gcs_prefix)
        if tcfg.save_interval and step % (tcfg.save_interval * 5) == 0:
            keep = sorted(out_dir.glob("ckpt_*.pt"))
            for old in keep[:-2]:
                old.unlink()
            ck = out_dir / f"ckpt_{step}.pt"
            torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(), "step": step}, ck)
        return path

    def evaluate():
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(tcfg.eval_iters):
                x, y = data.get_batch("val", tcfg.batch_size, device)
                with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                    _, loss, _, _, _ = model(x, y)
                losses.append(loss.item())
        model.train()
        return float(np.mean(losses))

    model.train()
    t_start = time.time()
    t_last = t_start
    ce_running, aux_running, z_running = [], [], []
    last_stats = None

    while step < tcfg.max_steps:
        lr = lr_at(step, tcfg.lr, tcfg.warmup_steps, tcfg.max_steps, tcfg.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        ce_acc, aux_acc, z_acc = 0.0, 0.0, 0.0
        # aux / z 是逐层求和后汇总的，这里除以层数换算成"每层平均"，
        # 这样仪表盘上的 aux 才能用"均匀=1.0"这把尺子量（README 第 4 节）。
        n_moe = mcfg.n_layer if mcfg.use_moe else 1
        for _ in range(tcfg.grad_accum):
            x, y = data.get_batch("train", tcfg.batch_size, device)
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                _, ce, aux, z, stats = model(x, y)
                loss = ce + mcfg.aux_coef * aux / n_moe + mcfg.z_coef * z / n_moe
            scaler.scale(loss / tcfg.grad_accum).backward()
            ce_acc += ce.item()
            aux_acc += (float(aux.detach()) / n_moe) if torch.is_tensor(aux) else aux
            z_acc += (float(z.detach()) / n_moe) if torch.is_tensor(z) else z
            if stats:
                last_stats = {"step": step, "stats": stats}
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        ce_mean = ce_acc / tcfg.grad_accum
        ce_running.append(ce_mean)
        aux_running.append(aux_acc / tcfg.grad_accum)
        z_running.append(z_acc / tcfg.grad_accum)
        step += 1

        if step % tcfg.log_interval == 0:
            now = time.time()
            tps = tcfg.log_interval * tokens_per_step / max(now - t_last, 1e-6)
            t_last = now
            aux_m = float(np.mean(aux_running))
            z_m = float(np.mean(z_running))
            util_min = util_max = float("nan")
            line = f"step {step:>6}/{tcfg.max_steps} | ce {float(np.mean(ce_running)):.4f} " \
                   f"| ppl {math.exp(min(20, float(np.mean(ce_running)))):.1f} | lr {lr:.2e} " \
                   f"| {tps/1000:.1f}k tok/s"
            if mcfg.use_moe and last_stats is not None:
                utils = [st["util"] for st in last_stats["stats"]]
                util_min = min(float(u.min()) for u in utils)
                util_max = max(float(u.max()) for u in utils)
                line += f" | aux {aux_m:.3f} z {z_m:.2f}（每层均值） | 利用率 min {util_min:.3f} max {util_max:.3f}"
            print(line)
            for i, st in enumerate(last_stats["stats"] if (mcfg.use_moe and last_stats) else []):
                u = st["util"].tolist()
                ent = float(st["entropy"])
                flag = "  <== 有专家饿死" if min(u) < 0.25 / mcfg.n_experts else ""
                print(f"    层{i} 专家负载 {spark(u, 1.0/mcfg.n_experts)}  "
                      f"(均匀={1/mcfg.n_experts:.3f}) 路由熵={ent:.3f}/{math.log(mcfg.n_experts):.3f}{flag}")
            csv_writer.writerow([step, f"{float(np.mean(ce_running)):.4f}", f"{math.exp(min(20, float(np.mean(ce_running)))):.2f}",
                                 f"{aux_m:.4f}", f"{z_m:.4f}", f"{lr:.2e}", f"{tps:.0f}",
                                 f"{util_min:.4f}", f"{util_max:.4f}"])
            csv_file.flush()
            ce_running, aux_running, z_running = [], [], []
            if mcfg.use_moe:
                snapshot(last_stats)

        if step % tcfg.eval_interval == 0:
            val_ce = evaluate()
            print(f"     [验证] step {step} val_ce {val_ce:.4f} val_ppl {math.exp(min(20, val_ce)):.2f}")

        if step % tcfg.save_interval == 0:
            p = save_ckpt()
            print(f"     [存档] {p}")

        if step % tcfg.sample_interval == 0:
            prompt, _ = encode_prompt("人工智能", args.data_dir)
            prompt = prompt.to(device)
            with torch.no_grad():
                gen = model.generate(prompt, 40, temperature=0.8, top_k=40)[0].tolist()
            print(f"     [生成] {decode(gen, args.data_dir).replace(chr(10), ' ')}")

    p = save_ckpt()
    val_ce = evaluate()
    dt = time.time() - t_start
    print("=" * 72)
    print(f"[完成] step={step} 训练 tokens={step*tokens_per_step/1e6:.1f}M 用时 {dt/60:.1f} 分钟")
    print(f"[完成] val_ce={val_ce:.4f} val_ppl={math.exp(min(20, val_ce)):.2f}")
    print(f"[完成] 总参数={total/1e6:.2f}M 激活参数={active/1e6:.2f}M "
          f"（{6*active*step*tokens_per_step/1e18:.2f}e18 FLOPs）")
    print(f"[存档] {p}  日志: {csv_path}  路由历史: {hist_path}")
    csv_file.close()
    hist_file.close()


if __name__ == "__main__":
    main()
