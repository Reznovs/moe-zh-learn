"""数据准备：中文语料 -> token id 二进制文件。

三种数据来源：
  --source builtin   内置小语料（含故事/百科/代码/数学/日常 4 种文体，仅用于冒烟测试）
  --source local     本地 txt（--path）
  --source hf        HuggingFace 流式数据集，例如维基百科中文

Tokenizer 两种：
  --tokenizer char   字符级（中文天然合适，词表 5k 左右，L0 用这个）
  --tokenizer bpe    BPE（词表 16k，L1 用这个，需要 pip install tokenizers）
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------- 内置小语料
BUILTIN_ZH = """从前有一座山，山脚下住着一位老木匠。他每天天不亮就起身，磨好刨子，走到院子里。
老木匠有一个徒弟，年纪很小，总是问不完的问题。「师父，木头为什么会浮在水上？」
老木匠笑着说：「因为木头里藏着小气泡，气泡比水轻，就托着木头往上走。」
徒弟把木块按进水里，松手，木块果然跳出来，溅了他一脸水花。他笑得直不起腰。

人工智能是一门研究如何让机器表现出智能行为的学科。它包含感知、推理、学习和决策等方向。
语言模型通过阅读大量文本，学习词语之间的统计规律，从而预测下一个词。
混合专家模型是一种稀疏激活的神经网络结构。它的前馈层被复制成许多份，每一份称为一个专家。
对于每一个输入词元，路由器只挑选其中少数几个专家参与计算，其余的专家保持沉默。
这样做的结果是：模型的总参数量可以很大，而每个词元实际消耗的算力却很小。
训练混合专家模型的关键难点在于路由器的负载均衡，如果所有词元都涌向同一个专家，其余专家就永远学不到东西。

def quick_sort(items):
    if len(items) <= 1:
        return items
    pivot = items[len(items) // 2]
    left = [x for x in items if x < pivot]
    mid = [x for x in items if x == pivot]
    right = [x for x in items if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)

设一个玩具混合专家模型有 E 个专家，每个词元激活其中 k 个，那么激活率等于 k 除以 E。
若总参数量为 N，则每个词元的前向算力近似正比于激活参数量，而不是总参数量。
例如 E 等于 8、k 等于 2、再额外添加一个共享专家，此时激活率约等于百分之三十三。
一块显卡能装下多少参数由总参数量决定，而训练速度由激活参数量决定，这两件事必须分开计算。

今天天气不错，你吃饭了吗？我等会儿去超市买点菜，顺便把快递取回来。
晚上要不要一起看电影？听说那部片子的评价挺高的，就是有点长，差不多三个小时。
算了，还是在家煮面吧，外面排队太久了。明天早上我八点叫你，别又睡过头。
"""


def load_text(args) -> str:
    if args.source == "builtin":
        return BUILTIN_ZH * args.repeat
    if args.source == "local":
        p = Path(args.path)
        if not p.exists():
            raise SystemExit(f"找不到文件：{p}")
        return p.read_text(encoding="utf-8", errors="ignore")
    if args.source == "hf":
        from datasets import load_dataset  # 延迟导入
        ds = load_dataset(args.hf_dataset, args.hf_config, split="train", streaming=True)
        chunks, total = [], 0
        limit = int(args.max_mb * 1_000_000)
        for ex in ds:
            t = ex.get("text") or ""
            if not t.strip():
                continue
            chunks.append(t)
            total += len(t)
            if total >= limit:
                break
        if not chunks:
            raise SystemExit("HF 数据集没有取到文本，检查 --hf_dataset / --hf_config")
        return "\n".join(chunks)
    raise SystemExit(f"未知 --source {args.source}")


def clean(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\ufeff", "")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


# ---------------------------------------------------------------- tokenizer
def build_char_tokenizer(text, vocab_size):
    """字符级：统计频次最高的字符，其余映射到 <unk>。中文里一个字≈一个 token。"""
    counter = Counter(text)
    keep = [c for c, _ in counter.most_common(vocab_size - 1)]
    itos = ["<unk>"] + keep
    stoi = {c: i for i, c in enumerate(itos)}
    return stoi, itos, lambda s: np.array([stoi.get(c, 0) for c in s], dtype=np.uint16)


def build_bpe_tokenizer(text, vocab_size, out_dir):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def it():
        # 分块喂给训练器，避免一次性把 200MB 全塞进内存
        step = 4_000_000
        for i in range(0, len(text), step):
            yield text[i:i + step]

    tok.train_from_iterator(it(), trainer=trainer)
    tok.save(str(Path(out_dir) / "tokenizer.json"))
    vocab = tok.get_vocab()
    itos = [None] * len(vocab)
    for k, v in vocab.items():
        itos[v] = k
    enc = lambda s: np.array(tok.encode(s).ids, dtype=np.uint16)
    return enc, itos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["builtin", "local", "hf"], default="builtin")
    ap.add_argument("--path", default="")
    ap.add_argument("--hf_dataset", default="wikimedia/wikipedia")
    ap.add_argument("--hf_config", default="20231101.zh")
    ap.add_argument("--max_mb", type=float, default=200.0)
    ap.add_argument("--repeat", type=int, default=1, help="builtin 语料重复次数")
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    ap.add_argument("--vocab_size", type=int, default=5376)
    ap.add_argument("--val_ratio", type=float, default=0.005)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    text = clean(load_text(args))
    print(f"[data] 原始字符数 = {len(text):,}")

    if args.tokenizer == "char":
        stoi, itos, enc = build_char_tokenizer(text, args.vocab_size)
    else:
        enc, itos = build_bpe_tokenizer(text, args.vocab_size, out)
        stoi = None

    ids = enc(text)
    n_val = max(1, int(len(ids) * args.val_ratio))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    train_ids.tofile(out / "train.bin")
    val_ids.tofile(out / "val.bin")
    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": len(itos),
        "itos": itos,
        "source": args.source,
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"[data] 词表 = {len(itos):,} | 训练 token = {len(train_ids):,} | 验证 token = {len(val_ids):,}")
    if len(train_ids) < 20_000:
        print("[data] 提示：语料偏小，模型很快会背下来（ce 直线掉到很低）。")
        print("       只想看路由行为可以接受；想看到正常的 ce 下降曲线，"
              "请加 --repeat 50，或换 --source hf 拉一份中文维基。")
    print(f"[data] 已写入 {out}/{{train.bin,val.bin,meta.json}}")


if __name__ == "__main__":
    main()
