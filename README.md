# MoE 中文小模型：从 0 到 1（学习项目）

**一句话**：在 10M 这个"看得清"的规模上，亲手把 MoE 训练的每个零件装起来，重点是看**路由怎么从随机分配变成有分工**——训练结果好不好无所谓，过程要看得见。

---

## 📚 不想看代码？先读 `docs/` 里的八篇讲义

| 文档 | 内容 |
|---|---|
| `docs/00-学习路线.md` | 五课路线图、每课产出、学习建议 |
| `docs/01-直觉篇-为什么有MoE.md` | MoE 改了哪里、两本账（总参数/激活参数）、什么时候反而不如稠密 |
| `docs/02-机制篇-路由怎么工作.md` | 路由的数学定义、L_aux 与 L_z 的构造、指标判读标准 |
| `docs/03-实操篇-第一次训练.md` | 4 条命令 + 屏幕上每一行怎么读（含 Colab/Kaggle 注意事项） |
| `docs/04-诊断篇-路由的四种病.md` | 塌缩/过度均匀/z 爆炸/不分化：症状 → 病因 → 处方（含实测病例） |
| `docs/05-实验手册-E1到E8.md` | 8 个对照实验：先写预测 → 跑命令 → 记录数字 → 得结论 |
| `docs/06-进阶篇-从10M到3B.md` | 免费算力能爬多高、稀疏上采样、三堵墙 |
| `docs/07-速查与术语.md` | 公式卡 / 术语中英对照 / 命令卡 / 踩坑清单 |

> 代码只是用来"跑实验"的，全部实验都能用文档里的命令行完成，不必读源码。

---

## 1. 先建立直觉（3 分钟，看不懂后面都是背单词）

- Transformer 一层 = **注意力** + **前馈 FFN**。参数大头在 FFN（约占 2/3）。
- **MoE 只改 FFN**：把它复制成 E 份（"专家"），每个 token 只走其中 top-k 份。
  - 8 专家 top-2 → 每个 token 只算 1/4 的 FFN ⇒ **总参数大、激活参数小**。
- **两个账本，必须分开算**：
  - 显存看**总参数**（8 个专家都得常驻显存，哪怕这步只用了 2 个）；
  - 算力看**激活参数**：`训练 FLOPs ≈ 6 × N_激活 × tokens`。
- **路由器**就是一个线性层：给每个 token 输出 E 个分数 → softmax → 取 top-k → 权重归一化 → 加权求和。
- 唯一必须打的补丁：**负载均衡损失**（别让所有 token 挤同一个专家）+ **z-loss**（别让路由分数越大越离谱）。

## 2. 文件地图

| 文件 | 作用 | 打开后先看什么 |
|---|---|---|
| `config.py` | 全部超参 + 三档预设（smoke / L0 / L1） | `ModelConfig` 里的 MoE 部分：`n_experts` `top_k` `expert_hidden` |
| `data_prep.py` | 造语料 → 训练 token 流 | `builtin` 语料（4 个文体，专门用来看分工） |
| `model.py` | 手写模型 | `MoEFFN.forward`：gate → top-k → 逐专家 index_add_（等价于 all-to-all，但可读） |
| `train.py` | 训练 + 实时路由仪表盘 | 每 `log_interval` 打出的利用率条 |
| `analyze.py` | 训练完做路由分析（均衡性/专业化/演化） | 三种模式：`util` `domain` `evolution` |
| `inspect_route.py` | 逐 token 追踪“这一层把哪个字交给了哪个专家” | `trace` 模式的路由表 |

## 3. 跑起来（三步）

```bash
pip install torch numpy                 # 只有用 BPE 分词才额外需要: pip install tokenizers

# ① 造一份 1 分钟就能用的小语料（内置 4 文体：故事/百科/代码/对话）
python data_prep.py --source builtin --tokenizer char --vocab_size 2048 --out_dir data/demo

# ② 先冒烟 30 步验证管道（CPU 也行），再正式跑
python train.py --preset smoke --data_dir data/demo --out_dir out_smoke
python train.py --preset L0    --data_dir data/demo --out_dir out_L0 --max_steps 2000

# ③ 训练完做分析
python analyze.py  --ckpt out_L0/latest.pt --data_dir data/demo --mode all
python inspect_route.py --ckpt out_L0/latest.pt --data_dir data/demo --text "人工智能是计算机科学的分支"
```

> 内置 demo 语料只有大约 950 字，先加 `--repeat 50` 让它到几万字，否则模型秒学会背课文（`ce` 一路掉到很低，那不是真本事）。
> 只想看路由行为的话，小语料完全够用。

想换真语料（中文维基，约 200MB 切片 → 约 3000 万汉字）：

```bash
python data_prep.py --source hf --hf_dataset wikimedia/wikipedia --hf_config 20231101.zh \
       --max_mb 200 --tokenizer bpe --vocab_size 8192 --out_dir data/wiki200mb
```

**Colab / Kaggle 注意**：离线环境要先把语料做成 Dataset 上传；12 小时会断线，续训用
`python train.py ... --resume out_L0/latest.pt`；Kaggle 的 checkpoint 要落在 `/kaggle/working`
（20GB 会自动保存）；想更稳就 `pip install google-cloud-storage` 后加 `--gcs_prefix gs://桶名/前缀`。

## 4. 屏幕上那些数字怎么读（本项目的核心教材）

| 看到什么 | 含义 | 该做什么 |
|---|---|---|
| `ce` / `ppl` | 交叉熵 / 困惑度，语言建模能力 | 本项目次要指标，只看趋势别追分数 |
| `aux` | 负载均衡损失，**每层平均后均匀时 ≈ 1.0** | > 1.3 说明明显偏科；继续涨就加大 `--aux_coef` |
| `z` | 路由 z-loss（每层平均），通常 2~8 缓慢变化 | 突然涨到几十 ⇒ 路由分数失控，加 `--z_coef` 或降 lr |
| 利用率条 `▅▅▂▅▅▅▅▅` | 8 个专家各自的 token 份额（均匀水位 = 1/E） | 某格掉到 0.25/E 以下 → 打印会标 `有专家饿死` |
| `熵 x/y` | 路由熵 / logE，越接近越均匀 | 从 logE 一路掉 = router 开始"挑专家"，属于正常分工，但掉太狠要点检 aux |
| 层间差异 | 浅层通常更均匀，深层更容易分工 | 若"最后一层最乱"，往往是深层开始学语义了 |

**每次跑完至少记录这三行**：最终 `val_ce`、`aux` 均值、每层利用率的最小/最大。写在纸上，实验阶梯要靠它对比。

## 5. 实验阶梯：一次只改一个变量

> 建议格式：**先写预测 → 再跑 → 记录观察 → 写一句结论**。这一步比训练本身值钱。

| 编号 | 命令（在 L0 基础上加） | 想看什么 |
|---|---|---|
| E1 | `--dense` | 等激活参数的稠密对照：参数量和算力是两笔账，val_ce 谁更好、tok/s 谁更快 |
| E2 | `--aux_coef 0` | **必做**：亲眼看到路由塌缩——某个专家吃掉几乎所有 token |
| E3 | `--top_k 1` | top-1 更容易塌缩、但每 token 更省算力 |
| E4 | `--n_experts 16 --expert_hidden 256` | 细粒度专家（多而小，激活参数不变）vs 粗粒度 |
| E5 | `--n_shared 0` | 共享专家承担"通用知识"后，路由专家是否更专注 |
| E6 | `--z_coef 0` | 看 z 值能不能飙起来，以及 ce 是否变差 |
| E7 | 改 `config.py` 里 gate 初始化 `std=0.02 → 0.5` | 初始路由越乱，训练需要多久才回归均匀 |
| E8 | `--source builtin --repeat 8`（单文体）vs 4 文体混 | 语料越单一，专业化越难看出来（`analyze.py --mode domain`） |

## 6. 术语对照表（撞到再看）

| 术语 | 大白话 | 项目里的位置 |
|---|---|---|
| token / 词表 | 模型眼里的最小单位（这里是一个汉字） | `data_prep.py` 产出 `meta.json` |
| 专家 Expert | 一个被复制的 FFN 分支 | `Expert` 类 |
| 路由器 Router / Gate | 决定 token 去哪个专家的小线性层 | `MoEFFN.gate` |
| top-k | 每个 token 只走分数最高的 k 个专家 | `cfg.top_k` |
| 共享专家 Shared Expert | 所有 token 都必过的那份 FFN（装通用知识） | `cfg.n_shared` |
| 负载均衡损失 aux | 防止"都挤一个专家"的罚款 | `MoEFFN.forward` 里的 `aux` |
| z-loss | 防止路由分数越来越大的罚款 | `MoEFFN.forward` 里的 `z` |
| 激活参数 / 稀疏率 | 每 token 真正参与运算的参数量 / 它占比 | `model.param_report()` |
| 路由塌缩 | 少数专家吃掉全部 token，其余白占显存 | 仪表盘的 `有专家饿死` |
| 专家并行 / all-to-all | 大规模时专家分到不同 GPU 上，token 需要跨卡通信 | 本项目用逐专家 `index_add_` 等价实现 |
| 稀疏上采样 upcycling | 拿训好的稠密模型复制 FFN 变成 MoE，再续训 | 路线图 L2，`--resume` 的进阶玩法 |

## 7. 之后往哪走

| 级别 | 规模 | 资源 | 学到什么 |
|---|---|---|---|
| **L0（本目录）** | 10.6M 总 / 4.4M 激活（字符级小词表时） | 一晚上，本地或 Colab | 管道 + 路由可视化 + 消融 |
| **L1**（`--preset L1`） | 60M 级 | Kaggle T4 几小时 | 真语料下专家分化；和 `--dense` 对照 |
| **L2 稀疏上采样** | 稠密 60M → 复制专家 | 半天 | 不用从零重训也能得到 MoE；这是产业界常用做法 |
| **L3 规模化** | 3B 总 / 0.8B 激活 / 20B tokens | Kaggle + Colab + GCS，约一个月 | 断点续训、禅让式实验管理、真实预算感 |

## 8. 已知坑

- **“激活率 = k/E” 只是一部分真相**：这个公式只算 FFN。词嵌入、注意力、共享专家每步都在参与，
  所以实测激活率（L0 预设约 41%）总比 k/E=25% 高。要看真实数字就跑 `python -c "import config;from model import MoELM;print(MoELM(config.get_preset('L0')[0]).param_report())"`。
- 小语料 + char 分词会让模型"看起来还行"（几个字就能猜下一个字）；换 BPE 后 `ce` 的绝对值**不可比**，只能同类比同类。
- 10M 级别 MoE 不会比同算力稠密更准（效率优势随规模扩大才显现，见 Krajewski et al. ICML 2024）——本项目练的是工程与路由直觉，不是刷分。
- T4 无 bf16：`dtype auto` 会退到 fp16，路由器 logits 已在 fp32 里算（`model.py` 里有注释），少动它。
- 冒烟测试请用 `--preset smoke`：真实 L0 配置词表 5376，`--vocab_size` 给小于它的值也没事（多余 embedding 行闲置）。
