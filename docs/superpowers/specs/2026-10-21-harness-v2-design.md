# 评测协议 v2：标定噪声、冻结尺子、强制复核

日期：2026-10-21
状态：设计，待审阅。本文确认前不写实现计划，也不改代码。
前身：`2026-10-19-search-claims-design.md`（下称 v1）。本文替换 v1 的第 4、5、7、8 节，保留其余。

## 1. 目的

v1 把「夜里留下的提交」和「可以称为成立的结论」分成了两件事。`autoresearch/flower14` 这一夜（2026-10-20 15:22 → 10-21 06:24，138 次训练，15 小时）证明这个区分成立，但三条路被走穿了：

1. `MIN_IMPROVEMENT` 交付值是 `None`，`claim_verdict` 的第 3 条因此让**任何** `claim` 都返回 `还判不了`。一夜没有产出任何已验证结论，不是因为 agent 偷懒，是因为这条路被留了个未填的洞。
2. `data_hash()` 走 `prepare.py` 的 `DATA_DIR`，哈希的是文本任务的 parquet 分片。flower14 的 `train.py` 读的是图片目录，两者无关。该 run 中途把验证集从 98 张改成 1498 张（`ac0a661`），148 行记录的 `data_hash` 一个字节都没变。尺子长在被测物身上。
3. 被 `judge.py` 机械强制的规则全部生效（工作区干净、同 commit 同 seed 不重复、父提交必须在名单里、seed 2 机制名一致）；只写在 `program.md` 里的规则全部失效（「不要总从最近的父本出发」被违反 97 次；「进入名单不是成立」被报告写成「纪录可复现」；三事实如实描述机制被 `517b245` 违反——它机制名叫「半精度」，diff 里同时改了 `IMAGE_SIZE = 128 → 160`）。在无人值守的 15 小时里，劝告的半衰期是零。

另有一条不是洞但是设计缺口：`compare_seed1` 是精确浮点比较，搜索阶段没有任何噪声带。1/1498 张图（`0.000668`）就能触发 `accept`。flower18:05 之后的 100 轮只净赚 2 张图，其中 97 轮对同一个父本全部判负，而判负行的错误数标准差是 12.7 张——搜索在自己的噪声地板下面跑了 12 个小时。

v2 要达到的四件事（验收标准）：

- **A** 一夜必须落地至少一个 judge 判定，而不只是一串提交。
- **B** `accept` 的门槛建立在实测噪声之上，拒绝噪声级别的「改进」。
- **C** harness 与任务解耦：换任务时数据身份、评测协议、时间预算都被正确捕获，协议中途变更会被拒绝或标记。
- **D** 父本轮换、机制名与 diff 的一致性、「进入名单 ≠ 成立」的措辞，由 `judge.py` 机械执行，而不是写在 `program.md` 里。

## 2. 保留的现有规则

- 一次训练的墙钟预算仍是固定值（v2 迁到 `task.py`，见第 5 节），默认仍是 300 秒。
- 一张 GPU，一次跑一个实验。
- 研究循环开始后，agent 只改 `train.py`。
- agent 不安装新依赖。
- 运行超过 10 分钟就杀掉（`TIMEOUT_SECONDS = 600`）。
- 实验在 `autoresearch/<tag>` 分支上进行。
- `judge.py` 不执行 `git commit`、`git reset` 或 `git checkout`。留下还是退回提交，由 agent 做。
- `results.tsv`、`parents.tsv`、`run.log` 不提交。
- 不设 t 检验的 p 值门槛。
- 不检测 agent 在读完种子之后再次手动调用 `manual_seed`。这是已知残留。

## 3. 不做什么

- 不引入 expctl、Praxist 或任何外部实验账本。
- 不让 `judge.py` 替 agent 决定下一个该试什么机制。四个验收目标里没有「指导搜索方向」这一条；`flower14` 的 `peak_vram` 只有 830 MB、容量方向几乎没被探索，这个问题 v2 **不**用机械约束解决。父本轮换（第 7.3 节）会间接拓宽探索面，更直接的办法留作后续议题。
- 不做 v1 → v2 的记录格式兼容读取。见第 13 节。
- 不检测 agent 通过纯结构性代码改动（而非常量赋值）引入的混淆。见第 7.2 节的明确盲区声明。

## 4. 文件与职责

| 文件 | 谁改 | 作用 |
|---|---|---|
| `task.py` | 开跑前写定，之后冻结 | **新增。** 数据根、训练/验证切分、指标计算、时间预算。换任务只换这个文件。 |
| `prepare.py` | 冻结 | 保留 `seed_everything()` 与通用工具。数据与评测常量迁往 `task.py`。 |
| `train.py` | agent | 模型、优化器、训练循环。只能从 `task.py` 取 val 与指标。 |
| `judge.py` | 人 | 标定、启动训练、解析分数、判定、改写名单与队列、生成报告。 |
| `program.md` | 人 | 把循环改成第 12 节的命令表。 |
| `run.json` | 只由 `judge.py` 写 | **新增。** 本次 run 的配置与标定结果。见第 6 节。 |
| `results.tsv` | 只由 `judge.py` 追加 | 全部历史。列见第 11 节。 |
| `parents.tsv` | 只由 `judge.py` 整文件替换 | 当前最多三个父本。列见第 11 节。 |
| `queue.tsv` | 只由 `judge.py` 整文件替换 | **新增。** 待复核队列。见第 8 节。 |
| `REPORT.md` | 只由 `judge.py` 整文件替换 | **新增。** 数据驱动的报告。见第 10 节。 |

`run.json`、`queue.tsv`、`REPORT.md` 加入 `.gitignore`。

写盘顺序不变：先把决定追加到 `results.tsv` 并 `fsync`，成功之后才替换 `parents.tsv` / `queue.tsv` / `REPORT.md`。整文件替换一律写临时文件再 `os.replace`。

开跑前 `train.py`、`prepare.py`、`judge.py`、`task.py` 相对 `HEAD` 有未提交改动时，所有子命令拒绝执行，且不追加记录。

## 5. `task.py`：冻结的尺子

新增文件，与 `prepare.py` 同级，开跑前写定，此后与 `prepare.py` 一样冻结。它导出：

```python
DATA_ROOT: Path          # 数据根目录
SPLIT_SEED: int          # 切分用的固定种子，与训练种子无关
TIME_BUDGET: float       # 墙钟训练预算，秒

def load_train() -> 训练集
def load_val() -> 验证集          # 唯一的验证集来源
def score(predict_fn) -> float    # 返回 val_bpb，越低越好
```

`train.py` 必须通过 `from task import load_train, load_val, score, TIME_BUDGET` 取得这些，不得自行遍历 `DATA_ROOT`、不得自行切分验证集、不得自行计算指标。

这一条是 flower14 第 2 号根因的正面修法：该 run 的 `train.py` 自己装载图片、自己持有 `EXTRA_PER_CLASS = 100` 和 `SPLIT_SEED = 0`，所以「不要改评测 harness」这句话落不到实处。v2 之后，改验证集的唯一办法是改 `task.py`，而改 `task.py` 会改变 `eval_hash`（第 6 节），会被 judge 当场拒绝。

`task.py` 不被 `judge.py` 导入执行除常量读取之外的逻辑；`judge.py` 只需要 `DATA_ROOT`。

## 6. 标定：`judge.py calibrate` 与 `run.json`

`calibrate` 取代 v1 的 `baseline`。在干净工作区的 baseline 提交上，连续跑 `R` 次训练，种子分别为 `1..R`。`R` 默认 **5**。

`R` 行都追加进 `results.tsv`，`mechanism` 为 `基线`，`decision` 为 `基线`，`seed` 分别为 `1`..`R`。任何一次训练崩溃或测量失败，`calibrate` 把该行按 `代码坏了` / `测量失败` 记下并**整体失败**，不写 `run.json`，不写 `parents.tsv`。

全部成功时：

```
σ   = R 个 val_bpb 的样本标准差（除数 R − 1）
Z   = 2.0
band_base      = Z · σ · √2
min_improvement = Z · σ
```

写 `run.json`：

```json
{
  "run_tag": "<分支名的 tag 部分>",
  "run_start": "<ISO 8601 带时区>",
  "baseline_commit": "<完整 40 位哈希>",
  "eval_hash": "<64 位十六进制>",
  "repeats": 5,
  "z": 2.0,
  "sigma": 0.0,
  "min_improvement": 0.0,
  "queue_limit": 2,
  "rotate_after": 8
}
```

`sigma` 与 `min_improvement` 写实际计算值。`band_base` 不入文件，由 `z` 和 `sigma` 现算。`queue_limit`（第 8 节的 K）默认 **2**，`rotate_after`（第 7.3 节的 M）默认 **8**。人可以在 `calibrate` 之后、搜索开始之前手改这两个值和 `z`；`sigma`、`repeats`、`eval_hash`、`baseline_commit` 手改视为篡改，但 judge 不检测。

`parents.tsv` 写入唯一一行：baseline 提交，`val_bpb` 是 `R` 个分数的**平均**，`seeds` 是 `1,2,...,R`，`reviewed` 是 `yes`，`simpler` 是 `no`。

### `eval_hash`

```
eval_hash = sha256(
    sha256(task.py 字节).hexdigest() + "\n" +
    sha256(prepare.py 字节).hexdigest() + "\n" +
    data_manifest(task.DATA_ROOT)
)
```

`data_manifest` 是 `DATA_ROOT` 下全部普通文件按相对路径排序后，每行 `<相对路径> <字节数>`，行间以 `\n` 拼接，末尾带一个 `\n`。不读取文件内容。目录不存在时 `eval_hash` 不可用。

`eval_hash` 取代 v1 的 `prepare_hash` + `data_hash` 两列，成为唯一的尺子身份。

任何子命令在当前 `eval_hash` 与 `run.json` 里的不一致时拒绝执行，两个例外：`calibrate` 负责重新标定，`report` 是只读的，尺子换了仍要能看报告。这是验收目标 C 的闸门：`task.py` 一个字节的改动、数据目录多一个文件、验证集换一种切法，都会让整条搜索链停下来，要求重新 `calibrate`。

`eval_hash` 不可用（`DATA_ROOT` 缺失）时，`trial` 把该行记成 `代码坏了`，细节 `数据身份不可用`。

## 7. 搜索：`judge.py trial`

### 7.1 噪声带判定

`trial --seed 1` 完成后，设候选分数为 `c`，父本在 `parents.tsv` 上的 `val_bpb` 为 `p`，父本的 `seeds` 条目数为 `n`：

```
band = Z · σ · √(1 + 1/n)
Δ    = c − p

Δ < −band    → 好于父本
|Δ| ≤ band   → 分不开
Δ > band     → 差于父本
```

`n` 来自父本自己被测了几次：刚 `accept` 进来的父本 `n = 1`（`band = Z·σ·√2`）；复核过的父本 `n = 2`（`band = Z·σ·√1.5`）；`calibrate` 产生的 baseline `n = R = 5`（`band = Z·σ·√1.2`）。父本被测得越准，带越窄，对候选越灵敏。

v1 的 `持平`（精确相等）一词取消。`分不开` 覆盖它。

> 这条规则的后果要在实现前知道：`517b245` 对 `39c96f8` 的领先是 `0.001335`。只要实测 `σ ≥ 0.00047`，`band` 就超过它，flower14 的冠军在 v2 下判 `分不开`，进不了名单。接受链后段的「兰索斯」「轻色彩」两次持平更是直接出局。这是验收目标 B 买到的东西，不是副作用。

### 7.2 `--knobs`：机制名与 diff 的一致性

`trial` 新增**必填**参数 `--knobs`，逗号分隔的模块级常量名集合，允许空串 `--knobs ""`。

judge 取 `git diff <HEAD^> <HEAD> -- train.py`，在 diff 的 `+` 与 `-` 行里匹配零缩进的模块级常量赋值：

```
^[+-]([A-Z][A-Z0-9_]*)[ \t]*=[^=]
```

末尾的 `[^=]` 排除 `==` 比较式。diff 自己的 `+++ b/train.py` 与 `--- a/train.py` 头行不匹配，因为第二个字符不是大写字母。

收集被触及的常量名集合 `touched`。`--knobs` 声明的集合为 `declared`。

```
touched ≠ declared → 拒绝，退出码 2，不训练，不追加记录
```

> `517b245` 重放：`declared = {}`（机制名「半精度」），`touched = {IMAGE_SIZE}`。当场拒绝。

**明确的盲区**：这条规则只覆盖模块级大写常量赋值。纯结构性改动（新增 `GradScaler`、把 `loss.backward()` 换成 `scaler.scale(loss).backward()`）不在覆盖范围，`--knobs ""` 对它们永远通过。这部分只能靠第 10 节 report 的机制审计段事后暴露，不做机械拦截——任何基于 diff 规模的阈值都会误伤天然需要改多行的单一机制，而且 agent 拆成两次提交就能绕过。

`calibrate` 没有父提交，不做 knobs 检查。`--seed 2` 的 `trial` 是对已有提交的复核，不做 knobs 检查。

### 7.3 父本轮换

设 `M = run.json.rotate_after`（默认 8）。取 `results.tsv` 里最近 `M` 行满足 `seed == "1"` 且 `decision ∈ {好于父本, 分不开, 差于父本, 代码坏了, 测量失败}` 的记录，读它们的 `parent` 列。

```
parents.tsv 行数 ≥ 2
且 这 M 个 parent 全部相同
且 该 parent == 当前 HEAD^
→ 拒绝，退出码 2，不训练，不追加记录
```

满足条件的记录不足 `M` 行时不拒绝。`parents.tsv` 只有一行时不拒绝（无处可换）。

> flower14 重放：平台期第 9 次从 `517b245` 出发就会被拒，强制切到 `151ca69` 或 `39c96f8`。

### 7.4 保留的 v1 检查

- 工作区（`train.py`、`prepare.py`、`judge.py`、`task.py`）必须干净。
- 同一完整提交哈希加同一种子已有记录时拒绝。
- `--seed 1` 的父提交必须在 `parents.tsv` 里，且该父本有可用分数。
- `--seed 2` 要求同提交已有正常完成的 seed-1 记录，且机制名一致。
- `--expect` 只允许 `lower` / `higher`，只作记录。`--fail` 非空、不含制表符与换行。
- 机制名非空、不含制表符与换行；`基线` 保留给 `calibrate`。
- 种子通过 `AUTORES_SEED` 传给训练进程；分数从 `run.log` 里最后一次 `seed:` 与 `val_bpb:` 行解析。`seed:` 不匹配记 `测量失败`；非零退出、缺 `val_bpb:`、或超 10 分钟记 `代码坏了`。

`trial` 允许的种子仍只有 `1` 和 `2`。`results.tsv` 的 `seed` 列可以出现 `1..R`（由 `calibrate` 写入）和 `*`（`accept` / `claim` 行）。

## 8. 接受与复核队列

### 8.1 `queue.tsv`

```
commit	parent	mechanism	accepted_at	state
```

`state` 目前只有 `待复核` 一个取值，保留列以便将来扩展。出队即删行。

### 8.2 `accept`

前置条件（相对 v1 的变化以粗体标出）：

- 工作区干净，`eval_hash` 与 `run.json` 一致。
- `HEAD` 有 seed-1 记录。
- 不带 `--simpler` 时，该记录的 `decision` 必须是 `好于父本`。
- 带 `--simpler --replace <哈希>` 时，该记录的 `decision` 必须是 **`分不开`**（v1 是 `持平`），且被替换的提交在名单里。

名单淘汰规则沿用 v1 第 4 节：优先去掉机制名重复者中分数最高的；无重复则去掉全体分数最高的；并列时不去掉刚进入的，其余并列者取哈希字典序较大的；应去掉的恰是刚进入者时名单不变，记 `未留下` / `名单已满`。

新进入或被替换写入 `parents.tsv` 的行：`val_bpb` 取该提交的 seed-1 分数，`seeds` 为 `1`，`reviewed` 为 `no`。

**`进入名单` 与 `替换父本` 都向 `queue.tsv` 追加一条 `待复核`。** v1 里 `持平` 是精确相等、极罕见；v2 里 `分不开` 是常态（flower14 平台期 97 轮全部落在带内），若 `--simpler` 替换不入队，agent 就能用噪声等价的提交无限次刷新名单。入队让背压天然封住这个口子。

### 8.3 背压

```
trial --seed 1 且 len(queue) > K  → 拒绝，退出码 2，不训练
                                     stderr 打印队首的 commit 与 mechanism
trial --seed 2                    → 不受队列限制
```

`K = run.json.queue_limit`，默认 2。seed-2 必须放行，否则 agent 被锁死——复核本身就要跑 seed-2。

> flower14 重放：accept 发生在 15:32 / 16:05 / 16:11，第三次之后队列长度 3 > 2，下一个 seed-1 trial 当场被拒。开跑 **49 分钟**就会被逼着做第一次复核，而不是 15 小时后交白卷。这是验收目标 A。

## 9. 成立：`judge.py claim`

`claim` 不训练。候选是 `HEAD`，父本是 `HEAD` 的 git 父提交，与 v1 一致。它读四个分数：候选 seed 1、候选 seed 2、父本 seed 1、父本 seed 2。

判定顺序（相对 v1 的变化以粗体标出）：

1. 四行的 **`eval_hash`** 不完全相同，或任一为空：`还判不了`，细节 `尺子或数据不同`。
2. 四个分数缺任何一个：`还判不了`，细节写明缺哪一个。
3. **`run.json` 缺失或 `min_improvement` ≤ 0**：`还判不了`，细节 `未标定`。（v1 这里是 `最小差距未写`，交付值 `None` 让这条必然命中；v2 的值由 `calibrate` 写入，正常流程下不可能为空。）
4. 两个种子的改进完全相等且不为 0：`还判不了`，细节 `两次改进没有分开`。
5. 平均改进 < 0：`更差`。
6. 平均改进 > `min_improvement`，且平均改进减碰运气扣减后仍 > `min_improvement`：`成立`。
7. 其余：`没有更好`。平均改进恰为 0 也落在这里。

改进定义、平均方式、碰运气扣减公式 `σ_cand · √(2 ln N)` 与 `N` 的计数口径全部沿用 v1 第 5 节，不变。`min_improvement = Z · σ` 的推导：claim 用两个种子的平均改进，其标准差是 `σ`（不是单次比较的 `σ√2`），所以同样的 `Z` 下门槛比 `trial` 的 `band` 窄 `√2` 倍。claim 更灵敏但更贵，trial 更粗但免费，两者内部一致。

### 9.1 claim 的副作用

得出**终局**判定（`成立` / `更差` / `没有更好`）时：

- 从 `queue.tsv` 删除该候选的行。
- 该候选若在 `parents.tsv` 里：
  - `成立` 或 `没有更好` → `val_bpb` 更新为两个种子的**平均**，`seeds` 改为 `1,2`，`reviewed` 改为 `yes`。
  - `更差` → 从 `parents.tsv` **移除**该行。不恢复当初被它挤掉的提交。移除后名单会变空时，保留该行并把 `reviewed` 改为 `yes`，`detail` 写 `名单不可为空`——没有父本，搜索就无从继续。

`还判不了` 不出队、不改名单——材料不够不算复核完成，否则一句 `claim` 就能卸掉背压。

> 「复核后用平均值更新父本基准」是本设计的一个附加收益。flower14 里 `517b245` 当了 97 次父本，而它的基准只是一次测量。若那一次恰好落在幸运的左尾，之后 97 次比较全部偏严，平台期就被人为制造出来了——72 错这个数字正好在 96 次重跑分布的最左端，这个可能性不低。复核把父本基准的噪声从 `σ` 降到 `σ/√2`，带也随之收紧（第 7.1 节的 `n`）。背压不只是逼出判定，它顺带把基准洗干净了。

## 10. 报告：`judge.py report`

写 `REPORT.md`，整文件替换。`calibrate`、`accept`、`claim` 成功之后 judge **自动**重写一次；`report` 子命令让人随时手动重算。不依赖 agent 记得敲命令——flower14 已经证明没被强制的事 agent 不会做。

段落，全部由数据决定：

1. **标定** — `R`、`σ`、`Z`、`band_base`、`min_improvement`、`run_start`、`baseline_commit`。
2. **轮次统计** — trial 总数与各 `decision` 计数，**按 `eval_hash` 分组**。协议一变，表自动劈成两段。flower14 把 98 张和 1498 张两版评测混进同一张接受链表、并把 `0.132653`（旧表 13/98）写成「199 错」（新表 `0.132844`）的事，结构上不可能再发生。
3. **搜索结果（未验证）** — `parents.tsv` 当前三行，每行标注 `reviewed` 与复核状态。
4. **已验证结论** — **只收 `decision = 成立` 的 claim 行**。一行都没有时，写「无」。
5. **未复核警告** — `queue.tsv` 余量与队首。
6. **机制审计** — 同一 `mechanism` 名下 `knobs` 取值不一致的记录行，逐条列出。

第 3 段与第 4 段的标题是硬编码的，`进入名单` 的提交只能出现在第 3 段。

这一条不是强制：agent 仍可以在别处写措辞超出数据的散文。它买到的是「数据表就在同一份文件里，矛盾一眼可见」——flower14 那句「纪录可复现」会直接挨着「已验证结论：无」出现。这是能做到的最硬的程度，spec 不假装它更硬。

## 11. 记录格式

`results.tsv`，制表符分隔，有表头：

```
commit	parent	mechanism	knobs	expect	fail	seed	val_bpb	eval_hash	decision	detail
```

相对 v1：删去 `prepare_hash` 与 `data_hash`，新增 `knobs` 与 `eval_hash`。

- `commit` / `parent` 是完整 40 位哈希，基线行 `parent` 为空。
- `knobs` 是逗号分隔的常量名，按字典序排列；空集合写空串。`calibrate` 行与 `accept` / `claim` 行写空串。
- `seed` 在 `accept` 与 `claim` 行写 `*`；`calibrate` 行写 `1`..`R`；`trial` 行写 `1` 或 `2`。
- 崩溃或测量失败时 `val_bpb` 留空，不写 `0`。

`decision` 取值：

| 来源 | 取值 |
|---|---|
| `calibrate` | `基线`、`代码坏了`、`测量失败` |
| `trial --seed 1` | `好于父本`、`分不开`、`差于父本`、`代码坏了`、`测量失败` |
| `trial --seed 2` | `已记录`、`代码坏了`、`测量失败` |
| `accept` | `进入名单`、`替换父本`、`未留下` |
| `claim` | `成立`、`还判不了`、`更差`、`没有更好` |

v1 的 `持平` 取消。

`parents.tsv`：

```
commit	mechanism	val_bpb	seeds	reviewed	simpler
```

新增 `seeds`（逗号分隔，如 `1` / `1,2` / `1,2,3,4,5`）与 `reviewed`（`yes` / `no`）。文件始终只有表头加最多三行数据。

`queue.tsv`：

```
commit	parent	mechanism	accepted_at	state
```

`accepted_at` 是 ISO 8601 带时区。

## 12. 命令表

```text
uv run judge.py calibrate                      # 取代 baseline，跑 R 个种子
uv run judge.py trial --seed 1 --mechanism 学习率 \
                      --knobs LEARNING_RATE \
                      --expect lower --fail 变高或崩溃
uv run judge.py trial --seed 2 --mechanism 学习率 \
                      --expect lower --fail 变高或崩溃
uv run judge.py accept
uv run judge.py accept --simpler --replace <父本完整哈希>
uv run judge.py claim
uv run judge.py report
```

`program.md` 的循环相应改为：

1. 读 `parents.tsv`、`results.tsv`、`queue.tsv`。队列满了就先去复核，别的都做不了。
2. 选父本，checkout，改 `train.py`，提交。
3. 写下三事实（机制名、方向、失败判据）**与 `--knobs`**。
4. `trial --seed 1`。
5. `好于父本` 且复杂度值得 → `accept`；`分不开` 且代码更简单 → `accept --simpler --replace`；`差于父本` 或不值得 → `git reset` 回父本。
6. 队列非空时，挑一条去做 `trial --seed 2` + `claim`。
7. `进入名单` 不是已验证结论。只有 `claim` 写出 `成立` 的那一行才是。

`program.md` 里不再写「不要总从最近的父本出发」这类劝告——该条已由第 7.3 节机械执行。同理「一次只动一个机制」由 `--knobs` 执行，「必须产出判定」由背压执行。凡是能机械化的都搬进 `judge.py`，`program.md` 只留下 judge 无法执行的部分。

## 13. 迁移

v2 的 `results.tsv` 与 v1 列不同、`decision` 词表不同，**不做兼容读取**。新 run 从 `calibrate` 开始，产生全新的 `results.tsv`。

`autoresearch/flower14` 的记录保留在它自己的分支与 worktree 里，不迁移、不重判。该 run 的结论按 v2 的标准重述一次即可：它产生了 0 个已验证结论，纪录提交 `517b245` 在 v2 的噪声带下大概率判 `分不开`。

`judge.py` 的 `baseline` 子命令删除。`MIN_IMPROVEMENT` 模块级常量删除，值从 `run.json` 读。

## 14. 测试

测试放在 `tests/test_judge.py`，只用标准库，不启动 GPU，不跑真实训练。沿用 v1 的全部用例（判定函数、淘汰规则、写盘顺序、脏工作区拒绝、重复 trial 拒绝、种子不符记 `测量失败`、超时记 `代码坏了`），另加：

**噪声带**
- `σ = 0.004`、`Z = 2`、父本 `n = 1` 时 `band = 0.004·2·√2`；`Δ = −band − 1e-9` 判 `好于父本`，`Δ = −band` 判 `分不开`，`Δ = band + 1e-9` 判 `差于父本`。边界取等号一律判 `分不开`。
- 父本 `n = 2` 与 `n = 5` 时 `band` 分别是 `Z·σ·√1.5` 与 `Z·σ·√1.2`，与公式的差在 `1e-12` 以内。
- 真实回放：`σ = 0.0005`、父本 `n = 1`、`Δ = 0.048064 − 0.049399` 判 `分不开`（flower14 冠军在 v2 下进不了名单）。

**标定**
- 5 个分数算出的 `σ` 等于 `statistics.stdev`，`min_improvement = 2σ`。
- `calibrate` 中任一次训练崩溃 → 整体失败，不写 `run.json`，不写 `parents.tsv`，但崩溃行已在 `results.tsv` 里。
- `parents.tsv` 首行 `val_bpb` 是 5 个分数的平均，`seeds = "1,2,3,4,5"`，`reviewed = "yes"`。

**`eval_hash`**
- `task.py` 改一个字节 → `eval_hash` 变 → `trial` 拒绝，退出码 2，无新行。
- `DATA_ROOT` 下新增一个文件 → `eval_hash` 变 → 同上。
- `DATA_ROOT` 不存在 → `trial` 记 `代码坏了` / `数据身份不可用`。
- 四行 `eval_hash` 不一致 → `claim` 判 `还判不了` / `尺子或数据不同`。

**`--knobs`**
- `declared = {}`、diff 改了 `IMAGE_SIZE` → 拒绝，无新行（`517b245` 回放）。
- `declared = {IMAGE_SIZE}`、diff 只改了 `IMAGE_SIZE` → 通过。
- `declared = {IMAGE_SIZE}`、diff 改了 `IMAGE_SIZE` 和 `BATCH_SIZE` → 拒绝。
- diff 只有结构性改动、`declared = {}` → 通过（盲区的正向断言）。
- 函数体内缩进的 `IMAGE_SIZE = 160` 不计入 `touched`。

**队列与背压**
- `accept` 成功 → 队列加一行；`accept` 判 `未留下` → 队列不变。
- `accept --simpler --replace` 成功 → 队列加一行。
- `len(queue) = K` → seed-1 `trial` 通过；`len(queue) = K + 1` → 拒绝，无新行。
- `len(queue) = K + 1` 时 seed-2 `trial` 仍通过。
- `claim` 判 `成立` / `更差` / `没有更好` → 出队；判 `还判不了` → 不出队。

**claim 的副作用**
- `成立` → `parents.tsv` 该行 `val_bpb` 变成两种子平均、`seeds = "1,2"`、`reviewed = "yes"`。
- `没有更好` → 同上（仍留在名单里）。
- `更差` → 该行从 `parents.tsv` 移除，其余两行不动。

**父本轮换**
- `parents.tsv` 一行时，连续 20 次同父本不拒绝。
- `parents.tsv` 三行、`M = 8`、最近 8 次 seed-1 记录父本相同且等于 `HEAD^` → 拒绝。
- 最近 8 次里有一次父本不同 → 通过。
- 满足条件的记录只有 7 行 → 通过。

**报告**
- 两种不同 `eval_hash` 的记录 → `REPORT.md` 的轮次统计分成两段。
- 没有 `成立` 行 → 「已验证结论」段写「无」。
- 有 `进入名单` 但无 `成立` → 该提交只出现在「搜索结果（未验证）」段。
- 同一 `mechanism` 名下 `knobs` 不一致 → 机制审计段列出这些行。

## 15. 做完时的样子

agent 按 `program.md` 可以完成：`calibrate` 建立基线与噪声底 → 用种子 1 搜索 → `accept` 更新最多三个父本 → 队列满时被迫复核 → 得到 `成立` / `还判不了` / `更差` / `没有更好` 之一 → `REPORT.md` 自动保持最新。agent 不需要手改任何表。

对照第 1 节的四个验收标准：

- **A** 队列背压让一夜不可能只留下提交而不落地判定。flower14 的节奏下，开跑 49 分钟就会触发第一次复核。
- **B** `band` 与 `min_improvement` 都由实测 `σ` 导出，噪声级别的「改进」判 `分不开`，进不了名单。
- **C** `task.py` 冻结尺子，`eval_hash` 覆盖 `task.py` + `prepare.py` + 数据清单，协议中途变更会让搜索停下来。
- **D** 父本轮换、`--knobs` 与 diff 的一致性、`REPORT.md` 对「进入名单」与「成立」的物理分段，全部在 `judge.py` 里；`program.md` 只留 judge 执行不了的部分。

父本名单上的提交仍然只是搜索结果。只有 `claim` 写出 `成立` 的那一行，才是已验证结论。
