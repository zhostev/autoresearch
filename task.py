"""任务身份：数据根、时间预算、验证集与指标。开跑前写定，此后冻结。

每个任务必须导出 NAME、DATA_ROOT、TIME_BUDGET、should_stop、load_train、load_val、score。

should_stop 是预算的执行权本身。它住在这里而不是 train.py，否则 agent 既能
定义预算又能决定什么时候不守它——那正是 flower14 二号根因的形状。judge 另外
核对训练打印的步数，所以即使 agent 不调它也瞒不过去。
DATA_ROOT 必须覆盖一切决定分数的数据，包括 tokenizer。
SPLIT_SEED 是任务自定义的可选项：nanochat 的验证集是固定分片
shard_06542.parquet，不按种子切分，因此它在这里不存在。

train.py 只能从这里取验证集与指标，不得自行遍历 DATA_ROOT、自行切分
验证集、自行计算 val_bpb。
"""

from pathlib import Path

import prepare

NAME = "nanochat"
DATA_ROOT = Path(prepare.CACHE_DIR)
TIME_BUDGET = prepare.TIME_BUDGET


def load_train(tokenizer, batch_size):
    return prepare.make_dataloader(tokenizer, batch_size, prepare.MAX_SEQ_LEN, "train")


def load_val(tokenizer, batch_size):
    return prepare.make_dataloader(tokenizer, batch_size, prepare.MAX_SEQ_LEN, "val")


def score(model, tokenizer, batch_size) -> float:
    return prepare.evaluate_bpb(model, tokenizer, batch_size)


HARD_SECONDS = 540.0
"""墙钟硬上限。judge.py 在 600 秒强杀，这里留 60 秒给训练之后的评测。"""


def should_stop(step: int, elapsed: float) -> bool:
    """训练循环唯一的终止条件。train.py 不得自己再写一个。"""
    return elapsed >= TIME_BUDGET or elapsed >= HARD_SECONDS
