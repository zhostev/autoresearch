"""任务身份：数据根、时间预算、验证集与指标。开跑前写定，此后冻结。

每个任务必须导出 NAME、DATA_ROOT、TIME_BUDGET、load_val、score。
SPLIT_SEED 与 load_train 是任务自定义的可选项：nanochat 的验证集是固定
分片 shard_06542.parquet，不按种子切分，因此两者在这里不存在。

train.py 只能从这里取验证集与指标，不得自行遍历 DATA_ROOT、自行切分
验证集、自行计算 val_bpb。
"""

from pathlib import Path

import prepare

NAME = "nanochat"
DATA_ROOT = Path(prepare.DATA_DIR)
TIME_BUDGET = prepare.TIME_BUDGET


def load_val(tokenizer, batch_size):
    return prepare.make_dataloader(tokenizer, batch_size, prepare.MAX_SEQ_LEN, "val")


def score(model, tokenizer, batch_size) -> float:
    return prepare.evaluate_bpb(model, tokenizer, batch_size)
