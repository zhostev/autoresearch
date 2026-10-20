"""Record seed-1 search runs and write a separate morning claim."""

from __future__ import annotations

import math
import statistics

MIN_IMPROVEMENT: float | None = None
TIMEOUT_SECONDS = 600

COUNTED_DECISIONS = frozenset({
    "好于父本",
    "持平",
    "差于父本",
    "已记录",
    "代码坏了",
    "测量失败",
})


def compare_seed1(candidate: float, parent: float) -> str:
    if candidate < parent:
        return "好于父本"
    if candidate == parent:
        return "持平"
    return "差于父本"


def luck_penalty(candidate_a: float, candidate_b: float, n: int) -> float:
    if n <= 1:
        return 0.0
    sigma = statistics.stdev([candidate_a, candidate_b])
    return sigma * math.sqrt(2.0 * math.log(n))


def count_n(rows: list[dict]) -> int:
    names = {
        row["mechanism"]
        for row in rows
        if row["decision"] in COUNTED_DECISIONS and row["mechanism"] != "基线"
    }
    return len(names)


def claim_verdict(
    candidate_seed1: float | None,
    candidate_seed2: float | None,
    parent_seed1: float | None,
    parent_seed2: float | None,
    prepare_hashes: list[str],
    data_hashes: list[str],
    min_improvement: float | None,
    n: int,
) -> tuple[str, str]:
    if (
        len(prepare_hashes) != 4
        or len(set(prepare_hashes)) != 1
        or len(data_hashes) != 4
        or len(set(data_hashes)) != 1
        or any(item == "" for item in prepare_hashes + data_hashes)
    ):
        return "还判不了", "尺子或数据不同"
    missing = []
    if candidate_seed1 is None:
        missing.append("候选种子1")
    if candidate_seed2 is None:
        missing.append("候选种子2")
    if parent_seed1 is None:
        missing.append("父本种子1")
    if parent_seed2 is None:
        missing.append("父本种子2")
    if missing:
        return "还判不了", "缺" + "、".join(missing)
    if min_improvement is None or min_improvement <= 0:
        return "还判不了", "最小差距未写"
    improvement_1 = parent_seed1 - candidate_seed1
    improvement_2 = parent_seed2 - candidate_seed2
    if improvement_1 == improvement_2 and improvement_1 != 0:
        return "还判不了", "两次改进没有分开"
    average = (improvement_1 + improvement_2) / 2.0
    if average < 0:
        return "更差", ""
    penalty = luck_penalty(candidate_seed1, candidate_seed2, n)
    if average > min_improvement and average - penalty > min_improvement:
        return "成立", ""
    return "没有更好", ""


def _drop_one(rows: list[dict], newcomer_commit: str) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["mechanism"]] = counts.get(row["mechanism"], 0) + 1
    duplicated = [row for row in rows if counts[row["mechanism"]] > 1]
    pool = duplicated if duplicated else list(rows)
    worst = max(float(row["val_bpb"]) for row in pool)
    tied = [row for row in pool if float(row["val_bpb"]) == worst]
    if len(tied) == 1:
        return tied[0]
    others = [row for row in tied if row["commit"] != newcomer_commit]
    if not others:
        return tied[0]
    others.sort(key=lambda row: row["commit"])
    return others[-1]


def apply_accept(
    parents: list[dict], newcomer: dict, replace_commit: str | None
) -> tuple[list[dict], str, str]:
    if replace_commit is not None:
        if not any(row["commit"] == replace_commit for row in parents):
            raise ValueError("父本不在名单")
        replaced = [
            newcomer if row["commit"] == replace_commit else row for row in parents
        ]
        return replaced, "替换父本", replace_commit
    if any(row["commit"] == newcomer["commit"] for row in parents):
        return list(parents), "未留下", "已在名单"
    proposed = list(parents) + [newcomer]
    if len(proposed) <= 3:
        return proposed, "进入名单", ""
    victim = _drop_one(proposed, newcomer["commit"])
    if victim["commit"] == newcomer["commit"]:
        return list(parents), "未留下", "名单已满"
    kept = [row for row in proposed if row["commit"] != victim["commit"]]
    return kept, "进入名单", victim["commit"]
