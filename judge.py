"""Record seed-1 search runs and write a separate morning claim."""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT_SECONDS = 600

LEDGER_FILE = "ledger.jsonl"
"""append-only 的 run 账本。results.tsv 是给人读的投影，这里是机器核的身份。"""

ALPHA = 0.05
"""配对检验的显著性水平。与 expctl 的 G4 同口径：预注册，跑完不许改。"""

RUN_FILE = "run.json"
RUN_FIELDS: tuple[tuple[str, type], ...] = (
    ("run_tag", str),
    ("run_start", str),
    ("baseline_commit", str),
    ("eval_hash", str),
    ("repeats", int),
    ("z", float),
    ("sigma", float),
    ("min_improvement", float),
    ("queue_limit", int),
    ("rotate_after", int),
)


def read_run(repo: Path) -> dict | None:
    path = repo / RUN_FILE
    if not path.is_file():
        return None
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(config, dict):
        return None
    for name, kind in RUN_FIELDS:
        if name not in config:
            return None
        value = config[name]
        if kind is float and isinstance(value, int) and not isinstance(value, bool):
            config[name] = float(value)
            continue
        if not isinstance(value, kind) or isinstance(value, bool):
            return None
    return config


def write_run(repo: Path, config: dict) -> None:
    path = repo / RUN_FILE
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def arm_sigma(samples: list[float], calibrated: float) -> float:
    """一臂的噪声估计。

    少于两个样本就只能用标定值。有两个以上时取观测与标定的较大者：
    三个样本看着安静往往只是抽样不足，看着吵一定是真吵。σ 是配置的属性
    而不是全局常数——同一基线上，两个机制单独用的 sd 都在 3 张左右，
    合起来能跳到 7-9 张。
    """
    if len(samples) < 2:
        return calibrated
    return max(statistics.stdev(samples), calibrated)


def band_for(
    sigma_candidate: float, sigma_parent: float, z: float, m: int = 1, n: int = 1
) -> float:
    """两个均值之差的判定带。两臂各用自己的噪声，不假设同方差。"""
    return z * standard_error(sigma_candidate, sigma_parent, m, n)


def standard_error(
    sigma_candidate: float, sigma_parent: float, m: int, n: int
) -> float:
    candidate = m if m >= 1 else 1
    parent = n if n >= 1 else 1
    return math.sqrt(
        sigma_candidate ** 2 / candidate + sigma_parent ** 2 / parent
    )


def claim_threshold(
    sigma_candidate: float, sigma_parent: float, z: float, m: int, n: int, n_mech: int
) -> float:
    """终局门槛：把多重比较做成标准误的倍数，而不是一个固定偏移。

    固定偏移不随样本数缩小，会让「多测几次」对 claim 毫无用处。写成倍数后，
    门槛随样本数下降、随已试机制数上升，并且恒大于 accept 用的 z·SE。
    """
    crowd = math.sqrt(2.0 * math.log(max(n_mech, 2)))
    return (z + crowd) * standard_error(sigma_candidate, sigma_parent, m, n)


def _betacf(a: float, b: float, x: float) -> float:
    """不完全 beta 函数的连分式（Numerical Recipes 6.4）。

    stdlib 没有 t 分布；配对样本只有 2–5 对时用正态近似会让 p 值偏小，
    所以这里补一个不引第三方依赖的精确 t 尾概率。
    """
    max_iter, eps, tiny = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    beta = math.exp(log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return beta * _betacf(a, b, x) / a
    return 1.0 - beta * _betacf(b, a, 1.0 - x) / b


def paired_p_value(deltas: list[float]) -> float | None:
    """双侧配对 t 检验的 p 值。

    样本不足或差值完全同值（方差 0）时返回 None：两次测量给出完全相同的
    差值，更可能是测量被钉死，而不是「无穷显著」。expctl 对这条判 inconclusive。
    """
    if len(deltas) < 2:
        return None
    spread = statistics.stdev(deltas)
    if spread == 0.0:
        return None
    t_stat = statistics.fmean(deltas) / (spread / math.sqrt(len(deltas)))
    df = len(deltas) - 1
    return _betai(df / 2.0, 0.5, df / (df + t_stat * t_stat))


COUNTED_DECISIONS = frozenset({
    "好于父本",
    "分不开",
    "差于父本",
    "已记录",
    "代码坏了",
    "测量失败",
})


def sample_scores(rows: list[dict], commit: str) -> dict[str, float]:
    """这个提交所有正常跑完的种子分数，种子号 -> val_bpb。"""
    found: dict[str, float] = {}
    for row in rows:
        if row["commit"] != commit or row["val_bpb"] == "":
            continue
        if row["decision"] in COMPLETED_DECISIONS:
            found[row["seed"]] = float(row["val_bpb"])
    return found


def parent_n(parent_row: dict) -> int:
    raw = parent_row.get("seeds", "")
    parts = [item for item in raw.split(",") if item.strip() != ""]
    return len(parts) if parts else 1


def compare_to_parent(candidate: float, parent: float, band: float) -> str:
    delta = candidate - parent
    if delta < -band:
        return "好于父本"
    if delta > band:
        return "差于父本"
    return "分不开"


def count_n(rows: list[dict]) -> int:
    names = {
        row["mechanism"]
        for row in rows
        if row["decision"] in COUNTED_DECISIONS and row["mechanism"] != "基线"
    }
    return len(names)


def claim_verdict(
    candidate: list[float],
    parent: list[float],
    eval_hashes: list[str],
    sigma: float | None,
    z: float,
    policy_floor: float | None,
    n_mech: int,
) -> tuple[str, str]:
    """终局判定。两道独立的闸：统计门槛，和人设的政策下限。

    统计门槛 = (z + √(2 ln N)) · SE，SE 按两臂各自的噪声算。它恒大于 accept
    用的 z·SE，所以 claim 永远严于 accept；同时它随样本数缩小，所以多测
    几次真的能换来灵敏度。

    政策下限是「人不关心小于这个幅度的改进」，与统计无关，可以是 0。
    """
    if len(candidate) < 2:
        return "还判不了", "缺候选样本"
    if len(parent) < 2:
        return "还判不了", "缺父本样本"
    if (
        len(eval_hashes) < 2
        or len(set(eval_hashes)) != 1
        or any(item == "" for item in eval_hashes)
    ):
        return "还判不了", "尺子或数据不同"
    if sigma is None or sigma <= 0:
        return "还判不了", "未标定"
    improvement = statistics.fmean(parent) - statistics.fmean(candidate)
    if improvement < 0:
        return "更差", ""
    threshold = claim_threshold(
        arm_sigma(candidate, sigma), arm_sigma(parent, sigma),
        z, len(candidate), len(parent), n_mech,
    )
    if improvement <= threshold:
        return "没有更好", ""
    if policy_floor is not None and improvement <= policy_floor:
        return "没有更好", ""
    return "成立", ""


ATTEMPT_DECISIONS = frozenset({
    "好于父本", "分不开", "差于父本", "代码坏了", "测量失败",
})


def rotation_blocked(
    rows: list[dict], parent_count: int, head_parent: str, rotate_after: int
) -> bool:
    if parent_count < 2 or rotate_after < 1:
        return False
    attempts = [
        row for row in rows
        if row["seed"] == "1" and row["decision"] in ATTEMPT_DECISIONS
    ]
    window = attempts[-rotate_after:]
    if len(window) < rotate_after:
        return False
    seen = {row["parent"] for row in window}
    return len(seen) == 1 and head_parent in seen


FINAL_CLAIMS = frozenset({"成立", "更差", "没有更好"})


def apply_claim(
    parents: list[dict], commit: str, decision: str, mean: float,
    seeds: str = "1,2",
) -> list[dict]:
    if decision not in FINAL_CLAIMS:
        return list(parents)
    if not any(row["commit"] == commit for row in parents):
        return list(parents)
    if decision == "更差" and len(parents) > 1:
        return [row for row in parents if row["commit"] != commit]
    updated = []
    for row in parents:
        if row["commit"] != commit:
            updated.append(row)
            continue
        fresh = dict(row)
        fresh["reviewed"] = "yes"
        if decision != "更差":
            fresh["val_bpb"] = f"{mean:.6f}"
            fresh["seeds"] = seeds
        updated.append(fresh)
    return updated


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


RESULT_HEADER = (
    "commit\tparent\tmechanism\tknobs\texpect\tfail\tseed\tval_bpb\t"
    "eval_hash\tdecision\tdetail"
)
PARENT_HEADER = "commit\tmechanism\tval_bpb\tseeds\treviewed\tsimpler"
RESULT_FIELDS = RESULT_HEADER.split("\t")
PARENT_FIELDS = PARENT_HEADER.split("\t")


def _line(fields: list[str], row: dict) -> str:
    return "\t".join(row[field] for field in fields)


QUEUE_HEADER = "commit\tparent\tmechanism\taccepted_at\tstate"
QUEUE_FIELDS = QUEUE_HEADER.split("\t")


def read_queue(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[1:]:
        if line == "":
            continue
        rows.append(dict(zip(QUEUE_FIELDS, line.split("\t"))))
    return rows


def write_queue(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(QUEUE_HEADER + "\n")
        for row in rows:
            handle.write(_line(QUEUE_FIELDS, row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def enqueue(rows: list[dict], entry: dict) -> list[dict]:
    if any(row["commit"] == entry["commit"] for row in rows):
        return list(rows)
    return list(rows) + [entry]


def dequeue(rows: list[dict], commit: str) -> list[dict]:
    return [row for row in rows if row["commit"] != commit]


def prune_queue(repo: Path, rows: list[dict]) -> list[dict]:
    kept = []
    for row in rows:
        if git_output(repo, "cat-file", "-e", row["commit"] + "^{commit}") is None:
            continue
        kept.append(row)
    return kept


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[1:]:
        if line == "":
            continue
        parts = line.split("\t")
        if len(parts) < len(RESULT_FIELDS):
            parts = parts + [""] * (len(RESULT_FIELDS) - len(parts))
        rows.append(dict(zip(RESULT_FIELDS, parts)))
    return rows


def append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write(RESULT_HEADER + "\n")
        handle.write(_line(RESULT_FIELDS, row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_parents(path: Path) -> list[dict]:
    if not path.exists() or path.is_dir():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[1:]:
        if line == "":
            continue
        parts = line.split("\t")
        rows.append(dict(zip(PARENT_FIELDS, parts)))
    return rows


def write_parents(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(PARENT_HEADER + "\n")
        for row in rows:
            handle.write(_line(PARENT_FIELDS, row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def record_accept(
    results_path: Path,
    parents_path: Path,
    result_row: dict,
    new_parents: list[dict] | None,
) -> None:
    append_result(results_path, result_row)
    if new_parents is not None:
        write_parents(parents_path, new_parents)


def build_report(
    config: dict | None,
    rows: list[dict],
    parents: list[dict],
    queue: list[dict],
) -> str:
    out: list[str] = ["# 搜索报告", "", "## 标定", ""]
    if config is None:
        out.append("未标定。先运行 `uv run judge.py calibrate`。")
    else:
        out.append(f"- run_tag：{config['run_tag']}")
        out.append(f"- run_start：{config['run_start']}")
        out.append(f"- baseline：{config['baseline_commit'][:7]}")
        out.append(f"- R = {config['repeats']}，Z = {config['z']}")
        out.append(f"- σ = {config['sigma']:.6f}")
        out.append(
            f"- band（两臂各一次、均为标定噪声）= "
            f"{band_for(config['sigma'], config['sigma'], config['z'], 1, 1):.6f}"
        )
        out.append(f"- min_improvement = {config['min_improvement']:.6f}")
        out.append(
            f"- queue_limit = {config['queue_limit']}，"
            f"rotate_after = {config['rotate_after']}"
        )
    out += ["", "## 轮次统计", ""]
    groups: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["seed"] == "*":
            continue
        key = row["eval_hash"]
        groups.setdefault(key, {})
        groups[key][row["decision"]] = groups[key].get(row["decision"], 0) + 1
    if not groups:
        out.append("还没有记录。")
    for key in sorted(groups):
        out += [f"### 尺子 {key[:12]}", "", "| 判定 | 次数 |", "|---|---|"]
        for decision in sorted(groups[key]):
            out.append(f"| {decision} | {groups[key][decision]} |")
        out.append("")
    out += ["", "## 搜索结果（未验证）", ""]
    if not parents:
        out.append("名单为空。")
    else:
        out += ["| commit | 机制 | val_bpb | seeds | 已复核 |", "|---|---|---|---|---|"]
        for row in parents:
            out.append(
                f"| {row['commit'][:7]} | {row['mechanism']} | {row['val_bpb']} "
                f"| {row['seeds']} | {row['reviewed']} |"
            )
    out += ["", "## 已验证结论", ""]
    holds = [row for row in rows if row["decision"] == "成立"]
    if not holds:
        out.append("无。进入名单不是已验证结论。")
    else:
        out += ["| commit | 机制 |", "|---|---|"]
        for row in holds:
            out.append(f"| {row['commit'][:7]} | {row['mechanism']} |")
    out += ["", "## 未复核警告", ""]
    if not queue:
        out.append("队列为空。")
    else:
        out.append(f"还有 {len(queue)} 条待复核：")
        for row in queue:
            out.append(
                f"- {row['commit'][:7]}（{row['mechanism']}，{row['accepted_at']}）"
            )
    out += ["", "## 机制审计", ""]
    knobs_by_mechanism: dict[str, set[str]] = {}
    for row in rows:
        if row["seed"] != "1" or row["mechanism"] == "基线":
            continue
        knobs_by_mechanism.setdefault(row["mechanism"], set()).add(row["knobs"])
    flagged = {
        name: values for name, values in knobs_by_mechanism.items() if len(values) > 1
    }
    if not flagged:
        out.append("同名机制的旋钮声明一致。")
    else:
        for name in sorted(flagged):
            out.append(f"- {name}：{'、'.join(sorted(flagged[name]))}")
    out.append("")
    return "\n".join(out)


def write_report(repo: Path) -> None:
    text = build_report(
        read_run(repo),
        read_results(repo / "results.tsv"),
        read_parents(repo / "parents.tsv"),
        read_queue(repo / "queue.tsv"),
    )
    path = repo / "REPORT.md"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_summary(text: str) -> tuple[str | None, str | None]:
    seed = None
    val_bpb = None
    for line in text.splitlines():
        if line.startswith("seed:"):
            parts = line.split()
            seed = parts[1] if len(parts) > 1 else None
        elif line.startswith("val_bpb:"):
            parts = line.split()
            val_bpb = parts[1] if len(parts) > 1 else None
    return seed, val_bpb


def will_overrun(
    step: int,
    elapsed: float,
    step_budget: int,
    hard_seconds: float,
    probe_steps: int = 1000,
) -> bool:
    """按当前速率投影，这次训练是否注定跑不完步数预算。

    步数预算与墙钟上限是两条互不知情的约束。没有投影时，一个超出算力的配置
    要烧满整个墙钟才被发现跑不完步数——一次约九分钟，而且每个贴着上限的机制
    都要交一次这笔学费。跑够 probe_steps 让速率稳定之后，投影把这笔学费压到
    一分钟以内；判定仍是 `测量失败`，只是来得早得多。

    task.py 的 should_stop 调它。放在这里是因为预算的执行是 harness 的政策，
    task.py 只提供参数。
    """
    if step < probe_steps or step <= 0:
        return False
    return elapsed / step * step_budget > hard_seconds


def step_budget(task_py: Path) -> int | None:
    """从 task.py 源码里读 STEP_BUDGET，不执行它。任务没声明就返回 None。"""
    if not task_py.is_file():
        return None
    try:
        tree = ast.parse(task_py.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, ValueError):
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "STEP_BUDGET":
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, int):
                    return value.value
    return None


def parse_steps(text: str) -> int | None:
    found = None
    for line in text.splitlines():
        if line.startswith("num_steps:"):
            parts = line.split()
            if len(parts) > 1:
                try:
                    found = int(parts[1])
                except ValueError:
                    found = None
    return found


def short_run(log_text: str, task_py: Path) -> tuple[int | None, int] | None:
    """训练没跑满步数预算时返回 (实际步数, 预算)，否则 None。"""
    budget = step_budget(task_py)
    if budget is None:
        return None
    steps = parse_steps(log_text)
    if steps is None or steps < budget:
        return steps, budget
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def data_manifest(data_root: Path) -> str | None:
    if not data_root.is_dir():
        return None
    entries = []
    for path in data_root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if path.name.startswith(".") or path.name.endswith(".tmp"):
            continue
        entries.append((path.relative_to(data_root).as_posix(), path.stat().st_size))
    entries.sort(key=lambda item: item[0])
    return "".join(f"{name} {size}\n" for name, size in entries)


def eval_hash(task_py: Path, prepare_py: Path, data_root: Path) -> str | None:
    if not task_py.is_file() or not prepare_py.is_file():
        return None
    manifest = data_manifest(data_root)
    if manifest is None:
        return None
    payload = file_sha256(task_py) + "\n" + file_sha256(prepare_py) + "\n" + manifest
    digest = hashlib.sha256()
    digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def refuse(reason: str) -> int:
    print(f"拒绝：{reason}", file=sys.stderr)
    return 2


COMPLETED_DECISIONS = frozenset({
    "基线", "好于父本", "分不开", "差于父本", "已记录",
})


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def canonical_json(payload: dict) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def run_identity(
    commit: str,
    parent: str,
    eval_hash: str,
    seed: str,
    knobs: str,
    step_budget: int | None,
) -> str:
    """内容寻址的 run 身份：同一份代码、尺子、种子与预算只有一个名字。

    与 expctl 的 run_id 同构，但**不**把 mechanism/expect/fail 放进来：
    那是这条 run 的假设叙述，不是身份；改叙述不该换一条 run。
    预算进身份：同一提交在 300 秒与 10200 步下不是同一条测量。
    """
    payload = {
        "commit": commit,
        "parent": parent,
        "eval_hash": eval_hash,
        "seed": seed,
        "knobs": knobs,
        "step_budget": "" if step_budget is None else step_budget,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def read_ledger(repo: Path) -> tuple[list[dict], int]:
    """读 append-only 账本，返回 (事件, 坏行数)。坏行不吞掉，交给 audit 报。"""
    path = repo / LEDGER_FILE
    if not path.is_file():
        return [], 0
    events: list[dict] = []
    bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "":
            continue
        try:
            event = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if not isinstance(event, dict):
            bad += 1
            continue
        events.append(event)
    return events, bad


def append_ledger(repo: Path, event: dict) -> None:
    path = repo / LEDGER_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def ledger_status(events: list[dict]) -> dict[str, dict]:
    """run_id -> {admitted, completed}。只认 run_id 字段，不猜。"""
    status: dict[str, dict] = {}
    for event in events:
        run_id = event.get("run_id")
        if not isinstance(run_id, str) or run_id == "":
            continue
        slot = status.setdefault(run_id, {"admitted": False, "completed": False})
        if event.get("event") == "admit":
            slot["admitted"] = True
        elif event.get("event") == "result" and event.get("decision") in COMPLETED_DECISIONS:
            slot["completed"] = True
    return status


def ruler_moved(repo: Path, config: dict | None, test_root: bool) -> bool:
    if config is None:
        return True
    task_py, prepare_py, data_root = eval_paths(repo, test_root)
    return eval_hash(task_py, prepare_py, data_root) != config["eval_hash"]


def eval_paths(repo: Path, test_root: bool) -> tuple[Path, Path, Path]:
    task_py = repo / "task.py"
    prepare_py = repo / "prepare.py"
    if test_root:
        return task_py, prepare_py, repo / "data"
    from task import DATA_ROOT

    return task_py, prepare_py, Path(DATA_ROOT)


def git_output(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def is_dirty(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--",
         "train.py", "prepare.py", "judge.py", "task.py"],
        cwd=root, capture_output=True, text=True,
    )
    return completed.returncode != 0 or completed.stdout.strip() != ""


def blank_row() -> dict:
    return {field: "" for field in RESULT_FIELDS}


CONSTANT_LINE = re.compile(
    r"^[+-]([A-Z][A-Z0-9_]*(?:[ \t]*,[ \t]*[A-Z][A-Z0-9_]*)*)"
    r"[ \t]*(?::[^=]*)?(?:\*\*|//|>>|<<|[-+*/%@|&^])?=(?!=)"
)


def touched_constants(diff_text: str) -> set[str]:
    found: set[str] = set()
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        match = CONSTANT_LINE.match(line)
        if match is not None:
            for name in match.group(1).split(","):
                found.add(name.strip())
    return found


def parse_knobs(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip() != ""}


def validate_text(seed: str, mechanism: str, expect: str, fail: str) -> str | None:
    if seed not in {"1", "2"}:
        return "种子只允许 1 或 2"
    if mechanism == "" or (mechanism == "基线" and seed != "2") or "\t" in mechanism or "\n" in mechanism:
        return "机制名无效"
    if expect not in {"lower", "higher"}:
        return "expect 只允许 lower 或 higher"
    if fail == "" or "\t" in fail or "\n" in fail:
        return "失败说明无效"
    return None


def run_command(command: list[str], seed: str, log_path: Path, timeout: float) -> int:
    env = os.environ.copy()
    env["AUTORES_SEED"] = seed
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=log_path.parent,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return 124


def main(
    argv: list[str] | None = None,
    command: list[str] | None = None,
    timeout_seconds: float | None = None,
    root: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="judge.py")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("claim", "report"):
        sub.add_parser(name)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--json", action="store_true")
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--base", required=True)
    compare_parser.add_argument("--treat", required=True)
    compare_parser.add_argument("--mde", type=float, default=None)
    compare_parser.add_argument("--json", action="store_true")
    calibrate_parser = sub.add_parser("calibrate")
    calibrate_parser.add_argument("--force", action="store_true")
    accept_parser = sub.add_parser("accept")
    accept_parser.add_argument("--simpler", action="store_true")
    accept_parser.add_argument("--replace", default=None)
    trial = sub.add_parser("trial")
    trial.add_argument("--seed", required=True)
    trial.add_argument("--mechanism", required=True)
    trial.add_argument("--knobs", default=None)
    trial.add_argument("--expect", required=True)
    trial.add_argument("--fail", required=True)
    args = parser.parse_args(argv)
    repo = root or Path.cwd()
    if args.cmd == "report":
        write_report(repo)
        return 0
    if args.cmd == "compare":
        return cmd_compare(repo, args.base, args.treat, args.mde, args.json)
    if args.cmd == "audit":
        return cmd_audit(repo, args.json)
    if args.cmd == "calibrate":
        return cmd_calibrate(
            repo,
            command or ["uv", "run", "train.py"],
            TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds,
            root is not None,
            args.force,
        )
    if args.cmd == "accept":
        return cmd_accept(repo, args.simpler, args.replace, root is not None)
    if args.cmd == "claim":
        return cmd_claim(repo, root is not None)
    if args.cmd != "trial":
        return refuse("不认识这个子命令")
    timeout = TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    train = command or ["uv", "run", "train.py"]
    error = validate_text(args.seed, args.mechanism, args.expect, args.fail)
    if error is not None:
        return refuse(error)
    if is_dirty(repo):
        return refuse("工作区不干净：train.py / prepare.py / judge.py / task.py 有未提交改动")
    config = read_run(repo)
    if config is None:
        return refuse("还没标定：先运行 judge.py calibrate")
    if ruler_moved(repo, config, root is not None):
        return refuse("尺子变了：task.py、prepare.py 或数据目录与 run.json 不一致，需重新 calibrate")
    head = git_output(repo, "rev-parse", "HEAD")
    if head is None:
        return refuse("读不出 HEAD")
    results_path = repo / "results.tsv"
    rows = read_results(results_path)
    if any(
        row["commit"] == head
        and row["seed"] == args.seed
        and row["decision"] in COMPLETED_DECISIONS
        for row in rows
    ):
        return refuse("这个提交的该种子已经有完成记录了")
    if args.seed == "2":
        seed1 = [
            row for row in rows
            if row["commit"] == head and row["seed"] == "1"
            and row["decision"] in {"基线", "好于父本", "分不开", "差于父本"}
        ]
        if not seed1 or seed1[-1]["mechanism"] != args.mechanism:
            return refuse("种子 2 要求这个提交已有同名机制的种子 1 完成记录")
    else:
        parent = git_output(repo, "rev-parse", "HEAD^")
        listed = {row["commit"] for row in read_parents(repo / "parents.tsv")}
        if parent is None or parent not in listed:
            return refuse("这个提交的父本不在 parents.tsv 名单里")
        parent_scores = [
            item for item in rows
            if item["commit"] == parent and item["seed"] == "1" and item["val_bpb"] != ""
        ]
        if not parent_scores:
            return refuse("父本还没有种子 1 的分数")
        if args.knobs is None:
            return refuse("种子 1 必须声明 --knobs（没改常量就写 --knobs \"\"）")
        diff = git_output(repo, "diff", parent, head, "--", "train.py")
        if diff is None:
            return refuse("读不出 train.py 相对父本的 diff")
        touched = touched_constants(diff)
        declared = parse_knobs(args.knobs)
        if touched != declared:
            return refuse(
                "声明的旋钮和 diff 对不上：diff 改了 "
                f"{'、'.join(sorted(touched)) or '（无）'}，声明的是 "
                f"{'、'.join(sorted(declared)) or '（无）'}"
            )
        if rotation_blocked(
            rows, len(read_parents(repo / "parents.tsv")), parent,
            config["rotate_after"],
        ):
            return refuse(
                f"连续 {config['rotate_after']} 次都从这个父本出发了，换一个起点"
            )
        queue_path = repo / "queue.tsv"
        queue_rows = read_queue(queue_path)
        pruned = prune_queue(repo, queue_rows)
        if pruned != queue_rows:
            write_queue(queue_path, pruned)
        if len(pruned) > config["queue_limit"]:
            head_entry = pruned[0]
            return refuse(
                f"队列积压 {len(pruned)} 条，先复核 {head_entry['commit'][:7]}"
                f"（{head_entry['mechanism']}）"
            )
    task_py, prepare_py, data_root = eval_paths(repo, root is not None)
    identity = eval_hash(task_py, prepare_py, data_root)
    row = blank_row()
    row.update({
        "commit": head,
        "parent": git_output(repo, "rev-parse", "HEAD^") or "",
        "mechanism": args.mechanism,
        "knobs": ",".join(sorted(parse_knobs(args.knobs or ""))) if args.seed == "1" else "",
        "expect": args.expect,
        "fail": args.fail,
        "seed": args.seed,
    })
    if identity is None:
        row["decision"] = "代码坏了"
        row["detail"] = "数据身份不可用"
        append_result(results_path, row)
        return 0
    row["eval_hash"] = identity
    budget = step_budget(task_py)
    run_id = run_identity(
        head, row["parent"], identity, args.seed, row["knobs"], budget
    )
    ledger_events, _ = read_ledger(repo)
    if ledger_status(ledger_events).get(run_id, {}).get("completed"):
        return refuse(f"这个 run 身份已经完成过：{run_id[:12]}")
    append_ledger(repo, {
        "event": "admit",
        "run_id": run_id,
        "at": _now(),
        "commit": head,
        "parent": row["parent"],
        "seed": args.seed,
        "eval_hash": identity,
        "knobs": row["knobs"],
        "step_budget": budget,
        "mechanism": args.mechanism,
        "expect": args.expect,
        "fail": args.fail,
    })

    def record(
        result_row: dict,
        steps: int | None = None,
        seconds: float | None = None,
        run_key: str | None = None,
    ) -> None:
        """结果表是投影，账本是身份；两边同一次写入，不各写各的。"""
        append_result(results_path, result_row)
        append_ledger(repo, {
            "event": "result",
            "run_id": run_key or run_id,
            "at": _now(),
            "commit": result_row["commit"],
            "seed": result_row["seed"],
            "decision": result_row["decision"],
            "val_bpb": result_row["val_bpb"],
            "steps": steps,
            "seconds": None if seconds is None else round(seconds, 3),
            "detail": result_row["detail"],
        })

    started = time.monotonic()
    code = run_command(train, args.seed, repo / "run.log", timeout)
    elapsed = time.monotonic() - started
    log_text = (repo / "run.log").read_text(encoding="utf-8") if (repo / "run.log").exists() else ""
    steps = parse_steps(log_text)
    printed_seed, printed_bpb = parse_summary(log_text)
    if code != 0 or printed_bpb is None:
        row["decision"] = "代码坏了"
        record(row, steps=steps, seconds=elapsed)
        return 0
    if printed_seed != args.seed:
        row["decision"] = "测量失败"
        record(row, steps=steps, seconds=elapsed)
        return 0
    cut = short_run(log_text, task_py)
    if cut is not None:
        row["decision"] = "测量失败"
        row["detail"] = f"只跑了 {cut[0]} 步，预算 {cut[1]} 步"
        record(row, steps=cut[0], seconds=elapsed)
        return 0
    row["val_bpb"] = printed_bpb
    if args.seed == "2":
        row["decision"] = "已记录"
        record(row, steps=steps, seconds=elapsed)
        return 0
    parent_commit = row["parent"]
    parent_rows = [
        item for item in rows
        if item["commit"] == parent_commit and item["seed"] == "1" and item["val_bpb"] != ""
    ]
    parent_listed = next(
        (item for item in read_parents(repo / "parents.tsv")
         if item["commit"] == parent_commit),
        None,
    )
    parent_bpb = (
        float(parent_listed["val_bpb"]) if parent_listed is not None
        else float(parent_rows[-1]["val_bpb"])
    )
    n_parent = parent_n(parent_listed or {})
    samples = [float(printed_bpb)]
    if float(printed_bpb) < parent_bpb:
        # 有苗头就追加测量：一次抽样的带宽既放得进幸运样本，
        # 也挡得住比它小的真实改进。多抽几次两边都治。
        measured = sample_scores(rows, head)
        for index in range(2, int(config.get("trial_repeats", 3)) + 1):
            seed = str(index)
            if seed in measured:
                samples.append(measured[seed])
                continue
            extra = blank_row()
            extra.update({
                "commit": head, "parent": parent_commit,
                "mechanism": args.mechanism, "knobs": "",
                "expect": args.expect, "fail": args.fail,
                "seed": seed, "eval_hash": identity,
            })
            extra_id = run_identity(
                head, parent_commit, identity, seed, "", budget
            )
            append_ledger(repo, {
                "event": "admit",
                "run_id": extra_id,
                "at": _now(),
                "commit": head,
                "parent": parent_commit,
                "seed": seed,
                "eval_hash": identity,
                "knobs": "",
                "step_budget": budget,
                "mechanism": args.mechanism,
                "expect": args.expect,
                "fail": args.fail,
                "detail": "追加测量",
            })
            started = time.monotonic()
            code = run_command(train, seed, repo / "run.log", timeout)
            extra_elapsed = time.monotonic() - started
            log_path = repo / "run.log"
            text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
            extra_steps = parse_steps(text)
            got_seed, got_bpb = parse_summary(text)
            if code != 0 or got_bpb is None:
                extra["decision"] = "代码坏了"
                record(extra, steps=extra_steps, seconds=extra_elapsed, run_key=extra_id)
                break
            if got_seed != seed:
                extra["decision"] = "测量失败"
                record(extra, steps=extra_steps, seconds=extra_elapsed, run_key=extra_id)
                break
            cut = short_run(text, task_py)
            if cut is not None:
                extra["decision"] = "测量失败"
                extra["detail"] = f"只跑了 {cut[0]} 步，预算 {cut[1]} 步"
                record(extra, steps=cut[0], seconds=extra_elapsed, run_key=extra_id)
                break
            extra["val_bpb"] = got_bpb
            extra["decision"] = "已记录"
            record(extra, steps=extra_steps, seconds=extra_elapsed, run_key=extra_id)
            samples.append(float(got_bpb))
    mean = statistics.fmean(samples)
    parent_samples = list(sample_scores(rows, parent_commit).values())
    band = band_for(
        arm_sigma(samples, config["sigma"]),
        arm_sigma(parent_samples, config["sigma"]),
        config["z"],
        len(samples),
        n_parent,
    )
    row["decision"] = compare_to_parent(mean, parent_bpb, band)
    if len(samples) > 1:
        row["detail"] = f"{len(samples)} 次均值 {mean:.6f}"
    record(row, steps=steps, seconds=elapsed)
    return 0


def _latest(rows: list[dict], commit: str, seed: str) -> dict | None:
    found = [row for row in rows if row["commit"] == commit and row["seed"] == seed]
    return found[-1] if found else None


def _bpb(row: dict | None) -> float | None:
    if row is None or row["val_bpb"] == "":
        return None
    return float(row["val_bpb"])


def calibration(scores: list[float], z: float) -> tuple[float, float]:
    sigma = statistics.stdev(scores)
    return sigma, z * sigma


def cmd_calibrate(
    repo: Path, command: list[str], timeout: float, test_root: bool,
    force: bool = False,
) -> int:
    if is_dirty(repo):
        return refuse("工作区不干净")
    if read_run(repo) is not None and not force:
        return refuse(
            "已经标定过了：重新标定会清空父本名单和复核队列，确认请加 --force"
        )
    head = git_output(repo, "rev-parse", "HEAD")
    if head is None:
        return refuse("读不出 HEAD")
    task_py, prepare_py, data_root = eval_paths(repo, test_root)
    identity = eval_hash(task_py, prepare_py, data_root)
    results_path = repo / "results.tsv"
    if identity is None:
        row = blank_row()
        row.update({
            "commit": head, "mechanism": "基线", "expect": "lower",
            "fail": "基线", "seed": "1", "decision": "代码坏了",
            "detail": "数据身份不可用",
        })
        append_result(results_path, row)
        return refuse("数据身份不可用：task.DATA_ROOT 不存在")
    repeats = 5
    z = 2.0
    scores: list[float] = []
    for index in range(repeats):
        seed = str(index + 1)
        row = blank_row()
        row.update({
            "commit": head, "mechanism": "基线", "expect": "lower",
            "fail": "基线", "seed": seed, "eval_hash": identity,
        })
        code = run_command(command, seed, repo / "run.log", timeout)
        log_path = repo / "run.log"
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        printed_seed, printed_bpb = parse_summary(log_text)
        if code != 0 or printed_bpb is None:
            row["decision"] = "代码坏了"
            append_result(results_path, row)
            return refuse(f"标定第 {seed} 次训练崩了，整次标定作废")
        if printed_seed != seed:
            row["decision"] = "测量失败"
            append_result(results_path, row)
            return refuse(f"标定第 {seed} 次打印的种子不是 {seed}，整次标定作废")
        cut = short_run(log_text, task_py)
        if cut is not None:
            row["decision"] = "测量失败"
            row["detail"] = f"只跑了 {cut[0]} 步，预算 {cut[1]} 步"
            append_result(results_path, row)
            return refuse(
                f"标定第 {seed} 次只跑了 {cut[0]}/{cut[1]} 步，整次标定作废"
            )
        row["val_bpb"] = printed_bpb
        row["decision"] = "基线"
        append_result(results_path, row)
        scores.append(float(printed_bpb))
    sigma, min_improvement = calibration(scores, z)
    if sigma <= 0.0:
        return refuse("五次基线分数完全相同，测不出噪声；这个任务的判定会退化成精确比较")
    tag = git_output(repo, "rev-parse", "--abbrev-ref", "HEAD") or ""
    write_run(repo, {
        "run_tag": tag.rsplit("/", 1)[-1],
        "run_start": datetime.datetime.now().astimezone().isoformat(),
        "baseline_commit": head,
        "eval_hash": identity,
        "repeats": repeats,
        "z": z,
        "sigma": sigma,
        "min_improvement": min_improvement,
        "queue_limit": 2,
        "rotate_after": 8,
        "trial_repeats": 3,
    })
    mean = statistics.fmean(scores)
    write_parents(repo / "parents.tsv", [{
        "commit": head,
        "mechanism": "基线",
        "val_bpb": f"{mean:.6f}",
        "seeds": ",".join(str(index + 1) for index in range(repeats)),
        "reviewed": "yes",
        "simpler": "no",
    }])
    write_queue(repo / "queue.tsv", [])
    write_report(repo)
    return 0


def cmd_accept(
    repo: Path, simpler: bool, replace_commit: str | None, test_root: bool
) -> int:
    if is_dirty(repo):
        return refuse("工作区不干净")
    config = read_run(repo)
    if config is None:
        return refuse("还没标定：先运行 judge.py calibrate")
    if ruler_moved(repo, config, test_root):
        return refuse("尺子变了：需重新 calibrate")
    head = git_output(repo, "rev-parse", "HEAD")
    if head is None:
        return refuse("读不出 HEAD")
    seed1 = _latest(read_results(repo / "results.tsv"), head, "1")
    if seed1 is None:
        return refuse("这个提交没有种子 1 记录，不能 accept")
    if simpler:
        if seed1["decision"] != "分不开" or not replace_commit:
            return refuse("--simpler 只接受判定为「分不开」且指定了 --replace 的提交")
        if replace_commit != seed1["parent"]:
            return refuse("--replace 只能指名这个提交自己的父本")
    elif seed1["decision"] != "好于父本":
        return refuse(f"判定是「{seed1['decision']}」，只有「好于父本」才能 accept")
    measured = sample_scores(read_results(repo / "results.tsv"), head)
    basis = statistics.fmean(measured.values()) if measured else float(seed1["val_bpb"])
    newcomer = {
        "commit": head,
        "mechanism": seed1["mechanism"],
        "val_bpb": f"{basis:.6f}",
        "seeds": ",".join(sorted(measured, key=int)) or "1",
        "reviewed": "no",
        "simpler": "yes" if simpler else "no",
    }
    try:
        parents, decision, detail = apply_accept(
            read_parents(repo / "parents.tsv"),
            newcomer,
            replace_commit if simpler else None,
        )
    except ValueError:
        return refuse("--replace 指定的提交不在名单里")
    result = blank_row()
    result.update({
        "commit": head,
        "parent": seed1["parent"],
        "mechanism": seed1["mechanism"],
        "expect": seed1["expect"],
        "fail": seed1["fail"],
        "seed": "*",
        "knobs": seed1["knobs"],
        "eval_hash": seed1["eval_hash"],
        "decision": decision,
        "detail": detail,
    })
    record_accept(
        repo / "results.tsv",
        repo / "parents.tsv",
        result,
        None if decision == "未留下" else parents,
    )
    if decision != "未留下":
        write_queue(repo / "queue.tsv", enqueue(read_queue(repo / "queue.tsv"), {
            "commit": head,
            "parent": seed1["parent"],
            "mechanism": seed1["mechanism"],
            "accepted_at": datetime.datetime.now().astimezone().isoformat(),
            "state": "待复核",
        }))
    write_report(repo)
    return 0


def cmd_claim(repo: Path, test_root: bool) -> int:
    if is_dirty(repo):
        return refuse("工作区不干净")
    config = read_run(repo)
    if config is None:
        return refuse("还没标定：先运行 judge.py calibrate")
    if ruler_moved(repo, config, test_root):
        return refuse("尺子变了：需重新 calibrate")
    head = git_output(repo, "rev-parse", "HEAD")
    if head is None:
        return refuse("读不出 HEAD")
    parent = git_output(repo, "rev-parse", "HEAD^")
    rows = read_results(repo / "results.tsv")
    result = blank_row()
    result.update({"commit": head, "parent": parent or "", "seed": "*"})
    if parent is None:
        result["decision"] = "还判不了"
        result["detail"] = "缺父本提交"
        append_result(repo / "results.tsv", result)
        return 0
    candidate = sample_scores(rows, head)
    parent_samples = sample_scores(rows, parent)
    seed1 = _latest(rows, head, "1")
    if seed1 is not None:
        result["mechanism"] = seed1["mechanism"]
        result["expect"] = seed1["expect"]
        result["fail"] = seed1["fail"]
    used = [
        row for row in rows
        if row["commit"] in (head, parent)
        and row["val_bpb"] != ""
        and row["decision"] in COMPLETED_DECISIONS
    ]
    decision, detail = claim_verdict(
        [candidate[key] for key in sorted(candidate, key=int)],
        [parent_samples[key] for key in sorted(parent_samples, key=int)],
        [row["eval_hash"] for row in used],
        config["sigma"],
        config["z"],
        config.get("min_improvement", 0.0),
        count_n(rows),
    )
    result["decision"] = decision
    if decision in FINAL_CLAIMS:
        detail = f"候选 {len(candidate)} 次 vs 父本 {len(parent_samples)} 次"
    result["detail"] = detail
    append_result(repo / "results.tsv", result)
    if decision in FINAL_CLAIMS:
        mean = statistics.fmean(candidate.values())
        seeds = ",".join(sorted(candidate, key=int))
        write_parents(
            repo / "parents.tsv",
            apply_claim(
                read_parents(repo / "parents.tsv"), head, decision, mean, seeds
            ),
        )
        write_queue(repo / "queue.tsv", dequeue(read_queue(repo / "queue.tsv"), head))
    write_report(repo)
    return 0


def resolve_commit(repo: Path, ref: str, rows: list[dict]) -> str | None:
    """先把 ref 当 git 引用解析；解析不了再按结果表里的前缀唯一匹配。"""
    resolved = git_output(repo, "rev-parse", "--verify", ref + "^{commit}")
    if resolved:
        return resolved
    matches = sorted({row["commit"] for row in rows if row["commit"].startswith(ref)})
    return matches[0] if len(matches) == 1 else None


def paired_compare(rows: list[dict], base: str, treat: str, mde: float) -> dict:
    """两臂在共享种子上的配对检验。

    与 accept 用的独立噪声带不是一回事：配对能消掉「这一对共享的难度」，
    在种子数相同的情况下更灵敏；代价是要求两臂测过同一批种子。
    判据三条，缺一不可：同一把尺子、至少 2 对、p < ALPHA 且 |Δ| ≥ MDE。
    """
    base_samples = sample_scores(rows, base)
    treat_samples = sample_scores(rows, treat)
    shared = sorted(set(base_samples) & set(treat_samples), key=int)
    base_hashes = {
        row["eval_hash"] for row in rows
        if row["commit"] == base and row["seed"] in base_samples and row["eval_hash"]
    }
    treat_hashes = {
        row["eval_hash"] for row in rows
        if row["commit"] == treat and row["seed"] in treat_samples and row["eval_hash"]
    }
    result = {
        "base": base,
        "treat": treat,
        "base_samples": len(base_samples),
        "treat_samples": len(treat_samples),
        "pairs": len(shared),
        "seeds": shared,
        "mde": mde,
        "alpha": ALPHA,
        "mean_delta": None,
        "p_value": None,
        "pair_deltas": {},
        "verdict": "inconclusive",
        "reason": "",
    }
    if not base_hashes or not treat_hashes:
        result["verdict"] = "not_comparable"
        result["reason"] = "有一臂没有带尺子身份（eval_hash）的完成记录"
        return result
    if base_hashes != treat_hashes:
        result["verdict"] = "not_comparable"
        result["reason"] = "两臂的 eval_hash 不同：在不同考卷上测的，不能配对"
        return result
    if len(shared) < 2:
        result["reason"] = f"只有 {len(shared)} 对共享种子；配对检验至少需要 2 对"
        return result
    deltas = [treat_samples[seed] - base_samples[seed] for seed in shared]
    result["pair_deltas"] = {seed: deltas[index] for index, seed in enumerate(shared)}
    result["mean_delta"] = statistics.fmean(deltas)
    p_value = paired_p_value(deltas)
    result["p_value"] = p_value
    if p_value is None:
        result["reason"] = "配对差值方差为 0，t 检验无定义"
        return result
    if p_value >= ALPHA:
        result["reason"] = f"p={p_value:.5f} ≥ {ALPHA}"
        return result
    if abs(result["mean_delta"]) < mde:
        result["reason"] = (
            f"统计显著但 |Δ|={abs(result['mean_delta']):.6f} < MDE={mde:.6f}："
            "大到能被看见，还没大到人关心"
        )
        return result
    result["verdict"] = "significant"
    result["direction"] = "better" if result["mean_delta"] < 0 else "worse"
    result["reason"] = f"p={p_value:.5f} < {ALPHA} 且 |Δ| ≥ MDE"
    return result


def compute_ratio(repo: Path, base: str, treat: str) -> float | None:
    """两臂实测步数的比值；只在两边都有数字时给，缺一边就返回 None。

    这是 expctl V5-84 的算力轴：Δ 里可能混着「多跑了步数」的贡献。
    本例是软警告，不设阈值门——「差几倍算超」推不出公理，硬拦会改判历史。
    """
    events, _ = read_ledger(repo)
    steps: dict[str, list[int]] = {base: [], treat: []}
    for event in events:
        if event.get("event") != "result":
            continue
        if event.get("decision") not in COMPLETED_DECISIONS:
            continue
        commit = event.get("commit")
        value = event.get("steps")
        if commit in steps and isinstance(value, int) and value > 0:
            steps[commit].append(value)
    if not steps[base] or not steps[treat]:
        return None
    low = min(statistics.median(steps[base]), statistics.median(steps[treat]))
    high = max(statistics.median(steps[base]), statistics.median(steps[treat]))
    if low <= 0:
        return None
    return high / low


def cmd_compare(
    repo: Path, base_ref: str, treat_ref: str, mde_override: float | None,
    as_json: bool,
) -> int:
    rows = read_results(repo / "results.tsv")
    base = resolve_commit(repo, base_ref, rows)
    treat = resolve_commit(repo, treat_ref, rows)
    if base is None or treat is None:
        return refuse(f"解析不了提交：{base_ref if base is None else treat_ref}")
    config = read_run(repo)
    if mde_override is not None:
        mde = mde_override
    elif config is not None:
        mde = float(config.get("min_improvement", 0.0))
    else:
        mde = 0.0
    result = paired_compare(rows, base, treat, mde)
    result["compute_ratio"] = compute_ratio(repo, base, treat)
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"=== compare: {base[:12]} vs {treat[:12]} ===")
    print(f"Metric: val_bpb (lower is better)   MDE: {mde:.6f}")
    print(f"Pairs: {result['pairs']}   shared seeds: {','.join(result['seeds']) or '（无）'}")
    if result["mean_delta"] is not None:
        print(f"Mean Delta: {result['mean_delta']:+.6f}")
    if result["p_value"] is not None:
        print(f"p-value: {result['p_value']:.5f}   alpha: {ALPHA}")
    if result["compute_ratio"] is not None:
        line = f"Compute ratio (steps): {result['compute_ratio']:.2f}x"
        if result["compute_ratio"] >= 2.0:
            line += "  警告：两臂算力相差 ≥2x，Δ 里混着预算差"
        print(line)
    print(f"Verdict: {result['verdict'].upper()}")
    if result["reason"]:
        print(f"Reason: {result['reason']}")
    return 0


def build_audit(repo: Path) -> dict:
    rows = read_results(repo / "results.tsv")
    events, bad_lines = read_ledger(repo)
    parents = read_parents(repo / "parents.tsv")
    queue = read_queue(repo / "queue.tsv")
    config = read_run(repo)
    status = ledger_status(events)
    errors: list[str] = []
    warnings: list[str] = []

    if bad_lines:
        errors.append(f"ledger.jsonl 有 {bad_lines} 行不是合法 JSON 对象")

    search_rows = [row for row in rows if row["seed"] not in ("", "*")]
    completed = [
        row for row in search_rows if row["decision"] in COMPLETED_DECISIONS
    ]

    for row in completed:
        if row["eval_hash"] == "":
            errors.append(
                f"{row['commit'][:7]} seed {row['seed']} 有完成记录但没有 eval_hash"
            )

    seen: dict[tuple, int] = {}
    for row in completed:
        key = (row["commit"], row["eval_hash"], row["seed"], row["knobs"])
        seen[key] = seen.get(key, 0) + 1
    for key, count in seen.items():
        if count > 1:
            errors.append(
                f"同一 run 身份完成了 {count} 次：{key[0][:7]} seed {key[2]}"
                f"（knobs={key[3] or '无'}）"
            )

    admitted = {
        (
            event.get("commit"), event.get("seed"),
            event.get("eval_hash"), event.get("knobs", ""),
        )
        for event in events
        if event.get("event") == "admit"
    }
    if not events:
        if completed:
            warnings.append(
                f"{len(completed)} 条完成记录早于账本：没有 append-only 的准入证据"
            )
    else:
        legacy = [
            row for row in completed
            if row["mechanism"] != "基线"
            and (row["commit"], row["seed"], row["eval_hash"], row["knobs"]) not in admitted
        ]
        if legacy:
            warnings.append(
                f"{len(legacy)} 条完成记录没有对应的准入事件（可能是账本启用前的历史）"
            )
        open_runs = [
            run_id for run_id, slot in status.items()
            if slot["admitted"] and not slot["completed"]
        ]
        if open_runs:
            warnings.append(
                f"{len(open_runs)} 条准入没有完成记录：崩溃、中断，或正在运行"
            )

    for parent in parents:
        if parent.get("reviewed") != "yes":
            warnings.append(
                f"父本 {parent['commit'][:7]}（{parent['mechanism']}）尚未复核"
            )
        seed1 = next(
            (
                row for row in search_rows
                if row["commit"] == parent["commit"] and row["seed"] == "1"
            ),
            None,
        )
        if seed1 is not None and "," in seed1["knobs"]:
            warnings.append(
                f"父本 {parent['commit'][:7]} 的机制「{seed1['mechanism']}」"
                f"一次改了多个旋钮（{seed1['knobs']}）：单变量归因不成立"
            )

    parent_commits = {parent["commit"] for parent in parents}
    row_commits = {row["commit"] for row in rows}
    claimed = {row["commit"] for row in rows if row["decision"] in FINAL_CLAIMS}
    for entry in queue:
        commit = entry["commit"]
        if commit not in parent_commits and commit not in row_commits:
            warnings.append(f"队列里的 {commit[:7]} 在结果和名单里都不存在")
        elif commit in claimed:
            warnings.append(f"队列里的 {commit[:7]} 已经 claim 过，应该出队")

    if config is not None:
        other_rulers = {
            row["eval_hash"] for row in completed
            if row["eval_hash"] and row["eval_hash"] != config["eval_hash"]
        }
        if other_rulers:
            warnings.append(
                f"{len(other_rulers)} 把旧尺子下的完成记录与当前 run.json 不同："
                "比较时必须按 eval_hash 分组"
            )

    counts = {
        "results_rows": len(rows),
        "search_rows": len(search_rows),
        "completed_rows": len(completed),
        "ledger_events": len(events),
        "admissions": sum(1 for e in events if e.get("event") == "admit"),
        "ledger_results": sum(1 for e in events if e.get("event") == "result"),
        "parents": len(parents),
        "queue": len(queue),
    }
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": counts,
    }


def cmd_audit(repo: Path, as_json: bool) -> int:
    report = build_audit(repo)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("=== audit ===")
        print(f"ok: {report['ok']}")
        for name, value in report["counts"].items():
            print(f"  {name}: {value}")
        for label, items in (("错误", report["errors"]), ("警告", report["warnings"])):
            if not items:
                print(f"{label}: 0")
                continue
            print(f"{label}: {len(items)}")
            for item in items:
                print(f"- {item}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
