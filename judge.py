"""Record seed-1 search runs and write a separate morning claim."""

from __future__ import annotations

import argparse
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
from pathlib import Path

TIMEOUT_SECONDS = 600

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


def band_for(sigma: float, z: float, n: int) -> float:
    count = n if n >= 1 else 1
    return z * sigma * math.sqrt(1.0 + 1.0 / count)

COUNTED_DECISIONS = frozenset({
    "好于父本",
    "分不开",
    "差于父本",
    "已记录",
    "代码坏了",
    "测量失败",
})


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
    eval_hashes: list[str],
    min_improvement: float | None,
    n: int,
) -> tuple[str, str]:
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
    if (
        len(eval_hashes) != 4
        or len(set(eval_hashes)) != 1
        or any(item == "" for item in eval_hashes)
    ):
        return "还判不了", "尺子或数据不同"
    if min_improvement is None or min_improvement <= 0:
        return "还判不了", "未标定"
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
    parents: list[dict], commit: str, decision: str, mean: float
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
            fresh["seeds"] = "1,2"
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
            f"- band（新父本）= {band_for(config['sigma'], config['z'], 1):.6f}"
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
        if row["seed"] not in {"1", "2"} or row["mechanism"] == "基线":
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
    code = run_command(train, args.seed, repo / "run.log", timeout)
    log_text = (repo / "run.log").read_text(encoding="utf-8") if (repo / "run.log").exists() else ""
    printed_seed, printed_bpb = parse_summary(log_text)
    if code != 0 or printed_bpb is None:
        row["decision"] = "代码坏了"
        append_result(results_path, row)
        return 0
    if printed_seed != args.seed:
        row["decision"] = "测量失败"
        append_result(results_path, row)
        return 0
    row["val_bpb"] = printed_bpb
    if args.seed == "2":
        row["decision"] = "已记录"
        append_result(results_path, row)
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
    band = band_for(config["sigma"], config["z"], parent_n(parent_listed or {}))
    row["decision"] = compare_to_parent(float(printed_bpb), parent_bpb, band)
    append_result(results_path, row)
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
    newcomer = {
        "commit": head,
        "mechanism": seed1["mechanism"],
        "val_bpb": seed1["val_bpb"],
        "seeds": "1",
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
    slots = [
        _latest(rows, head, "1"),
        _latest(rows, head, "2"),
        _latest(rows, parent, "1"),
        _latest(rows, parent, "2"),
    ]
    seed1 = slots[0]
    if seed1 is not None:
        result["mechanism"] = seed1["mechanism"]
        result["expect"] = seed1["expect"]
        result["fail"] = seed1["fail"]
    decision, detail = claim_verdict(
        _bpb(slots[0]),
        _bpb(slots[1]),
        _bpb(slots[2]),
        _bpb(slots[3]),
        [row["eval_hash"] if row else "" for row in slots],
        config["min_improvement"],
        count_n(rows),
    )
    result["decision"] = decision
    result["detail"] = detail
    append_result(repo / "results.tsv", result)
    if decision in FINAL_CLAIMS:
        candidate = [_bpb(slots[0]), _bpb(slots[1])]
        mean = statistics.fmean([item for item in candidate if item is not None])
        write_parents(
            repo / "parents.tsv",
            apply_claim(read_parents(repo / "parents.tsv"), head, decision, mean),
        )
        write_queue(repo / "queue.tsv", dequeue(read_queue(repo / "queue.tsv"), head))
    write_report(repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
