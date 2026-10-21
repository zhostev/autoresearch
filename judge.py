"""Record seed-1 search runs and write a separate morning claim."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
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


RESULT_HEADER = (
    "commit\tparent\tmechanism\tknobs\texpect\tfail\tseed\tval_bpb\t"
    "eval_hash\tdecision\tdetail"
)
PARENT_HEADER = "commit\tmechanism\tval_bpb\tseeds\treviewed\tsimpler"
RESULT_FIELDS = RESULT_HEADER.split("\t")
PARENT_FIELDS = PARENT_HEADER.split("\t")


def _line(fields: list[str], row: dict) -> str:
    return "\t".join(row[field] for field in fields)


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[1:]:
        if line == "":
            continue
        parts = line.split("\t")
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
    for name in ("calibrate", "claim"):
        sub.add_parser(name)
    accept_parser = sub.add_parser("accept")
    accept_parser.add_argument("--simpler", action="store_true")
    accept_parser.add_argument("--replace", default=None)
    trial = sub.add_parser("trial")
    trial.add_argument("--seed", required=True)
    trial.add_argument("--mechanism", required=True)
    trial.add_argument("--expect", required=True)
    trial.add_argument("--fail", required=True)
    args = parser.parse_args(argv)
    repo = root or Path.cwd()
    if args.cmd == "calibrate":
        return cmd_calibrate(
            repo,
            command or ["uv", "run", "train.py"],
            TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds,
            root is not None,
        )
    if args.cmd == "accept":
        return cmd_accept(repo, args.simpler, args.replace)
    if args.cmd == "claim":
        return cmd_claim(repo)
    if args.cmd != "trial":
        return 2
    timeout = TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    train = command or ["uv", "run", "train.py"]
    error = validate_text(args.seed, args.mechanism, args.expect, args.fail)
    if error is not None or is_dirty(repo):
        return 2
    config = read_run(repo)
    if config is None:
        return 2
    head = git_output(repo, "rev-parse", "HEAD")
    if head is None:
        return 2
    results_path = repo / "results.tsv"
    rows = read_results(results_path)
    if any(row["commit"] == head and row["seed"] == args.seed for row in rows):
        return 2
    if args.seed == "2":
        seed1 = [
            row for row in rows
            if row["commit"] == head and row["seed"] == "1"
            and row["decision"] in {"基线", "好于父本", "持平", "差于父本"}
        ]
        if not seed1 or seed1[-1]["mechanism"] != args.mechanism:
            return 2
    else:
        parent = git_output(repo, "rev-parse", "HEAD^")
        listed = {row["commit"] for row in read_parents(repo / "parents.tsv")}
        if parent is None or parent not in listed:
            return 2
        parent_scores = [
            item for item in rows
            if item["commit"] == parent and item["seed"] == "1" and item["val_bpb"] != ""
        ]
        if not parent_scores:
            return 2
    task_py, prepare_py, data_root = eval_paths(repo, root is not None)
    identity = eval_hash(task_py, prepare_py, data_root)
    row = blank_row()
    row.update({
        "commit": head,
        "parent": git_output(repo, "rev-parse", "HEAD^") or "",
        "mechanism": args.mechanism,
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
    repo: Path, command: list[str], timeout: float, test_root: bool
) -> int:
    if is_dirty(repo):
        return 2
    head = git_output(repo, "rev-parse", "HEAD")
    if head is None:
        return 2
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
        return 2
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
            return 2
        if printed_seed != seed:
            row["decision"] = "测量失败"
            append_result(results_path, row)
            return 2
        row["val_bpb"] = printed_bpb
        row["decision"] = "基线"
        append_result(results_path, row)
        scores.append(float(printed_bpb))
    sigma, min_improvement = calibration(scores, z)
    if sigma <= 0.0:
        return 2
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
    return 0


def cmd_accept(repo: Path, simpler: bool, replace_commit: str | None) -> int:
    if is_dirty(repo):
        return 2
    head = git_output(repo, "rev-parse", "HEAD")
    if head is None:
        return 2
    seed1 = _latest(read_results(repo / "results.tsv"), head, "1")
    if seed1 is None:
        return 2
    if simpler:
        if seed1["decision"] != "分不开" or not replace_commit:
            return 2
    elif seed1["decision"] != "好于父本":
        return 2
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
        return 2
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
    return 0


def cmd_claim(repo: Path) -> int:
    if is_dirty(repo):
        return 2
    head = git_output(repo, "rev-parse", "HEAD")
    if head is None:
        return 2
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
        [row["eval_hash"] if row else "" for row in slots],
        (read_run(repo) or {}).get("min_improvement"),
        count_n(rows),
    )
    result["decision"] = decision
    result["detail"] = detail
    append_result(repo / "results.tsv", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
