# Search Parents and Validated Claims Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an in-repo judge that records seed-1 search attempts, keeps at most three parents, and writes a separate morning claim.

**Architecture:** `judge.py` is a standard-library program. Pure functions decide seed-1 comparisons, the luck penalty, claim words, and which parent to drop. The CLI appends `results.tsv` and only then replaces `parents.tsv`. `prepare.py` grows one seed helper; `train.py` calls it; `program.md` tells the agent the new loop. Nothing calls expctl or Praxist.

**Tech Stack:** Python 3.10 standard library, existing `unittest`, existing PyTorch already imported by `train.py` and `prepare.py`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-19-search-claims-design.md`

## Global Constraints

- Training wall clock stays `TIME_BUDGET = 300` in `prepare.py`. Evaluation stays `prepare.evaluate_bpb`.
- One GPU, one experiment at a time.
- After the loop starts, the agent edits only `train.py`.
- Do not add dependencies. Tests use the standard library.
- A run that exceeds 10 minutes is killed. `judge.TIMEOUT_SECONDS` is `600`.
- Experiments stay on a branch such as `autoresearch/<tag>`.
- `results.tsv` is not committed. `parents.tsv` and `run.log` are not committed.
- Do not import or shell out to expctl or Praxist.
- Do not add a t-test or a p-value gate.
- `judge.py` does not run `git commit`, `git reset`, or `git checkout`.
- Seeds are only `1` and `2`. `val_bpb` lower is better. `--expect` is stored and does not flip that direction.
- `MIN_IMPROVEMENT` ships as `None`. A missing or non-positive value yields `还判不了` / `最小差距未写`.
- A command that writes a row exits `0`. A command that refuses before writing a row exits `2`.
- Decision words are exactly the strings in the spec. Do not translate them.

## Review Focus

- Real training prints `val_bpb:` with wide padding (`val_bpb:          0.997900`). The parser must read that number; a strict `split()` that keeps the spaces glued to the value would drop a real run. Pinned in Task 4.
- `accept --simpler --replace` of a hash that is not in `parents.tsv` must exit `2`, leave the parent file unchanged, and append no results row. Pinned in Task 6.
- `trial --seed 2` whose `--mechanism` differs from that commit's seed-1 name must exit `2` and append no row. Pinned in Task 5.
- `claim` on a present row whose `val_bpb` cell is empty must write `还判不了` and name the missing score, not raise `ValueError`. Pinned in Task 6.
- `MIN_IMPROVEMENT <= 0` must be treated as unset. A zero gap must not be able to mint `成立`. Pinned in Task 1.

## File structure

- `judge.py` — decisions, TSV IO, data identity, and the CLI. One responsibility: the experiment protocol.
- `tests/test_judge.py` — unittest coverage. No GPU and no five-minute training.
- `tests/__init__.py` — empty package marker.
- `prepare.py` — add `seed_everything()` only.
- `train.py` — replace the hardcoded seed `42` with `seed_everything()`.
- `program.md` — agent loop.
- `.gitignore` — add `parents.tsv` and `run.log`.

---

### Task 1: Claim and seed-1 decisions

**Files:**
- Create: `judge.py`
- Create: `tests/__init__.py`
- Create: `tests/test_judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `MIN_IMPROVEMENT: float | None = None`
  - `TIMEOUT_SECONDS: int = 600`
  - `COUNTED_DECISIONS: frozenset[str]` containing `好于父本`, `持平`, `差于父本`, `已记录`, `代码坏了`, `测量失败`
  - `compare_seed1(candidate: float, parent: float) -> str`
  - `luck_penalty(candidate_a: float, candidate_b: float, n: int) -> float`
  - `count_n(rows: list[dict]) -> int` using each row's `decision` and `mechanism`
  - `claim_verdict(candidate_seed1: float | None, candidate_seed2: float | None, parent_seed1: float | None, parent_seed2: float | None, prepare_hashes: list[str], data_hashes: list[str], min_improvement: float | None, n: int) -> tuple[str, str]`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` as an empty file. Create `tests/test_judge.py`:

```python
import unittest

import judge


class TestClaim(unittest.TestCase):
    def test_seed1_directions(self):
        self.assertEqual(judge.compare_seed1(0.9, 1.0), "好于父本")
        self.assertEqual(judge.compare_seed1(1.0, 1.0), "持平")
        self.assertEqual(judge.compare_seed1(1.1, 1.0), "差于父本")

    def test_timeout_constant(self):
        self.assertEqual(judge.TIMEOUT_SECONDS, 600)
        self.assertIsNone(judge.MIN_IMPROVEMENT)

    def test_missing_inputs_have_distinct_details(self):
        hashes = ["a", "a", "a", "a"]
        cases = [
            (None, 0.9, 1.0, 1.0, "缺候选种子1"),
            (0.9, None, 1.0, 1.0, "缺候选种子2"),
            (0.9, 0.9, None, 1.0, "缺父本种子1"),
            (0.9, 0.9, 1.0, None, "缺父本种子2"),
        ]
        for scores in cases:
            decision, detail = judge.claim_verdict(
                *scores[:4], hashes, hashes, 0.001, 1
            )
            self.assertEqual(decision, "还判不了")
            self.assertEqual(detail, scores[4])
        decision, detail = judge.claim_verdict(
            0.9, 0.9, 1.0, 1.0, hashes, hashes, None, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "最小差距未写"))
        decision, detail = judge.claim_verdict(
            0.9, 0.9, 1.0, 1.0, hashes, hashes, 0.0, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "最小差距未写"))
        decision, detail = judge.claim_verdict(
            0.9, 0.9, 1.0, 1.0, hashes, hashes, -0.1, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "最小差距未写"))
        decision, detail = judge.claim_verdict(
            0.9, 0.9, 1.0, 1.0, ["a", "b", "a", "a"], hashes, 0.001, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "尺子或数据不同"))
        decision, detail = judge.claim_verdict(
            0.99, 0.99, 1.0, 1.0, hashes, hashes, 0.001, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "两次改进没有分开"))

    def test_worse_and_tie(self):
        hashes = ["a", "a", "a", "a"]
        decision, detail = judge.claim_verdict(
            1.1, 1.2, 1.0, 1.0, hashes, hashes, 0.001, 1
        )
        self.assertEqual(decision, "更差")
        decision, detail = judge.claim_verdict(
            1.0, 1.0, 1.0, 1.0, hashes, hashes, 0.001, 1
        )
        self.assertEqual(decision, "没有更好")

    def test_penalty_example(self):
        rows = [
            {"decision": "好于父本", "mechanism": "学习率"},
            {"decision": "差于父本", "mechanism": "深度"},
            {"decision": "代码坏了", "mechanism": "窗口"},
            {"decision": "已记录", "mechanism": "学习率"},
            {"decision": "基线", "mechanism": "基线"},
            {"decision": "进入名单", "mechanism": "优化器"},
        ]
        self.assertEqual(judge.count_n(rows), 4)
        penalty = judge.luck_penalty(0.990, 0.996, 4)
        self.assertAlmostEqual(penalty, 0.007064460135092854, delta=1e-12)
        hashes = ["h", "h", "h", "h"]
        self.assertEqual(
            judge.claim_verdict(0.990, 0.996, 1.000, 1.002, hashes, hashes, 0.001, 4)[0],
            "没有更好",
        )
        self.assertEqual(
            judge.claim_verdict(0.990, 0.996, 1.000, 1.002, hashes, hashes, 0.0005, 4)[0],
            "成立",
        )
        self.assertEqual(judge.luck_penalty(0.5, 0.6, 1), 0.0)
        self.assertEqual(judge.count_n([{"decision": "基线", "mechanism": "基线"}]), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_judge.TestClaim -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'judge'`

- [ ] **Step 3: Write the minimal implementation**

Create `judge.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_judge.TestClaim -v`

Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add judge.py tests/__init__.py tests/test_judge.py
git commit -m "Add claim and seed-1 decision functions."
```

---

### Task 2: Parent list updates

**Files:**
- Modify: `judge.py`
- Modify: `tests/test_judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: nothing from Task 1 except the same module
- Produces: `apply_accept(parents: list[dict], newcomer: dict, replace_commit: str | None) -> tuple[list[dict], str, str]`
  - Parent dict keys: `commit`, `mechanism`, `val_bpb`, `simpler`
  - `val_bpb` is a `str` of the printed number
  - Return is `(new_parents, decision, detail)`
  - `replace_commit is None` means a normal accept
  - Raises `ValueError` with message `父本不在名单` when the replace target is absent

- [ ] **Step 1: Write the failing test**

Append to `tests/test_judge.py`:

```python
class TestParents(unittest.TestCase):
    def _row(self, commit, mechanism, val_bpb, simpler="no"):
        return {
            "commit": commit,
            "mechanism": mechanism,
            "val_bpb": val_bpb,
            "simpler": simpler,
        }

    def test_drops_worse_duplicate_mechanism(self):
        parents = [
            self._row("a", "学习率", "0.500000"),
            self._row("b", "学习率", "0.900000"),
            self._row("c", "深度", "0.400000"),
        ]
        newcomer = self._row("d", "窗口", "0.600000")
        kept, decision, detail = judge.apply_accept(parents, newcomer, None)
        self.assertEqual(decision, "进入名单")
        self.assertEqual(detail, "b")
        self.assertEqual([row["commit"] for row in kept], ["a", "c", "d"])

    def test_drops_worst_when_mechanisms_differ(self):
        parents = [
            self._row("a", "学习率", "0.900000"),
            self._row("b", "深度", "0.800000"),
            self._row("c", "窗口", "0.700000"),
        ]
        newcomer = self._row("d", "优化器", "0.600000")
        kept, decision, detail = judge.apply_accept(parents, newcomer, None)
        self.assertEqual(decision, "进入名单")
        self.assertEqual(detail, "a")
        self.assertEqual({row["commit"] for row in kept}, {"b", "c", "d"})

    def test_rejects_newcomer_when_it_is_the_one_to_drop(self):
        parents = [
            self._row("a", "学习率", "0.900000"),
            self._row("b", "深度", "0.800000"),
            self._row("c", "窗口", "0.700000"),
        ]
        newcomer = self._row("d", "优化器", "1.100000")
        kept, decision, detail = judge.apply_accept(parents, newcomer, None)
        self.assertEqual(decision, "未留下")
        self.assertEqual(detail, "名单已满")
        self.assertEqual(kept, parents)

    def test_tie_keeps_the_newcomer(self):
        parents = [
            self._row("aaaa", "学习率", "1.000000"),
            self._row("bbbb", "学习率", "1.000000"),
            self._row("cccc", "深度", "0.500000"),
        ]
        newcomer = self._row("dddd", "窗口", "0.900000")
        kept, decision, detail = judge.apply_accept(parents, newcomer, None)
        self.assertEqual(detail, "bbbb")
        self.assertIn("dddd", [row["commit"] for row in kept])
        self.assertNotIn("bbbb", [row["commit"] for row in kept])

    def test_simpler_replaces_named_parent(self):
        parents = [self._row("a", "学习率", "1.000000")]
        newcomer = self._row("b", "学习率", "1.000000", simpler="yes")
        kept, decision, detail = judge.apply_accept(parents, newcomer, "a")
        self.assertEqual(decision, "替换父本")
        self.assertEqual(detail, "a")
        self.assertEqual(kept, [newcomer])

    def test_missing_replace_target_raises(self):
        parents = [self._row("a", "学习率", "1.000000")]
        newcomer = self._row("b", "深度", "1.000000", simpler="yes")
        with self.assertRaisesRegex(ValueError, "父本不在名单"):
            judge.apply_accept(parents, newcomer, "missing")

    def test_already_listed_does_not_duplicate(self):
        parents = [self._row("a", "学习率", "1.000000")]
        kept, decision, detail = judge.apply_accept(parents, parents[0], None)
        self.assertEqual(decision, "未留下")
        self.assertEqual(detail, "已在名单")
        self.assertEqual(kept, parents)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_judge.TestParents -v`

Expected: FAIL with `AttributeError: module 'judge' has no attribute 'apply_accept'`

- [ ] **Step 3: Write the minimal implementation**

Append to `judge.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_judge.TestParents -v`

Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add judge.py tests/test_judge.py
git commit -m "Add the three-parent accept rules."
```

---

### Task 3: Append-only results and parent replacement

**Files:**
- Modify: `judge.py`
- Modify: `tests/test_judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: parent dicts from `apply_accept`
- Produces:
  - `RESULT_HEADER: str` equal to `commit	parent	mechanism	expect	fail	seed	val_bpb	prepare_hash	data_hash	decision	detail`
  - `PARENT_HEADER: str` equal to `commit	mechanism	val_bpb	simpler`
  - `read_results(path: Path) -> list[dict]`
  - `append_result(path: Path, row: dict) -> None` flushes and fsyncs
  - `read_parents(path: Path) -> list[dict]`
  - `write_parents(path: Path, rows: list[dict]) -> None` writes a temp file in the same directory, fsyncs, then `os.replace`
  - `record_accept(results_path: Path, parents_path: Path, result_row: dict, new_parents: list[dict] | None) -> None`
    - Always appends the result row first
    - Writes parents only when `new_parents is not None`
    - If the parent write raises, the new result row is already on disk and the exception propagates

- [ ] **Step 1: Write the failing test**

Append to `tests/test_judge.py`:

```python
from pathlib import Path


class TestTsv(unittest.TestCase):
    def test_append_is_stable_and_creates_header(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.tsv"
            row = {name: name for name in judge.RESULT_HEADER.split("\t")}
            row["val_bpb"] = ""
            judge.append_result(path, row)
            judge.append_result(path, row)
            rows = judge.read_results(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(list(rows[0]), judge.RESULT_HEADER.split("\t"))
            self.assertEqual(rows[0]["val_bpb"], "")
            self.assertEqual(path.read_text(encoding="utf-8").splitlines()[0], judge.RESULT_HEADER)

    def test_parent_write_failure_keeps_the_appended_result(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results.tsv"
            parents = root / "parents.tsv"
            parents.mkdir()
            row = {name: "x" for name in judge.RESULT_HEADER.split("\t")}
            new_parents = [{
                "commit": "abc",
                "mechanism": "学习率",
                "val_bpb": "0.900000",
                "simpler": "no",
            }]
            with self.assertRaises(OSError):
                judge.record_accept(results, parents, row, new_parents)
            self.assertEqual(len(judge.read_results(results)), 1)
            self.assertTrue(parents.is_dir())

    def test_record_accept_can_skip_parent_rewrite(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results.tsv"
            parents = root / "parents.tsv"
            row = {name: "x" for name in judge.RESULT_HEADER.split("\t")}
            judge.record_accept(results, parents, row, None)
            self.assertEqual(len(judge.read_results(results)), 1)
            self.assertFalse(parents.exists())
```

`Path` is imported at module level in this test file from here on. Later tasks must not add a second `import os` just for this class.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_judge.TestTsv -v`

Expected: FAIL with `AttributeError: module 'judge' has no attribute 'RESULT_HEADER'`

- [ ] **Step 3: Write the minimal implementation**

Add these imports near the top of `judge.py`: `import os` and `from pathlib import Path`.

Append:

```python
RESULT_HEADER = (
    "commit\tparent\tmechanism\texpect\tfail\tseed\tval_bpb\t"
    "prepare_hash\tdata_hash\tdecision\tdetail"
)
PARENT_HEADER = "commit\tmechanism\tval_bpb\tsimpler"
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_judge.TestTsv tests.test_judge.TestClaim tests.test_judge.TestParents -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add judge.py tests/test_judge.py
git commit -m "Append results before replacing the parent list."
```

---

### Task 4: Training summary and data identity

**Files:**
- Modify: `judge.py`
- Modify: `tests/test_judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `parse_summary(text: str) -> tuple[str | None, str | None]` returning the last `seed` token and the last `val_bpb` token
  - `file_sha256(path: Path) -> str`
  - `data_hash(tokenizer_path: Path, val_path: Path, data_dir: Path) -> str | None`
    - Returns `None` if the tokenizer, the val shard, or `data_dir` is missing
    - Hashes, in order, one line `tokenizer.pkl <size>`, one line `shard_06542.parquet <size>`, then `name <size>` for every `*.parquet` in `data_dir` sorted by name
    - Does not read shard bytes

- [ ] **Step 1: Write the failing test**

Append to `tests/test_judge.py`:

```python
class TestParse(unittest.TestCase):
    def test_padded_val_bpb_line(self):
        text = "seed: 1\n---\nval_bpb:          0.997900\ntraining_seconds: 300.1\n"
        self.assertEqual(judge.parse_summary(text), ("1", "0.997900"))

    def test_last_occurrence_wins(self):
        text = "seed: 0\nval_bpb: 1.000000\nseed: 2\nval_bpb:          0.500000\n"
        self.assertEqual(judge.parse_summary(text), ("2", "0.500000"))

    def test_missing_lines(self):
        self.assertEqual(judge.parse_summary("no summary\n"), (None, None))

    def test_data_hash_uses_sizes_not_bytes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_dir = root / "tokenizer"
            token_dir.mkdir()
            tokenizer = token_dir / "tokenizer.pkl"
            tokenizer.write_bytes(b"abc")
            data = root / "data"
            data.mkdir()
            val = data / "shard_06542.parquet"
            val.write_bytes(b"12345")
            (data / "shard_00001.parquet").write_bytes(b"z")
            first = judge.data_hash(tokenizer, val, data)
            val.write_bytes(b"1234")
            second = judge.data_hash(tokenizer, val, data)
            self.assertIsNotNone(first)
            self.assertNotEqual(first, second)
            self.assertIsNone(judge.data_hash(tokenizer, root / "missing.parquet", data))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_judge.TestParse -v`

Expected: FAIL with `AttributeError: module 'judge' has no attribute 'parse_summary'`

- [ ] **Step 3: Write the minimal implementation**

Add `import hashlib` to `judge.py`. Append:

```python
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


def data_hash(tokenizer_path: Path, val_path: Path, data_dir: Path) -> str | None:
    if not tokenizer_path.is_file() or not val_path.is_file() or not data_dir.is_dir():
        return None
    lines = [
        f"tokenizer.pkl {tokenizer_path.stat().st_size}",
        f"shard_06542.parquet {val_path.stat().st_size}",
    ]
    for path in sorted(data_dir.glob("*.parquet")):
        lines.append(f"{path.name} {path.stat().st_size}")
    digest = hashlib.sha256()
    digest.update(("\n".join(lines) + "\n").encode("utf-8"))
    return digest.hexdigest()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_judge.TestParse -v`

Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add judge.py tests/test_judge.py
git commit -m "Parse training summaries and hash the data manifest."
```

---

### Task 5: Trial command guards

**Files:**
- Modify: `judge.py`
- Modify: `tests/test_judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `parse_summary`, `append_result`, `read_results`, `read_parents`, `file_sha256`, `data_hash`, `compare_seed1`, `RESULT_FIELDS`
- Produces: `main(argv: list[str] | None = None, command: list[str] | None = None, timeout_seconds: float | None = None, root: Path | None = None) -> int`
  - Default `command` is `["uv", "run", "train.py"]`
  - Default `timeout_seconds` is `TIMEOUT_SECONDS`
  - Default `root` is the current directory
  - `trial` is implemented in this task
  - `baseline`, `accept`, and `claim` may exist only as argparse destinations that return `2` until Task 6. Task 6 replaces `main` with the full command set; do not invent partial behavior for them here beyond returning `2` without writing a row

- [ ] **Step 1: Write the failing test**

Append to `tests/test_judge.py`:

Add this helper at module level in `tests/test_judge.py`:

```python
def git_head(root: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
```

When `main` is called with `root=`, data files are `<root>/tokenizer.pkl` and `<root>/data/shard_06542.parquet`. When `root` is omitted, `main` uses `prepare.DATA_DIR`, `prepare.TOKENIZER_DIR`, and `prepare.VAL_FILENAME`. Guard order for seed 1 is: invalid text, dirty tree, duplicate `(HEAD, seed)`, then "git parent is listed in `parents.tsv`".

```python
class TestTrial(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._git("init")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "user.name", "test")
        (self.root / "prepare.py").write_text("TIME_BUDGET = 300\n", encoding="utf-8")
        (self.root / "train.py").write_text("print('train')\n", encoding="utf-8")
        (self.root / "judge.py").write_text("# test copy\n", encoding="utf-8")
        data = self.root / "data"
        data.mkdir()
        (data / "shard_06542.parquet").write_bytes(b"val")
        (data / "shard_00001.parquet").write_bytes(b"train")
        (self.root / "tokenizer.pkl").write_bytes(b"tok")
        self._git("add", "prepare.py", "train.py", "judge.py")
        self._git("commit", "-m", "base")

    def tearDown(self):
        self._tmp.cleanup()

    def _git(self, *args):
        import subprocess

        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def _trial(self, args, command, timeout=5):
        return judge.main(
            ["trial", *args],
            command=command,
            timeout_seconds=timeout,
            root=self.root,
        )

    def _done_row(self, mechanism, seed, decision="好于父本", val="1.000000"):
        return {
            "commit": git_head(self.root),
            "parent": "",
            "mechanism": mechanism,
            "expect": "lower",
            "fail": "崩了",
            "seed": seed,
            "val_bpb": val,
            "prepare_hash": "p",
            "data_hash": "d",
            "decision": decision,
            "detail": "",
        }

    def _prepare_child(self):
        parent = git_head(self.root)
        judge.write_parents(self.root / "parents.tsv", [{
            "commit": parent,
            "mechanism": "基线",
            "val_bpb": "1.000000",
            "simpler": "no",
        }])
        base = self._done_row("基线", "1", "基线", "1.000000")
        base["commit"] = parent
        judge.append_result(self.root / "results.tsv", base)
        (self.root / "train.py").write_text("print('next')\n", encoding="utf-8")
        self._git("add", "train.py")
        self._git("commit", "-m", "try")
        return parent

    def test_seed2_mechanism_mismatch_writes_nothing(self):
        judge.append_result(self.root / "results.tsv", self._done_row("学习率", "1"))
        before = len(judge.read_results(self.root / "results.tsv"))
        code = self._trial(
            ["--seed", "2", "--mechanism", "深度", "--expect", "lower", "--fail", "崩了"],
            ["python", "-c", "print('should-not-run')"],
        )
        self.assertEqual(code, 2)
        self.assertEqual(len(judge.read_results(self.root / "results.tsv")), before)

    def test_bad_seed_writes_nothing(self):
        code = self._trial(
            ["--seed", "3", "--mechanism", "学习率", "--expect", "lower", "--fail", "崩了"],
            ["python", "-c", "print('no')"],
        )
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "results.tsv").exists())

    def test_tab_in_fail_writes_nothing(self):
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--expect", "lower", "--fail", "a\tb"],
            ["python", "-c", "print('no')"],
        )
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "results.tsv").exists())

    def test_dirty_train_py_writes_nothing(self):
        (self.root / "train.py").write_text("print('dirty')\n", encoding="utf-8")
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--expect", "lower", "--fail", "崩了"],
            ["python", "-c", "print('no')"],
        )
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "results.tsv").exists())

    def test_duplicate_seed_writes_nothing(self):
        judge.append_result(self.root / "results.tsv", self._done_row("学习率", "1"))
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--expect", "lower", "--fail", "崩了"],
            ["python", "-c", "print('no')"],
        )
        self.assertEqual(code, 2)
        self.assertEqual(len(judge.read_results(self.root / "results.tsv")), 1)

    def test_timeout_records_crash(self):
        import sys

        self._prepare_child()
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--expect", "lower", "--fail", "崩了"],
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=0.2,
        )
        self.assertEqual(code, 0)
        self.assertEqual(judge.read_results(self.root / "results.tsv")[-1]["decision"], "代码坏了")

    def test_seed_mismatch_blocks_accept(self):
        import sys

        self._prepare_child()
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--expect", "lower", "--fail", "崩了"],
            [sys.executable, "-c", "print('seed: 9'); print('val_bpb:          0.900000')"],
        )
        self.assertEqual(code, 0)
        self.assertEqual(judge.read_results(self.root / "results.tsv")[-1]["decision"], "测量失败")
        self.assertEqual(judge.read_results(self.root / "results.tsv")[-1]["val_bpb"], "")
        self.assertEqual(judge.main(["accept"], root=self.root), 2)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_judge.TestTrial -v`

Expected: FAIL with `AttributeError: module 'judge' has no attribute 'main'`

- [ ] **Step 3: Write the minimal implementation**

Add imports `import argparse`, `import subprocess`, `import sys`. Append the trial CLI. `blank_row()` fills every `RESULT_FIELDS` key with `""`.

```python
def trial_paths(root: Path) -> tuple[Path, Path, Path]:
    data_dir = root / "data"
    return root / "tokenizer.pkl", data_dir / "shard_06542.parquet", data_dir


def production_paths() -> tuple[Path, Path, Path]:
    from prepare import DATA_DIR, TOKENIZER_DIR, VAL_FILENAME

    return (
        Path(TOKENIZER_DIR) / "tokenizer.pkl",
        Path(DATA_DIR) / VAL_FILENAME,
        Path(DATA_DIR),
    )


def git_output(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def is_dirty(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--", "train.py", "prepare.py", "judge.py"],
        cwd=root, capture_output=True, text=True,
    )
    return completed.returncode != 0 or completed.stdout.strip() != ""


def blank_row() -> dict:
    return {field: "" for field in RESULT_FIELDS}


def validate_text(seed: str, mechanism: str, expect: str, fail: str) -> str | None:
    if seed not in {"1", "2"}:
        return "种子只允许 1 或 2"
    if mechanism == "" or mechanism == "基线" or "\t" in mechanism or "\n" in mechanism:
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
            command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=log_path.parent
        )
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
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
    for name in ("baseline", "accept", "claim"):
        sub.add_parser(name)
    trial = sub.add_parser("trial")
    trial.add_argument("--seed", required=True)
    trial.add_argument("--mechanism", required=True)
    trial.add_argument("--expect", required=True)
    trial.add_argument("--fail", required=True)
    args = parser.parse_args(argv)
    if args.cmd != "trial":
        return 2
    repo = root or Path.cwd()
    timeout = TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    train = command or ["uv", "run", "train.py"]
    error = validate_text(args.seed, args.mechanism, args.expect, args.fail)
    if error is not None or is_dirty(repo):
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
    tokenizer, val, data_dir = trial_paths(repo) if root is not None else production_paths()
    prepare_hash = file_sha256(repo / "prepare.py")
    identity = data_hash(tokenizer, val, data_dir)
    row = blank_row()
    row.update({
        "commit": head,
        "parent": git_output(repo, "rev-parse", "HEAD^") or "",
        "mechanism": args.mechanism,
        "expect": args.expect,
        "fail": args.fail,
        "seed": args.seed,
        "prepare_hash": prepare_hash,
    })
    if identity is None:
        row["decision"] = "代码坏了"
        row["detail"] = "数据身份不可用"
        append_result(results_path, row)
        return 0
    row["data_hash"] = identity
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
    parent_bpb = float(parent_rows[-1]["val_bpb"])
    row["decision"] = compare_seed1(float(printed_bpb), parent_bpb)
    append_result(results_path, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The parse result variable in `main` is `args`. Do not rename it back to `accept`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_judge.TestTrial tests.test_judge.TestParse tests.test_judge.TestTsv -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add judge.py tests/test_judge.py
git commit -m "Reject invalid trials before training."
```

---

### Task 6: Baseline, accept, and claim commands

**Files:**
- Modify: `judge.py`
- Modify: `tests/test_judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `main`, `apply_accept`, `record_accept`, `claim_verdict`, `count_n`, `read_results`, `read_parents`, `write_parents`, `blank_row`, `is_dirty`, `git_output`, `run_command`
- Produces: the same `main` signature, now implementing `baseline`, `trial`, `accept`, and `claim`. `accept` arguments are optional `--simpler` and `--replace`. `baseline` takes no seed argument and always trains seed `1`. `claim` takes no arguments and uses `HEAD` plus `HEAD^`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_judge.py`:

```python
class TestCommands(TestTrial):
    def test_replace_unknown_parent_writes_nothing(self):
        judge.write_parents(self.root / "parents.tsv", [{
            "commit": git_head(self.root),
            "mechanism": "基线",
            "val_bpb": "1.000000",
            "simpler": "no",
        }])
        before = (self.root / "parents.tsv").read_text(encoding="utf-8")
        code = judge.main(
            ["accept", "--simpler", "--replace", "deadbeef"],
            root=self.root,
        )
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "results.tsv").exists())
        self.assertEqual((self.root / "parents.tsv").read_text(encoding="utf-8"), before)

    def test_claim_empty_score_names_the_gap(self):
        head = git_head(self.root)
        self._git("commit", "--allow-empty", "-m", "child")
        child = git_head(self.root)
        for commit, seed, val in (
            (head, "1", "1.000000"),
            (head, "2", ""),
            (child, "1", "0.900000"),
            (child, "2", "0.900000"),
        ):
            row = self._done_row("学习率", seed, "已记录", val)
            row["commit"] = commit
            row["parent"] = head if commit == child else ""
            row["prepare_hash"] = "p"
            row["data_hash"] = "d"
            if seed == "1" and commit == head:
                row["decision"] = "基线"
            judge.append_result(self.root / "results.tsv", row)
        code = judge.main(["claim"], root=self.root)
        self.assertEqual(code, 0)
        claim = judge.read_results(self.root / "results.tsv")[-1]
        self.assertEqual(claim["decision"], "还判不了")
        self.assertEqual(claim["detail"], "缺父本种子2")
        self.assertEqual(claim["seed"], "*")

    def test_baseline_repair_does_not_train(self):
        head = git_head(self.root)
        row = self._done_row("基线", "1", "基线", "1.000000")
        row["commit"] = head
        judge.append_result(self.root / "results.tsv", row)
        code = judge.main(
            ["baseline"],
            command=["python", "-c", "raise SystemExit('trained')"],
            root=self.root,
        )
        self.assertEqual(code, 0)
        parents = judge.read_parents(self.root / "parents.tsv")
        self.assertEqual(parents[0]["commit"], head)
        self.assertEqual(len(judge.read_results(self.root / "results.tsv")), 1)

    def test_seed1_better_does_not_change_parents_or_head(self):
        import sys

        self._prepare_child()
        child = git_head(self.root)
        code = judge.main(
            ["trial", "--seed", "1", "--mechanism", "学习率", "--expect", "lower", "--fail", "变高"],
            command=[
                sys.executable, "-c",
                "print('seed: 1'); print('val_bpb:          0.900000')",
            ],
            root=self.root,
        )
        self.assertEqual(code, 0)
        self.assertEqual(judge.read_results(self.root / "results.tsv")[-1]["decision"], "好于父本")
        self.assertEqual(len(judge.read_parents(self.root / "parents.tsv")), 1)
        self.assertEqual(git_head(self.root), child)

    def _ok(self, seed, bpb):
        import sys

        return [
            sys.executable, "-c",
            f"print('seed: {seed}'); print('val_bpb:          {bpb}')",
        ]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_judge.TestCommands -v`

Expected: FAIL because `baseline`, `accept`, and `claim` return `2` or because `好于父本` is not written

- [ ] **Step 3: Write the minimal implementation**

Keep the Task 5 trial body. Change the parser setup so `accept` is a real subcommand, then add the three functions below and dispatch from `main`. The trial branch must stay byte-for-byte the same aside from `args.cmd == "trial"`.

Replace the subparser loop:

```python
    for name in ("baseline", "claim"):
        sub.add_parser(name)
    accept_parser = sub.add_parser("accept")
    accept_parser.add_argument("--simpler", action="store_true")
    accept_parser.add_argument("--replace", default=None)
```

Delete the Task 5 line `if args.cmd != "trial": return 2` that sits immediately after `parse_args`. Then dispatch:

```python
    repo = root or Path.cwd()
    if args.cmd == "baseline":
        return cmd_baseline(repo, command or ["uv", "run", "train.py"], TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds, root is not None)
    if args.cmd == "accept":
        return cmd_accept(repo, args.simpler, args.replace)
    if args.cmd == "claim":
        return cmd_claim(repo)
    if args.cmd != "trial":
        return 2
```

Remove the later `repo = root or Path.cwd()` line so it is not assigned twice. Append:

```python
def _latest(rows: list[dict], commit: str, seed: str) -> dict | None:
    found = [row for row in rows if row["commit"] == commit and row["seed"] == seed]
    return found[-1] if found else None


def _bpb(row: dict | None) -> float | None:
    if row is None or row["val_bpb"] == "":
        return None
    return float(row["val_bpb"])


def cmd_baseline(repo: Path, command: list[str], timeout: float, test_root: bool) -> int:
    if is_dirty(repo):
        return 2
    head = git_output(repo, "rev-parse", "HEAD")
    if head is None:
        return 2
    results_path = repo / "results.tsv"
    rows = read_results(results_path)
    existing = [
        row for row in rows
        if row["commit"] == head and row["seed"] == "1" and row["decision"] == "基线"
    ]
    if existing and not read_parents(repo / "parents.tsv"):
        write_parents(repo / "parents.tsv", [{
            "commit": head,
            "mechanism": "基线",
            "val_bpb": existing[-1]["val_bpb"],
            "simpler": "no",
        }])
        return 0
    if existing:
        return 2
    tokenizer, val, data_dir = trial_paths(repo) if test_root else production_paths()
    row = blank_row()
    row.update({
        "commit": head,
        "parent": "",
        "mechanism": "基线",
        "expect": "lower",
        "fail": "基线",
        "seed": "1",
        "prepare_hash": file_sha256(repo / "prepare.py"),
    })
    identity = data_hash(tokenizer, val, data_dir)
    if identity is None:
        row["decision"] = "代码坏了"
        row["detail"] = "数据身份不可用"
        append_result(results_path, row)
        return 0
    row["data_hash"] = identity
    code = run_command(command, "1", repo / "run.log", timeout)
    log_text = (repo / "run.log").read_text(encoding="utf-8") if (repo / "run.log").exists() else ""
    printed_seed, printed_bpb = parse_summary(log_text)
    if code != 0 or printed_bpb is None:
        row["decision"] = "代码坏了"
        append_result(results_path, row)
        return 0
    if printed_seed != "1":
        row["decision"] = "测量失败"
        append_result(results_path, row)
        return 0
    row["val_bpb"] = printed_bpb
    row["decision"] = "基线"
    append_result(results_path, row)
    write_parents(repo / "parents.tsv", [{
        "commit": head,
        "mechanism": "基线",
        "val_bpb": printed_bpb,
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
        if seed1["decision"] != "持平" or not replace_commit:
            return 2
    elif seed1["decision"] != "好于父本":
        return 2
    newcomer = {
        "commit": head,
        "mechanism": seed1["mechanism"],
        "val_bpb": seed1["val_bpb"],
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
        "prepare_hash": seed1["prepare_hash"],
        "data_hash": seed1["data_hash"],
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
        [row["prepare_hash"] if row else "" for row in slots],
        [row["data_hash"] if row else "" for row in slots],
        MIN_IMPROVEMENT,
        count_n(rows),
    )
    result["decision"] = decision
    result["detail"] = detail
    append_result(repo / "results.tsv", result)
    return 0
```

`main` must not call git commands other than `rev-parse` and `status`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_judge -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add judge.py tests/test_judge.py
git commit -m "Record baseline, accept, and claim without moving git."
```

---

### Task 7: Seed comes from the environment

**Files:**
- Modify: `prepare.py`
- Modify: `train.py:458-459`
- Modify: `tests/test_judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: nothing from `judge.py`
- Produces: `prepare.seed_everything() -> int`, which reads `AUTORES_SEED`, rejects a missing or blank value with `SystemExit`, calls `torch.manual_seed`, and calls `torch.cuda.manual_seed` only when CUDA is available

- [ ] **Step 1: Write the failing test**

Append to `tests/test_judge.py`:

```python
class TestSeedWiring(unittest.TestCase):
    def test_prepare_requires_the_env_var(self):
        import os
        import prepare

        os.environ.pop("AUTORES_SEED", None)
        with self.assertRaises(SystemExit):
            prepare.seed_everything()

    def test_prepare_returns_the_env_seed(self):
        import os
        import prepare

        os.environ["AUTORES_SEED"] = "2"
        self.assertEqual(prepare.seed_everything(), 2)

    def test_train_source_no_longer_hardcodes_seed_42(self):
        source = Path("train.py").read_text(encoding="utf-8")
        self.assertNotIn("manual_seed(42)", source)
        self.assertIn("seed_everything()", source)
        prepare_source = Path("prepare.py").read_text(encoding="utf-8")
        self.assertIn("TIME_BUDGET = 300", prepare_source)
        self.assertIn("def evaluate_bpb(", prepare_source)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_judge.TestSeedWiring -v`

Expected: FAIL with `AttributeError: module 'prepare' has no attribute 'seed_everything'`

- [ ] **Step 3: Write the minimal implementation**

In `prepare.py`, after the `import torch` line, add:

```python
def seed_everything():
    raw = os.environ.get("AUTORES_SEED")
    if raw is None or raw.strip() == "":
        raise SystemExit("AUTORES_SEED is required")
    seed = int(raw)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed
```

In `train.py`, change the import from `prepare` to include `seed_everything`. Replace these two lines:

```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)
```

with:

```python
seed = seed_everything()
print(f"seed: {seed}", flush=True)
```

Do not change `evaluate_bpb` or `TIME_BUDGET`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_judge.TestSeedWiring -v`

Expected: PASS, 3 tests

- [ ] **Step 5: Commit**

```bash
git add prepare.py train.py tests/test_judge.py
git commit -m "Read the training seed from AUTORES_SEED."
```

---

### Task 8: Agent instructions and ignored outputs

**Files:**
- Modify: `program.md:23-110`
- Modify: `.gitignore`
- Modify: `tests/test_judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: the CLI built in Tasks 5 and 6
- Produces: no Python functions. The agent-facing loop and the gitignore entries.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_judge.py`:

```python
class TestInstructions(unittest.TestCase):
    def test_program_describes_the_split(self):
        text = Path("program.md").read_text(encoding="utf-8")
        self.assertIn("uv run judge.py trial --seed 1", text)
        self.assertIn("uv run judge.py accept", text)
        self.assertIn("uv run judge.py claim", text)
        self.assertIn("judge.py", text)
        self.assertIn("parents.tsv", text)
        self.assertIn("do not call a parent-list update a validated claim", text)
        ignored = Path(".gitignore").read_text(encoding="utf-8")
        self.assertIn("results.tsv", ignored)
        self.assertIn("parents.tsv", ignored)
        self.assertIn("run.log", ignored)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_judge.TestInstructions -v`

Expected: FAIL because `program.md` does not mention `uv run judge.py trial --seed 1`

- [ ] **Step 3: Write the minimal implementation**

Add these lines to `.gitignore` under the results entry:

```text
parents.tsv
run.log
```

Replace the `program.md` sections from `## Experimentation` through the crashes paragraph with the following. Keep `## NEVER STOP` and the paragraphs after it.

```markdown
## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You do not launch `train.py` yourself. `judge.py` launches it and records the score.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
- Choose which parent commit to start from, then commit your edit.
- Run `judge.py` commands listed below.
- `git reset` a commit you decided not to keep.

**What you CANNOT do:**
- Modify `prepare.py` or `judge.py`.
- Edit `results.tsv` or `parents.tsv` by hand.
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.
- Hardcode a random seed. `train.py` must keep the `seed_everything()` call.
- Treat a parent-list update as a validated claim. Do not call a parent-list update a validated claim.

**The goal is simple: get the lowest val_bpb.** The time budget stays 5 minutes. VRAM is a soft constraint. Some increase is acceptable for a meaningful gain, but it should not blow up dramatically.

**Simplicity:** You judge whether added code is worth the gain. `judge.py` does not measure simplicity. A strictly lower seed-1 `val_bpb` may enter the parent list only if you then run `accept`. An equal seed-1 score can replace a listed parent only through `accept --simpler`. A higher seed-1 score never enters the list.

## The experiment loop

Work on a branch such as `autoresearch/mar5`. The first command on a clean baseline commit is:

```text
uv run judge.py baseline
```

LOOP FOREVER:

1. Read `parents.tsv` and `results.tsv`. Do not always continue from the most recently accepted parent.
2. Check out the chosen parent commit.
3. Write down the mechanism in three facts: a short mechanism name, whether you expect `val_bpb` to go `lower` or `higher`, and one sentence for what counts as failure. The same short name is the same mechanism even if the other two sentences change.
4. Edit `train.py` and git commit. The new commit's parent must be the chosen parent.
5. Run:

```text
uv run judge.py trial --seed 1 --mechanism 学习率 --expect lower --fail 变高或崩溃
```

6. Read the new `results.tsv` row. `judge.py` does not create commits and does not reset them.
7. If the decision is `好于父本` and the complexity is worth keeping, leave the commit and run `uv run judge.py accept`.
8. If the decision is `持平` and the code is simpler, leave the commit and run `uv run judge.py accept --simpler --replace <parent-hash>`.
9. If the decision is `差于父本`, or the complexity is not worth it, `git reset` back to the parent. Do not run `accept`.
10. If the decision is `代码坏了`, fix a typo or missing import and try again. If the idea itself cannot run, leave the crash row and move on. Do not describe that crash as the mechanism being worse.
11. If the decision is `测量失败`, do not accept the commit. Reset and fix the seed printing before counting the idea.

Overnight search uses seed 1 only. The parent list holds at most three commits.

## Morning claim

Pick one or two commits you might call validated. For the parent and then the candidate, check out that commit with a clean tree and run:

```text
uv run judge.py trial --seed 2 --mechanism <same short name> --expect lower --fail <same failure sentence>
```

Then, on the candidate commit, run:

```text
uv run judge.py claim
```

Only a `claim` row whose decision is `成立` is a validated result. `还判不了` means the materials were insufficient. `进入名单` is not that result.

`judge.py` kills a training run that exceeds 10 minutes and records `代码坏了`.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests.test_judge -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add program.md .gitignore tests/test_judge.py
git commit -m "Document the search loop and ignore judge outputs."
```
