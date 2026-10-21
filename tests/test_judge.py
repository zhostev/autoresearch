import math
import statistics
import sys
import unittest
from pathlib import Path

import judge


class TestClaim(unittest.TestCase):
    def test_seed1_directions(self):
        self.assertEqual(judge.compare_to_parent(0.9, 1.0, 0.0), "好于父本")
        self.assertEqual(judge.compare_to_parent(1.0, 1.0, 0.0), "分不开")
        self.assertEqual(judge.compare_to_parent(1.1, 1.0, 0.0), "差于父本")
        self.assertFalse(hasattr(judge, "compare_seed1"))

    def test_timeout_constant(self):
        self.assertEqual(judge.TIMEOUT_SECONDS, 600)
        self.assertFalse(hasattr(judge, "MIN_IMPROVEMENT"))

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
                *scores[:4], hashes, 0.001, 1
            )
            self.assertEqual(decision, "还判不了")
            self.assertEqual(detail, scores[4])
        decision, detail = judge.claim_verdict(
            0.9, 0.9, 1.0, 1.0, hashes, None, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "未标定"))
        decision, detail = judge.claim_verdict(
            0.9, 0.9, 1.0, 1.0, hashes, 0.0, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "未标定"))
        decision, detail = judge.claim_verdict(
            0.9, 0.9, 1.0, 1.0, hashes, -0.1, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "未标定"))
        decision, detail = judge.claim_verdict(
            0.9, 0.9, 1.0, 1.0, ["a", "b", "a", "a"], 0.001, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "尺子或数据不同"))
        decision, detail = judge.claim_verdict(
            0.99, 0.99, 1.0, 1.0, hashes, 0.001, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "两次改进没有分开"))

    def test_worse_and_tie(self):
        hashes = ["a", "a", "a", "a"]
        decision, detail = judge.claim_verdict(
            1.1, 1.2, 1.0, 1.0, hashes, 0.001, 1
        )
        self.assertEqual(decision, "更差")
        decision, detail = judge.claim_verdict(
            1.0, 1.0, 1.0, 1.0, hashes, 0.001, 1
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
        self.assertEqual(judge.count_n(rows), 3)
        penalty = judge.luck_penalty(0.990, 0.996, 4)
        self.assertAlmostEqual(penalty, 0.007064460135092854, delta=1e-12)
        hashes = ["h", "h", "h", "h"]
        self.assertEqual(
            judge.claim_verdict(0.990, 0.996, 1.000, 1.002, hashes, 0.001, 4)[0],
            "没有更好",
        )
        self.assertEqual(
            judge.claim_verdict(0.990, 0.996, 1.000, 1.002, hashes, 0.0005, 4)[0],
            "成立",
        )
        self.assertEqual(judge.luck_penalty(0.5, 0.6, 1), 0.0)
        self.assertEqual(judge.count_n([{"decision": "基线", "mechanism": "基线"}]), 0)


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
                "seeds": "1",
                "reviewed": "no",
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


if __name__ == "__main__":
    unittest.main()


class TestParse(unittest.TestCase):
    def test_padded_val_bpb_line(self):
        text = "seed: 1\n---\nval_bpb:          0.997900\ntraining_seconds: 300.1\n"
        self.assertEqual(judge.parse_summary(text), ("1", "0.997900"))

    def test_last_occurrence_wins(self):
        text = "seed: 0\nval_bpb: 1.000000\nseed: 2\nval_bpb:          0.500000\n"
        self.assertEqual(judge.parse_summary(text), ("2", "0.500000"))

    def test_missing_lines(self):
        self.assertEqual(judge.parse_summary("no summary\n"), (None, None))


def git_head(root: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


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
        (self.root / "task.py").write_text("NAME = 'test'\n", encoding="utf-8")
        data = self.root / "data"
        data.mkdir()
        (data / "shard_06542.parquet").write_bytes(b"val")
        (data / "shard_00001.parquet").write_bytes(b"train")
        (self.root / "tokenizer.pkl").write_bytes(b"tok")
        self._git("add", "prepare.py", "train.py", "judge.py", "task.py")
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

    def _identity(self):
        return judge.eval_hash(
            self.root / "task.py", self.root / "prepare.py", self.root / "data"
        )

    def _ok_command(self, value):
        script = self.root / "ok_train.py"
        script.write_text(
            "import os\n"
            "print('seed: ' + os.environ['AUTORES_SEED'])\n"
            f"print('val_bpb:          {value}')\n",
            encoding="utf-8",
        )
        return [sys.executable, str(script)]

    def _done_row(self, mechanism, seed, decision="好于父本", val="1.000000"):
        return {
            "commit": git_head(self.root),
            "parent": "",
            "mechanism": mechanism,
            "expect": "lower",
            "fail": "崩了",
            "seed": seed,
            "knobs": "",
            "val_bpb": val,
            "eval_hash": self._identity(),
            "decision": decision,
            "detail": "",
        }

    def _write_run(self, sigma=0.004, **overrides):
        config = {
            "run_tag": "test",
            "run_start": "2026-10-21T00:00:00+08:00",
            "baseline_commit": git_head(self.root),
            "eval_hash": self._identity(),
            "repeats": 5,
            "z": 2.0,
            "sigma": sigma,
            "min_improvement": 2.0 * sigma,
            "queue_limit": 2,
            "rotate_after": 8,
        }
        config.update(overrides)
        judge.write_run(self.root, config)
        return config

    def _prepare_child(self, sigma=0.004, train_text="print('next')\n"):
        parent = git_head(self.root)
        self._write_run(sigma)
        judge.write_parents(self.root / "parents.tsv", [{
            "commit": parent,
            "mechanism": "基线",
            "val_bpb": "1.000000",
            "seeds": "1,2,3,4,5",
            "reviewed": "yes",
            "simpler": "no",
        }])
        base = self._done_row("基线", "1", "基线", "1.000000")
        base["commit"] = parent
        judge.append_result(self.root / "results.tsv", base)
        (self.root / "train.py").write_text(train_text, encoding="utf-8")
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

    def test_seed1_baseline_mechanism_writes_nothing(self):
        import sys

        self._prepare_child()
        before = (self.root / "results.tsv").read_text(encoding="utf-8")
        code = self._trial(
            ["--seed", "1", "--mechanism", "基线", "--knobs", "", "--expect", "lower", "--fail", "崩了"],
            [sys.executable, "-c", "print('seed: 1'); print('val_bpb:          0.900000')"],
        )
        self.assertEqual(code, 2)
        self.assertEqual((self.root / "results.tsv").read_text(encoding="utf-8"), before)

    def test_seed2_baseline_mechanism_records_score(self):
        import sys

        self._write_run()
        judge.append_result(
            self.root / "results.tsv",
            self._done_row("基线", "1", "基线", "1.000000"),
        )
        code = self._trial(
            ["--seed", "2", "--mechanism", "基线", "--expect", "lower", "--fail", "崩了"],
            [sys.executable, "-c", "print('seed: 2'); print('val_bpb:          0.990000')"],
        )
        self.assertEqual(code, 0)
        row = judge.read_results(self.root / "results.tsv")[-1]
        self.assertEqual(row["decision"], "已记录")
        self.assertEqual(row["val_bpb"], "0.990000")
        self.assertEqual(row["mechanism"], "基线")
        self.assertEqual(row["seed"], "2")

    def test_seed2_baseline_name_on_other_mechanism_writes_nothing(self):
        import sys

        judge.append_result(self.root / "results.tsv", self._done_row("学习率", "1"))
        before = len(judge.read_results(self.root / "results.tsv"))
        code = self._trial(
            ["--seed", "2", "--mechanism", "基线", "--expect", "lower", "--fail", "崩了"],
            [sys.executable, "-c", "print('seed: 2'); print('val_bpb:          0.990000')"],
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
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "", "--expect", "lower", "--fail", "a\tb"],
            ["python", "-c", "print('no')"],
        )
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "results.tsv").exists())

    def test_dirty_train_py_writes_nothing(self):
        (self.root / "train.py").write_text("print('dirty')\n", encoding="utf-8")
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "", "--expect", "lower", "--fail", "崩了"],
            ["python", "-c", "print('no')"],
        )
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "results.tsv").exists())

    def test_duplicate_seed_writes_nothing(self):
        judge.append_result(self.root / "results.tsv", self._done_row("学习率", "1"))
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "", "--expect", "lower", "--fail", "崩了"],
            ["python", "-c", "print('no')"],
        )
        self.assertEqual(code, 2)
        self.assertEqual(len(judge.read_results(self.root / "results.tsv")), 1)

    def test_timeout_records_crash(self):
        import sys

        self._prepare_child()
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "", "--expect", "lower", "--fail", "崩了"],
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=0.2,
        )
        self.assertEqual(code, 0)
        self.assertEqual(judge.read_results(self.root / "results.tsv")[-1]["decision"], "代码坏了")

    def test_timeout_kills_child_sleeper(self):
        import os
        import sys
        import time

        self._prepare_child()
        pid_path = self.root / "sleeper.pid"
        script = "\n".join([
            "import subprocess, sys",
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])",
            f"handle = open({str(pid_path)!r}, 'w', encoding='utf-8')",
            "handle.write(str(child.pid))",
            "handle.close()",
            "child.wait()",
        ])
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "", "--expect", "lower", "--fail", "崩了"],
            [sys.executable, "-c", script],
            timeout=1,
        )
        self.assertEqual(code, 0)
        self.assertEqual(judge.read_results(self.root / "results.tsv")[-1]["decision"], "代码坏了")
        self.assertTrue(pid_path.is_file(), "sleeper pid was not recorded")
        pid = int(pid_path.read_text(encoding="utf-8"))
        try:
            deadline = time.monotonic() + 1
            running = True
            while time.monotonic() < deadline:
                if not self._sleeper_running(pid):
                    running = False
                    break
                time.sleep(0.02)
            self.assertFalse(running, "child sleeper still running after timeout")
        finally:
            if self._sleeper_running(pid):
                os.kill(pid, 9)

    def _sleeper_running(self, pid: int) -> bool:
        import os

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        stat_path = Path(f"/proc/{pid}/stat")
        try:
            stat = stat_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return False
        state = stat[stat.rfind(")") + 2]
        return state != "Z"

    def test_seed_mismatch_blocks_accept(self):
        import sys

        self._prepare_child()
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "", "--expect", "lower", "--fail", "崩了"],
            [sys.executable, "-c", "print('seed: 9'); print('val_bpb:          0.900000')"],
        )
        self.assertEqual(code, 0)
        self.assertEqual(judge.read_results(self.root / "results.tsv")[-1]["decision"], "测量失败")
        self.assertEqual(judge.read_results(self.root / "results.tsv")[-1]["val_bpb"], "")
        self.assertEqual(judge.main(["accept"], root=self.root), 2)


class TestCommands(TestTrial):
    def test_replace_unknown_parent_writes_nothing(self):
        judge.write_parents(self.root / "parents.tsv", [{
            "commit": git_head(self.root),
            "mechanism": "基线",
            "val_bpb": "1.000000",
            "seeds": "1,2,3,4,5",
            "reviewed": "yes",
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

    def test_seed1_better_does_not_change_parents_or_head(self):
        import sys

        self._prepare_child()
        child = git_head(self.root)
        code = judge.main(
            ["trial", "--seed", "1", "--mechanism", "学习率", "--knobs", "", "--expect", "lower", "--fail", "变高"],
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


class TestInstructions(unittest.TestCase):
    def test_program_describes_the_split(self):
        text = Path("program.md").read_text(encoding="utf-8")
        self.assertIn("uv run judge.py trial --seed 1", text)
        self.assertIn("uv run judge.py accept", text)
        self.assertIn("uv run judge.py claim", text)
        self.assertIn("judge.py", text)
        self.assertIn("parents.tsv", text)
        self.assertIn("Do not call a parent-list update a validated claim.", text)
        ignored = Path(".gitignore").read_text(encoding="utf-8")
        self.assertIn("results.tsv", ignored)
        self.assertIn("parents.tsv", ignored)
        self.assertIn("run.log", ignored)


class TestEvalHash(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "task.py").write_text("NAME = 'x'\n", encoding="utf-8")
        (self.root / "prepare.py").write_text("TIME_BUDGET = 300\n", encoding="utf-8")
        self.data = self.root / "data"
        (self.data / "sub").mkdir(parents=True)
        (self.data / "a.parquet").write_bytes(b"aaa")
        (self.data / "sub" / "b.parquet").write_bytes(b"bb")

    def tearDown(self):
        self._tmp.cleanup()

    def _hash(self):
        return judge.eval_hash(
            self.root / "task.py", self.root / "prepare.py", self.data
        )

    def test_manifest_lists_relative_paths_and_sizes(self):
        self.assertEqual(
            judge.data_manifest(self.data), "a.parquet 3\nsub/b.parquet 2\n"
        )

    def test_hash_is_stable(self):
        self.assertEqual(self._hash(), self._hash())

    def test_task_py_byte_changes_the_hash(self):
        before = self._hash()
        (self.root / "task.py").write_text("NAME = 'y'\n", encoding="utf-8")
        self.assertNotEqual(before, self._hash())

    def test_new_data_file_changes_the_hash(self):
        before = self._hash()
        (self.data / "c.parquet").write_bytes(b"c")
        self.assertNotEqual(before, self._hash())

    def test_content_change_at_same_size_does_not_change_the_hash(self):
        before = self._hash()
        (self.data / "a.parquet").write_bytes(b"zzz")
        self.assertEqual(before, self._hash())

    def test_missing_data_root_is_unavailable(self):
        import shutil

        shutil.rmtree(self.data)
        self.assertIsNone(judge.data_manifest(self.data))
        self.assertIsNone(self._hash())

    def test_symlink_and_directory_entries_are_skipped(self):
        (self.data / "link.parquet").symlink_to(self.data / "a.parquet")
        (self.data / "emptydir").mkdir()
        self.assertEqual(
            judge.data_manifest(self.data), "a.parquet 3\nsub/b.parquet 2\n"
        )


class TestRunConfig(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _good(self):
        return {
            "run_tag": "mar5",
            "run_start": "2026-10-21T15:22:00+08:00",
            "baseline_commit": "a" * 40,
            "eval_hash": "b" * 64,
            "repeats": 5,
            "z": 2.0,
            "sigma": 0.004,
            "min_improvement": 0.008,
            "queue_limit": 2,
            "rotate_after": 8,
        }

    def test_round_trip(self):
        judge.write_run(self.root, self._good())
        self.assertEqual(judge.read_run(self.root), self._good())

    def test_missing_file_is_none(self):
        self.assertIsNone(judge.read_run(self.root))

    def test_broken_json_is_none(self):
        (self.root / "run.json").write_text("{not json", encoding="utf-8")
        self.assertIsNone(judge.read_run(self.root))

    def test_missing_field_is_none(self):
        import json

        bad = self._good()
        del bad["sigma"]
        (self.root / "run.json").write_text(json.dumps(bad), encoding="utf-8")
        self.assertIsNone(judge.read_run(self.root))

    def test_wrong_type_is_none(self):
        import json

        bad = self._good()
        bad["repeats"] = "five"
        (self.root / "run.json").write_text(json.dumps(bad), encoding="utf-8")
        self.assertIsNone(judge.read_run(self.root))

    def test_band_widens_as_the_parent_gets_noisier(self):
        self.assertAlmostEqual(
            judge.band_for(0.004, 2.0, 1), 2.0 * 0.004 * math.sqrt(2.0), places=12
        )
        self.assertAlmostEqual(
            judge.band_for(0.004, 2.0, 2), 2.0 * 0.004 * math.sqrt(1.5), places=12
        )
        self.assertAlmostEqual(
            judge.band_for(0.004, 2.0, 5), 2.0 * 0.004 * math.sqrt(1.2), places=12
        )
        self.assertGreater(judge.band_for(0.004, 2.0, 1), judge.band_for(0.004, 2.0, 5))

    def test_band_falls_back_to_one_sample_on_bad_n(self):
        expected = judge.band_for(0.004, 2.0, 1)
        self.assertEqual(judge.band_for(0.004, 2.0, 0), expected)
        self.assertEqual(judge.band_for(0.004, 2.0, -3), expected)


class TestCalibrate(TestTrial):
    def _fake_scores(self, values):
        table = ", ".join(
            f"'{index + 1}': '{value}'" for index, value in enumerate(values)
        )
        script = self.root / "fake_train.py"
        script.write_text(
            "import os\n"
            f"table = {{{table}}}\n"
            "seed = os.environ['AUTORES_SEED']\n"
            "print('seed: ' + seed)\n"
            "print('val_bpb:          ' + table[seed])\n",
            encoding="utf-8",
        )
        return [sys.executable, str(script)]

    def _calibrate(self, values, timeout=5):
        return judge.main(
            ["calibrate"],
            command=self._fake_scores(values),
            timeout_seconds=timeout,
            root=self.root,
        )

    def test_five_seeds_are_recorded_and_sigma_is_written(self):
        values = ["1.000000", "1.004000", "0.996000", "1.002000", "0.998000"]
        self.assertEqual(self._calibrate(values), 0)
        rows = judge.read_results(self.root / "results.tsv")
        self.assertEqual(len(rows), 5)
        self.assertEqual([row["seed"] for row in rows], ["1", "2", "3", "4", "5"])
        self.assertTrue(all(row["decision"] == "基线" for row in rows))
        self.assertTrue(all(row["mechanism"] == "基线" for row in rows))
        config = judge.read_run(self.root)
        expected = statistics.stdev([float(value) for value in values])
        self.assertAlmostEqual(config["sigma"], expected, places=12)
        self.assertAlmostEqual(config["min_improvement"], 2.0 * expected, places=12)
        self.assertEqual(config["repeats"], 5)
        self.assertEqual(config["queue_limit"], 2)
        self.assertEqual(config["rotate_after"], 8)
        self.assertEqual(config["baseline_commit"], git_head(self.root))

    def test_parent_list_holds_the_mean_of_every_seed(self):
        values = ["1.000000", "1.004000", "0.996000", "1.002000", "0.998000"]
        self._calibrate(values)
        parents = judge.read_parents(self.root / "parents.tsv")
        self.assertEqual(len(parents), 1)
        self.assertAlmostEqual(float(parents[0]["val_bpb"]), 1.0, places=9)
        self.assertEqual(parents[0]["seeds"], "1,2,3,4,5")
        self.assertEqual(parents[0]["reviewed"], "yes")

    def test_zero_variance_is_refused(self):
        values = ["1.000000"] * 5
        self.assertEqual(self._calibrate(values), 2)
        self.assertIsNone(judge.read_run(self.root))
        self.assertEqual(judge.read_parents(self.root / "parents.tsv"), [])

    def test_a_crash_aborts_the_whole_calibration(self):
        script = self.root / "boom.py"
        script.write_text("raise SystemExit(1)\n", encoding="utf-8")
        code = judge.main(
            ["calibrate"],
            command=[sys.executable, str(script)],
            timeout_seconds=5,
            root=self.root,
        )
        self.assertEqual(code, 2)
        rows = judge.read_results(self.root / "results.tsv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision"], "代码坏了")
        self.assertIsNone(judge.read_run(self.root))

    def test_baseline_subcommand_is_gone(self):
        with self.assertRaises(SystemExit):
            judge.main(["baseline"], command=["true"], timeout_seconds=1, root=self.root)


class TestNoiseBand(unittest.TestCase):
    def test_band_edges_are_indistinguishable(self):
        band = judge.band_for(0.004, 2.0, 1)
        self.assertEqual(judge.compare_to_parent(1.0 - band - 1e-9, 1.0, band), "好于父本")
        self.assertEqual(judge.compare_to_parent(1.0 - band, 1.0, band), "分不开")
        self.assertEqual(judge.compare_to_parent(1.0, 1.0, band), "分不开")
        self.assertEqual(judge.compare_to_parent(1.0 + band, 1.0, band), "分不开")
        self.assertEqual(judge.compare_to_parent(1.0 + band + 1e-9, 1.0, band), "差于父本")

    def test_flower14_champion_is_indistinguishable(self):
        band = judge.band_for(0.0005, 2.0, 1)
        self.assertEqual(judge.compare_to_parent(0.048064, 0.049399, band), "分不开")

    def test_a_real_gain_still_wins(self):
        band = judge.band_for(0.0005, 2.0, 1)
        self.assertEqual(judge.compare_to_parent(0.048064, 0.132844, band), "好于父本")

    def test_parent_n_counts_the_seed_list(self):
        self.assertEqual(judge.parent_n({"seeds": "1"}), 1)
        self.assertEqual(judge.parent_n({"seeds": "1,2"}), 2)
        self.assertEqual(judge.parent_n({"seeds": "1,2,3,4,5"}), 5)

    def test_parent_n_falls_back_to_one_on_bad_input(self):
        self.assertEqual(judge.parent_n({"seeds": ""}), 1)
        self.assertEqual(judge.parent_n({"seeds": "   "}), 1)
        self.assertEqual(judge.parent_n({}), 1)
        self.assertEqual(judge.parent_n({"seeds": ",,"}), 1)


class TestKnobs(unittest.TestCase):
    def test_constant_assignment_on_both_sides(self):
        diff = (
            "--- a/train.py\n"
            "+++ b/train.py\n"
            "-IMAGE_SIZE = 128\n"
            "+IMAGE_SIZE = 160\n"
        )
        self.assertEqual(judge.touched_constants(diff), {"IMAGE_SIZE"})

    def test_diff_headers_are_not_constants(self):
        diff = "--- a/train.py\n+++ b/train.py\n"
        self.assertEqual(judge.touched_constants(diff), set())

    def test_comparison_is_not_an_assignment(self):
        diff = "+IMAGE_SIZE == 160\n+    if BATCH_SIZE == 2:\n"
        self.assertEqual(judge.touched_constants(diff), set())

    def test_indented_assignment_is_not_module_level(self):
        diff = "+    IMAGE_SIZE = 160\n"
        self.assertEqual(judge.touched_constants(diff), set())

    def test_structural_change_touches_nothing(self):
        diff = (
            "-        loss.backward()\n"
            "-        optimizer.step()\n"
            "+        scaler.scale(loss).backward()\n"
            "+        scaler.step(optimizer)\n"
        )
        self.assertEqual(judge.touched_constants(diff), set())

    def test_several_constants(self):
        diff = "+IMAGE_SIZE = 176\n+BATCH_SIZE = 48\n"
        self.assertEqual(judge.touched_constants(diff), {"IMAGE_SIZE", "BATCH_SIZE"})

    def test_assignment_at_end_of_line(self):
        self.assertEqual(judge.touched_constants("+IMAGE_SIZE =\n"), {"IMAGE_SIZE"})

    def test_parse_knobs_normalises(self):
        self.assertEqual(judge.parse_knobs(""), set())
        self.assertEqual(judge.parse_knobs("  "), set())
        self.assertEqual(judge.parse_knobs("A, B ,A"), {"A", "B"})
        self.assertEqual(judge.parse_knobs("IMAGE_SIZE"), {"IMAGE_SIZE"})


class TestKnobsGate(TestTrial):
    def test_undeclared_constant_is_refused(self):
        self._prepare_child(train_text="IMAGE_SIZE = 160\n")
        code = self._trial(
            ["--seed", "1", "--mechanism", "半精度", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=["true"],
        )
        self.assertEqual(code, 2)
        self.assertEqual(len(judge.read_results(self.root / "results.tsv")), 1)

    def test_matching_declaration_is_allowed(self):
        self._prepare_child(train_text="IMAGE_SIZE = 160\n")
        code = self._trial(
            ["--seed", "1", "--mechanism", "分辨率", "--knobs", "IMAGE_SIZE",
             "--expect", "lower", "--fail", "崩了"],
            command=self._ok_command("0.900000"),
        )
        self.assertEqual(code, 0)
        rows = judge.read_results(self.root / "results.tsv")
        self.assertEqual(rows[-1]["knobs"], "IMAGE_SIZE")

    def test_extra_constant_is_refused(self):
        self._prepare_child(train_text="IMAGE_SIZE = 160\nBATCH_SIZE = 48\n")
        code = self._trial(
            ["--seed", "1", "--mechanism", "分辨率", "--knobs", "IMAGE_SIZE",
             "--expect", "lower", "--fail", "崩了"],
            command=["true"],
        )
        self.assertEqual(code, 2)
        self.assertEqual(len(judge.read_results(self.root / "results.tsv")), 1)


class TestKnobsRequiredOnSeed1(TestTrial):
    def test_seed1_without_knobs_is_refused(self):
        self._prepare_child()
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--expect", "lower", "--fail", "崩了"],
            command=["true"],
        )
        self.assertEqual(code, 2)
        self.assertEqual(len(judge.read_results(self.root / "results.tsv")), 1)


class TestRotation(unittest.TestCase):
    def _rows(self, parents):
        return [
            {"seed": "1", "decision": "差于父本", "parent": parent}
            for parent in parents
        ]

    def test_single_parent_never_blocks(self):
        self.assertFalse(judge.rotation_blocked(self._rows(["p"] * 20), 1, "p", 8))

    def test_eight_in_a_row_blocks_the_same_parent(self):
        self.assertTrue(judge.rotation_blocked(self._rows(["p"] * 8), 3, "p", 8))

    def test_a_different_parent_in_the_window_clears_it(self):
        parents = ["p"] * 7 + ["q"]
        self.assertFalse(judge.rotation_blocked(self._rows(parents), 3, "p", 8))
        parents = ["q"] + ["p"] * 7
        self.assertFalse(judge.rotation_blocked(self._rows(parents), 3, "p", 8))

    def test_a_short_history_never_blocks(self):
        self.assertFalse(judge.rotation_blocked(self._rows(["p"] * 7), 3, "p", 8))

    def test_switching_to_another_parent_is_allowed(self):
        self.assertFalse(judge.rotation_blocked(self._rows(["p"] * 8), 3, "q", 8))

    def test_only_seed1_trial_rows_count(self):
        rows = self._rows(["p"] * 8)
        rows.insert(0, {"seed": "*", "decision": "进入名单", "parent": "q"})
        rows.insert(0, {"seed": "2", "decision": "已记录", "parent": "q"})
        self.assertTrue(judge.rotation_blocked(rows, 3, "p", 8))

    def test_crashes_count_as_attempts(self):
        parents = ["p"] * 8
        rows = self._rows(parents)
        rows[3]["decision"] = "代码坏了"
        rows[4]["decision"] = "测量失败"
        self.assertTrue(judge.rotation_blocked(rows, 3, "p", 8))


class TestQueue(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "queue.tsv"

    def tearDown(self):
        self._tmp.cleanup()

    def _entry(self, commit):
        return {
            "commit": commit, "parent": "p", "mechanism": "学习率",
            "accepted_at": "2026-10-21T00:00:00+08:00", "state": "待复核",
        }

    def test_round_trip_and_header(self):
        judge.write_queue(self.path, [self._entry("a"), self._entry("b")])
        self.assertEqual(
            self.path.read_text(encoding="utf-8").splitlines()[0], judge.QUEUE_HEADER
        )
        self.assertEqual(
            [row["commit"] for row in judge.read_queue(self.path)], ["a", "b"]
        )

    def test_missing_file_is_empty(self):
        self.assertEqual(judge.read_queue(self.path), [])

    def test_enqueue_appends_and_never_duplicates(self):
        rows = judge.enqueue([], self._entry("a"))
        rows = judge.enqueue(rows, self._entry("b"))
        rows = judge.enqueue(rows, self._entry("a"))
        self.assertEqual([row["commit"] for row in rows], ["a", "b"])

    def test_dequeue_removes_only_that_commit(self):
        rows = [self._entry("a"), self._entry("b")]
        self.assertEqual([row["commit"] for row in judge.dequeue(rows, "a")], ["b"])
        self.assertEqual(
            [row["commit"] for row in judge.dequeue(rows, "zz")], ["a", "b"]
        )


class TestBackpressure(TestTrial):
    def _fill_queue(self, count):
        parent = git_head(self.root)
        judge.write_queue(self.root / "queue.tsv", [{
            "commit": parent, "parent": "p", "mechanism": f"m{index}",
            "accepted_at": "2026-10-21T00:00:00+08:00", "state": "待复核",
        } for index in range(count)])

    def test_queue_at_the_limit_still_runs(self):
        self._prepare_child()
        self._fill_queue(2)
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=self._ok_command("0.900000"),
        )
        self.assertEqual(code, 0)

    def test_queue_over_the_limit_refuses_seed1(self):
        self._prepare_child()
        self._fill_queue(3)
        before = len(judge.read_results(self.root / "results.tsv"))
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=["true"],
        )
        self.assertEqual(code, 2)
        self.assertEqual(len(judge.read_results(self.root / "results.tsv")), before)

    def test_queue_over_the_limit_still_allows_seed2(self):
        self._prepare_child()
        head = git_head(self.root)
        seed1 = self._done_row("学习率", "1", "好于父本", "0.900000")
        seed1["commit"] = head
        judge.append_result(self.root / "results.tsv", seed1)
        self._fill_queue(3)
        code = self._trial(
            ["--seed", "2", "--mechanism", "学习率",
             "--expect", "lower", "--fail", "崩了"],
            command=self._ok_command("0.910000"),
        )
        self.assertEqual(code, 0)

    def test_simpler_replacement_also_enqueues(self):
        parent = self._prepare_child()
        head = git_head(self.root)
        tie = self._done_row("学习率", "1", "分不开", "1.000000")
        tie["commit"] = head
        tie["parent"] = parent
        judge.append_result(self.root / "results.tsv", tie)
        code = judge.main(["accept", "--simpler", "--replace", parent], root=self.root)
        self.assertEqual(code, 0)
        self.assertEqual(
            [row["commit"] for row in judge.read_queue(self.root / "queue.tsv")], [head]
        )

    def test_accept_enqueues_a_better_commit(self):
        parent = self._prepare_child()
        head = git_head(self.root)
        better = self._done_row("学习率", "1", "好于父本", "0.500000")
        better["commit"] = head
        better["parent"] = parent
        judge.append_result(self.root / "results.tsv", better)
        self.assertEqual(judge.main(["accept"], root=self.root), 0)
        self.assertEqual(
            [row["commit"] for row in judge.read_queue(self.root / "queue.tsv")], [head]
        )

    def test_unreachable_commits_are_pruned_instead_of_deadlocking(self):
        self._prepare_child()
        judge.write_queue(self.root / "queue.tsv", [{
            "commit": "0" * 40, "parent": "p", "mechanism": f"m{index}",
            "accepted_at": "2026-10-21T00:00:00+08:00", "state": "待复核",
        } for index in range(3)])
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=self._ok_command("0.900000"),
        )
        self.assertEqual(code, 0)
        self.assertEqual(judge.read_queue(self.root / "queue.tsv"), [])


class TestClaimV2(unittest.TestCase):
    def _hashes(self):
        return ["e"] * 4

    def test_one_hash_list_replaces_two(self):
        decision, detail = judge.claim_verdict(
            0.9, 0.9, 1.0, 1.0, ["e", "f", "e", "e"], 0.001, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "尺子或数据不同"))

    def test_uncalibrated_says_so(self):
        decision, detail = judge.claim_verdict(
            0.9, 0.9, 1.0, 1.0, self._hashes(), None, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "未标定"))
        decision, detail = judge.claim_verdict(
            0.9, 0.9, 1.0, 1.0, self._hashes(), 0.0, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "未标定"))

    def test_penalty_example_still_holds(self):
        decision, _ = judge.claim_verdict(
            0.990, 0.996, 1.000, 1.002, self._hashes(), 0.001, 4
        )
        self.assertEqual(decision, "没有更好")
        decision, _ = judge.claim_verdict(
            0.990, 0.996, 1.000, 1.002, self._hashes(), 0.0005, 4
        )
        self.assertEqual(decision, "成立")


class TestClaimSideEffects(unittest.TestCase):
    def _parents(self):
        return [
            {"commit": "a" * 40, "mechanism": "基线", "val_bpb": "1.000000",
             "seeds": "1,2,3,4,5", "reviewed": "yes", "simpler": "no"},
            {"commit": "b" * 40, "mechanism": "学习率", "val_bpb": "0.900000",
             "seeds": "1", "reviewed": "no", "simpler": "no"},
        ]

    def test_holds_updates_to_the_mean(self):
        rows = judge.apply_claim(self._parents(), "b" * 40, "成立", 0.905)
        target = rows[1]
        self.assertEqual(target["val_bpb"], "0.905000")
        self.assertEqual(target["seeds"], "1,2")
        self.assertEqual(target["reviewed"], "yes")
        self.assertEqual(len(rows), 2)

    def test_no_better_also_updates_and_stays(self):
        rows = judge.apply_claim(self._parents(), "b" * 40, "没有更好", 0.905)
        self.assertEqual(rows[1]["reviewed"], "yes")
        self.assertEqual(len(rows), 2)

    def test_worse_is_removed(self):
        rows = judge.apply_claim(self._parents(), "b" * 40, "更差", 1.100)
        self.assertEqual([row["commit"] for row in rows], ["a" * 40])

    def test_worse_never_empties_the_list(self):
        only = [self._parents()[1]]
        rows = judge.apply_claim(only, "b" * 40, "更差", 1.100)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reviewed"], "yes")

    def test_not_yet_decidable_never_dequeues(self):
        queue = [{"commit": "b" * 40, "parent": "a" * 40, "mechanism": "学习率",
                  "accepted_at": "t", "state": "待复核"}]
        self.assertNotIn("还判不了", judge.FINAL_CLAIMS)
        self.assertEqual(judge.dequeue(queue, "z" * 40), queue)

    def test_unknown_commit_changes_nothing(self):
        before = self._parents()
        self.assertEqual(judge.apply_claim(before, "z" * 40, "成立", 0.5), before)

    def test_not_yet_decidable_changes_nothing(self):
        before = self._parents()
        self.assertEqual(judge.apply_claim(before, "b" * 40, "还判不了", 0.9), before)


class TestReport(unittest.TestCase):
    def _config(self):
        return {
            "run_tag": "mar5", "run_start": "2026-10-21T15:22:00+08:00",
            "baseline_commit": "a" * 40, "eval_hash": "e" * 64,
            "repeats": 5, "z": 2.0, "sigma": 0.004,
            "min_improvement": 0.008, "queue_limit": 2, "rotate_after": 8,
        }

    def _row(self, commit, decision, eval_hash="e" * 64, mechanism="学习率", knobs=""):
        return {
            "commit": commit, "parent": "a" * 40, "mechanism": mechanism,
            "knobs": knobs, "expect": "lower", "fail": "崩了", "seed": "1",
            "val_bpb": "0.900000", "eval_hash": eval_hash,
            "decision": decision, "detail": "",
        }

    def test_calibration_numbers_appear(self):
        text = judge.build_report(self._config(), [], [], [])
        self.assertIn("0.004", text)
        self.assertIn("0.008", text)

    def test_two_eval_hashes_split_the_table(self):
        rows = [
            self._row("b" * 40, "好于父本", "e" * 64),
            self._row("c" * 40, "差于父本", "f" * 64),
        ]
        text = judge.build_report(self._config(), rows, [], [])
        self.assertIn("e" * 12, text)
        self.assertIn("f" * 12, text)

    def test_no_holds_says_none(self):
        rows = [self._row("b" * 40, "好于父本")]
        text = judge.build_report(self._config(), rows, [], [])
        self.assertIn("## 已验证结论", text)
        section = text.split("## 已验证结论", 1)[1].split("##", 1)[0]
        self.assertIn("无", section)

    def test_listed_commit_without_a_hold_stays_unverified(self):
        parents = [{
            "commit": "b" * 40, "mechanism": "学习率", "val_bpb": "0.900000",
            "seeds": "1", "reviewed": "no", "simpler": "no",
        }]
        rows = [self._row("b" * 40, "进入名单")]
        text = judge.build_report(self._config(), rows, parents, [])
        unverified = text.split("## 搜索结果（未验证）", 1)[1].split("##", 1)[0]
        verified = text.split("## 已验证结论", 1)[1].split("##", 1)[0]
        self.assertIn("b" * 7, unverified)
        self.assertNotIn("b" * 7, verified)

    def test_a_hold_is_listed_as_verified(self):
        rows = [self._row("b" * 40, "成立")]
        text = judge.build_report(self._config(), rows, [], [])
        verified = text.split("## 已验证结论", 1)[1].split("##", 1)[0]
        self.assertIn("b" * 7, verified)

    def test_queue_backlog_is_warned(self):
        queue = [{
            "commit": "b" * 40, "parent": "a" * 40, "mechanism": "学习率",
            "accepted_at": "2026-10-21T16:00:00+08:00", "state": "待复核",
        }]
        text = judge.build_report(self._config(), [], [], queue)
        self.assertIn("待复核", text)
        self.assertIn("b" * 7, text)

    def test_inconsistent_knobs_are_audited(self):
        rows = [
            self._row("b" * 40, "差于父本", mechanism="分辨率", knobs="IMAGE_SIZE"),
            self._row("c" * 40, "差于父本", mechanism="分辨率", knobs="BATCH_SIZE"),
        ]
        text = judge.build_report(self._config(), rows, [], [])
        audit = text.split("## 机制审计", 1)[1]
        self.assertIn("分辨率", audit)

    def test_consistent_knobs_are_not_audited(self):
        rows = [
            self._row("b" * 40, "差于父本", mechanism="分辨率", knobs="IMAGE_SIZE"),
            self._row("c" * 40, "差于父本", mechanism="分辨率", knobs="IMAGE_SIZE"),
        ]
        text = judge.build_report(self._config(), rows, [], [])
        audit = text.split("## 机制审计", 1)[1]
        self.assertNotIn("分辨率", audit)

    def test_uncalibrated_report_still_renders(self):
        self.assertIn("未标定", judge.build_report(None, [], [], []))


class TestReportIsWritten(TestTrial):
    def _report(self):
        path = self.root / "REPORT.md"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def test_report_subcommand_writes_the_file(self):
        self.assertEqual(judge.main(["report"], root=self.root), 0)
        self.assertIn("未标定", self._report())

    def test_calibrate_writes_the_report(self):
        values = ["1.000000", "1.004000", "0.996000", "1.002000", "0.998000"]
        table = ", ".join(
            f"'{index + 1}': '{value}'" for index, value in enumerate(values)
        )
        script = self.root / "fake_train.py"
        script.write_text(
            "import os\n"
            f"table = {{{table}}}\n"
            "seed = os.environ['AUTORES_SEED']\n"
            "print('seed: ' + seed)\n"
            "print('val_bpb:          ' + table[seed])\n",
            encoding="utf-8",
        )
        judge.main(
            ["calibrate"],
            command=[sys.executable, str(script)],
            timeout_seconds=5,
            root=self.root,
        )
        text = self._report()
        self.assertIn("## 已验证结论", text)
        self.assertIn("无。进入名单不是已验证结论。", text)

    def test_accept_refreshes_the_report(self):
        parent = self._prepare_child()
        head = git_head(self.root)
        better = self._done_row("学习率", "1", "好于父本", "0.500000")
        better["commit"] = head
        better["parent"] = parent
        judge.append_result(self.root / "results.tsv", better)
        self.assertEqual(judge.main(["accept"], root=self.root), 0)
        text = self._report()
        self.assertIn(head[:7], text)
        self.assertIn("待复核", text)


class TestInstructionsV2(unittest.TestCase):
    def setUp(self):
        self.text = Path("program.md").read_text(encoding="utf-8")

    def test_commands_are_v2(self):
        self.assertIn("judge.py calibrate", self.text)
        self.assertIn("--knobs", self.text)
        self.assertIn("judge.py report", self.text)
        self.assertNotIn("judge.py baseline", self.text)

    def test_mechanised_rules_are_no_longer_advice(self):
        self.assertNotIn("不要总从最近", self.text)
        self.assertNotIn("Do not always continue from the most recently", self.text)

    def test_the_old_tie_word_is_gone(self):
        self.assertNotIn("持平", self.text)
        self.assertIn("分不开", self.text)

    def test_the_hold_is_still_spelled_out(self):
        self.assertIn("成立", self.text)
        self.assertIn("进入名单", self.text)

    def test_gitignore_covers_the_new_state_files(self):
        ignored = Path(".gitignore").read_text(encoding="utf-8")
        for name in ("run.json", "queue.tsv", "REPORT.md"):
            self.assertIn(name, ignored)


class TestSeed2OnATie(TestTrial):
    def test_an_indistinguishable_commit_can_be_reviewed(self):
        parent = self._prepare_child()
        head = git_head(self.root)
        tie = self._done_row("学习率", "1", "分不开", "1.000000")
        tie["commit"] = head
        tie["parent"] = parent
        judge.append_result(self.root / "results.tsv", tie)
        code = self._trial(
            ["--seed", "2", "--mechanism", "学习率",
             "--expect", "lower", "--fail", "崩了"],
            command=self._ok_command("0.998000"),
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            judge.read_results(self.root / "results.tsv")[-1]["decision"], "已记录"
        )
