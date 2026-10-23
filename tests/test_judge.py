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
        self._write_run()
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
            if seed == "1" and commit == head:
                row["decision"] = "基线"
            judge.append_result(self.root / "results.tsv", row)
        code = judge.main(["claim"], root=self.root)
        self.assertEqual(code, 0)
        claim = judge.read_results(self.root / "results.tsv")[-1]
        self.assertEqual(claim["decision"], "还判不了")
        self.assertEqual(claim["detail"], "缺父本样本")
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
            judge.band_for(0.004, 0.004, 2.0, 1, 1), 2.0 * 0.004 * math.sqrt(2.0), places=12
        )
        self.assertAlmostEqual(
            judge.band_for(0.004, 0.004, 2.0, 1, 2), 2.0 * 0.004 * math.sqrt(1.5), places=12
        )
        self.assertAlmostEqual(
            judge.band_for(0.004, 0.004, 2.0, 1, 5), 2.0 * 0.004 * math.sqrt(1.2), places=12
        )
        self.assertGreater(judge.band_for(0.004, 0.004, 2.0, 1, 1), judge.band_for(0.004, 0.004, 2.0, 1, 5))

    def test_band_falls_back_to_one_sample_on_bad_n(self):
        expected = judge.band_for(0.004, 0.004, 2.0, 1, 1)
        self.assertEqual(judge.band_for(0.004, 0.004, 2.0, 1, 0), expected)
        self.assertEqual(judge.band_for(0.004, 0.004, 2.0, 1, -3), expected)


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
        band = judge.band_for(0.004, 0.004, 2.0, 1, 1)
        self.assertEqual(judge.compare_to_parent(1.0 - band - 1e-9, 1.0, band), "好于父本")
        self.assertEqual(judge.compare_to_parent(1.0 - band, 1.0, band), "分不开")
        self.assertEqual(judge.compare_to_parent(1.0, 1.0, band), "分不开")
        self.assertEqual(judge.compare_to_parent(1.0 + band, 1.0, band), "分不开")
        self.assertEqual(judge.compare_to_parent(1.0 + band + 1e-9, 1.0, band), "差于父本")

    def test_flower14_champion_is_indistinguishable(self):
        band = judge.band_for(0.0005, 0.0005, 2.0, 1, 1)
        self.assertEqual(judge.compare_to_parent(0.048064, 0.049399, band), "分不开")

    def test_a_real_gain_still_wins(self):
        band = judge.band_for(0.0005, 0.0005, 2.0, 1, 1)
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


class TestEvalHashGate(TestTrial):
    def _move_the_ruler(self):
        (self.root / "data" / "extra.parquet").write_bytes(b"x")

    def test_trial_is_refused_when_the_ruler_moved(self):
        self._prepare_child()
        before = len(judge.read_results(self.root / "results.tsv"))
        self._move_the_ruler()
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=self._ok_command("0.500000"),
        )
        self.assertEqual(code, 2)
        self.assertEqual(len(judge.read_results(self.root / "results.tsv")), before)

    def test_seed2_is_refused_when_the_ruler_moved(self):
        parent = self._prepare_child()
        head = git_head(self.root)
        seed1 = self._done_row("学习率", "1", "好于父本", "0.900000")
        seed1["commit"] = head
        seed1["parent"] = parent
        judge.append_result(self.root / "results.tsv", seed1)
        self._move_the_ruler()
        code = self._trial(
            ["--seed", "2", "--mechanism", "学习率",
             "--expect", "lower", "--fail", "崩了"],
            command=self._ok_command("0.910000"),
        )
        self.assertEqual(code, 2)

    def test_accept_is_refused_when_the_ruler_moved(self):
        parent = self._prepare_child()
        head = git_head(self.root)
        better = self._done_row("学习率", "1", "好于父本", "0.500000")
        better["commit"] = head
        better["parent"] = parent
        judge.append_result(self.root / "results.tsv", better)
        self._move_the_ruler()
        self.assertEqual(judge.main(["accept"], root=self.root), 2)
        self.assertEqual(
            [row["commit"] for row in judge.read_parents(self.root / "parents.tsv")],
            [parent],
        )

    def test_claim_is_refused_when_the_ruler_moved(self):
        self._prepare_child()
        self._move_the_ruler()
        self.assertEqual(judge.main(["claim"], root=self.root), 2)

    def test_report_still_works_when_the_ruler_moved(self):
        self._prepare_child()
        self._move_the_ruler()
        self.assertEqual(judge.main(["report"], root=self.root), 0)
        self.assertTrue((self.root / "REPORT.md").is_file())


class TestReviewRetry(TestTrial):
    def _accepted_head(self):
        parent = self._prepare_child()
        head = git_head(self.root)
        seed1 = self._done_row("学习率", "1", "好于父本", "0.900000")
        seed1["commit"] = head
        seed1["parent"] = parent
        judge.append_result(self.root / "results.tsv", seed1)
        return parent, head

    def test_a_crashed_seed2_can_be_retried(self):
        parent, head = self._accepted_head()
        crash = self._done_row("学习率", "2", "代码坏了", "")
        crash["commit"] = head
        judge.append_result(self.root / "results.tsv", crash)
        code = self._trial(
            ["--seed", "2", "--mechanism", "学习率",
             "--expect", "lower", "--fail", "崩了"],
            command=self._ok_command("0.910000"),
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            judge.read_results(self.root / "results.tsv")[-1]["decision"], "已记录"
        )

    def test_a_completed_seed2_still_cannot_be_repeated(self):
        parent, head = self._accepted_head()
        done = self._done_row("学习率", "2", "已记录", "0.910000")
        done["commit"] = head
        judge.append_result(self.root / "results.tsv", done)
        code = self._trial(
            ["--seed", "2", "--mechanism", "学习率",
             "--expect", "lower", "--fail", "崩了"],
            command=self._ok_command("0.920000"),
        )
        self.assertEqual(code, 2)


class TestReplaceMustNameTheOwnParent(TestTrial):
    def test_a_tie_cannot_evict_an_unrelated_parent(self):
        parent = self._prepare_child()
        head = git_head(self.root)
        other = "f" * 40
        judge.write_parents(self.root / "parents.tsv", [
            {"commit": parent, "mechanism": "基线", "val_bpb": "1.000000",
             "seeds": "1,2,3,4,5", "reviewed": "yes", "simpler": "no"},
            {"commit": other, "mechanism": "冠军", "val_bpb": "0.100000",
             "seeds": "1,2", "reviewed": "yes", "simpler": "no"},
        ])
        tie = self._done_row("学习率", "1", "分不开", "1.000000")
        tie["commit"] = head
        tie["parent"] = parent
        judge.append_result(self.root / "results.tsv", tie)
        code = judge.main(["accept", "--simpler", "--replace", other], root=self.root)
        self.assertEqual(code, 2)
        self.assertIn(
            other,
            [row["commit"] for row in judge.read_parents(self.root / "parents.tsv")],
        )

    def test_replacing_its_own_parent_is_allowed(self):
        parent = self._prepare_child()
        head = git_head(self.root)
        tie = self._done_row("学习率", "1", "分不开", "1.000000")
        tie["commit"] = head
        tie["parent"] = parent
        judge.append_result(self.root / "results.tsv", tie)
        self.assertEqual(
            judge.main(["accept", "--simpler", "--replace", parent], root=self.root), 0
        )


class TestKnobsRegexCoverage(unittest.TestCase):
    def test_annotated_assignment_counts(self):
        self.assertEqual(
            judge.touched_constants("+IMAGE_SIZE: int = 160\n"), {"IMAGE_SIZE"}
        )

    def test_augmented_assignment_counts(self):
        self.assertEqual(
            judge.touched_constants("+LEARNING_RATE *= 2\n"), {"LEARNING_RATE"}
        )
        self.assertEqual(judge.touched_constants("+WARMUP //= 2\n"), {"WARMUP"})
        self.assertEqual(judge.touched_constants("+DECAY **= 2\n"), {"DECAY"})

    def test_tuple_target_counts_every_name(self):
        self.assertEqual(
            judge.touched_constants("+IMAGE_SIZE, BATCH_SIZE = 160, 64\n"),
            {"IMAGE_SIZE", "BATCH_SIZE"},
        )

    def test_comparison_is_still_not_an_assignment(self):
        self.assertEqual(judge.touched_constants("+IMAGE_SIZE == 160\n"), set())
        self.assertEqual(judge.touched_constants("+    if BATCH_SIZE == 2:\n"), set())


class TestRecalibrationGuard(TestCalibrate):
    def _values(self):
        return ["1.000000", "1.004000", "0.996000", "1.002000", "0.998000"]

    def test_recalibrating_without_force_is_refused(self):
        self.assertEqual(self._calibrate(self._values()), 0)
        before = judge.read_parents(self.root / "parents.tsv")
        rows_before = len(judge.read_results(self.root / "results.tsv"))
        self.assertEqual(self._calibrate(self._values()), 2)
        self.assertEqual(judge.read_parents(self.root / "parents.tsv"), before)
        self.assertEqual(
            len(judge.read_results(self.root / "results.tsv")), rows_before
        )

    def test_forced_recalibration_clears_the_queue(self):
        self.assertEqual(self._calibrate(self._values()), 0)
        judge.write_queue(self.root / "queue.tsv", [{
            "commit": git_head(self.root), "parent": "p", "mechanism": "学习率",
            "accepted_at": "t", "state": "待复核",
        }])
        code = judge.main(
            ["calibrate", "--force"],
            command=self._fake_scores(self._values()),
            timeout_seconds=5,
            root=self.root,
        )
        self.assertEqual(code, 0)
        self.assertEqual(judge.read_queue(self.root / "queue.tsv"), [])


class TestTruncatedRows(TestTrial):
    def test_read_results_pads_a_short_row(self):
        path = self.root / "results.tsv"
        path.write_text(judge.RESULT_HEADER + "\nabc\tdef\n", encoding="utf-8")
        row = judge.read_results(path)[0]
        self.assertEqual(row["commit"], "abc")
        self.assertEqual(row["detail"], "")
        self.assertEqual(row["eval_hash"], "")

    def test_the_report_survives_a_truncated_row(self):
        path = self.root / "results.tsv"
        path.write_text(judge.RESULT_HEADER + "\nabc\tdef\n", encoding="utf-8")
        self.assertEqual(judge.main(["report"], root=self.root), 0)
        self.assertIn("## 已验证结论", (self.root / "REPORT.md").read_text(encoding="utf-8"))


class TestManifestIgnoresScratchFiles(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.data = Path(self._tmp.name) / "data"
        self.data.mkdir()
        (self.data / "a.parquet").write_bytes(b"aaa")

    def tearDown(self):
        self._tmp.cleanup()

    def test_partial_downloads_do_not_move_the_ruler(self):
        before = judge.data_manifest(self.data)
        (self.data / "shard_00007.parquet.tmp").write_bytes(b"half")
        self.assertEqual(judge.data_manifest(self.data), before)

    def test_dotfiles_do_not_move_the_ruler(self):
        before = judge.data_manifest(self.data)
        (self.data / ".DS_Store").write_bytes(b"junk")
        self.assertEqual(judge.data_manifest(self.data), before)


class TestTaskContract(unittest.TestCase):
    """task.py 的契约对任何任务都成立；只有最后一条是 nanochat 专有的。"""

    def setUp(self):
        self.task = Path("task.py").read_text(encoding="utf-8")
        self.train = Path("train.py").read_text(encoding="utf-8")

    def test_task_exports_the_required_names(self):
        for name in ("NAME", "DATA_ROOT", "TIME_BUDGET", "def score(", "def should_stop("):
            self.assertIn(name, self.task)

    def test_train_takes_its_data_from_task(self):
        self.assertIn("from task import", self.train)

    def test_train_does_not_reach_past_task_for_a_loader(self):
        self.assertNotIn("make_dataloader", self.train)

    def test_the_nanochat_ruler_covers_the_tokenizer(self):
        if 'NAME = "nanochat"' not in self.task:
            self.skipTest("这个分支跑的不是 nanochat 任务")
        self.assertIn("DATA_ROOT = Path(prepare.CACHE_DIR)", self.task)
        self.assertNotIn("DATA_ROOT = Path(prepare.DATA_DIR)", self.task)


class TestReportCountsTrialsOnly(unittest.TestCase):
    def _config(self):
        return {
            "run_tag": "mar5", "run_start": "t", "baseline_commit": "a" * 40,
            "eval_hash": "e" * 64, "repeats": 5, "z": 2.0, "sigma": 0.004,
            "min_improvement": 0.008, "queue_limit": 2, "rotate_after": 8,
        }

    def _row(self, seed, decision, eval_hash):
        return {
            "commit": "b" * 40, "parent": "a" * 40, "mechanism": "学习率",
            "knobs": "", "expect": "lower", "fail": "崩了", "seed": seed,
            "val_bpb": "0.9", "eval_hash": eval_hash, "decision": decision,
            "detail": "",
        }

    def test_bookkeeping_rows_do_not_form_a_ruler_group(self):
        rows = [
            self._row("1", "好于父本", "e" * 64),
            self._row("*", "进入名单", "e" * 64),
            self._row("*", "成立", ""),
        ]
        text = judge.build_report(self._config(), rows, [], [])
        counts = text.split("## 轮次统计", 1)[1].split("## 搜索结果", 1)[0]
        self.assertIn("好于父本", counts)
        self.assertNotIn("进入名单", counts)
        self.assertNotIn("### 尺子 \n", counts)

    def test_a_hold_is_still_listed_as_verified(self):
        rows = [self._row("*", "成立", "")]
        text = judge.build_report(self._config(), rows, [], [])
        verified = text.split("## 已验证结论", 1)[1].split("##", 1)[0]
        self.assertIn("b" * 7, verified)


class TestInstructionsNameTheFrozenTask(unittest.TestCase):
    def setUp(self):
        self.text = Path("program.md").read_text(encoding="utf-8")

    def test_task_py_is_listed_as_out_of_bounds(self):
        self.assertIn("task.py", self.text)

    def test_no_v1_exact_score_semantics_remain(self):
        self.assertNotIn("A strictly lower seed-1", self.text)
        self.assertNotIn("An equal seed-1 score", self.text)


class TestRefusalsSpeak(TestTrial):
    def _stderr_of(self, call):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            code = call()
        return code, buffer.getvalue()

    def test_a_dirty_tree_says_so(self):
        self._prepare_child()
        (self.root / "train.py").write_text("dirty\n", encoding="utf-8")
        code, err = self._stderr_of(lambda: self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=["true"],
        ))
        self.assertEqual(code, 2)
        self.assertIn("工作区", err)

    def test_an_undeclared_knob_says_so(self):
        self._prepare_child(train_text="IMAGE_SIZE = 160\n")
        code, err = self._stderr_of(lambda: self._trial(
            ["--seed", "1", "--mechanism", "半精度", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=["true"],
        ))
        self.assertEqual(code, 2)
        self.assertIn("IMAGE_SIZE", err)

    def test_a_parent_outside_the_list_says_so(self):
        self._write_run()
        (self.root / "train.py").write_text("print('x')\n", encoding="utf-8")
        self._git("add", "train.py")
        self._git("commit", "-m", "orphan")
        code, err = self._stderr_of(lambda: self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=["true"],
        ))
        self.assertEqual(code, 2)
        self.assertIn("父本", err)

    def test_rotation_refusal_says_so(self):
        parent = self._prepare_child()
        head = git_head(self.root)
        judge.write_parents(self.root / "parents.tsv", [
            {"commit": parent, "mechanism": "基线", "val_bpb": "1.000000",
             "seeds": "1,2,3,4,5", "reviewed": "yes", "simpler": "no"},
            {"commit": "f" * 40, "mechanism": "别的", "val_bpb": "0.5",
             "seeds": "1", "reviewed": "no", "simpler": "no"},
        ])
        for index in range(8):
            row = self._done_row("机制%d" % index, "1", "差于父本", "1.100000")
            row["commit"] = "%040d" % index
            row["parent"] = parent
            judge.append_result(self.root / "results.tsv", row)
        code, err = self._stderr_of(lambda: self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=["true"],
        ))
        self.assertEqual(code, 2)
        self.assertIn("换", err)

    def test_every_refusal_path_is_wired_to_the_helper(self):
        import re

        source = Path("judge.py").read_text(encoding="utf-8")
        body = source.split("def refuse(", 1)[1].split("\n\n\n", 1)[1]
        self.assertEqual(re.findall(r"^\s*return 2$", body, re.M), [])


class TestTwoRulersAreNeverCompared(TestTrial):
    """端到端：候选和父本的分数必须出自同一把尺子。

    flower14 的二号根因：`ac0a661` 在搜索中途把验证集从 98 张换成 1498 张，
    而 148 行记录的数据哈希一个字节都没变，于是两版评测下的分数被直接比大小。
    v2 把哈希算对了，这条测试钉住的是「拿它挡人」那一步。
    """

    def test_a_commit_that_moves_the_ruler_cannot_reach_the_parent_list(self):
        baseline = git_head(self.root)
        calibrated = self._write_run()["eval_hash"]
        judge.write_parents(self.root / "parents.tsv", [{
            "commit": baseline, "mechanism": "基线", "val_bpb": "1.000000",
            "seeds": "1,2,3,4,5", "reviewed": "yes", "simpler": "no",
        }])
        base_row = self._done_row("基线", "1", "基线", "1.000000")
        base_row["commit"] = baseline
        judge.append_result(self.root / "results.tsv", base_row)

        # 一个提交里同时改 train.py 和 task.py。两道旧闸门都看不见它：
        # is_dirty 只查未提交的改动，knobs 只看 `git diff -- train.py`。
        (self.root / "train.py").write_text("print('v2')\n", encoding="utf-8")
        (self.root / "task.py").write_text(
            "NAME = 'test'\nHOLDOUT = 900\n", encoding="utf-8"
        )
        self._git("add", "train.py", "task.py")
        self._git("commit", "-m", "mechanism and a quietly different ruler")
        self.assertNotEqual(self._identity(), calibrated)

        code = self._trial(
            ["--seed", "1", "--mechanism", "机制", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=self._ok_command("0.500000"),
        )
        self.assertEqual(code, 2)
        self.assertEqual(judge.main(["accept"], root=self.root), 2)
        self.assertEqual(
            [row["commit"] for row in judge.read_parents(self.root / "parents.tsv")],
            [baseline],
        )

        scored = [
            row for row in judge.read_results(self.root / "results.tsv")
            if row["val_bpb"] != ""
        ]
        self.assertTrue(scored)
        self.assertEqual({row["eval_hash"] for row in scored}, {calibrated})


class TestStepBudgetIsEnforced(TestTrial):
    """步数预算在 task.py 里冻着，但终止条件在 agent 可改的 train.py 里。
    judge 必须自己核对训练真的跑满了，否则一次提前退出会被记成有效测量。"""

    def _train_printing(self, steps):
        script = self.root / "fake_train.py"
        script.write_text(
            "import os\n"
            "print('seed: ' + os.environ['AUTORES_SEED'])\n"
            f"print('num_steps:        {steps}')\n"
            "print('val_bpb:          0.500000')\n",
            encoding="utf-8",
        )
        return [sys.executable, str(script)]

    def _with_budget(self, budget):
        (self.root / "task.py").write_text(
            f"NAME = 'test'\nSTEP_BUDGET = {budget}\n", encoding="utf-8"
        )
        self._git("add", "task.py")
        self._git("commit", "-m", "budget")

    def test_a_short_run_is_a_failed_measurement(self):
        self._with_budget(100)
        self._prepare_child()
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=self._train_printing(93),
        )
        self.assertEqual(code, 0)
        row = judge.read_results(self.root / "results.tsv")[-1]
        self.assertEqual(row["decision"], "测量失败")
        self.assertEqual(row["val_bpb"], "")
        self.assertIn("93", row["detail"])

    def test_a_full_run_is_recorded(self):
        self._with_budget(100)
        self._prepare_child()
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=self._train_printing(100),
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            judge.read_results(self.root / "results.tsv")[-1]["decision"], "好于父本"
        )

    def test_a_task_without_a_step_budget_is_not_checked(self):
        self._prepare_child()
        code = self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=self._train_printing(7),
        )
        self.assertEqual(code, 0)
        self.assertEqual(
            judge.read_results(self.root / "results.tsv")[-1]["decision"], "好于父本"
        )


class TestAuditOnlySearchRows(unittest.TestCase):
    def _row(self, seed, knobs):
        return {
            "commit": "b" * 40, "parent": "a" * 40, "mechanism": "容量",
            "knobs": knobs, "expect": "lower", "fail": "崩了", "seed": seed,
            "val_bpb": "0.9", "eval_hash": "e" * 64, "decision": "好于父本",
            "detail": "",
        }

    def test_a_reviewed_mechanism_is_not_flagged(self):
        rows = [self._row("1", "IMAGE_SIZE,WIDTH"), self._row("2", "")]
        text = judge.build_report(None, rows, [], [])
        audit = text.split("## 机制审计", 1)[1]
        self.assertNotIn("容量", audit)

    def test_a_genuinely_inconsistent_mechanism_is_still_flagged(self):
        rows = [self._row("1", "IMAGE_SIZE"), self._row("1", "BATCH_SIZE")]
        rows[1]["commit"] = "c" * 40
        text = judge.build_report(None, rows, [], [])
        audit = text.split("## 机制审计", 1)[1]
        self.assertIn("容量", audit)


class TestBandAcrossBothArms(unittest.TestCase):
    def test_more_candidate_samples_tighten_the_band(self):
        one = judge.band_for(0.004, 0.004, 2.0, 1, 5)
        three = judge.band_for(0.004, 0.004, 2.0, 3, 5)
        self.assertAlmostEqual(one, 2.0 * 0.004 * math.sqrt(1.0 + 1.0 / 5), places=12)
        self.assertAlmostEqual(
            three, 2.0 * 0.004 * math.sqrt(1.0 / 3 + 1.0 / 5), places=12
        )
        self.assertLess(three, one)

    def test_one_candidate_sample_is_the_old_behaviour(self):
        self.assertEqual(judge.band_for(0.004, 0.004, 2.0, 1, 2), judge.band_for(0.004, 0.004, 2.0, 1, 2))

    def test_bad_counts_fall_back_to_one(self):
        expected = judge.band_for(0.004, 0.004, 2.0, 1, 1)
        self.assertEqual(judge.band_for(0.004, 0.004, 2.0, 0, 0), expected)


class TestEscalation(TestTrial):
    """Δ<0 的候选追加测量，用均值重判。"""

    def _train_table(self, values: dict):
        table = ", ".join(f"'{k}': '{v}'" for k, v in values.items())
        script = self.root / "table_train.py"
        script.write_text(
            "import os\n"
            f"table = {{{table}}}\n"
            "seed = os.environ['AUTORES_SEED']\n"
            "print('seed: ' + seed)\n"
            "print('val_bpb:          ' + table[seed])\n",
            encoding="utf-8",
        )
        return [sys.executable, str(script)]

    def _run(self, values):
        self._prepare_child()
        return self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=self._train_table(values),
            timeout=10,
        )

    def _rows(self):
        return judge.read_results(self.root / "results.tsv")

    def test_a_candidate_no_better_than_its_parent_is_measured_once(self):
        self.assertEqual(self._run({"1": "1.002000"}), 0)
        trials = [r for r in self._rows() if r["mechanism"] == "学习率"]
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0]["seed"], "1")
        self.assertEqual(trials[0]["decision"], "分不开")

    def test_a_lucky_first_draw_is_pulled_back_to_its_mean(self):
        # 单次抽样下 0.990 会判「好于父本」；三次均值 0.99767 落回带内。
        self.assertEqual(
            self._run({"1": "0.990000", "2": "1.002000", "3": "1.001000"}), 0
        )
        trials = [r for r in self._rows() if r["mechanism"] == "学习率"]
        self.assertEqual([r["seed"] for r in trials], ["2", "3", "1"])
        verdict = trials[-1]
        self.assertEqual(verdict["seed"], "1")
        self.assertEqual(verdict["decision"], "分不开")
        self.assertIn("3 次均值", verdict["detail"])
        self.assertEqual([r["decision"] for r in trials[:2]], ["已记录", "已记录"])

    def test_a_small_but_consistent_gain_becomes_visible(self):
        # 单次抽样下 -0.0065 埋在 0.00876 的带里；三次均值把带收到 0.00584。
        self.assertEqual(
            self._run({"1": "0.993500", "2": "0.994000", "3": "0.993000"}), 0
        )
        verdict = [r for r in self._rows() if r["seed"] == "1"][-1]
        self.assertEqual(verdict["decision"], "好于父本")

    def test_an_already_measured_seed_is_not_run_twice(self):
        self._prepare_child()
        head = git_head(self.root)
        seed2 = self._done_row("学习率", "2", "已记录", "1.002000")
        seed2["commit"] = head
        judge.append_result(self.root / "results.tsv", seed2)
        self._trial(
            ["--seed", "1", "--mechanism", "学习率", "--knobs", "",
             "--expect", "lower", "--fail", "崩了"],
            command=self._train_table({"1": "0.990000", "3": "1.001000"}),
            timeout=10,
        )
        seeds = [r["seed"] for r in self._rows() if r["mechanism"] == "学习率"]
        self.assertEqual(seeds.count("2"), 1)
        verdict = [r for r in self._rows() if r["seed"] == "1"][-1]
        self.assertEqual(verdict["decision"], "分不开")


class TestParentBasisUsesEverySample(TestTrial):
    def test_accept_lists_every_measured_seed(self):
        parent = self._prepare_child()
        head = git_head(self.root)
        for seed, val in (("2", "0.902000"), ("3", "0.898000")):
            row = self._done_row("学习率", seed, "已记录", val)
            row["commit"] = head
            judge.append_result(self.root / "results.tsv", row)
        win = self._done_row("学习率", "1", "好于父本", "0.900000")
        win["commit"] = head
        win["parent"] = parent
        judge.append_result(self.root / "results.tsv", win)
        self.assertEqual(judge.main(["accept"], root=self.root), 0)
        listed = [
            r for r in judge.read_parents(self.root / "parents.tsv")
            if r["commit"] == head
        ][0]
        self.assertEqual(listed["seeds"], "1,2,3")
        self.assertAlmostEqual(float(listed["val_bpb"]), 0.900000, places=6)


class TestInstructionsDescribeEscalation(unittest.TestCase):
    def test_the_agent_is_told_a_trial_may_measure_three_times(self):
        text = Path("program.md").read_text(encoding="utf-8")
        self.assertIn("追加测量", text)
        self.assertIn("trial_repeats", text)


class TestClaimUsesEverySample(unittest.TestCase):
    """缺陷 5：claim 原本只配对 seed1/seed2，比 accept 还弱。"""

    def _hashes(self, count):
        return ["e"] * count

    def test_both_arms_are_pooled(self):
        # 候选 3 个样本均值 0.9，父本 5 个样本均值 1.0，改进 0.1。
        decision, detail = judge.claim_verdict(
            [0.88, 0.92, 0.90], [0.98, 1.00, 1.02, 0.99, 1.01],
            self._hashes(8), 0.004, 2.0, 0.008, 1,
        )
        self.assertEqual((decision, detail), ("成立", ""))

    def test_a_single_bad_second_seed_no_longer_decides_alone(self):
        # 旧设计只看 seed1/seed2：(1.00-0.88) 与 (1.02-0.95) 平均 0.095 本来也成立，
        # 但若父本 seed2 恰好是好抽样，旧设计会被它一个样本拖垮。合并后看全部。
        decision, _ = judge.claim_verdict(
            [0.88, 0.95, 0.89], [1.00, 0.96, 1.02, 1.01, 1.03],
            self._hashes(8), 0.004, 2.0, 0.008, 1,
        )
        self.assertEqual(decision, "成立")

    def test_claim_is_strictly_stricter_than_accept(self):
        sigma, z = 0.004, 2.0
        candidate = [0.9900, 0.9910, 0.9890]
        parent = [1.0000, 1.0010, 0.9990, 1.0005, 0.9995]
        band = judge.band_for(sigma, sigma, z, len(candidate), len(parent))
        improvement = statistics.fmean(parent) - statistics.fmean(candidate)
        # accept 会放行：改进超过带宽。
        self.assertGreater(improvement, band)
        self.assertEqual(
            judge.compare_to_parent(
                statistics.fmean(candidate), statistics.fmean(parent), band
            ),
            "好于父本",
        )
        # claim 不放行：门槛是 max(band, min_improvement) 再加碰运气扣减。
        decision, _ = judge.claim_verdict(
            candidate, parent, self._hashes(8), sigma, z, z * sigma, 4
        )
        self.assertEqual(decision, "没有更好")

    def test_more_mechanisms_tried_raise_the_bar(self):
        # 改进固定在 0.012，SE 约 0.00292。N=2 时门槛 0.0093 判成立；
        # N=400 时门槛 0.0160，同一个改进就不够了。
        args = ([0.988, 0.990, 0.986], [1.000, 1.001, 0.999, 1.002, 0.998],
                self._hashes(8), 0.004, 2.0, 0.008)
        self.assertEqual(judge.claim_verdict(*args, 2)[0], "成立")
        self.assertEqual(judge.claim_verdict(*args, 400)[0], "没有更好")

    def test_identical_candidate_scores_no_longer_block(self):
        # 错误率是离散的，两个种子给出同一个分数很常见；旧设计会判「还判不了」。
        decision, _ = judge.claim_verdict(
            [0.900, 0.900], [1.000, 1.000, 1.000, 1.000, 1.000],
            self._hashes(7), 0.004, 2.0, 0.008, 1,
        )
        self.assertEqual(decision, "成立")

    def test_too_few_samples_names_the_gap(self):
        decision, detail = judge.claim_verdict(
            [0.9], [1.0, 1.0], self._hashes(3), 0.004, 2.0, 0.008, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "缺候选样本"))
        decision, detail = judge.claim_verdict(
            [0.9, 0.9], [1.0], self._hashes(3), 0.004, 2.0, 0.008, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "缺父本样本"))

    def test_a_moved_ruler_still_alarms(self):
        decision, detail = judge.claim_verdict(
            [0.9, 0.9], [1.0, 1.0], ["e", "f", "e", "e"], 0.004, 2.0, 0.008, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "尺子或数据不同"))

    def test_uncalibrated_says_so(self):
        decision, detail = judge.claim_verdict(
            [0.9, 0.9], [1.0, 1.0], self._hashes(4), None, 2.0, 0.008, 1
        )
        self.assertEqual((decision, detail), ("还判不了", "未标定"))

    def test_worse_is_still_worse(self):
        decision, _ = judge.claim_verdict(
            [1.1, 1.2], [1.0, 1.0], self._hashes(4), 0.004, 2.0, 0.008, 1
        )
        self.assertEqual(decision, "更差")


class TestStopRuleIsFrozen(unittest.TestCase):
    """缺陷 6：预算的执行权原本一半在 agent 可改的 train.py 里。"""

    def test_task_exports_the_stop_rule(self):
        text = Path("task.py").read_text(encoding="utf-8")
        self.assertIn("def should_stop(", text)

    def test_train_defers_to_it(self):
        text = Path("train.py").read_text(encoding="utf-8")
        self.assertIn("should_stop(", text)
        self.assertNotIn("elapsed >= TIME_BUDGET", text)
        self.assertNotIn("* TIME_BUDGET:", text)


class TestPerArmNoise(unittest.TestCase):
    """缺陷 7：σ 是配置的属性，两臂不能假设同方差。"""

    def test_a_quiet_arm_never_goes_below_the_calibration(self):
        # 三个样本看着安静往往只是抽样不足。
        self.assertEqual(judge.arm_sigma([1.0, 1.0, 1.0], 0.004), 0.004)
        self.assertEqual(judge.arm_sigma([1.000, 1.001], 0.004), 0.004)

    def test_a_noisy_arm_is_believed(self):
        samples = [0.990, 0.998, 1.012]
        self.assertAlmostEqual(
            judge.arm_sigma(samples, 0.004), statistics.stdev(samples), places=12
        )

    def test_one_sample_falls_back_to_the_calibration(self):
        self.assertEqual(judge.arm_sigma([1.0], 0.004), 0.004)
        self.assertEqual(judge.arm_sigma([], 0.004), 0.004)

    def test_the_band_uses_each_arm_separately(self):
        band = judge.band_for(0.008, 0.002, 2.0, 3, 5)
        expected = 2.0 * math.sqrt(0.008**2 / 3 + 0.002**2 / 5)
        self.assertAlmostEqual(band, expected, places=12)

    def test_equal_sigmas_reproduce_the_old_formula(self):
        band = judge.band_for(0.004, 0.004, 2.0, 3, 5)
        expected = 2.0 * 0.004 * math.sqrt(1.0 / 3 + 1.0 / 5)
        self.assertAlmostEqual(band, expected, places=12)

    def test_the_night_two_accept_flips_to_indistinguishable(self):
        # 候选 65/78/67（sd 7.0 张）对父本 72..80（sd 3.16 张），改进 6.0 张。
        # 假设同方差时 band 4.62 张 → 好于父本；按各自方差 band 8.56 张 → 分不开。
        cand = [65 / 1498, 78 / 1498, 67 / 1498]
        parent = [n / 1498 for n in (72, 74, 76, 78, 80)]
        calibrated = 3.16 / 1498
        band = judge.band_for(
            judge.arm_sigma(cand, calibrated), judge.arm_sigma(parent, calibrated),
            2.0, len(cand), len(parent),
        )
        self.assertGreater(band * 1498, 8.0)
        self.assertEqual(
            judge.compare_to_parent(
                statistics.fmean(cand), statistics.fmean(parent), band
            ),
            "分不开",
        )


class TestSelectionScalesTheError(unittest.TestCase):
    """缺陷 8：扣减原本是固定偏移，不随样本数缩小，使多测样本对 claim 无用。"""

    def _args(self, cand, parent, n):
        return (cand, parent, ["e"] * (len(cand) + len(parent)), 0.004, 2.0, 0.0, n)

    def test_more_samples_lower_the_threshold(self):
        few = judge.claim_threshold(0.004, 0.004, 2.0, 2, 2, 4)
        many = judge.claim_threshold(0.004, 0.004, 2.0, 10, 10, 4)
        self.assertLess(many, few)

    def test_more_mechanisms_raise_the_threshold(self):
        quiet = judge.claim_threshold(0.004, 0.004, 2.0, 3, 5, 2)
        crowded = judge.claim_threshold(0.004, 0.004, 2.0, 3, 5, 400)
        self.assertGreater(crowded, quiet)

    def test_the_threshold_always_exceeds_the_accept_band(self):
        for m in (2, 3, 10):
            for n in (2, 5, 10):
                for mech in (1, 2, 50):
                    band = judge.band_for(0.004, 0.004, 2.0, m, n)
                    thr = judge.claim_threshold(0.004, 0.004, 2.0, m, n, mech)
                    self.assertGreater(thr, band, f"m={m} n={n} N={mech}")

    def test_a_real_gain_can_now_be_certified(self):
        # 改进 10 张，两臂都安静：旧规则门槛 11.6 张判不成立，新规则 8.5 张判成立。
        cand = [(76 - 10) / 1498] * 3
        parent = [n / 1498 for n in (72, 74, 76, 78, 80)]
        decision, _ = judge.claim_verdict(
            cand, parent, ["e"] * 8, 3.16 / 1498, 2.0, 0.0, 4
        )
        self.assertEqual(decision, "成立")

    def test_the_policy_floor_is_a_separate_gate(self):
        cand = [(76 - 10) / 1498] * 3
        parent = [n / 1498 for n in (72, 74, 76, 78, 80)]
        # 统计上够了，但人设的政策下限是 20 张 —— 仍然不成立。
        decision, detail = judge.claim_verdict(
            cand, parent, ["e"] * 8, 3.16 / 1498, 2.0, 20.0 / 1498, 4
        )
        self.assertEqual(decision, "没有更好")

    def test_a_zero_floor_is_a_valid_policy(self):
        cand = [(76 - 10) / 1498] * 3
        parent = [n / 1498 for n in (72, 74, 76, 78, 80)]
        decision, detail = judge.claim_verdict(
            cand, parent, ["e"] * 8, 3.16 / 1498, 2.0, 0.0, 4
        )
        self.assertNotEqual(detail, "未标定")
        self.assertEqual(decision, "成立")


class TestOverrunProjection(unittest.TestCase):
    """缺陷 9：步数预算与墙钟上限互不知情，注定跑不完的配置要烧满墙钟才被发现。"""

    def test_a_run_on_pace_is_not_cut(self):
        # 10200 步、540 秒上限；当前速率投影 464 秒，跑得完。
        self.assertFalse(judge.will_overrun(1000, 45.5, 10200, 540.0))

    def test_a_run_off_pace_is_cut_early(self):
        # 第四夜轮 2 的真实速率：9990 步用了 542 秒，投影 553 秒 > 540。
        self.assertTrue(judge.will_overrun(1000, 54.3, 10200, 540.0))

    def test_nothing_is_judged_before_the_probe(self):
        # 探测期内不下结论，哪怕速率看着很糟——前几百步还没稳定。
        self.assertFalse(judge.will_overrun(999, 500.0, 10200, 540.0))
        self.assertFalse(judge.will_overrun(0, 0.0, 10200, 540.0))

    def test_the_probe_length_is_adjustable(self):
        self.assertFalse(judge.will_overrun(200, 11.0, 10200, 540.0, probe_steps=500))
        self.assertTrue(judge.will_overrun(500, 27.2, 10200, 540.0, probe_steps=500))

    def test_the_saving_is_an_order_of_magnitude(self):
        # 按第四夜轮 2 的速率，投影在 1000 步（约 54 秒）就能下结论，
        # 而不是跑满 540 秒才发现。
        rate = 542.5 / 9990
        self.assertTrue(judge.will_overrun(1000, rate * 1000, 10200, 540.0))
        self.assertLess(rate * 1000, 540.0 / 9)
