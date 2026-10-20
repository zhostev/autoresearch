import unittest
from pathlib import Path

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
        self.assertEqual(judge.count_n(rows), 3)
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
            ["--seed", "1", "--mechanism", "学习率", "--expect", "lower", "--fail", "崩了"],
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
            ["--seed", "1", "--mechanism", "学习率", "--expect", "lower", "--fail", "崩了"],
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
