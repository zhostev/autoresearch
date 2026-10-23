"""expctl 能力补充：内容寻址 run 身份、append-only 账本、配对比较、审计。

这三样的共同点：判据的输入必须是可复算的事实，而不是 agent 的叙述。
- run_id 由 commit / eval_hash / seed / knobs / 预算算出；
- 每次 trial 在训练**之前**落一条 admit，训练之后落一条 result；
- compare 用共享种子做配对 t 检验，至少 2 对，且两臂必须同一把尺子；
- audit 只读，把账本、结果表、名单、队列之间的不一致挑出来。

Covers: run_identity, ledger, compare, audit
"""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import judge


def git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    )
    return completed.stdout.strip()


class ParityFixture(unittest.TestCase):
    def setUp(self):
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
        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def _identity(self):
        return judge.eval_hash(
            self.root / "task.py", self.root / "prepare.py", self.root / "data"
        )

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
            "trial_repeats": 3,
        }
        config.update(overrides)
        judge.write_run(self.root, config)
        return config

    def _prepare_child(self):
        parent = git_head(self.root)
        self._write_run()
        judge.write_parents(self.root / "parents.tsv", [{
            "commit": parent,
            "mechanism": "基线",
            "val_bpb": "1.000000",
            "seeds": "1,2,3,4,5",
            "reviewed": "yes",
            "simpler": "no",
        }])
        base = judge.blank_row()
        base.update({
            "commit": parent,
            "mechanism": "基线",
            "expect": "lower",
            "fail": "基线",
            "seed": "1",
            "val_bpb": "1.000000",
            "eval_hash": self._identity(),
            "decision": "基线",
        })
        judge.append_result(self.root / "results.tsv", base)
        (self.root / "train.py").write_text("print('next')\n", encoding="utf-8")
        self._git("add", "train.py")
        self._git("commit", "-m", "try")
        return parent

    def _trial(self, command, seed="1", mechanism="学习率"):
        return judge.main(
            ["trial", "--seed", seed, "--mechanism", mechanism,
             "--knobs", "", "--expect", "lower", "--fail", "变高"],
            command=command, timeout_seconds=5, root=self.root,
        )

    def _row(self, commit, seed, value, decision="已记录", eval_hash=None, knobs=""):
        row = judge.blank_row()
        row.update({
            "commit": commit,
            "parent": "",
            "mechanism": "学习率",
            "knobs": knobs,
            "expect": "lower",
            "fail": "变高",
            "seed": seed,
            "val_bpb": value,
            "eval_hash": self._identity() if eval_hash is None else eval_hash,
            "decision": decision,
        })
        return row


class TestRunIdentity(unittest.TestCase):
    def test_identity_is_content_addressed(self):
        first = judge.run_identity("c", "p", "e", "1", "LR", 100)
        self.assertEqual(first, judge.run_identity("c", "p", "e", "1", "LR", 100))
        variants = {
            judge.run_identity("c", "p", "e", "2", "LR", 100),
            judge.run_identity("c", "p", "e2", "1", "LR", 100),
            judge.run_identity("c", "p", "e", "1", "WD", 100),
            judge.run_identity("c", "p", "e", "1", "LR", 200),
            judge.run_identity("c2", "p", "e", "1", "LR", 100),
        }
        self.assertNotIn(first, variants)
        self.assertEqual(len(variants), 5)

    def test_missing_budget_is_distinct_from_zero(self):
        self.assertNotEqual(
            judge.run_identity("c", "p", "e", "1", "LR", None),
            judge.run_identity("c", "p", "e", "1", "LR", 0),
        )


class TestPairedPValue(unittest.TestCase):
    def test_matches_student_t_for_three_pairs(self):
        # deltas 1,2,3 → t=3.4641, df=2 → 双侧 p=0.07418
        self.assertAlmostEqual(judge.paired_p_value([1.0, 2.0, 3.0]), 0.07418, places=4)

    def test_zero_variance_is_undefined(self):
        self.assertIsNone(judge.paired_p_value([0.1, 0.1, 0.1]))

    def test_single_pair_is_undefined(self):
        self.assertIsNone(judge.paired_p_value([0.1]))


class TestPairedCompare(ParityFixture):
    def _pairs(self):
        rows = []
        for seed, base, treat in (
            ("1", "1.000", "0.900"),
            ("2", "0.990", "0.895"),
            ("3", "1.010", "0.905"),
        ):
            rows.append(self._row("a", seed, base))
            rows.append(self._row("b", seed, treat))
        return rows

    def test_significant_when_deltas_are_consistent(self):
        result = judge.paired_compare(self._pairs(), "a", "b", 0.01)
        self.assertEqual(result["verdict"], "significant")
        self.assertEqual(result["direction"], "better")
        self.assertAlmostEqual(result["mean_delta"], -0.1, places=6)
        self.assertEqual(result["pairs"], 3)

    def test_needs_at_least_two_pairs(self):
        rows = [self._row("a", "1", "1.0"), self._row("b", "1", "0.9")]
        result = judge.paired_compare(rows, "a", "b", 0.0)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertIn("1 对", result["reason"])

    def test_different_rulers_are_not_comparable(self):
        rows = [
            self._row("a", "1", "1.0"), self._row("a", "2", "1.0"),
            self._row("b", "1", "0.9", eval_hash="other"),
            self._row("b", "2", "0.9", eval_hash="other"),
        ]
        result = judge.paired_compare(rows, "a", "b", 0.0)
        self.assertEqual(result["verdict"], "not_comparable")
        self.assertIn("eval_hash", result["reason"])

    def test_below_mde_stays_inconclusive(self):
        result = judge.paired_compare(self._pairs(), "a", "b", 0.5)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertIn("MDE", result["reason"])


class TestTrialLedger(ParityFixture):
    def test_trial_writes_admit_before_result(self):
        parent = self._prepare_child()
        command = [
            sys.executable, "-c",
            "print('seed: 1')\nprint('num_steps: 10200')\n"
            "print('val_bpb:          1.005000')",
        ]
        self.assertEqual(self._trial(command), 0)
        events, bad = judge.read_ledger(self.root)
        self.assertEqual(bad, 0)
        self.assertEqual([event["event"] for event in events], ["admit", "result"])
        child = git_head(self.root)
        expected = judge.run_identity(
            child, parent, self._identity(), "1", "", None
        )
        self.assertEqual(events[0]["run_id"], expected)
        self.assertEqual(events[1]["run_id"], expected)
        self.assertEqual(events[1]["val_bpb"], "1.005000")
        self.assertEqual(events[1]["steps"], 10200)
        self.assertIsInstance(events[1]["seconds"], float)

    def test_completed_identity_is_refused_before_training(self):
        self._prepare_child()
        marker = self.root / "runs.txt"
        script = self.root / "counting_train.py"
        script.write_text(
            "import os\n"
            "with open('runs.txt', 'a') as handle:\n"
            "    handle.write(os.environ['AUTORES_SEED'] + '\\n')\n"
            "print('seed: ' + os.environ['AUTORES_SEED'])\n"
            "print('val_bpb:          1.005000')\n",
            encoding="utf-8",
        )
        command = [sys.executable, str(script)]
        self.assertEqual(self._trial(command), 0)
        first = marker.read_text(encoding="utf-8")
        child = git_head(self.root)
        kept = [
            row for row in judge.read_results(self.root / "results.tsv")
            if row["commit"] != child
        ]
        path = self.root / "results.tsv"
        path.unlink()
        for row in kept:
            judge.append_result(path, row)
        self.assertEqual(self._trial(command), 2)
        self.assertEqual(marker.read_text(encoding="utf-8"), first)

    def test_failed_measurement_can_be_retried(self):
        self._prepare_child()
        broken = [sys.executable, "-c", "raise SystemExit(1)"]
        self.assertEqual(self._trial(broken), 0)
        events, _ = judge.read_ledger(self.root)
        self.assertEqual(events[-1]["decision"], "代码坏了")
        good = [
            sys.executable, "-c",
            "print('seed: 1')\nprint('val_bpb:          1.005000')",
        ]
        # 同一条身份在失败后允许重试：账本只拦已完成的 run
        code = judge.main(
            ["trial", "--seed", "1", "--mechanism", "学习率",
             "--knobs", "", "--expect", "lower", "--fail", "变高"],
            command=good, timeout_seconds=5, root=self.root,
        )
        self.assertEqual(code, 0)
        last = judge.read_results(self.root / "results.tsv")[-1]
        self.assertNotEqual(last["decision"], "代码坏了")


class TestAudit(ParityFixture):
    def test_duplicate_completed_rows_are_errors(self):
        row = self._row("abc", "1", "1.0")
        judge.append_result(self.root / "results.tsv", row)
        judge.append_result(self.root / "results.tsv", row)
        report = judge.build_audit(self.root)
        self.assertFalse(report["ok"])
        self.assertTrue(any("完成了 2 次" in item for item in report["errors"]))

    def test_missing_eval_hash_is_an_error(self):
        judge.append_result(
            self.root / "results.tsv", self._row("abc", "1", "1.0", eval_hash="")
        )
        report = judge.build_audit(self.root)
        self.assertFalse(report["ok"])
        self.assertTrue(any("eval_hash" in item for item in report["errors"]))

    def test_legacy_rows_are_warnings_not_errors(self):
        judge.append_result(self.root / "results.tsv", self._row("abc", "1", "1.0"))
        report = judge.build_audit(self.root)
        self.assertTrue(report["ok"])
        self.assertTrue(any("早于账本" in item for item in report["warnings"]))

    def test_orphan_queue_entry_is_flagged(self):
        judge.write_queue(self.root / "queue.tsv", [{
            "commit": "deadbeef" * 5,
            "parent": "",
            "mechanism": "x",
            "accepted_at": "2026-01-01T00:00:00+08:00",
            "state": "待复核",
        }])
        report = judge.build_audit(self.root)
        self.assertTrue(any("不存在" in item for item in report["warnings"]))

    def test_multi_knob_accept_is_flagged(self):
        parent = git_head(self.root)
        judge.write_parents(self.root / "parents.tsv", [{
            "commit": parent,
            "mechanism": "半精度",
            "val_bpb": "1.000000",
            "seeds": "1",
            "reviewed": "yes",
            "simpler": "no",
        }])
        judge.append_result(
            self.root / "results.tsv",
            self._row(parent, "1", "1.0", knobs="IMAGE_SIZE,AMP"),
        )
        report = judge.build_audit(self.root)
        self.assertTrue(any("多个旋钮" in item for item in report["warnings"]))

    def test_audit_cli_is_nonzero_when_errors_exist(self):
        row = self._row("abc", "1", "1.0")
        judge.append_result(self.root / "results.tsv", row)
        judge.append_result(self.root / "results.tsv", row)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = judge.main(["audit", "--json"], root=self.root)
        self.assertEqual(code, 1)
        payload = json.loads(out.getvalue())
        self.assertFalse(payload["ok"])
        self.assertGreaterEqual(len(payload["errors"]), 1)


class TestCompareCli(ParityFixture):
    def _run(self, base, treat, mde="0.01"):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = judge.main(
                ["compare", "--base", base, "--treat", treat,
                 "--mde", mde, "--json"],
                root=self.root,
            )
        self.assertEqual(code, 0)
        return json.loads(out.getvalue())

    def test_json_verdict_is_machine_readable(self):
        for seed, base, treat in (
            ("1", "1.000", "0.900"),
            ("2", "0.990", "0.895"),
            ("3", "1.010", "0.905"),
        ):
            judge.append_result(
                self.root / "results.tsv", self._row("a" * 40, seed, base)
            )
            judge.append_result(
                self.root / "results.tsv", self._row("b" * 40, seed, treat)
            )
        payload = self._run("a" * 7, "b" * 7)
        self.assertEqual(payload["verdict"], "significant")
        self.assertAlmostEqual(payload["mean_delta"], -0.1, places=6)
        self.assertEqual(payload["pairs"], 3)

    def test_zero_variance_is_inconclusive(self):
        for seed in ("1", "2", "3"):
            judge.append_result(
                self.root / "results.tsv", self._row("a" * 40, seed, "1.000")
            )
            judge.append_result(
                self.root / "results.tsv", self._row("b" * 40, seed, "0.900")
            )
        payload = self._run("a" * 7, "b" * 7)
        self.assertEqual(payload["verdict"], "inconclusive")
        self.assertIn("方差为 0", payload["reason"])

    def test_unknown_commit_is_refused(self):
        code = judge.main(
            ["compare", "--base", "deadbee", "--treat", "deadbee"],
            root=self.root,
        )
        self.assertEqual(code, 2)


class TestComputeRatio(ParityFixture):
    def test_ratio_comes_from_ledger_steps(self):
        for commit, steps in (("a", 1000), ("b", 3000)):
            judge.append_ledger(self.root, {
                "event": "result",
                "run_id": commit,
                "commit": commit,
                "seed": "1",
                "decision": "已记录",
                "val_bpb": "1.0",
                "steps": steps,
            })
        self.assertAlmostEqual(judge.compute_ratio(self.root, "a", "b"), 3.0)

    def test_failed_rows_do_not_contribute_steps(self):
        judge.append_ledger(self.root, {
            "event": "result", "run_id": "a", "commit": "a", "seed": "1",
            "decision": "代码坏了", "val_bpb": "", "steps": 100,
        })
        judge.append_ledger(self.root, {
            "event": "result", "run_id": "b", "commit": "b", "seed": "1",
            "decision": "已记录", "val_bpb": "1.0", "steps": 3000,
        })
        self.assertIsNone(judge.compute_ratio(self.root, "a", "b"))


if __name__ == "__main__":
    unittest.main()
