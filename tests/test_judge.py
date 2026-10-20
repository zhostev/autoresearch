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
