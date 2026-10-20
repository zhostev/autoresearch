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


if __name__ == "__main__":
    unittest.main()
