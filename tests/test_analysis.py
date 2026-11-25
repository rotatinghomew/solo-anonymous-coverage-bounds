import unittest
from pathlib import Path

from coverplan.analysis import (VERDICT_EVIDENCE, VERDICT_OK, VERDICT_SEATS,
                                free_slots)
from coverplan.evidence import build_evidence
from coverplan.model import load_records
from coverplan.report import build_report

DATA = Path(__file__).resolve().parents[1] / "data" / "records.json"


class AnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_records(DATA)
        cls.cells = build_evidence(cls.records)
        cls.report = build_report(cls.records)

    def verdict_of(self, label):
        return next(v for v in self.report["verdicts"] if v.label == label)

    # ---- 相容性 ----
    def test_active_evidence_consistent(self):
        for key, info in self.report["consistency"]["cells"].items():
            self.assertTrue(info["consistent"], key)

    def test_revoked_certificate_not_in_evidence(self):
        active = self.report["consistency"]["cells"][("夜间", "2026-09")]["active_refs"]
        self.assertNotIn("夜间-2026-09-4", active)      # 已撤回
        self.assertIn("夜间-2026-09-8", active)         # 补发仍有效

    def test_new_certificate_conflict(self):
        nc = self.report["consistency"]["new_certificate"]
        self.assertEqual(nc["ref"], "N1")
        pairwise = nc["conflict_with"]["pairwise"]
        # 并集上限 27 与下单点 >27 的凭证及并集凭证两两冲突
        self.assertIn("夜间-2026-09-8", pairwise)
        self.assertIn("回件F2", pairwise)
        self.assertNotIn("夜间-2026-09-3", pairwise)    # 竹 [25,25] 与 [22,27] 可相容

    def test_periods_not_mixed(self):
        # 不同期间没有可比观察窗：08 与 09 各自独立成立
        self.assertIn(("夜间", "2026-08"), self.report["consistency"]["cells"])
        self.assertIn(("夜间", "2026-09"), self.report["consistency"]["cells"])

    # ---- 覆盖查询 ----
    def test_unique_and_displaced_ranges(self):
        q = self.report["queries"][("夜间", "2026-09")]
        self.assertEqual(q["unique"]["松"], (2, 30))
        self.assertEqual(q["unique"]["杉"], (0, 26))
        self.assertEqual(q["displaced"][("松",)], (2, 30))
        # 同时停两站的缺口大于各自相加的下界：重叠家庭一起失去服务
        self.assertEqual(q["displaced"][("杉", "松")], (30, 51))
        self.assertEqual(q["overlap_only"], (0, 13))

    # ---- 计划评估 ----
    def test_free_slots_after_confirmed(self):
        self.assertEqual(free_slots(self.records), {"J1": 5, "J2": 7})

    def test_request_verdicts(self):
        self.assertEqual(self.verdict_of("K1").verdict, VERDICT_EVIDENCE)
        self.assertEqual(self.verdict_of("K2").verdict, VERDICT_EVIDENCE)
        self.assertEqual(self.verdict_of("K3").verdict, VERDICT_EVIDENCE)
        # 分别可办不代表一起可办：一起办是席位不足
        joint = self.verdict_of("K1+K2")
        self.assertEqual(joint.verdict, VERDICT_SEATS)
        self.assertLess(joint.best_case, joint.group_min)

    def test_guarantee_numbers(self):
        k1 = self.verdict_of("K1")
        self.assertEqual(k1.guarantee, 53)   # 剩余并集下界 48 + 确认 3 + 可分配 2
        self.assertEqual(k1.allocatable["K1"], 2)
        k3 = self.verdict_of("K3")
        self.assertEqual(k3.guarantee, 56)
        self.assertEqual(k3.allocatable["K3"], 0)

    # ---- 征询比较 ----
    def appraisal_of(self, iid):
        return next(a for a in self.report["inquiries"] if a.id == iid)

    def test_inquiry_f3_decisive(self):
        f3 = self.appraisal_of("F3")
        self.assertEqual(f3.status, "可期待")
        self.assertTrue(f3.changes_anything)
        self.assertEqual(f3.effects["K1"], 95)   # 回件下界 ≥95 时 K1 可确认
        self.assertEqual(f3.effects["K2"], 96)   # 回件下界 ≥96 时 K2 可确认
        self.assertIsNone(f3.effects["K1+K2"])   # 一起办席位不足，与回件无关

    def test_inquiry_f5_irrelevant(self):
        f5 = self.appraisal_of("F5")
        self.assertEqual(f5.status, "可期待")
        self.assertFalse(f5.changes_anything)    # 无论返回什么都不改变决定

    def test_inquiry_f4_too_late(self):
        self.assertEqual(self.appraisal_of("F4").status, "不可期待")

    def test_pending_not_zero(self):
        pend = self.report["consistency"]["cells"][("夜间", "2026-09")]["pending_inquiries"]
        self.assertEqual(pend, ["F3"])           # 缺回件，不算零

    def test_mutual_return_consistency(self):
        # 多份征询的有利回件须联合相容，不能拼接各自有利回答
        mutual = self.report["mutual"]
        self.assertTrue(all(mutual["consistent"].values()))
        # 同一层内若强行拼接不相容回答应判为不可行
        from coverplan.analysis import mutual_return_consistency
        bad = mutual_return_consistency(self.records, self.cells, {"F3": (96, 97)})
        self.assertTrue(bad[("夜间", "2026-09")])
        # F3 回件若越出证据范围（下界 98 超过并集上界 97）则不相容
        bad2 = mutual_return_consistency(self.records, self.cells, {"F3": (98, 99)})
        self.assertFalse(bad2[("夜间", "2026-09")])

    # ---- 共享与授权 ----
    def test_parallel_release_ok(self):
        cell = self.report["sharing"]["cells"][("夜间", "2026-09")]
        self.assertTrue(cell["ok"])              # 并行发出不泄露未授权细分
        self.assertIn("回件F2", cell["shared"])  # F2 授权共享
        self.assertNotIn("回件F1", cell["shared"])  # F1 仅授权规划

    # ---- 原有承诺 ----
    def test_commitments(self):
        text = "\n".join(self.report["commitments"])
        self.assertIn("服务继续占用原席位", text)
        self.assertIn("不得用于新承诺", text)


if __name__ == "__main__":
    unittest.main()
