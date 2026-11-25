"""汇总分析报告：把各项核对与评估整理成规划人员可读的结论。"""
from __future__ import annotations

from . import analysis
from .analysis import venn
from .evidence import build_evidence
from .model import EVT_CONFIRMED, EVT_REISSUED, EVT_REVOKED


def _fmt_interval(iv):
    if iv is None:
        return "不可行"
    return f"[{iv[0]}, {iv[1]}]"


def build_report(records):
    cells = build_evidence(records)
    consistency = analysis.consistency_report(records, cells)
    queries = {key: analysis.coverage_queries(records, cells, *key)
               for key in cells}
    verdicts = analysis.evaluate_requests(records, cells)
    inquiries = analysis.evaluate_inquiries(records, cells)
    sharing = analysis.sharing_check(records, cells)
    commitments = _commitments(records)
    mutual = _mutual_check(records, cells, inquiries)
    return {
        "consistency": consistency,
        "queries": queries,
        "verdicts": verdicts,
        "inquiries": inquiries,
        "sharing": sharing,
        "commitments": commitments,
        "mutual": mutual,
        "free_slots": analysis.free_slots(records),
    }


def _mutual_check(records, cells, inquiries):
    """可期待征询的有利回件能否联合相容（不能拼各自有利回答来承诺）。"""
    chosen = {}
    inq = {i.id: i for i in records.inquiries}
    for ap in inquiries:
        if ap.status != "可期待":
            continue
        iq = inq[ap.id]
        thresholds = [t for t in ap.effects.values() if t is not None]
        ev = cells[(iq.group, iq.period)]
        vec = ev.cell.inter_vec(iq.points) if iq.relation == "交集" else ev.cell.union_vec(iq.points)
        qlo, qhi = venn.bound(ev.cell, ev.constraints, vec)
        if thresholds:
            a = max(thresholds)  # 最苛刻的有利回件
            chosen[ap.id] = (a, min(a + iq.precision, qhi))
        else:
            chosen[ap.id] = (qlo, qhi)  # 与决定无关：取当前全范围
    if not chosen:
        return {"chosen": {}, "consistent": {}}
    return {"chosen": chosen,
            "consistent": analysis.mutual_return_consistency(records, cells, chosen)}


def _commitments(records):
    """原有承诺的处置：已确认服务继续；撤回凭证不得支持新承诺。"""
    out = []
    reqs = {r.request: r for r in records.requests}
    for h in records.history:
        if h.event == EVT_CONFIRMED:
            out.append(f"{h.date} {h.request} 已确认 {h.slots} 席（凭证 {h.certificate}）："
                       f"服务继续占用原席位")
        elif h.event == EVT_REVOKED:
            out.append(f"{h.date} 撤回凭证 {h.certificate}：不再作为证据，"
                       f"已确认服务不自动取消，但不得用于新承诺")
        elif h.event == EVT_REISSUED:
            out.append(f"{h.date} 补发凭证 {h.certificate}：继续有效")
    return out


def render_text(report, records):
    lines = []
    add = lines.append
    add("# 家长服务覆盖承诺分析报告")
    add("")
    add("## 一、凭证相容性")
    for (group, period), info in sorted(report["consistency"]["cells"].items()):
        mark = "相容" if info["consistent"] else "不相容"
        add(f"- {group}/{period}：{mark}"
            f"（有效凭证 {len(info['active_refs'])} 份"
            f"＋已收到回件 {len(info['received_returns'])} 份）")
        if info["pending_inquiries"]:
            add(f"    未收到（缺回件，不算零）：{', '.join(info['pending_inquiries'])}")
        if info["late_inquiries"]:
            add(f"    超过最迟收件日，不可期待：{', '.join(info['late_inquiries'])}")
    nc = report["consistency"]["new_certificate"]
    if nc["conflict_with"]:
        cw = nc["conflict_with"]
        add(f"- 待核来件 {nc['ref']}：矛盾来件，无法同时成立——")
        if cw["pairwise"]:
            add(f"    与单份凭证即冲突：{'、'.join(cw['pairwise'])}")
        if cw["joint"]:
            add(f"    与这组凭证联合冲突：{'、'.join(cw['joint'])}")
        add("    相关凭证全部保留待核对，不自行删除任何一张。")
    else:
        add(f"- 待核来件 {nc['ref']}：与现有证据相容。")
    add("  注：不同期间没有可比观察窗，各（分层, 期间）单独成立，不互相借用。")
    add("")
    add("## 二、覆盖查询（当前证据下的数量区间）")
    for (group, period), q in sorted(report["queries"].items()):
        add(f"- {group}/{period}：")
        add(f"    独有覆盖：" + "，".join(f"{p} {_fmt_interval(iv)}" for p, iv in q["unique"].items()))
        if q["joint"]:
            add("    联合覆盖：" + "，".join(f"{'∩'.join(k)} {_fmt_interval(iv)}"
                                          for k, iv in q["joint"].items()))
        if q["displaced"]:
            add("    改时段挤出：" + "，".join(f"停{'+'.join(k)} {_fmt_interval(iv)}"
                                            for k, iv in q["displaced"].items()))
        if "overlap_only" in q:
            add(f"    其中两站重叠家庭（只到访这两站，同时停开一起失去服务）："
                f"{_fmt_interval(q['overlap_only'])}")
    add("")
    add("## 三、计划评估（撤回与来件后重做同一组计划）")
    add(f"    剩余席位：" + "，".join(f"{k} {v} 席" for k, v in report["free_slots"].items()))
    for pv in report["verdicts"]:
        add(f"- {pv.label}（{pv.group}，停 {'+'.join(pv.removed)}）：{pv.verdict}")
        for r in pv.requests:
            lo, hi, fl, ok = pv.floors[r]
            add(f"    {r} 独有人数 {_fmt_interval((lo, hi))}，保底 {fl}"
                f"（{'已证明' if ok else '未证明'}），可分配新席位 {pv.allocatable.get(r, 0)}")
        add(f"    保留人数保证 {pv.guarantee} / 层保底 {pv.group_min}"
            f"（最有利构成可达 {pv.best_case}）")
        for reason in pv.reasons:
            add(f"    · {reason}")
    add("")
    add("## 四、征询方案比较"
        f"（预算 {records.requirements.inquiry_budget}，最迟收件 {records.requirements.last_received}）")
    for ap in report["inquiries"]:
        add(f"- {ap.id}（{ap.status}，费用 {ap.cost}，交付 {ap.ready_by}）：")
        if ap.status == "可期待":
            for label, threshold in ap.effects.items():
                if threshold is None:
                    add(f"    任何回件都不能使 {label} 可确认")
                else:
                    add(f"    回件下界 ≥ {threshold} 时 {label} 可确认")
            if not ap.changes_anything:
                add("    无论返回什么都不改变当前决定")
        if ap.note:
            add(f"    · {ap.note}")
    mutual = report.get("mutual", {})
    if mutual.get("chosen"):
        combo = "，".join(f"{k}∈[{v[0]},{v[1]}]" for k, v in mutual["chosen"].items())
        ok = all(mutual["consistent"].values())
        verdict = "联合相容，可共同支持承诺" if ok else "互不相容，不能拼接各自有利回答"
        add(f"- 多份征询联合核对：{combo} → {verdict}。")
    add("")
    add("## 五、共享与授权")
    for (group, period), cellrep in sorted(report["sharing"]["cells"].items()):
        shared = "、".join(cellrep["shared"]) or "（无）"
        if cellrep["ok"]:
            add(f"- {group}/{period}：已共享（{shared}）可并行发出，"
                f"未唯一确定任何未授权原子细分或仅规划量。")
        else:
            add(f"- {group}/{period}：不得并行发出！可反推：")
            for kind, name, val in cellrep["breaches"]:
                add(f"    · {kind} {name} 被确定为 {val}")
    add("  仅授权规划的凭证与回件不得对外，也不得经反复加减共享结果反推细分人数。")
    add("")
    add("## 六、原有承诺的处置")
    for line in report["commitments"]:
        add(f"- {line}")
    return "\n".join(lines)
