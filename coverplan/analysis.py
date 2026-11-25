"""业务分析：相容性、覆盖查询、计划评估、征询比较、共享核对。

判定语义（与 README 资料字典一致）：
- 请求的 unique_floor 是证据要求：须由当前证据证明独有人数下界 >= floor，
  否则为证据不足；
- 席位用于保住各层保底（group_retained_minimum）：保留人数 =
  |剩余服务点并集| + 已确认席位 + 实际可分配席位，须对全部相容构成成立；
- 实际可分配席位 <= 独有人数下界（联办方确认每席必接住一个尚未被承接的
  家庭，不能多于该点独有人数），同一席位只被一个请求占用，历史耗用继续扣；
- 席位不足：连最有利构成加全部剩余席位都够不到层保底（再补证据也没用）；
- 证据不足：有利构成可达、不利构成不可达（补足证据可能改变决定）；
- 一起停开的重叠家庭会一起失去服务，联合评估按停开集合的并集缺口计算。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import venn
from .evidence import build_evidence, shared_constraints
from .model import (AUD_SHARE, EVT_CONFIRMED, REL_INTERSECT)

VERDICT_OK = "可确认"
VERDICT_EVIDENCE = "证据不足"
VERDICT_SEATS = "席位不足"
VERDICT_CONFLICT = "矛盾来件"


# ---------------------------------------------------------------- 基础查询

def coverage_queries(records, cells, group, period, extra=()):
    """独有覆盖、联合覆盖、改时段挤出数量的区间（对全部相容构成）。"""
    ev = cells[(group, period)]
    cons = ev.constraints + list(extra)
    cell = ev.cell
    out = {"unique": {}, "joint": {}, "displaced": {}}
    for p in cell.points:
        out["unique"][p] = venn.bound(cell, cons, cell.unique_vec(p))
    pairs = sorted({tuple(sorted(c.points)) for c in records.certificates
                    if c.group == group and c.period == period and len(c.points) == 2})
    for pair in pairs:
        out["joint"][pair] = venn.bound(cell, cons, cell.inter_vec(pair))
    removed_sets = {r.remove_point for r in records.requests
                    if r.group == group and r.period == period}
    for p in sorted(removed_sets):
        out["displaced"][(p,)] = venn.bound(cell, cons, cell.displaced_vec([p]))
    if len(removed_sets) >= 2:
        both = sorted(removed_sets)
        out["displaced"][tuple(both)] = venn.bound(cell, cons, cell.displaced_vec(both))
        # 两站重叠家庭：只到访这两点的家庭，同时停开时一起失去服务
        out["overlap_only"] = venn.bound(cell, cons, cell.region_vec(both))
    return out


def consistency_report(records, cells):
    """各（分层, 期间）证据相容性 + 待核来件矛盾核对。"""
    report = {}
    for key, ev in cells.items():
        report[key] = {
            "consistent": venn.feasible(ev.cell, ev.constraints),
            "active_refs": list(ev.active_refs),
            "received_returns": list(ev.received_returns),
            "pending_inquiries": list(ev.pending_inquiries),
            "late_inquiries": list(ev.late_inquiries),
        }
    n1 = records.new_certificate
    ev = cells[(n1.group, n1.period)]
    vec = ev.cell.inter_vec(n1.points) if n1.relation == REL_INTERSECT else ev.cell.union_vec(n1.points)
    n1c = venn.Constraint(vec, n1.lower, n1.upper, label=n1.ref)
    if venn.feasible(ev.cell, ev.constraints + [n1c]):
        conflict = None
    else:
        pairwise, joint = venn.conflicts_with(ev.cell, ev.constraints, n1c)
        conflict = {"pairwise": pairwise, "joint": joint}
    return {"cells": report, "new_certificate": {"ref": n1.ref, "conflict_with": conflict}}


# ---------------------------------------------------------------- 计划评估

def free_slots(records):
    """每槽剩余席位 = 席位 - 已确认占用（历史耗用继续扣除）。"""
    free = {s.slot: s.seats for s in records.slots}
    reqs = {r.request: r for r in records.requests}
    for h in records.history:
        if h.event == EVT_CONFIRMED and h.slots:
            free[reqs[h.request].offsets[0]] -= h.slots
    return free


@dataclass
class PlanVerdict:
    label: str
    requests: list
    group: str
    period: str
    removed: list
    floors: dict          # request -> (独有下界, 独得上界, floor, 是否证明)
    allocatable: dict     # request -> 实际可分配新席位
    confirmed: dict       # request -> 已确认席位
    retained_lo: int      # 剩余并集下界
    guarantee: int        # 保留人数保证（对全部相容构成）
    best_case: int        # 席位不受独界限制时的最大保留
    group_min: int
    verdict: str
    reasons: list = field(default_factory=list)


def evaluate_plan_set(records, cells, group, period, req_list, extra=()):
    """评估同一（分层, 期间）的一组请求（单个或一起办）。"""
    ev = cells[(group, period)]
    cell = ev.cell
    cons = ev.constraints + list(extra)
    conf_all = records.confirmed_slots()
    free = free_slots(records)
    removed = []
    for r in req_list:
        if r.remove_point not in removed:
            removed.append(r.remove_point)
    remaining = [p for p in cell.points if p not in removed]
    floors, caps, offsets, confirmed = {}, {}, {}, {}
    for r in req_list:
        lo, hi = venn.bound(cell, cons, cell.unique_vec(r.remove_point))
        floors[r.request] = (lo, hi, r.unique_floor, lo >= r.unique_floor)
        caps[r.request] = lo
        offsets[r.request] = r.offsets
        confirmed[r.remove_point] = conf_all.get(r.request, 0)
    alloc_total, alloc = venn.allocatable_seats(cell, cons, caps, free, offsets)
    rem_lo, rem_hi = venn.bound(cell, cons, cell.union_vec(remaining))
    conf_total = sum(confirmed.values())
    guarantee = rem_lo + conf_total + alloc_total
    best = venn.max_retained_best_case(
        cell, cons, remaining, removed, confirmed,
        {j: free[j] for j in free}, {r.remove_point: offsets[r.request] for r in req_list})
    gmin = records.requirements.group_retained_minimum[group]
    floors_ok = all(f[3] for f in floors.values())
    reasons = []
    for r in req_list:
        lo, hi, fl, ok = floors[r.request]
        if not ok:
            reasons.append(f"{r.request} 独有人数下界 {lo} < 保底 {fl}")
    if best < gmin:
        verdict = VERDICT_SEATS
        reasons.append(f"连最有利构成加全部剩余席位最多保留 {best} < 层保底 {gmin}")
    elif not floors_ok or guarantee < gmin:
        verdict = VERDICT_EVIDENCE
        if guarantee < gmin:
            reasons.append(f"保留人数保证 {guarantee} < 层保底 {gmin}（最有利构成可达 {best}）")
    else:
        verdict = VERDICT_OK
    label = "+".join(r.request for r in req_list)
    return PlanVerdict(label, [r.request for r in req_list], group, period, removed,
                       floors, alloc, confirmed, rem_lo, guarantee, best, gmin, verdict, reasons)


def evaluate_requests(records, cells, extra=()):
    """重做同一组计划：每个请求单独评估 + 同层共用席位的一起办评估。"""
    by_cell = {}
    for r in records.requests:
        by_cell.setdefault((r.group, r.period), []).append(r)
    verdicts = []
    for (group, period), reqs in sorted(by_cell.items()):
        for r in reqs:
            verdicts.append(evaluate_plan_set(records, cells, group, period, [r], extra))
        if len(reqs) > 1:
            verdicts.append(evaluate_plan_set(records, cells, group, period, reqs, extra))
    return verdicts


# ---------------------------------------------------------------- 征询比较

@dataclass
class InquiryAppraisal:
    id: str
    status: str            # 已收到 / 可期待 / 不可期待
    cost: int
    ready_by: str
    affordable: bool
    effects: dict          # 计划集 label -> 使可确认的回件下界阈值（无则 None）
    changes_anything: bool
    note: str = ""


def _return_vec(ev, inquiry):
    if inquiry.relation == REL_INTERSECT:
        return ev.cell.inter_vec(inquiry.points)
    return ev.cell.union_vec(inquiry.points)


def mutual_return_consistency(records, cells, chosen):
    """核对一组假设回件能否与证据联合相容（按层分别）。

    chosen: {inquiry_id: (lower, upper)}。多份征询不能拼几个各自可能的
    有利回答来声称可以承诺：只有联合相容的回答组合才能共同支持承诺。
    """
    inq = {i.id: i for i in records.inquiries}
    by_cell = {}
    for iid, (lo, hi) in chosen.items():
        iq = inq[iid]
        by_cell.setdefault((iq.group, iq.period), []).append((iq, lo, hi))
    out = {}
    for key, items in by_cell.items():
        ev = cells[key]
        extra = [venn.Constraint(_return_vec(ev, iq), lo, hi, label=f"假设{iq.id}")
                 for iq, lo, hi in items]
        out[key] = venn.feasible(ev.cell, ev.constraints + extra)
    return out


def evaluate_inquiries(records, cells):
    """比较征询方案：哪些能改变决定、哪些无论返回什么都不改变。"""
    inq_returns = {r.inquiry for r in records.returns}
    budget = records.requirements.inquiry_budget
    last = records.requirements.last_received
    out = []
    for iq in records.inquiries:
        if iq.id in inq_returns:
            out.append(InquiryAppraisal(iq.id, "已收到", iq.cost, iq.ready_by, True, {}, True,
                                        "回件已并入证据"))
            continue
        if iq.ready_by > last:
            out.append(InquiryAppraisal(iq.id, "不可期待", iq.cost, iq.ready_by, False, {},
                                        False, f"到达 {iq.ready_by} 晚于最迟收件日 {last}"))
            continue
        ev = cells[(iq.group, iq.period)]
        vec = _return_vec(ev, iq)
        qlo, qhi = venn.bound(ev.cell, ev.constraints, vec)
        reqs = [r for r in records.requests if r.group == iq.group and r.period == iq.period]
        sets = [[r] for r in reqs] + ([reqs] if len(reqs) > 1 else [])
        # 枚举有效回件 [a, min(a+precision, qhi)]，找各计划集可确认的下界阈值
        effects = {}
        for req_set in sets:
            label = "+".join(r.request for r in req_set)
            threshold = None
            a = qlo
            while a <= qhi:
                b = min(a + iq.precision, qhi)
                extra = [venn.Constraint(vec, a, b, label=f"假设{iq.id}")]
                pv = evaluate_plan_set(records, cells, iq.group, iq.period, req_set, extra)
                if pv.verdict == VERDICT_OK:
                    threshold = a
                    break
                a += 1
            effects[label] = threshold
        changes = any(t is not None for t in effects.values())
        out.append(InquiryAppraisal(iq.id, "可期待", iq.cost, iq.ready_by,
                                    iq.cost <= budget, effects, changes))
    return out


# ---------------------------------------------------------------- 共享核对

def sharing_check(records, cells):
    """并行发出核对：已共享 + 候选共享是否唯一确定未授权原子细分或仅规划量。"""
    # 候选：已收到且授权共享的回件
    inq = {i.id: i for i in records.inquiries}
    candidates = [r.inquiry for r in records.returns if AUD_SHARE in inq[r.inquiry].audience]
    shared = shared_constraints(records, cells, extra_shared=candidates)
    report = {"candidates": candidates, "cells": {}}
    for key, cons in shared.items():
        ev = cells[key]
        cell = ev.cell
        base = [c for c in ev.constraints if c.label == "基数"]
        allc = base + cons
        breaches = []
        for subset in cell.regions:
            lo, hi = venn.bound(cell, allc, cell.region_vec(subset))
            if lo == hi:
                breaches.append(("原子细分", "".join(subset), lo))
        # 仅授权规划的量不得被推导为确定人数
        for cert in records.certificates:
            if (cert.group, cert.period) != key or AUD_SHARE in cert.audience:
                continue
            vec = cell.inter_vec(cert.points) if cert.relation == REL_INTERSECT else cell.union_vec(cert.points)
            lo, hi = venn.bound(cell, allc, vec)
            if lo == hi:
                breaches.append(("仅规划凭证", cert.ref, lo))
        for r in records.returns:
            iq = inq[r.inquiry]
            if (iq.group, iq.period) != key or AUD_SHARE in iq.audience:
                continue
            vec = cell.inter_vec(iq.points) if iq.relation == REL_INTERSECT else cell.union_vec(iq.points)
            lo, hi = venn.bound(cell, allc, vec)
            if lo == hi:
                breaches.append(("仅规划回件", r.inquiry, lo))
        report["cells"][key] = {"ok": not breaches, "breaches": breaches,
                                "shared": sorted(c.label for c in cons)}
    return report
