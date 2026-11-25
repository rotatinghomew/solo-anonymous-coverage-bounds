"""证据装配：有效凭证、已收到回件、待核来件、待收征询的区分。

- 撤回的凭证不再作为证据（但已确认服务不自动取消）；
- 已收到回件按来件区间并入证据，audience 只影响能否共享，不影响规划使用；
- 未收到的征询不计为零，只记"缺回件"；
- 待核来件（new_certificate）不覆盖同名凭证，单独做相容核对。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import venn
from .model import AUD_SHARE, REL_INTERSECT


@dataclass
class CellEvidence:
    group: str
    period: str
    cell: venn.Cell
    constraints: list          # 有效凭证 + 已收到回件
    active_refs: list          # 参与证据的凭证编号
    received_returns: list     # 已收到的征询 id
    pending_inquiries: list    # 可期待而未收到的征询 id（缺回件，不算零）
    late_inquiries: list       # 到达时间超过最迟收件日，不可期待


def _cert_constraint(cell, cert):
    vec = cell.inter_vec(cert.points) if cert.relation == REL_INTERSECT else cell.union_vec(cert.points)
    return venn.Constraint(vec, cert.lower, cert.upper, label=cert.ref)


def build_evidence(records):
    """按（分层, 期间）装配证据。返回 {(group, period): CellEvidence}。"""
    revoked = records.revoked_refs()
    inq = {i.id: i for i in records.inquiries}
    returns_by_inq = {r.inquiry: r for r in records.returns}
    cells = {}
    for f in records.frames:
        key = (f.group, f.period)
        points = records.points_of(f.group, f.period)
        cells[key] = CellEvidence(
            group=f.group, period=f.period,
            cell=venn.Cell(f.group, f.period, points, f.population),
            constraints=[], active_refs=[], received_returns=[],
            pending_inquiries=[], late_inquiries=[],
        )
    # 总体基数约束：非空区域合计 <= population（空区域为未到访家庭）
    for ev in cells.values():
        ev.constraints.append(venn.Constraint(
            np.ones(ev.cell.dim), 0.0, ev.cell.population, label="基数"))
    # 有效凭证（撤回的不计入）
    for cert in records.certificates:
        if cert.ref in revoked:
            continue
        ev = cells[(cert.group, cert.period)]
        ev.constraints.append(_cert_constraint(ev.cell, cert))
        ev.active_refs.append(cert.ref)
    # 已收到回件并入证据
    for r in records.returns:
        iq = inq[r.inquiry]
        ev = cells[(iq.group, iq.period)]
        vec = ev.cell.inter_vec(iq.points) if iq.relation == REL_INTERSECT else ev.cell.union_vec(iq.points)
        ev.constraints.append(venn.Constraint(vec, r.lower, r.upper, label=f"回件{r.inquiry}"))
        ev.received_returns.append(r.inquiry)
    # 征询的期待性
    for iq in records.inquiries:
        ev = cells[(iq.group, iq.period)]
        if iq.id in returns_by_inq:
            continue
        if iq.ready_by <= records.requirements.last_received:
            ev.pending_inquiries.append(iq.id)
        else:
            ev.late_inquiries.append(iq.id)
    return cells


def shared_constraints(records, cells, extra_shared=()):
    """已共享（及候选共享）凭证/回件在某（分层, 期间）上形成的约束。

    只纳入 audience 含"共享"的项目；仅授权规划的一律不进入共享集。
    """
    inq = {i.id: i for i in records.inquiries}
    returns_by_inq = {r.inquiry: r for r in records.returns}
    by_cell = {}
    refs = set(records.already_shared) | set(extra_shared)
    for ref in refs:
        cert = next((c for c in records.certificates if c.ref == ref), None)
        if cert is None or AUD_SHARE not in cert.audience:
            continue
        ev = cells[(cert.group, cert.period)]
        by_cell.setdefault((cert.group, cert.period), []).append(_cert_constraint(ev.cell, cert))
    for rid, r in returns_by_inq.items():
        iq = inq[rid]
        if AUD_SHARE not in iq.audience:
            continue
        if rid not in extra_shared and rid not in refs:
            continue
        ev = cells[(iq.group, iq.period)]
        vec = ev.cell.inter_vec(iq.points) if iq.relation == REL_INTERSECT else ev.cell.union_vec(iq.points)
        by_cell.setdefault((iq.group, iq.period), []).append(
            venn.Constraint(vec, r.lower, r.upper, label=f"回件{rid}"))
    return by_cell
