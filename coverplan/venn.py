"""Venn 区域构成求解器。

同一（分层, 期间）内，每个家庭按其到访服务点子集落入一个 Venn 区域；
凭证（交集/并集区间）翻译成区域上的线性约束。所有凭证必须能由同一份
区域构成解释（相容），查询与保证都对全部相容构成取最小/最大。

不同（分层, 期间）之间没有可比观察窗，绝不合并求解。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

_BIG = 100000.0


class Cell:
    """一个（分层, 期间）的 Venn 区域骨架。"""

    def __init__(self, group, period, points, population):
        self.group = group
        self.period = period
        self.points = tuple(points)
        self.population = population
        self.regions = []
        for r in range(1, len(self.points) + 1):
            self.regions += [tuple(c) for c in combinations(self.points, r)]
        self.index = {frozenset(s): i for i, s in enumerate(self.regions)}
        self.dim = len(self.regions)

    def _vec(self, pred):
        v = np.zeros(self.dim)
        for s, i in self.index.items():
            if pred(s):
                v[i] = 1.0
        return v

    def inter_vec(self, points):
        """|∩ points|：到访集合包含 points 全部的家庭数。"""
        ps = frozenset(points)
        return self._vec(lambda s: ps <= s)

    def union_vec(self, points):
        """|∪ points|：到访过 points 中至少一点的家庭数。"""
        ps = frozenset(points)
        return self._vec(lambda s: len(ps & s) > 0)

    def unique_vec(self, point):
        """独有人数：只到访 point 一点的家庭数。"""
        target = frozenset([point])
        return self._vec(lambda s: s == target)

    def displaced_vec(self, points):
        """被挤出人数：到访集合非空且全部落在 points 内（停开后无处可去）。"""
        ps = frozenset(points)
        return self._vec(lambda s: len(s) > 0 and s <= ps)

    def region_vec(self, subset):
        target = frozenset(subset)
        return self._vec(lambda s: s == target)


@dataclass
class Constraint:
    vec: np.ndarray
    lower: float
    upper: float
    label: str = ""


def _milp(cell, constraints, objective, maximize=False):
    rows = [c.vec for c in constraints]
    lb = [c.lower for c in constraints]
    ub = [c.upper for c in constraints]
    A = np.vstack(rows) if rows else np.zeros((0, cell.dim))
    c = -objective if maximize else objective
    res = milp(c=c, constraints=LinearConstraint(A, np.array(lb), np.array(ub)),
               bounds=Bounds(np.zeros(cell.dim), np.full(cell.dim, np.inf)),
               integrality=np.ones(cell.dim))
    return res


def feasible(cell, constraints):
    """全部约束是否可由同一份区域构成满足。"""
    res = _milp(cell, constraints, np.zeros(cell.dim))
    return res.status == 0


def bound(cell, constraints, vec):
    """某线性覆盖量在全部相容构成下的 [下界, 上界]（含端点）。不可行返回 None。"""
    lo = _milp(cell, constraints, vec)
    if lo.status != 0:
        return None
    hi = _milp(cell, constraints, vec, maximize=True)
    return round(lo.fun), round(-hi.fun)


def minimal_conflict(cell, constraints):
    """不可行约束集的一个极小不可行子集（贪心删减），用于指出矛盾来件。

    只报告，不删除任何凭证。
    """
    keep = list(constraints)
    i = 0
    while i < len(keep):
        trial = keep[:i] + keep[i + 1:]
        if trial and not feasible(cell, trial):
            keep = trial
        else:
            i += 1
    return keep


def conflicts_with(cell, base_constraints, new_constraint):
    """新来件与现有证据的冲突定位。

    返回 (两两冲突的标签列表, 联合冲突的标签列表)：
    先列出每个单独就与新来件不相容的凭证；若不存在两两冲突而整体仍
    不相容，则给出一个极小联合冲突集。只报告，不删除任何凭证。
    """
    pairwise = sorted(
        c.label for c in base_constraints
        if c.label != "基数" and not feasible(cell, [c, new_constraint])
    )
    if pairwise:
        return pairwise, []
    core = minimal_conflict(cell, base_constraints + [new_constraint])
    joint = sorted(c.label for c in core if c.label != new_constraint.label)
    return [], joint


def allocatable_seats(cell, constraints, caps, slot_free, offsets):
    """实际可分配的新席位：s_K <= 独有人数下界（联办方确认每席必接住家庭），
    同一联办席位只被一个请求占用，同槽合计 <= 剩余席位。

    caps: {request: 独有人数下界}; slot_free: {slot: 剩余席位};
    offsets: {request: (slot, ...)}。返回 (总量, {request: 分配})。
    """
    reqs = list(caps)
    n = len(reqs)
    if n == 0:
        return 0, {}
    rows, lb, ub = [], [], []
    slots = sorted(slot_free)
    for j in slots:
        row = np.zeros(n)
        for k, rq in enumerate(reqs):
            if j in offsets[rq]:
                row[k] = 1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(slot_free[j])
    A = np.vstack(rows) if rows else np.zeros((0, n))
    bounds = Bounds(np.zeros(n), np.array([caps[r] for r in reqs], dtype=float))
    res = milp(c=-np.ones(n),
               constraints=LinearConstraint(A, np.array(lb), np.array(ub)),
               bounds=bounds, integrality=np.ones(n))
    if res.status != 0:
        return 0, {r: 0 for r in reqs}
    alloc = {r: int(round(res.x[k])) for k, r in enumerate(reqs)}
    return sum(alloc.values()), alloc


def max_retained_best_case(cell, constraints, remaining, removed, confirmed, slot_free, offsets):
    """席位不受独有人数下界限制时（证据可再补）保留人数的最大值。

    用于区分席位不足与证据不足：若连最有利构成 + 全部剩余席位都够不到
    层保底，则瓶颈是席位而非证据。confirmed 为已确认席位（确定接住）。
    """
    reqs = list(removed)
    n_r = len(reqs)
    # 变量: x(dim) + s_K + y_K(接住) + b_K
    n = cell.dim + 3 * n_r
    rows, lb, ub = [], [], []
    for c in constraints:
        rows.append(np.concatenate([c.vec, np.zeros(n - cell.dim)]))
        lb.append(c.lower)
        ub.append(c.upper)
    uvec = {p: cell.unique_vec(p) for p in reqs}
    for k, p in enumerate(reqs):
        i_s = cell.dim + k
        i_y = cell.dim + n_r + k
        i_b = cell.dim + 2 * n_r + k
        # y <= s
        row = np.zeros(n); row[i_y] = 1.0; row[i_s] = -1.0
        rows.append(row); lb.append(-np.inf); ub.append(0.0)
        # y <= u_p
        row = np.zeros(n); row[i_y] = 1.0; row[:cell.dim] = -uvec[p]
        rows.append(row); lb.append(-np.inf); ub.append(0.0)
        # y >= s - M*b
        row = np.zeros(n); row[i_y] = 1.0; row[i_s] = -1.0; row[i_b] = _BIG
        rows.append(row); lb.append(0.0); ub.append(np.inf)
        # y >= u_p - M*(1-b)
        row = np.zeros(n); row[i_y] = 1.0; row[:cell.dim] = -uvec[p]; row[i_b] = -_BIG
        rows.append(row); lb.append(-_BIG); ub.append(np.inf)
    # 槽位合计约束
    for j, free in slot_free.items():
        row = np.zeros(n)
        for k, p in enumerate(reqs):
            if j in offsets[p]:
                row[cell.dim + k] = 1.0
        rows.append(row); lb.append(-np.inf); ub.append(free)
    A = np.vstack(rows)
    obj = np.zeros(n)
    obj[:cell.dim] = cell.union_vec(remaining)
    for k in range(n_r):
        obj[cell.dim + n_r + k] = 1.0
    bounds_arr = np.full(n, np.inf)
    bounds_arr[cell.dim + 2 * n_r:] = 1.0  # 二元
    res = milp(c=-obj, constraints=LinearConstraint(A, np.array(lb), np.array(ub)),
               bounds=Bounds(np.zeros(n), bounds_arr), integrality=np.ones(n))
    if res.status != 0:
        return None
    return round(-res.fun) + sum(confirmed.get(p, 0) for p in reqs)
