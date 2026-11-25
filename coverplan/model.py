"""数据模型与载入核对。

对应 README 资料字典：
- frames 分层互斥，population 为同期间该层基数，没有跨月身份；
- certificates 的日期是事实期间，relation 规定集合怎样计数，上下界含端点；
- audience 是允许使用凭证作出结论的范围；
- slots 同一席位只可被一个请求占用，历史耗用继续扣除；
- history 中撤回是撤销证据用途，不是撤销服务承诺；
- new_certificate 是独立待核来件，不覆盖同名凭证；
- inquiries 的 precision 是覆盖真实人数的连续整数区间宽度上限；
- returns 是真实来件，known 是收到日期，不随征询选项自动发生。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REL_INTERSECT = "交集"
REL_UNION = "并集"
RELATIONS = (REL_INTERSECT, REL_UNION)

AUD_PLAN = "规划"
AUD_SHARE = "共享"

EVT_CONFIRMED = "已确认"
EVT_REVOKED = "撤回凭证"
EVT_REISSUED = "补发凭证"


@dataclass(frozen=True)
class Frame:
    group: str
    period: str
    population: int


@dataclass(frozen=True)
class Certificate:
    ref: str
    group: str
    period: str
    points: tuple
    relation: str
    lower: int
    upper: int
    audience: tuple


@dataclass(frozen=True)
class Slot:
    slot: str
    local: str
    points: tuple
    seats: int
    group: str


@dataclass(frozen=True)
class Request:
    request: str
    period: str
    remove_point: str
    group: str
    unique_floor: int
    offsets: tuple


@dataclass(frozen=True)
class HistoryEvent:
    date: str
    request: str
    event: str
    certificate: str
    slots: int | None


@dataclass(frozen=True)
class Requirements:
    inquiry_budget: int
    last_received: str
    group_retained_minimum: dict


@dataclass(frozen=True)
class Inquiry:
    id: str
    group: str
    period: str
    points: tuple
    relation: str
    cost: int
    ready_by: str
    precision: int
    audience: tuple


@dataclass(frozen=True)
class Return:
    inquiry: str
    known: str
    lower: int
    upper: int


@dataclass
class Records:
    frames: list
    certificates: list
    slots: list
    requests: list
    history: list
    requirements: Requirements
    inquiries: list
    returns: list
    already_shared: list
    new_certificate: Certificate

    def frame(self, group, period):
        for f in self.frames:
            if f.group == group and f.period == period:
                return f
        raise KeyError((group, period))

    def revoked_refs(self):
        return {h.certificate for h in self.history if h.event == EVT_REVOKED}

    def confirmed_slots(self):
        """request -> 已确认占用席位数（服务继续，不随撤回取消）。"""
        out = {}
        for h in self.history:
            if h.event == EVT_CONFIRMED and h.slots:
                out[h.request] = out.get(h.request, 0) + h.slots
        return out

    def points_of(self, group, period):
        """该（分层, 期间）出现过的服务点，按首次出现顺序，保证确定性。"""
        seen = []
        for c in self.certificates:
            if c.group == group and c.period == period:
                for p in c.points:
                    if p not in seen:
                        seen.append(p)
        return tuple(seen)


def _certificate(raw) -> Certificate:
    return Certificate(
        ref=raw["ref"], group=raw["group"], period=raw["period"],
        points=tuple(raw["points"]), relation=raw["relation"],
        lower=int(raw["lower"]), upper=int(raw["upper"]),
        audience=tuple(raw["audience"]),
    )


def load_records(path) -> Records:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = [Frame(g["group"], g["period"], int(g["population"])) for g in payload["frames"]]
    certificates = [_certificate(c) for c in payload["certificates"]]
    slots = [Slot(s["slot"], s["local"], tuple(s["points"]), int(s["seats"]), s["group"])
             for s in payload["slots"]]
    requests = [Request(r["request"], r["period"], r["remove_point"], r["group"],
                        int(r["unique_floor"]), tuple(r["offsets"]))
                for r in payload["requests"]]
    history = [HistoryEvent(h["date"], h["request"], h["event"], h["certificate"], h.get("slots"))
               for h in payload["history"]]
    req = payload["requirements"]
    requirements = Requirements(int(req["inquiry_budget"]), req["last_received"],
                                dict(req["group_retained_minimum"]))
    inquiries = [Inquiry(i["id"], i["group"], i["period"], tuple(i["points"]), i["relation"],
                         int(i["cost"]), i["ready_by"], int(i["precision"]), tuple(i["audience"]))
                 for i in payload["inquiries"]]
    returns = [Return(r["inquiry"], r["known"], int(r["lower"]), int(r["upper"]))
               for r in payload["returns"]]
    new_certificate = _certificate(payload["new_certificate"])
    records = Records(frames, certificates, slots, requests, history, requirements,
                      inquiries, returns, list(payload["already_shared"]), new_certificate)
    validate(records)
    return records


def validate(records: Records):
    """与 tests/test_records.py 相同的输入契约，另加席位与回件核对。"""
    frames = {(f.group, f.period): f.population for f in records.frames}
    for item in list(records.certificates) + [records.new_certificate]:
        assert item.relation in RELATIONS, item.ref
        assert 0 <= item.lower <= item.upper <= frames[item.group, item.period], item.ref
        assert len(item.points) == len(set(item.points)), item.ref
    assert any(h.event == EVT_REVOKED for h in records.history)
    used = {k for r in records.requests for k in r.offsets}
    assert len(used) < len(records.requests)  # 存在共用席位
    inq = {i.id: i for i in records.inquiries}
    for r in records.returns:
        iq = inq[r.inquiry]
        assert 0 <= r.lower <= r.upper
        assert r.upper - r.lower <= iq.precision, r.inquiry  # 回件遵守约定精度
    slot_seats = {s.slot: s.seats for s in records.slots}
    for r in records.requests:
        for off in r.offsets:
            assert off in slot_seats, (r.request, off)
