"""匿名重叠统计与家长服务覆盖承诺：业务分析引擎。"""
from .model import load_records
from .report import build_report, render_text

__all__ = ["load_records", "build_report", "render_text"]
