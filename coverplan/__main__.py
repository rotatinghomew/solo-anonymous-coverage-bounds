"""命令行入口：python3 -m coverplan [records.json]"""
from __future__ import annotations

import sys
from pathlib import Path

from .model import load_records
from .report import build_report, render_text


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    path = Path(argv[0]) if argv else Path(__file__).resolve().parents[1] / "data" / "records.json"
    records = load_records(path)
    report = build_report(records)
    print(render_text(report, records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
