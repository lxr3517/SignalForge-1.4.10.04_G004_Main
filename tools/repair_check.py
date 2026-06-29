from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.health import build_health_report
from app.main import app  # noqa: F401


def main() -> int:
    report = build_health_report(deep=True)
    print(f"{report['app_name']} {report['app_version']}")
    print(f"overall: {report['status']}")

    for name, check in report['checks'].items():
        if isinstance(check, dict) and 'status' in check:
            print(f"{name}: {check['status']} - {check['message']}")
            continue
        if isinstance(check, dict):
            for sub_name, sub_check in check.items():
                print(f"{name}.{sub_name}: {sub_check['status']} - {sub_check['message']}")

    return 1 if report['status'] == 'fail' else 0


if __name__ == '__main__':
    sys.exit(main())
