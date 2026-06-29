from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text as sql_text

from app.config import (
    APP_NAME,
    APP_VERSION,
    BASE_DIR,
    DATABASE_DIR,
    PROJECTS_DIR,
    STATIC_DIR,
    TEMPLATES_DIR,
    UPLOADS_DIR,
)
from app.db import engine


DEPENDENCY_MODULES = {
    'fastapi': 'fastapi',
    'jinja2': 'jinja2',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'sqlalchemy': 'sqlalchemy',
    'uvicorn': 'uvicorn',
}

REQUIRED_TEMPLATES = ['base.html', 'home.html']


def _check(status: str, message: str, **details: Any) -> dict[str, Any]:
    payload = {'status': status, 'message': message}
    payload.update(details)
    return payload


def _dependency_checks() -> dict[str, dict[str, Any]]:
    checks = {}
    for name, module_name in DEPENDENCY_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            checks[name] = _check('ok', 'importable', version=getattr(module, '__version__', None))
        except Exception as exc:
            checks[name] = _check('fail', f'{module_name} could not be imported', error=str(exc))
    return checks


def _directory_check(path: Path, writable: bool = False) -> dict[str, Any]:
    if not path.exists():
        return _check('fail', 'directory is missing', path=str(path))
    if not path.is_dir():
        return _check('fail', 'path is not a directory', path=str(path))
    if writable:
        probe = path / '.healthcheck'
        try:
            probe.write_text('ok', encoding='utf-8')
            probe.unlink(missing_ok=True)
        except Exception as exc:
            return _check('fail', 'directory is not writable', path=str(path), error=str(exc))
    return _check('ok', 'directory is available', path=str(path))


def _database_check() -> dict[str, Any]:
    try:
        with engine.connect() as conn:
            conn.execute(sql_text('SELECT 1')).scalar_one()
        return _check('ok', 'database is reachable', path=str(DATABASE_DIR / 'app.db'))
    except Exception as exc:
        return _check('fail', 'database check failed', error=str(exc), path=str(DATABASE_DIR / 'app.db'))


def _template_check() -> dict[str, Any]:
    missing = [name for name in REQUIRED_TEMPLATES if not (TEMPLATES_DIR / name).is_file()]
    if missing:
        return _check('fail', 'required templates are missing', missing=missing, path=str(TEMPLATES_DIR))
    return _check('ok', 'required templates are present', templates=REQUIRED_TEMPLATES)


def _runtime_check() -> dict[str, Any]:
    expected_venv = BASE_DIR / '.venv'
    executable = Path(sys.executable).resolve()
    try:
        in_project_venv = executable.is_relative_to(expected_venv.resolve())
    except AttributeError:
        in_project_venv = str(executable).lower().startswith(str(expected_venv.resolve()).lower())
    if not in_project_venv:
        return _check(
            'warn',
            'app is not running from the project-local .venv',
            executable=str(executable),
            expected=str(expected_venv),
        )
    return _check('ok', 'app is running from the project-local .venv', executable=str(executable))


def build_health_report(deep: bool = False) -> dict[str, Any]:
    checks: dict[str, Any] = {
        'runtime': _runtime_check(),
        'database': _database_check(),
        'templates': _template_check(),
    }

    if deep:
        checks.update(
            {
                'dependencies': _dependency_checks(),
                'static_dir': _directory_check(STATIC_DIR),
                'uploads_dir': _directory_check(UPLOADS_DIR, writable=True),
                'projects_dir': _directory_check(PROJECTS_DIR, writable=True),
            }
        )

    flat_checks = []
    for value in checks.values():
        if isinstance(value, dict) and 'status' in value:
            flat_checks.append(value)
        elif isinstance(value, dict):
            flat_checks.extend(item for item in value.values() if isinstance(item, dict) and 'status' in item)

    has_failure = any(item['status'] == 'fail' for item in flat_checks)
    has_warning = any(item['status'] == 'warn' for item in flat_checks)
    overall = 'fail' if has_failure else 'degraded' if has_warning else 'ok'

    return {
        'status': overall,
        'app_name': APP_NAME,
        'app_version': APP_VERSION,
        'checks': checks,
    }
