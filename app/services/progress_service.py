from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import UPLOADS_DIR


def _progress_path(project_id: int, run_id: int) -> Path:
    folder = UPLOADS_DIR / f'project_{project_id}'
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f'run_{run_id}_progress.json'


DEFAULT_STATE = {
    'status': 'idle',
    'percent': 0,
    'step': 'Waiting to start',
    'detail': '',
    'started_at': None,
    'updated_at': None,
    'finished_at': None,
    'error': None,
    'result_url': None,
    'model_activity': {},
}


def _safe_load_progress_text(raw: str) -> dict[str, Any]:
    raw = (raw or '').strip()
    if not raw:
        return dict(DEFAULT_STATE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Repair files where multiple JSON documents were accidentally concatenated.
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(raw)
            if isinstance(obj, dict):
                repaired = dict(DEFAULT_STATE)
                repaired.update(obj)
                return repaired
        except Exception:
            pass
        return dict(DEFAULT_STATE)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    last_error: OSError | None = None
    for _ in range(8):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.1)
    if last_error is not None:
        raise last_error


def read_progress(project_id: int, run_id: int) -> dict[str, Any]:
    path = _progress_path(project_id, run_id)
    if not path.exists():
        return dict(DEFAULT_STATE)
    state = _safe_load_progress_text(path.read_text(encoding='utf-8'))
    return state


def write_progress(project_id: int, run_id: int, **updates: Any) -> dict[str, Any]:
    state = read_progress(project_id, run_id)
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    if state.get('started_at') is None and updates.get('status') in {'queued', 'running'}:
        state['started_at'] = now
    state.update(updates)
    state['updated_at'] = now
    if updates.get('status') in {'completed', 'failed'}:
        state['finished_at'] = now

    log = state.get('log') or []
    step = updates.get('step', state.get('step', ''))
    detail = updates.get('detail', state.get('detail', ''))
    percent = updates.get('percent', state.get('percent', 0))
    if step:
        log.append({'ts': now, 'percent': percent, 'step': step, 'detail': detail})
    state['log'] = log[-80:]

    path = _progress_path(project_id, run_id)
    _atomic_write_json(path, state)
    return state


def reset_progress(project_id: int, run_id: int) -> dict[str, Any]:
    path = _progress_path(project_id, run_id)
    fresh = dict(DEFAULT_STATE)
    fresh.update({
        'status': 'queued',
        'percent': 0,
        'step': 'Queued',
        'detail': 'Your forecast job is waiting to start.',
        'error': None,
        'result_url': None,
        'started_at': None,
        'finished_at': None,
        'log': [],
        'model_activity': {},
    })
    fresh['updated_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    _atomic_write_json(path, fresh)
    return fresh
