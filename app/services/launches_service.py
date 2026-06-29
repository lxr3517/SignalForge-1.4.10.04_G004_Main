from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app import models
from app.config import BASE_DIR, LAUNCHES_DIR, PROJECTS_DIR


def _safe_iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, 'isoformat') else None


def _json_default(value: Any):
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    raise TypeError(f'Not JSON serializable: {type(value)!r}')


def _path(name: str) -> Path:
    LAUNCHES_DIR.mkdir(parents=True, exist_ok=True)
    return LAUNCHES_DIR / name


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(BASE_DIR.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _project_payload(project: models.Project) -> dict:
    return {
        'id': int(project.id),
        'name': project.name,
        'description': project.description,
        'frequency': project.frequency,
        'years_of_history': project.years_of_history,
        'grouped_forecast': bool(project.grouped_forecast),
        'forecast_scope': getattr(project, 'forecast_scope', 'company_total') or 'company_total',
        'created_at': _safe_iso(project.created_at),
        'updated_at': _safe_iso(project.updated_at),
        'project_dir': _rel(PROJECTS_DIR / f'project_{project.id}'),
    }


def _run_payload(run: models.Run) -> dict:
    return {
        'id': int(run.id),
        'project_id': int(run.project_id),
        'label': run.label,
        'status': run.status,
        'target_column': run.target_column,
        'date_column': run.date_column,
        'category_column': run.category_column,
        'selected_models': run.selected_models,
        'scoring_metric': run.scoring_metric,
        'forecast_horizon': run.forecast_horizon,
        'output_mode': run.output_mode,
        'created_at': _safe_iso(run.created_at),
        'updated_at': _safe_iso(run.updated_at),
        'project_dir': _rel(PROJECTS_DIR / f'project_{run.project_id}'),
        'forecast_settings_path': _rel(PROJECTS_DIR / f'project_{run.project_id}' / f'run_{run.id}_forecast_settings.json'),
        'mapping_path': _rel(PROJECTS_DIR / f'project_{run.project_id}' / f'run_{run.id}_mapping.json'),
        'upload_meta_path': _rel(PROJECTS_DIR / f'project_{run.project_id}' / f'run_{run.id}_upload.json'),
        'visuals_snapshot_path': _rel(PROJECTS_DIR / f'project_{run.project_id}' / f'run_{run.id}_visuals.json'),
    }


def sync_launch_registry(db: Session) -> None:
    try:
        projects = db.query(models.Project).order_by(models.Project.updated_at.desc()).all()
        runs = db.query(models.Run).order_by(models.Run.updated_at.desc()).all()
        project_rows = [_project_payload(p) for p in projects]
        run_rows = [_run_payload(r) for r in runs]
        recent_projects = project_rows[:25]
        recent_runs = run_rows[:50]
        manifest = {
            'projects': project_rows,
            'runs': run_rows,
            'recent_projects': recent_projects,
            'recent_runs': recent_runs,
        }
        _path('projects.json').write_text(json.dumps(project_rows, indent=2, default=_json_default), encoding='utf-8')
        _path('runs.json').write_text(json.dumps(run_rows, indent=2, default=_json_default), encoding='utf-8')
        _path('recent_projects.json').write_text(json.dumps(recent_projects, indent=2, default=_json_default), encoding='utf-8')
        _path('recent_runs.json').write_text(json.dumps(recent_runs, indent=2, default=_json_default), encoding='utf-8')
        _path('manifest.json').write_text(json.dumps(manifest, indent=2, default=_json_default), encoding='utf-8')
    except Exception:
        pass


def restore_launch_registry(db: Session) -> None:
    manifest_path = _path('manifest.json')
    if not manifest_path.exists():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    except Exception:
        return
    projects = payload.get('projects') or []
    runs = payload.get('runs') or []
    existing_project_ids = {row[0] for row in db.query(models.Project.id).all()}
    existing_run_ids = {row[0] for row in db.query(models.Run.id).all()}
    inserted = False
    for item in projects:
        pid = int(item.get('id') or 0)
        if not pid or pid in existing_project_ids:
            continue
        proj = models.Project(
            id=pid,
            name=item.get('name') or f'Project {pid}',
            description=item.get('description'),
            frequency=item.get('frequency') or 'D',
            years_of_history=int(item.get('years_of_history') or 5),
            grouped_forecast=bool(item.get('grouped_forecast')),
            forecast_scope=item.get('forecast_scope') or 'company_total',
        )
        db.add(proj)
        inserted = True
    if inserted:
        db.commit()
        existing_project_ids = {row[0] for row in db.query(models.Project.id).all()}
    inserted_runs = False
    for item in runs:
        rid = int(item.get('id') or 0)
        pid = int(item.get('project_id') or 0)
        if not rid or rid in existing_run_ids or pid not in existing_project_ids:
            continue
        run = models.Run(
            id=rid,
            project_id=pid,
            label=item.get('label') or f'Run {rid}',
            status=item.get('status') or 'draft',
            target_column=item.get('target_column'),
            date_column=item.get('date_column'),
            category_column=item.get('category_column'),
            selected_models=item.get('selected_models'),
            scoring_metric=item.get('scoring_metric'),
            forecast_horizon=item.get('forecast_horizon'),
            output_mode=item.get('output_mode'),
        )
        db.add(run)
        inserted_runs = True
    if inserted_runs:
        db.commit()
