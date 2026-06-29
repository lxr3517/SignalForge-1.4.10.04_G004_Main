from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import BASE_DIR, UPLOADS_DIR
from app.services.file_service import load_dataframe
from app.services.fix_rules import (
    fix_drop_blank_rows,
    fix_parse_numeric_columns,
    fix_remove_exact_duplicates,
    fix_remove_future_dates,
    fix_sort_dates,
    fix_standardize_null_tokens,
    fix_trim_headers,
    fix_trim_object_values,
)


def _project_dir(project_id: int) -> Path:
    return UPLOADS_DIR / f'project_{project_id}'


def _upload_meta_path(project_id: int, run_id: int) -> Path:
    return _project_dir(project_id) / f'run_{run_id}_upload.json'


def _auto_fix_path(project_id: int, run_id: int) -> Path:
    return _project_dir(project_id) / f'run_{run_id}_auto_fixes.json'


def _cleaned_source_key(file_key: str | None, sheet_name: str | None) -> str:
    return f"{file_key or ''}|{sheet_name or '__default__'}"


def _metadata_path(value: str | Path | None) -> Path:
    path = Path(str(value or ''))
    return path if path.is_absolute() else BASE_DIR / path


def _cleaned_dataset_path(project_id: int, run_id: int, file_key: str, sheet_name: str | None) -> Path:
    run_dir = _project_dir(project_id) / f'run_{run_id}' / 'cleaned'
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_sheet = re.sub(r'[^A-Za-z0-9._-]+', '_', (sheet_name or 'default')).strip('._') or 'default'
    safe_key = re.sub(r'[^A-Za-z0-9._-]+', '_', (file_key or 'dataset')).strip('._') or 'dataset'
    return run_dir / f'{safe_key}__{safe_sheet}_cleaned.csv'


def _mapping_sections(mapping: dict) -> list[tuple[str, dict]]:
    keys = ['revenue', 'leads', 'cost', 'events', 'cohort_revenue', 'custom_regressors']
    for key, val in mapping.items():
        if re.match(r'^custom_\d+$', str(key)):
            keys.append(key)
    out = []
    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        cfg = mapping.get(key)
        if isinstance(cfg, dict):
            out.append((key, cfg))
    return out


def _source_specs(mapping: dict) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for section_name, cfg in _mapping_sections(mapping):
        file_key = cfg.get('file_key') or ('revenue' if section_name == 'revenue' else None)
        if not file_key:
            continue
        sheet_name = cfg.get('sheet_name') or None
        comp_key = _cleaned_source_key(file_key, sheet_name)
        spec = specs.setdefault(comp_key, {
            'file_key': file_key,
            'sheet_name': sheet_name,
            'dataset_names': [],
            'date_columns': set(),
            'numeric_columns': set(),
            'dimension_columns': set(),
        })
        spec['dataset_names'].append(section_name)
        for date_key in ('date_column', 'lead_month_column', 'transaction_month_column'):
            col = cfg.get(date_key)
            if col:
                spec['date_columns'].add(col)
        for num_key in ('target_column', 'value_column', 'event_flag_column', 'outage_flag_column'):
            col = cfg.get(num_key)
            if col:
                spec['numeric_columns'].add(col)
        for num_list_key in ('value_columns',):
            for col in (cfg.get(num_list_key) or []):
                if col:
                    spec['numeric_columns'].add(col)
        for dim_key in ('category_column', 'platform_column', 'affiliate_id_column', 'campaign_id_column', 'ad_id_column', 'profile_id_column'):
            col = cfg.get(dim_key)
            if col:
                spec['dimension_columns'].add(col)
    return specs


def _apply_safe_fixes(df: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    fixes: list[dict[str, Any]] = []
    work = df.copy()
    for fn in (fix_trim_headers, fix_standardize_null_tokens, fix_drop_blank_rows, fix_remove_exact_duplicates):
        work, meta = fn(work)
        fixes.append(meta)
    work, meta = fix_parse_numeric_columns(work, list(spec.get('numeric_columns', [])))
    fixes.append(meta)
    work, meta = fix_trim_object_values(work, list(spec.get('dimension_columns', [])))
    fixes.append(meta)
    date_candidates = [c for c in spec.get('date_columns', []) if c in work.columns]
    for date_col in date_candidates[:1]:
        work, meta = fix_sort_dates(work, date_col)
        fixes.append(dict(meta, date_column=date_col))
        work, meta = fix_remove_future_dates(work, date_col)
        fixes.append(dict(meta, date_column=date_col))
    return work, fixes


def load_auto_fix_audit(project_id: int, run_id: int) -> dict:
    path = _auto_fix_path(project_id, run_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def run_auto_fix_engine(project_id: int, run_id: int, upload_meta: dict, mapping: dict) -> dict:
    files = dict(upload_meta.get('files', {}))
    specs = _source_specs(mapping)
    cleaned_sources = dict(upload_meta.get('cleaned_sources', {}))
    summary: dict[str, Any] = {
        'fixes_applied_count': 0,
        'sources_processed': 0,
        'rows_before_total': 0,
        'rows_after_total': 0,
        'sources': {},
    }
    changed = False
    for comp_key, spec in specs.items():
        file_key = spec['file_key']
        file_meta = files.get(file_key)
        if not file_meta:
            continue
        raw_path = _metadata_path(file_meta.get('raw_path') or file_meta.get('path'))
        if not raw_path.exists():
            continue
        try:
            df = load_dataframe(raw_path, sheet_name=spec.get('sheet_name'))
        except Exception as exc:
            summary['sources'][comp_key] = {
                'status': 'error',
                'error': str(exc),
                'file_key': file_key,
                'sheet_name': spec.get('sheet_name'),
            }
            continue
        rows_before = int(len(df))
        cleaned_df, fixes = _apply_safe_fixes(df, spec)
        rows_after = int(len(cleaned_df))
        cleaned_path = _cleaned_dataset_path(project_id, run_id, file_key, spec.get('sheet_name'))
        cleaned_df.to_csv(cleaned_path, index=False)
        cleaned_sources[comp_key] = {
            'path': str(cleaned_path),
            'raw_path': str(raw_path),
            'file_key': file_key,
            'sheet_name': spec.get('sheet_name'),
            'dataset_names': spec.get('dataset_names', []),
        }
        summary['sources'][comp_key] = {
            'status': 'ok',
            'file_key': file_key,
            'sheet_name': spec.get('sheet_name'),
            'dataset_names': spec.get('dataset_names', []),
            'rows_before': rows_before,
            'rows_after': rows_after,
            'applied_fixes': fixes,
            'cleaned_path': str(cleaned_path),
        }
        summary['sources_processed'] += 1
        summary['rows_before_total'] += rows_before
        summary['rows_after_total'] += rows_after
        summary['fixes_applied_count'] += sum(1 for f in fixes if any(v not in (0, False, None, [], {}) for k, v in f.items() if k != 'action'))
        changed = True
    if changed or cleaned_sources:
        new_meta = dict(upload_meta)
        new_meta['cleaned_sources'] = cleaned_sources
        _upload_meta_path(project_id, run_id).write_text(json.dumps(new_meta, indent=2), encoding='utf-8')
    _auto_fix_path(project_id, run_id).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
