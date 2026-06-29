
from __future__ import annotations
from pathlib import Path
import shutil
import pandas as pd
import re
from fastapi import UploadFile
from app.config import UPLOADS_DIR

ALLOWED = {'.csv', '.xlsx', '.xls', '.parquet'}


def save_upload(project_id: int, run_id: int, file: UploadFile) -> Path:
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in ALLOWED:
        raise ValueError(f'Unsupported file type: {suffix}')
    target_dir = UPLOADS_DIR / f'project_{project_id}' / f'run_{run_id}'
    target_dir.mkdir(parents=True, exist_ok=True)
    original_name = file.filename or f'upload{suffix}'
    stem = Path(original_name).stem or 'upload'
    safe_stem = re.sub(r'[^A-Za-z0-9._-]+', '_', stem).strip('._') or 'upload'
    target_path = target_dir / f'{safe_stem}{suffix}'
    with target_path.open('wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    return target_path


def get_sheet_names(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {'.xlsx', '.xls'}:
        return list(pd.ExcelFile(path).sheet_names)
    return []


def load_dataframe(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    if suffix == '.parquet':
        return pd.read_parquet(path)
    raise ValueError(f'Unsupported file type: {suffix}')
