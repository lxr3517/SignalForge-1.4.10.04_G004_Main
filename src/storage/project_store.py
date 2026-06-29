# LEGACY STORAGE MODULE: not used by the active app/ runtime. Kept for historical reference only.\nfrom __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]


def bootstrap_storage(base_dir: Path, config: dict) -> None:
    db_path = base_dir / config["storage"]["db_path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT PRIMARY KEY,
            project_name TEXT NOT NULL,
            project_dir TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            file_name TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            run_path TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def _db_path() -> Path:
    with open(BASE_DIR / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return BASE_DIR / config["storage"]["db_path"]


def create_project(project_name: str) -> dict:
    project_id = uuid.uuid4().hex[:12]
    project_dir = BASE_DIR / "data" / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "project_id": project_id,
        "project_name": project_name,
        "project_dir": str(project_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(project_dir / "project.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    conn = sqlite3.connect(_db_path())
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO projects (project_id, project_name, project_dir, created_at) VALUES (?, ?, ?, ?)",
        (metadata["project_id"], metadata["project_name"], metadata["project_dir"], metadata["created_at"]),
    )
    conn.commit()
    conn.close()
    return metadata


def list_projects() -> list[dict]:
    conn = sqlite3.connect(_db_path())
    cur = conn.cursor()
    cur.execute("SELECT project_id, project_name, project_dir, created_at FROM projects ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "project_id": r[0],
            "project_name": r[1],
            "project_dir": r[2],
            "created_at": r[3],
        }
        for r in rows
    ]


def load_project_metadata(project_id: str) -> dict:
    project_dir = BASE_DIR / "data" / "projects" / project_id
    with open(project_dir / "project.json", "r", encoding="utf-8") as f:
        return json.load(f)


def save_uploaded_dataset(project_id: str, dataset_name: str, uploaded_file, df: pd.DataFrame) -> str:
    dataset_id = uuid.uuid4().hex[:12]
    project_dir = BASE_DIR / "data" / "projects" / project_id
    raw_dir = project_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stored_path = raw_dir / f"{dataset_name}_{dataset_id}.parquet"
    df.to_parquet(stored_path, index=False)

    conn = sqlite3.connect(_db_path())
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO datasets (dataset_id, project_id, dataset_name, file_name, stored_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (dataset_id, project_id, dataset_name, uploaded_file.name, str(stored_path), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return str(stored_path)
