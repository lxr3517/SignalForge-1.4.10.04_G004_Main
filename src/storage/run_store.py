# LEGACY STORAGE MODULE: not used by the active app/ runtime. Kept for historical reference only.\nfrom __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _db_path(base_dir: Path) -> Path:
    with open(base_dir / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return base_dir / config["storage"]["db_path"]


def save_run_bundle(base_dir: Path, project_id: str, bundle: dict) -> str:
    run_id = uuid.uuid4().hex[:12]
    run_dir = base_dir / "data" / "projects" / project_id / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / f"{run_id}.json"

    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str)

    conn = sqlite3.connect(_db_path(base_dir))
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runs (run_id, project_id, run_path, created_at) VALUES (?, ?, ?, ?)",
        (run_id, project_id, str(run_path), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return run_id


def list_saved_runs(base_dir: Path, project_id: str) -> list[dict]:
    conn = sqlite3.connect(_db_path(base_dir))
    cur = conn.cursor()
    cur.execute(
        "SELECT run_id, project_id, run_path, created_at FROM runs WHERE project_id = ? ORDER BY created_at DESC",
        (project_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "run_id": r[0],
            "project_id": r[1],
            "run_path": r[2],
            "created_at": r[3],
        }
        for r in rows
    ]


def load_run_bundle(base_dir: Path, project_id: str, run_id: str) -> dict:
    run_path = base_dir / "data" / "projects" / project_id / "runs" / f"{run_id}.json"
    with open(run_path, "r", encoding="utf-8") as f:
        return json.load(f)
