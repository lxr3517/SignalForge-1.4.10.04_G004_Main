from sqlalchemy import text as sql_text
from app.db import Base, engine, SessionLocal
from app.config import DATABASE_DIR, PROJECTS_DIR, UPLOADS_DIR, STATIC_DIR, TEMPLATES_DIR, LAUNCHES_DIR
from app.services.launches_service import restore_launch_registry, sync_launch_registry


def _ensure_project_columns() -> None:
    with engine.begin() as conn:
        info = conn.execute(sql_text("PRAGMA table_info(projects)")).fetchall()
        columns = {row[1] for row in info}
        if 'forecast_scope' not in columns:
            conn.execute(sql_text("ALTER TABLE projects ADD COLUMN forecast_scope VARCHAR(50) DEFAULT 'company_total'"))


def bootstrap() -> None:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    LAUNCHES_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _ensure_project_columns()
    db = SessionLocal()
    try:
        restore_launch_registry(db)
        sync_launch_registry(db)
    finally:
        db.close()
