
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app import models
from app.schemas import ProjectCreate
from app.services.launches_service import sync_launch_registry


def create_project(db: Session, payload: ProjectCreate) -> models.Project:
    project = models.Project(**payload.model_dump())
    db.add(project)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        msg = str(getattr(exc, 'orig', exc)).lower()
        if 'unique constraint failed' in msg and 'projects.name' in msg:
            raise ValueError('duplicate_project_name') from exc
        raise
    db.refresh(project)
    sync_launch_registry(db)
    return project


def update_project(db: Session, project: models.Project, payload: ProjectCreate) -> models.Project:
    for key, value in payload.model_dump().items():
        setattr(project, key, value)
    db.add(project)
    db.commit()
    db.refresh(project)
    sync_launch_registry(db)
    return project


def list_projects(db: Session) -> list[models.Project]:
    return db.query(models.Project).order_by(models.Project.updated_at.desc()).all()


def get_project(db: Session, project_id: int) -> models.Project | None:
    return db.query(models.Project).filter(models.Project.id == project_id).first()


def create_run(db: Session, project_id: int, label: str) -> models.Run:
    run = models.Run(project_id=project_id, label=label)
    db.add(run)
    db.commit()
    db.refresh(run)
    sync_launch_registry(db)
    return run


def update_run_status(db: Session, run: models.Run, status: str) -> models.Run:
    run.status = status
    db.add(run)
    db.commit()
    db.refresh(run)
    sync_launch_registry(db)
    return run


def list_runs(db: Session, project_id: int | None = None) -> list[models.Run]:
    query = db.query(models.Run)
    if project_id is not None:
        query = query.filter(models.Run.project_id == project_id)
    return query.order_by(models.Run.updated_at.desc()).all()


def get_run(db: Session, run_id: int) -> models.Run | None:
    return db.query(models.Run).filter(models.Run.id == run_id).first()


def delete_project(db: Session, project: models.Project) -> None:
    db.query(models.Run).filter(models.Run.project_id == project.id).delete(synchronize_session=False)
    db.delete(project)
    db.commit()
    sync_launch_registry(db)
