from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.health import build_health_report


router = APIRouter()


@router.get('/health')
def health():
    report = build_health_report(deep=False)
    return JSONResponse(report, status_code=503 if report['status'] == 'fail' else 200)


@router.get('/health/deep')
def deep_health():
    report = build_health_report(deep=True)
    return JSONResponse(report, status_code=503 if report['status'] == 'fail' else 200)
