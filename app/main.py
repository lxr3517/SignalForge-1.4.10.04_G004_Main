from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.config import STATIC_DIR, APP_NAME, APP_VERSION
from app.routes.health import router as health_router
from app.routes.web import router as web_router
from app.services.bootstrap import bootstrap

bootstrap()
app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')
app.include_router(health_router)
app.include_router(web_router)


@app.get('/robots.txt', include_in_schema=False)
def robots_txt():
    return FileResponse(STATIC_DIR / 'robots.txt', media_type='text/plain')
