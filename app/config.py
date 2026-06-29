from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_NAME = 'SignalForge'
APP_ID = 'signalforge'
APP_VERSION = 'Model 1.4.10.04-G004'

DATABASE_DIR = BASE_DIR / 'database'
PROJECTS_DIR = BASE_DIR / 'data' / 'projects'
UPLOADS_DIR = BASE_DIR / 'data' / 'uploads'
LAUNCHES_DIR = BASE_DIR / 'data' / 'launches'
STATIC_DIR = BASE_DIR / 'app' / 'static'
TEMPLATES_DIR = BASE_DIR / 'app' / 'templates'

DATABASE_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
LAUNCHES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DATABASE_DIR / 'app.db'}"
SECRET_KEY = 'local-dev-key'
