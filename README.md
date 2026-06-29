# Forecast App - FastAPI Edition

Local forecasting app using **FastAPI + Jinja2 + vanilla JS**.

## Run

Windows users can double-click `run_local.bat`. It uses the project-local `.venv`
and creates it automatically if needed.

Manual setup:

```bash
setup.bat
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`

Mac users can run `run_local_mac.command`. If macOS says it is not executable, open Terminal in this folder once and run:

```bash
chmod +x run_local_mac.command
```

## Health and repair

- Basic health: `http://127.0.0.1:8000/health`
- Deep health: `http://127.0.0.1:8000/health/deep`
- One-click Windows repair: double-click `repair.bat`

`repair.bat` rebuilds or refreshes the project-local `.venv`, installs
dependencies, imports the app, and runs the deep health checks.

## What is included

- Multi-page local web app
- Project creation and saved runs
- Revenue CSV upload and preview
- Column mapping form
- Data quality checks
- Forecast setup form
- Hooks into the existing forecasting engine
- Blend-vs-best-model results page
- CSV export endpoint

## Notes

This is the **FastAPI scaffold** replacing Streamlit. It keeps the forecasting engine structure under `src/` so the UI can grow without changing your core model code.


## GPU option

Version 1 includes a **Use GPU acceleration when available** option on the forecast setup page.

- Current GPU support is wired to **LightGBM**
- If GPU training is not available on the machine, the app automatically falls back to CPU
- StatsForecast and Prophet remain CPU-based in this version


This packaged build includes runtime folders (`database/`, `data/uploads/`, `data/projects/`) and a pre-created blank `database/app.db`.


B075 fix: preserve platform/category dimensions when expanding monthly or weekly helper datasets to daily so cost/leads values still join into the merged modeling table.
