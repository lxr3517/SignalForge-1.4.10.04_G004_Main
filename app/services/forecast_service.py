
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd

from src.models.baselines import run_baseline_forecasts
from src.models.statsforecast_models import run_statsforecast_forecast
from src.models.prophet_model import run_prophet_forecast
from src.models.lightgbm_model import run_lightgbm_forecast
from src.models.xgboost_model import run_xgboost_forecast
from src.models.linear_models import run_linear_forecast
from src.models.random_forest_model import run_random_forest_forecast
from src.models.lightgbm_intelligence import build_lightgbm_intelligence
from src.evaluation.ranking import choose_best_and_blend
from src.models.ensemble import make_scenario_frame
from src.pipeline.data_contracts import enforce_schema, dedupe_columns_keep_first, first_series
from src.models.utils import frequency_family


@dataclass
class ForecastRequest:
    df: pd.DataFrame
    date_column: str
    target_column: str
    category_column: str | None
    frequency: str
    horizon: int
    scoring_metric: str
    selected_models: list[str]
    output_mode: str
    use_gpu: bool = False
    regressor_columns: list[str] | None = None
    enable_revenue_lag_modeling: bool = False
    revenue_lag_profile: str = 'standard'
    business_profile: str = 'general'
    training_window_mode: str = 'all_history'
    training_window_start: str | None = None
    training_window_end: str | None = None
    lightgbm_artifact_dir: str | None = None




def _apply_training_window(df: pd.DataFrame, payload: ForecastRequest, date_col: str) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    meta = {
        'mode': payload.training_window_mode or 'all_history',
        'requested_start': payload.training_window_start,
        'requested_end': payload.training_window_end,
        'used_start': None,
        'used_end': None,
        'rows_before': int(len(out)),
        'rows_after': int(len(out)),
    }
    if date_col not in out.columns:
        return out, meta

    series = pd.to_datetime(first_series(out, date_col), errors='coerce')
    out[date_col] = series
    if series.dropna().empty:
        return out, meta

    mode = (payload.training_window_mode or 'all_history').strip() or 'all_history'
    end_dt = pd.Timestamp(series.max()).normalize()
    start_dt = None

    if mode == 'custom_range':
        if payload.training_window_start:
            start_dt = pd.to_datetime(payload.training_window_start, errors='coerce')
        if payload.training_window_end:
            end_dt = pd.to_datetime(payload.training_window_end, errors='coerce')
        if pd.isna(start_dt) and not pd.isna(end_dt):
            start_dt = pd.Timestamp(series.min()).normalize()
    else:
        preset_days = {
            'last_90_days': 90,
            'last_180_days': 180,
            'last_365_days': 365,
        }
        if mode in preset_days:
            start_dt = (pd.Timestamp(end_dt) - pd.Timedelta(days=preset_days[mode] - 1)).normalize()

    if start_dt is not None and not pd.isna(start_dt):
        out = out[out[date_col] >= pd.Timestamp(start_dt)].copy()
    if end_dt is not None and not pd.isna(end_dt):
        out = out[out[date_col] <= pd.Timestamp(end_dt)].copy()

    if not out.empty:
        used_series = pd.to_datetime(first_series(out, date_col), errors='coerce').dropna()
        if not used_series.empty:
            meta['used_start'] = str(pd.Timestamp(used_series.min()).date())
            meta['used_end'] = str(pd.Timestamp(used_series.max()).date())
    meta['rows_after'] = int(len(out))
    return out, meta

def _enforce_daily_rollup(df: pd.DataFrame, category_column: str | None) -> pd.DataFrame:
    out = df.copy()
    out['ds'] = pd.to_datetime(first_series(out, 'ds'), errors='coerce').dt.normalize()
    out = out.dropna(subset=['ds']).copy()
    if category_column and category_column in out.columns:
        out['series_id'] = out[category_column].astype(str)
        out['category'] = out[category_column].astype(str)
        group_cols = ['series_id', 'ds']
    else:
        out['series_id'] = 'company_total'
        out['category'] = 'company_total'
        group_cols = ['ds']

    numeric_cols = [c for c in out.columns if c not in {'series_id', 'category', 'ds'} and pd.api.types.is_numeric_dtype(out[c])]
    agg = {c: 'sum' for c in numeric_cols}
    for flag_col in ['event_flag', 'outage_flag', 'promo_flag', 'holiday_flag']:
        if flag_col in agg:
            agg[flag_col] = 'max'
    text_cols = [c for c in out.columns if c not in {'series_id', 'category', 'ds'} and c not in agg]
    for c in text_cols:
        agg[c] = 'first'
    rolled = out.groupby(group_cols, as_index=False).agg(agg)
    if 'series_id' not in rolled.columns:
        rolled['series_id'] = 'company_total'
    if 'category' not in rolled.columns:
        rolled['category'] = rolled['series_id']
    return dedupe_columns_keep_first(rolled)


def _clip_series(series: pd.Series, lower: float | None = None, upper_quantile: float | None = None) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    if lower is not None:
        s = s.clip(lower=lower)
    if upper_quantile is not None and s.notna().any():
        upper = float(s.quantile(upper_quantile))
        if math.isfinite(upper):
            s = s.clip(upper=upper)
    return s


def _stabilize_regressors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ['leads', 'cost', 'roas']:
        if col in out.columns:
            lower = 0 if col in {'leads', 'cost'} else None
            out[col] = _clip_series(first_series(out, col), lower=lower, upper_quantile=0.995)
            out[f'{col}_log1p'] = np.log1p(pd.to_numeric(out[col], errors='coerce').clip(lower=0)) if col in {'leads', 'cost'} else pd.to_numeric(out[col], errors='coerce')
    return out


def _history_stats(df: pd.DataFrame) -> dict:
    clean_y = pd.to_numeric(first_series(df, 'y'), errors='coerce').dropna()
    recent = clean_y.tail(min(30, len(clean_y)))
    return {
        'history_rows': int(len(df)),
        'recent_avg_y': float(recent.mean()) if not recent.empty else None,
        'recent_median_y': float(recent.median()) if not recent.empty else None,
        'recent_max_y': float(recent.max()) if not recent.empty else None,
        'overall_avg_y': float(clean_y.mean()) if not clean_y.empty else None,
        'max_leads': float(pd.to_numeric(first_series(df, 'leads'), errors='coerce').max()) if 'leads' in df.columns else None,
        'avg_leads': float(pd.to_numeric(first_series(df, 'leads'), errors='coerce').mean()) if 'leads' in df.columns else None,
        'max_cost': float(pd.to_numeric(first_series(df, 'cost'), errors='coerce').max()) if 'cost' in df.columns else None,
        'avg_cost': float(pd.to_numeric(first_series(df, 'cost'), errors='coerce').mean()) if 'cost' in df.columns else None,
        'duplicate_ds_rows': int(df.duplicated(subset=['series_id', 'ds']).sum()) if {'series_id','ds'}.issubset(df.columns) else None,
    }


def _forecast_passes_sanity(frame: pd.DataFrame, history: pd.DataFrame) -> tuple[bool, str]:
    if frame is None or frame.empty:
        return False, 'empty forecast'
    if not {'ds', 'yhat'}.issubset(frame.columns):
        return False, 'missing ds/yhat'
    yhat = pd.to_numeric(first_series(frame, 'yhat'), errors='coerce')
    if yhat.notna().sum() == 0:
        return False, 'all forecast values are null'
    if (yhat < 0).any():
        return False, 'negative forecast values'
    stats = _history_stats(history)
    recent_avg = stats.get('recent_avg_y') or stats.get('overall_avg_y') or 0.0
    recent_max = stats.get('recent_max_y') or recent_avg or 0.0
    allowed = max(recent_avg * 10.0, recent_max * 5.0, 1_000_000.0)
    forecast_max = float(yhat.max())
    if math.isfinite(forecast_max) and forecast_max > allowed:
        return False, f'failed sanity check: forecast max {forecast_max:,.2f} exceeded allowed {allowed:,.2f}'
    growth_ratio = forecast_max / max(recent_avg, 1.0) if recent_avg else None
    if growth_ratio and growth_ratio > 10:
        return False, f'failed sanity check: forecast/history ratio {growth_ratio:,.2f}x is too large'
    return True, 'ok'


def _sanitize_family_result(result: dict, history: pd.DataFrame, family_key: str) -> tuple[dict, str | None]:
    clean_forecasts = []
    for records in result.get('forecasts', []):
        frame = pd.DataFrame(records)
        passed, reason = _forecast_passes_sanity(frame, history)
        if passed:
            clean_forecasts.append(records)
        else:
            return {'forecasts': [], 'metrics': []}, f'{family_key} rejected: {reason}'
    result['forecasts'] = clean_forecasts
    return result, None

def _detect_gpu_hardware(requested: bool) -> dict:
    info = {
        'gpu_requested': bool(requested),
        'gpu_available': False,
        'gpu_name': None,
        'gpu_backend': 'cpu',
        'gpu_strategy': 'off',
        'gpu_memory_mb': None,
    }

    if not requested:
        return info

    # Try torch first if present.
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update({
                'gpu_available': True,
                'gpu_name': getattr(props, 'name', 'CUDA GPU'),
                'gpu_backend': 'cuda',
                'gpu_strategy': 'prefer_gpu',
                'gpu_memory_mb': int(getattr(props, 'total_memory', 0) / (1024 * 1024)),
            })
            return info
    except Exception:
        pass

    # Fallback to nvidia-smi if available.
    try:
        cmd = ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits']
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if completed.returncode == 0 and completed.stdout.strip():
            first = completed.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(',')]
            info.update({
                'gpu_available': True,
                'gpu_name': parts[0] if parts else 'NVIDIA GPU',
                'gpu_backend': 'cuda',
                'gpu_strategy': 'prefer_gpu',
                'gpu_memory_mb': int(float(parts[1])) if len(parts) > 1 and parts[1] else None,
            })
    except Exception:
        pass

    return info


def _gpu_tuning_profile(config: dict, normalized: pd.DataFrame | None = None) -> dict:
    rows = int(len(normalized)) if normalized is not None else 0
    cols = int(len(normalized.columns)) if normalized is not None else 0
    heavy = rows >= 5000 or cols >= 25
    use_gpu = bool(config.get('use_gpu'))
    return {
        'xgb_n_estimators': 700 if use_gpu and heavy else 450 if use_gpu else 300,
        'xgb_max_depth': 8 if use_gpu and heavy else 6,
        'lgbm_n_estimators': 700 if use_gpu and heavy else 450 if use_gpu else 300,
        'lgbm_num_leaves': 63 if use_gpu and heavy else 31,
        'rf_n_estimators': 500 if heavy else 350,
        'rf_max_depth': 16 if heavy else 12,
        'gpu_heavy_profile': heavy,
    }

MODEL_GROUPS = {
    'baseline': {'naive', 'historical_average', 'seasonal_naive', 'trend_drift'},
    'statsforecast': {'autoarima', 'autoets', 'theta', 'historicaverage', 'statsforecast',
                      'autotheta', 'autoces', 'dot', 'mstl'},
    'prophet': {'prophet'},
    'lightgbm': {'lightgbm'},
    'xgboost': {'xgboost'},
    'random_forest': {'random_forest', 'randomforest', 'rf'},
    'linear': {'ridge', 'elasticnet'},
}


def _selected_model_activity(df: pd.DataFrame, config: dict, selected_models: list[str]) -> dict[str, dict]:
    chosen = {m.lower() for m in selected_models}
    activity: dict[str, dict] = {}
    for family, family_models in MODEL_GROUPS.items():
        selected = sorted(chosen & set(family_models))
        if not selected:
            continue
        family_device = 'gpu' if family in {'lightgbm', 'xgboost'} and config.get('use_gpu') else 'cpu'
        activity[family] = {
            'label': family.title() if family != 'statsforecast' else 'StatsForecast',
            'family': family,
            'models': selected,
            'session_total': len(selected),
            'queued': len(selected),
            'running': 0,
            'completed': 0,
            'failed': 0,
            'status': 'queued',
            'device': family_device,
            'note': '',
            'last_error': '',
        }
    return activity


def _emit_progress(progress_callback, percent: int, step: str, detail: str, model_activity: dict | None = None) -> None:
    if progress_callback:
        try:
            progress_callback(percent, step, detail, model_activity=model_activity)
        except TypeError:
            progress_callback(percent, step, detail)


def _safe_error_hint(message: str) -> str:
    text = str(message or '').strip()
    lowered = text.lower()
    if not text:
        return 'No specific error text was returned by the runner.'
    if 'sanity check' in lowered:
        return 'The forecast finished, but its output looked unrealistic against recent history so SignalForge skipped it.'
    if 'not enough' in lowered and 'row' in lowered:
        return 'There was not enough usable history left after cleanup and filtering.'
    if 'missing' in lowered and ('date' in lowered or 'target' in lowered):
        return 'Required mapped date or target fields were missing after normalization.'
    if 'cuda' in lowered or 'gpu' in lowered:
        return 'GPU acceleration was requested, but the required backend or device was unavailable.'
    if 'seasonality' in lowered:
        return 'The model could not support the requested seasonality with the available history.'
    if 'nan' in lowered or 'inf' in lowered:
        return 'Invalid numeric values appeared in the training features or target series.'
    return text


def _family_error_summary(metric_rows: list[dict]) -> str:
    reasons: list[str] = []
    for row in metric_rows or []:
        error = str(row.get('error') or '').strip()
        if not error:
            continue
        model = str(row.get('model') or 'model')
        reasons.append(f'{model}: {_safe_error_hint(error)}')
    unique: list[str] = []
    for item in reasons:
        if item not in unique:
            unique.append(item)
    return ' | '.join(unique[:2])


def _normalize_input(payload: ForecastRequest) -> pd.DataFrame:
    df = dedupe_columns_keep_first(payload.df.copy())

    date_col = payload.date_column if payload.date_column in df.columns else ('ds' if 'ds' in df.columns else None)
    target_col = payload.target_column if payload.target_column in df.columns else ('y' if 'y' in df.columns else None)
    if date_col is None or target_col is None:
        raise ValueError(f"The modeling table is missing the mapped date/target columns. date={payload.date_column!r}, target={payload.target_column!r}, available={list(df.columns)}")

    df[date_col] = pd.to_datetime(first_series(df, date_col), errors='coerce')
    df, training_window_meta = _apply_training_window(df, payload, date_col)
    df[target_col] = pd.to_numeric(first_series(df, target_col), errors='coerce')
    df = df.dropna(subset=[date_col, target_col]).copy()
    df.attrs['training_window_meta'] = training_window_meta
    if date_col != 'ds' or target_col != 'y':
        df = df.rename(columns={date_col: 'ds', target_col: 'y'})

    if payload.category_column and payload.category_column in df.columns:
        df['series_id'] = df[payload.category_column].astype(str)
        df['category'] = df[payload.category_column].astype(str)
    elif 'series_id' not in df.columns:
        df['series_id'] = 'company_total'
        df['category'] = 'company_total'
    elif 'category' not in df.columns:
        df['category'] = df['series_id'].astype(str)

    # Coerce configured regressors and standard helper fields safely.
    for col in set((payload.regressor_columns or []) + ['leads', 'cost', 'roas', 'event_flag', 'outage_flag', 'promo_flag', 'holiday_flag']):
        if col in df.columns:
            df[col] = pd.to_numeric(first_series(df, col), errors='coerce')

    # Guard against accidental duplicate target columns or 2D target selection.
    if isinstance(df.get('y'), pd.DataFrame):
        df['y'] = df['y'].iloc[:, 0]
    if 'y' in df.columns:
        df['y'] = pd.to_numeric(first_series(df, 'y'), errors='coerce')

    # B078: enforce a final daily rollup before modeling so the forecast engine
    # always receives one row per day for company totals, or one row per day/series
    # for grouped forecasts. This keeps helper merges from leaking extra detail rows
    # into training.
    df = enforce_schema(df)
    df = _enforce_daily_rollup(df, payload.category_column)
    if 'category' not in df.columns:
        df['category'] = df['series_id']
    df = _stabilize_regressors(df)
    return df.sort_values(['series_id', 'ds']).reset_index(drop=True)


def _infer_effective_frequency(requested_frequency: str, normalized: pd.DataFrame | None) -> tuple[str, str | None]:
    requested = (requested_frequency or 'D').upper()
    if normalized is None or normalized.empty or 'ds' not in normalized.columns:
        return requested, None
    dates = pd.to_datetime(first_series(normalized, 'ds'), errors='coerce').dropna().sort_values().drop_duplicates()
    if len(dates) < 3:
        return requested, None
    gaps = dates.diff().dt.days.dropna()
    if gaps.empty:
        return requested, None
    median_gap = float(gaps.median())
    observed: str | None = None
    if 25 <= median_gap <= 45:
        day_counts = dates.dt.day.value_counts(normalize=True)
        month_start_like = float(day_counts[day_counts.index <= 7].sum()) if not day_counts.empty else 0.0
        observed = 'MS' if month_start_like >= 0.75 else 'M'
    elif 5 <= median_gap <= 10:
        observed = 'W'
    elif median_gap <= 2:
        observed = 'D'
    if observed and frequency_family(observed) != frequency_family(requested):
        family_label = {'D': 'daily', 'W': 'weekly', 'M': 'monthly'}.get(frequency_family(observed), observed)
        return observed, f"Observed cadence looks {family_label} (median gap {median_gap:.0f} days), so the engine used {observed} instead of requested {requested}."
    if observed == 'MS' and requested == 'M':
        return observed, "Observed monthly dates are month-start, so the engine used MS to keep forecast dates aligned."
    return requested, None


def _build_config(payload: ForecastRequest, normalized: pd.DataFrame | None = None) -> dict:
    freq, frequency_note = _infer_effective_frequency(payload.frequency, normalized)
    freq_family = frequency_family(freq)
    h = payload.horizon
    # Map the user-facing horizon value to the correct engine label based on frequency.
    # The UI sends horizon=12 to mean "12 months" regardless of frequency selected.
    if h == 12 and freq_family == 'M':
        horizon_label = '12M'
    elif h == 12 and freq_family == 'W':
        horizon_label = '84D'   # 12 weeks
    elif h == 12 and freq_family == 'D':
        horizon_label = '365D'  # 12 months expressed in days
    elif freq_family == 'M':
        horizon_label = f'{h}M'
    elif freq_family == 'W':
        horizon_label = f'{h * 7}D'
    else:
        horizon_label = f'{h}D'
    hardware = _detect_gpu_hardware(payload.use_gpu)
    return {
        'frequency': freq,
        'requested_frequency': payload.frequency,
        'frequency_note': frequency_note,
        'horizon': horizon_label,
        'tuning_depth': 'aggressive' if hardware.get('gpu_available') else 'balanced',
        'use_gpu': bool(hardware.get('gpu_available')),
        'gpu_requested': payload.use_gpu,
        'gpu_backend': hardware.get('gpu_backend', 'cpu'),
        'gpu_strategy': hardware.get('gpu_strategy', 'off'),
        'gpu_name': hardware.get('gpu_name'),
        'gpu_memory_mb': hardware.get('gpu_memory_mb'),
        'enable_revenue_lag_modeling': payload.enable_revenue_lag_modeling,
        'revenue_lag_profile': payload.revenue_lag_profile,
        'business_profile': (payload.business_profile or 'general'),
        'lightgbm_artifact_dir': payload.lightgbm_artifact_dir,
        'enable_shap_root_cause': True,
        'enable_ranking_engine': True,
        'enable_lead_quality_scoring': True,
        'enable_whale_prediction': True,
        'enable_model_anomaly_detection': True,
        'enable_ml_diminishing_returns': True,
        'enable_scenario_simulation': True,
        'enable_affiliate_quality_score': True,
        'enable_confidence_layer': True,
        'enable_multistage_modeling': True,
    }


def _trend_profile_for_history(history: pd.DataFrame, series_id: str | None = None) -> dict | None:
    if history is None or history.empty or not {'ds', 'y'}.issubset(history.columns):
        return None
    hist = history.copy()
    if series_id and 'series_id' in hist.columns:
        hist = hist[hist['series_id'].astype(str) == str(series_id)].copy()
    if hist.empty:
        return None
    hist['ds'] = pd.to_datetime(first_series(hist, 'ds'), errors='coerce')
    hist['y'] = pd.to_numeric(first_series(hist, 'y'), errors='coerce')
    hist = hist.dropna(subset=['ds', 'y']).groupby('ds', as_index=False)['y'].sum().sort_values('ds')
    if len(hist) < 8:
        return None
    dates = hist['ds'].drop_duplicates().sort_values()
    gaps = dates.diff().dt.days.dropna()
    median_gap = float(gaps.median()) if not gaps.empty else 1.0
    window_size = min(len(hist), 12 if median_gap >= 20 else 180)
    window_size = max(min(window_size, len(hist)), min(8, len(hist)))
    segment = hist.tail(window_size).copy()
    y = pd.to_numeric(segment['y'], errors='coerce').to_numpy(dtype=float)
    if len(y) < 8 or not np.isfinite(y).all():
        return None
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    early_mean = float(np.mean(y[:max(2, len(y) // 3)]))
    recent_mean = float(np.mean(y[-max(2, len(y) // 3):]))
    mean_y = float(np.mean(y))
    if mean_y <= 0:
        return None
    total_change = (recent_mean - early_mean) / max(abs(early_mean), 1.0)
    slope_strength = (slope * len(y)) / max(abs(mean_y), 1.0)
    clear = abs(total_change) >= 0.12 and abs(slope_strength) >= 0.08
    if not clear:
        return None
    direction = 1 if slope > 0 else -1
    return {
        'direction': direction,
        'slope': float(slope),
        'recent_mean': recent_mean,
        'last_value': float(y[-1]),
        'total_change_pct': float(total_change * 100.0),
        'median_gap_days': median_gap,
        'window_points': int(len(y)),
    }


def _guardrail_forecast_records(records: list[dict], history: pd.DataFrame, label: str) -> tuple[list[dict], dict | None]:
    if not records:
        return records, None
    frame = pd.DataFrame(records).copy()
    if frame.empty or not {'ds', 'yhat'}.issubset(frame.columns):
        return records, None
    frame['ds'] = pd.to_datetime(first_series(frame, 'ds'), errors='coerce')
    frame['yhat'] = pd.to_numeric(first_series(frame, 'yhat'), errors='coerce')
    if frame['yhat'].notna().sum() < 3:
        return records, None
    applied = []
    series_values = frame['series_id'].astype(str).unique().tolist() if 'series_id' in frame.columns else [None]
    for series_id in series_values:
        mask = frame['series_id'].astype(str).eq(str(series_id)) if series_id is not None and 'series_id' in frame.columns else pd.Series(True, index=frame.index)
        idx = frame[mask].sort_values('ds').index
        if len(idx) < 3:
            continue
        profile = _trend_profile_for_history(history, series_id)
        if not profile:
            continue
        yhat = frame.loc[idx, 'yhat'].astype(float).to_numpy()
        if not np.isfinite(yhat).all():
            continue
        future_x = np.arange(len(yhat), dtype=float)
        future_slope = float(np.polyfit(future_x, yhat, 1)[0]) if len(yhat) >= 3 else 0.0
        future_change = (future_slope * len(yhat)) / max(abs(float(np.mean(yhat))), 1.0)
        direction = int(profile['direction'])
        contradicts = (future_change * direction) < 0.04
        under_recent = direction > 0 and float(np.mean(yhat[:min(3, len(yhat))])) < profile['recent_mean'] * 0.98
        over_recent = direction < 0 and float(np.mean(yhat[:min(3, len(yhat))])) > profile['recent_mean'] * 1.02
        if not (contradicts or under_recent or over_recent):
            continue
        anchor = profile['recent_mean']
        if direction > 0:
            target = anchor + (profile['slope'] * np.arange(1, len(yhat) + 1) * 0.75)
            adjusted = np.maximum(yhat, target)
        else:
            target = anchor + (profile['slope'] * np.arange(1, len(yhat) + 1) * 0.75)
            adjusted = np.minimum(yhat, target)
        adjusted = np.clip(adjusted, 0.0, None)
        ratio = np.divide(adjusted, np.where(np.abs(yhat) > 1e-9, yhat, np.nan))
        frame.loc[idx, 'yhat'] = adjusted
        for col, fallback in [('yhat_lower', 0.92), ('yhat_upper', 1.08)]:
            if col in frame.columns:
                vals = pd.to_numeric(frame.loc[idx, col], errors='coerce').to_numpy(dtype=float)
                scaled = np.where(np.isfinite(ratio), vals * ratio, adjusted * fallback)
                frame.loc[idx, col] = np.clip(scaled, 0.0, None)
        applied.append({
            'forecast': label,
            'series_id': series_id or 'company_total',
            'direction': 'up' if direction > 0 else 'down',
            'history_change_pct': round(profile['total_change_pct'], 2),
            'history_window_points': profile['window_points'],
        })
    if not applied:
        return records, None
    return frame.sort_values(['series_id', 'ds'] if 'series_id' in frame.columns else ['ds']).to_dict(orient='records'), {
        'applied': True,
        'adjustments': applied,
    }


def _apply_trend_guardrail(ranked: dict, history: pd.DataFrame) -> tuple[dict, dict | None]:
    updated = dict(ranked or {})
    notes = []
    for key in ['best_forecast', 'blended_forecast']:
        adjusted, note = _guardrail_forecast_records(updated.get(key, []), history, key)
        updated[key] = adjusted
        if note:
            notes.extend(note.get('adjustments', []))
    if not notes:
        return updated, None
    return updated, {'applied': True, 'adjustments': notes}


def _month_edge_lift_ratio(df: pd.DataFrame, value_col: str) -> float | None:
    if df is None or df.empty or not {'ds', value_col}.issubset(df.columns):
        return None
    work = df[['ds', value_col]].copy()
    work['ds'] = pd.to_datetime(first_series(work, 'ds'), errors='coerce')
    work[value_col] = pd.to_numeric(first_series(work, value_col), errors='coerce')
    work = work.dropna(subset=['ds', value_col]).sort_values('ds')
    if len(work) < 28:
        return None

    work['day_of_month'] = work['ds'].dt.day
    work['days_in_month'] = work['ds'].dt.days_in_month
    edge_mask = (work['day_of_month'] <= 4) | (work['day_of_month'] > (work['days_in_month'] - 4))
    mid_mask = work['day_of_month'].between(11, 20)
    if int(edge_mask.sum()) < 6 or int(mid_mask.sum()) < 6:
        return None

    edge_mean = float(work.loc[edge_mask, value_col].mean())
    mid_mean = float(work.loc[mid_mask, value_col].mean())
    if not math.isfinite(edge_mean) or not math.isfinite(mid_mean):
        return None
    return (edge_mean - mid_mean) / max(abs(mid_mean), 1.0)


def _choose_display_forecast(
    ranked: dict,
    history: pd.DataFrame,
    frequency: str,
) -> tuple[list[dict], str, dict | None]:
    best_records = ranked.get('best_forecast', []) or []
    blended_records = ranked.get('blended_forecast', []) or []
    if not blended_records:
        return best_records, 'best_forecast', None
    if not best_records:
        return blended_records, 'blended_forecast', None
    if frequency_family(frequency) != 'D':
        return blended_records, 'blended_forecast', None

    hist_frame = history[['ds', 'y']].copy() if {'ds', 'y'}.issubset(history.columns) else pd.DataFrame()
    hist_edge_lift = _month_edge_lift_ratio(hist_frame, 'y')
    if hist_edge_lift is None or abs(hist_edge_lift) < 0.05:
        return blended_records, 'blended_forecast', None

    best_frame = pd.DataFrame(best_records)
    blended_frame = pd.DataFrame(blended_records)
    best_edge_lift = _month_edge_lift_ratio(best_frame, 'yhat')
    blended_edge_lift = _month_edge_lift_ratio(blended_frame, 'yhat')

    if best_edge_lift is None:
        return blended_records, 'blended_forecast', None

    hist_sign = 1.0 if hist_edge_lift >= 0 else -1.0
    best_alignment = hist_sign * best_edge_lift
    blended_alignment = hist_sign * (blended_edge_lift or 0.0)
    best_gap = abs(best_edge_lift - hist_edge_lift)
    blended_gap = abs((blended_edge_lift or 0.0) - hist_edge_lift)
    blend_is_flat = blended_edge_lift is None or abs(blended_edge_lift) < 0.02
    best_is_meaningful = abs(best_edge_lift) >= 0.03
    preserves_shape_better = (
        best_alignment > blended_alignment + 0.03
        and best_gap <= (blended_gap * 0.9 if blended_gap > 0 else best_gap)
    )

    if best_is_meaningful and (blend_is_flat or preserves_shape_better):
        return best_records, 'best_forecast', {
            'trigger': 'month_edge_pattern_preserved',
            'history_edge_lift_pct': round(hist_edge_lift * 100.0, 2),
            'best_edge_lift_pct': round(best_edge_lift * 100.0, 2),
            'blended_edge_lift_pct': round((blended_edge_lift or 0.0) * 100.0, 2),
        }
    return blended_records, 'blended_forecast', None


def _derive_day_of_month_factors(history: pd.DataFrame) -> tuple[dict[int, float], dict | None]:
    if history is None or history.empty or not {'ds', 'y'}.issubset(history.columns):
        return {}, None
    work = history[['ds', 'y']].copy()
    work['ds'] = pd.to_datetime(first_series(work, 'ds'), errors='coerce')
    work['y'] = pd.to_numeric(first_series(work, 'y'), errors='coerce')
    work = work.dropna(subset=['ds', 'y']).sort_values('ds').tail(180)
    if len(work) < 45:
        return {}, None

    work['month_period'] = work['ds'].dt.to_period('M')
    work['day_of_month'] = work['ds'].dt.day
    work['month_mean'] = work.groupby('month_period')['y'].transform('mean')
    work = work[work['month_mean'].abs() > 1e-9].copy()
    if work.empty:
        return {}, None
    work['relative_level'] = work['y'] / work['month_mean']
    day_profile = work.groupby('day_of_month').agg(
        relative_level=('relative_level', 'mean'),
        samples=('relative_level', 'size'),
    ).reset_index()
    day_profile = day_profile[day_profile['samples'] >= 2].copy()
    if len(day_profile) < 10:
        return {}, None

    raw_factors = {
        int(row['day_of_month']): float(row['relative_level'])
        for _, row in day_profile.iterrows()
        if math.isfinite(float(row['relative_level']))
    }
    if not raw_factors:
        return {}, None

    avg_factor = float(np.mean(list(raw_factors.values())))
    if not math.isfinite(avg_factor) or abs(avg_factor) < 1e-9:
        return {}, None

    normalized = {day: max(0.6, min(1.4, factor / avg_factor)) for day, factor in raw_factors.items()}
    values = np.array(list(normalized.values()), dtype=float)
    signal_strength = float(np.std(values))
    if signal_strength < 0.03:
        return {}, None

    edge_days = [d for d in normalized if d <= 4 or d >= 27]
    mid_days = [d for d in normalized if 11 <= d <= 20]
    edge_avg = float(np.mean([normalized[d] for d in edge_days])) if edge_days else None
    mid_avg = float(np.mean([normalized[d] for d in mid_days])) if mid_days else None
    meta = {
        'signal_strength': round(signal_strength, 4),
        'edge_avg_factor': round(edge_avg, 4) if edge_avg is not None else None,
        'mid_avg_factor': round(mid_avg, 4) if mid_avg is not None else None,
        'days_profiled': int(len(normalized)),
        'history_rows_used': int(len(work)),
    }
    return normalized, meta


def _apply_daily_calendar_shape(
    forecast_records: list[dict],
    history: pd.DataFrame,
    frequency: str,
) -> tuple[list[dict], dict | None]:
    if not forecast_records or frequency_family(frequency) != 'D':
        return forecast_records, None
    factors, meta = _derive_day_of_month_factors(history)
    if not factors:
        return forecast_records, None

    frame = pd.DataFrame(forecast_records).copy()
    if frame.empty or not {'ds', 'yhat'}.issubset(frame.columns):
        return forecast_records, None
    frame['ds'] = pd.to_datetime(first_series(frame, 'ds'), errors='coerce')
    frame['yhat'] = pd.to_numeric(first_series(frame, 'yhat'), errors='coerce')
    frame = frame.dropna(subset=['ds', 'yhat']).sort_values('ds').reset_index(drop=True)
    if frame.empty:
        return forecast_records, None

    frame['day_factor'] = frame['ds'].dt.day.map(factors).fillna(1.0)
    factor_mean = float(frame['day_factor'].mean()) if frame['day_factor'].notna().any() else 1.0
    if not math.isfinite(factor_mean) or abs(factor_mean) < 1e-9:
        factor_mean = 1.0
    frame['day_factor'] = frame['day_factor'] / factor_mean
    frame['yhat'] = (frame['yhat'] * frame['day_factor']).clip(lower=0.0)

    for col in ['yhat_lower', 'yhat_upper']:
        if col in frame.columns:
            frame[col] = (pd.to_numeric(first_series(frame, col), errors='coerce') * frame['day_factor']).clip(lower=0.0)

    if {'yhat_lower', 'yhat_upper'}.issubset(frame.columns):
        frame['yhat_lower'] = np.minimum(frame['yhat_lower'], frame['yhat'])
        frame['yhat_upper'] = np.maximum(frame['yhat_upper'], frame['yhat'])

    note = {
        'applied': True,
        'method': 'daily_day_of_month_profile',
        'signal_strength': meta.get('signal_strength'),
        'edge_avg_factor': meta.get('edge_avg_factor'),
        'mid_avg_factor': meta.get('mid_avg_factor'),
    }
    return frame.to_dict(orient='records'), note


def _collect_results(df: pd.DataFrame, config: dict, selected_models: list[str], progress_callback: Callable | None = None) -> tuple[list[dict], list[list[dict]], list[dict], dict[str, dict]]:
    chosen = {m.lower() for m in selected_models}
    all_metrics: list[dict] = []
    all_forecasts: list[list[dict]] = []
    model_errors: list[dict] = []
    model_activity = _selected_model_activity(df, config, selected_models)
    _emit_progress(progress_callback, 12, 'Preparing model sessions', 'Registering selected model families, session counts, and compute device choices.', model_activity=model_activity)

    def _run_family(family_key: str, percent: int, step: str, detail: str, runner):
        if not (chosen & MODEL_GROUPS[family_key]):
            return
        family_state = model_activity.get(family_key)
        if family_state:
            family_state['status'] = 'running'
            family_state['running'] = family_state.get('session_total', 0)
            family_state['queued'] = 0
            family_state['note'] = detail
            family_state['last_error'] = ''
        _emit_progress(progress_callback, percent, step, detail, model_activity=model_activity)
        try:
            result = runner(df, config)
            result, sanity_error = _sanitize_family_result(result, df, family_key)
            if sanity_error:
                model_errors.append({'family': family_key, 'step': step, 'error': sanity_error})
                if family_state:
                    family_state['status'] = 'skipped'
                    family_state['running'] = 0
                    family_state['failed'] = family_state.get('session_total', 0)
                    family_state['completed'] = 0
                    family_state['last_error'] = _safe_error_hint(sanity_error)
                    family_state['note'] = 'Skipped because the finished forecast failed post-run sanity checks.'
                _emit_progress(progress_callback, percent, f'{step} skipped', f'{_safe_error_hint(sanity_error)} Raw detail: {sanity_error}', model_activity=model_activity)
                return
            all_metrics.extend(result.get('metrics', []))
            all_forecasts.extend(result.get('forecasts', []))
            if family_state:
                metric_rows = result.get('metrics', []) or []
                failed = sum(1 for m in metric_rows if m.get('error'))
                completed = max(0, family_state.get('session_total', 0) - failed)
                failure_summary = _family_error_summary(metric_rows)
                family_state['running'] = 0
                family_state['completed'] = completed
                family_state['failed'] = failed
                family_state['status'] = 'completed' if failed == 0 else ('partial' if completed > 0 else 'failed')
                family_state['last_error'] = failure_summary
                family_state['note'] = (
                    f'Finished {completed} of {family_state.get("session_total", 0)} model sessions successfully.'
                    if failed == 0
                    else f'Finished with partial coverage. {failed} session(s) failed.'
                )
            status_detail = f"{family_state.get('completed', 0) if family_state else 0} completed, {family_state.get('failed', 0) if family_state else 0} failed."
            if family_state and family_state.get('failed') and family_state.get('last_error'):
                status_detail += f" Why failures happened: {family_state.get('last_error')}"
            _emit_progress(progress_callback, percent + 1 if percent < 89 else percent, f'{step} status', status_detail, model_activity=model_activity)
        except Exception as exc:
            model_errors.append({'family': family_key, 'step': step, 'error': str(exc)})
            if family_state:
                family_state['status'] = 'failed'
                family_state['running'] = 0
                family_state['failed'] = family_state.get('session_total', 0)
                family_state['completed'] = 0
                family_state['last_error'] = _safe_error_hint(str(exc))
                family_state['note'] = 'This family stopped before producing usable metrics.'
            _emit_progress(progress_callback, percent, f'{step} failed', f'{step} hit an error and was skipped. Likely reason: {_safe_error_hint(str(exc))}. Raw detail: {exc}', model_activity=model_activity)

    _run_family('baseline', 15, 'Running baseline models', 'Calculating naive and seasonal benchmark forecasts so every advanced model has a simple business baseline to beat.', run_baseline_forecasts)
    _run_family('statsforecast', 35, 'Running StatsForecast models', 'Training AutoARIMA, AutoETS, Theta, and related statistical models in the background to test fast classical signal patterns.', run_statsforecast_forecast)
    _run_family('prophet', 58, 'Running Prophet', 'Fitting trend, seasonality, and interval-based forecasts while checking whether calendar structure improves the blend.', run_prophet_forecast)

    gpu_note = ' using GPU acceleration when available across LightGBM and XGBoost.' if config.get('use_gpu') else ' on CPU.'
    lag_note = ' Revenue lag modeling is enabled.' if config.get('enable_revenue_lag_modeling') else ''
    _run_family('lightgbm', 78, 'Running LightGBM', 'Building lag-based machine-learning forecasts and recursive predictions' + gpu_note + lag_note + ' SignalForge is testing whether tree models can capture nonlinear revenue patterns.', run_lightgbm_forecast)
    _run_family('xgboost', 84, 'Running XGBoost', 'Training gradient-boosted trees with the shared feature pipeline to compare against LightGBM and classical models.', run_xgboost_forecast)
    _run_family('random_forest', 87, 'Running Random Forest', 'Training bagged decision trees on lag, calendar, and driver features to check for a more stable non-parametric fit.', run_random_forest_forecast)
    _run_family('linear', 89, 'Running Linear models', 'Fitting Ridge and ElasticNet with scaled lag features to test simpler regularized machine-learning baselines.', run_linear_forecast)

    return all_metrics, all_forecasts, model_errors, model_activity


def run_forecast(payload: ForecastRequest, progress_callback: Callable | None = None) -> dict:
    _emit_progress(progress_callback, 5, 'Preparing data', 'Normalizing dates, targets, and series structure so the engine can remove unusable rows before training begins.')
    normalized = _normalize_input(payload)
    required_cols = {"ds", "y", "series_id"}
    if not required_cols.issubset(normalized.columns):
        missing = sorted(required_cols - set(normalized.columns))
        raise ValueError(f"Normalized forecast data is missing required columns: {', '.join(missing)}")
    if normalized.empty:
        raise ValueError("No usable rows were found after applying the date and target column mappings.")
    history_start = normalized['ds'].min()
    history_end = normalized['ds'].max()
    series_count = int(normalized['series_id'].nunique()) if 'series_id' in normalized.columns else 1
    _emit_progress(
        progress_callback,
        10,
        'Building configuration',
        f'Preparing horizon, frequency, and tuning settings for {len(normalized):,} clean rows across {series_count} series from {history_start:%Y-%m-%d} to {history_end:%Y-%m-%d}.'
    )
    config = _build_config(payload, normalized)
    config.update(_gpu_tuning_profile(config, normalized))
    hardware_mode = f"{config.get('gpu_backend') or 'cpu'} / {config.get('gpu_strategy') or 'off'}"
    cadence_note = f" {config.get('frequency_note')}" if config.get('frequency_note') else ''
    _emit_progress(
        progress_callback,
        11,
        'Configuration ready',
        f"Forecast horizon is {config.get('horizon')}, frequency is {config.get('frequency')}, tuning depth is {config.get('tuning_depth')}, and compute mode is {hardware_mode}.{cadence_note}"
    )
    all_metrics, all_forecasts, model_errors, model_activity = _collect_results(normalized, config, payload.selected_models, progress_callback=progress_callback)
    _emit_progress(progress_callback, 90, 'Ranking models', 'Comparing backtests, weighting the strongest candidates, and removing weak performers from the final blend.', model_activity=model_activity)
    if not all_metrics:
        if model_errors:
            error_text = '; '.join(f"{e['family']}: {e['error']}" for e in model_errors)
            raise ValueError('No model family completed successfully. Errors: ' + error_text)
        raise ValueError('No metrics were produced.')
    ranked = choose_best_and_blend(all_metrics, all_forecasts, payload.scoring_metric)
    ranked, trend_guardrail = _apply_trend_guardrail(ranked, normalized)

    display_forecast, display_forecast_source, display_forecast_note = _choose_display_forecast(
        ranked,
        normalized,
        config.get('frequency', payload.frequency),
    )
    display_forecast, daily_calendar_shape_note = _apply_daily_calendar_shape(
        display_forecast,
        normalized,
        config.get('frequency', payload.frequency),
    )

    if payload.output_mode == 'scenario_bands':
        ensemble_preview = make_scenario_frame(display_forecast)
    elif payload.output_mode == 'confidence_intervals':
        ensemble_preview = display_forecast
    else:
        ensemble_preview = [
            {'ds': row.get('ds'), 'expected': row.get('yhat')}
            for row in display_forecast
        ]

    lightgbm_intelligence = {'available': False, 'error': 'LightGBM intelligence was not run.'}
    _emit_progress(progress_callback, 94, 'Building LightGBM intelligence', 'Creating additive explainability, ranking, lead-quality, and anomaly intelligence artifacts without changing the core forecast.', model_activity=model_activity)
    try:
        lightgbm_intelligence = build_lightgbm_intelligence(normalized, config)
    except Exception as exc:
        lightgbm_intelligence = {'available': False, 'error': str(exc)}
        model_errors.append({'family': 'lightgbm_intelligence', 'step': 'Building LightGBM intelligence', 'error': str(exc)})

    _emit_progress(progress_callback, 96, 'Formatting output', 'Preparing scenario bands, preview tables, diagnostics, and hardware notes for the results dashboard.', model_activity=model_activity)
    gpu_used_for = []
    top_models = ranked.get('model_ranking', [])[:5]
    lightgbm_metric = next((m for m in all_metrics if m.get('model') == 'lightgbm'), None)
    xgboost_metric = next((m for m in all_metrics if m.get('model') == 'xgboost'), None)
    gpu_actual = lightgbm_metric.get('lightgbm_device') if lightgbm_metric else None
    xgb_actual = xgboost_metric.get('xgboost_device') if xgboost_metric else None

    if gpu_actual == 'gpu':
        gpu_used_for.append('lightgbm')
    if xgb_actual == 'gpu':
        gpu_used_for.append('xgboost')

    if config.get('use_gpu') and gpu_used_for:
        hardware_note = (
            f"GPU mode was requested and detected ({config.get('gpu_name') or 'GPU'}). "
            f"Accelerated models: {', '.join(gpu_used_for)}. "
            f"Profile: {'heavy' if config.get('gpu_heavy_profile') else 'standard'}."
        )
    elif payload.use_gpu and not config.get('use_gpu'):
        hardware_note = 'GPU mode was requested, but no compatible GPU backend was detected. CPU fallback was used.'
    elif payload.use_gpu and (gpu_actual == 'cpu_fallback' or xgb_actual == 'cpu_fallback'):
        hardware_note = 'GPU mode was requested, but one or more models fell back to CPU after a GPU attempt.'
    else:
        hardware_note = 'CPU mode used for all models.'
    diagnostics = _history_stats(normalized)
    diagnostics['requested_frequency'] = config.get('requested_frequency')
    diagnostics['effective_frequency'] = config.get('frequency')
    if config.get('frequency_note'):
        diagnostics['frequency_note'] = config.get('frequency_note')
    if trend_guardrail:
        diagnostics['trend_guardrail'] = trend_guardrail
    training_window_meta = normalized.attrs.get('training_window_meta', {}) if hasattr(normalized, 'attrs') else {}
    if training_window_meta:
        diagnostics['training_window_used'] = training_window_meta
    if ranked.get('blended_forecast'):
        forecast_diag = pd.DataFrame(ranked.get('blended_forecast'))
        if 'yhat' in forecast_diag.columns:
            expected_vals = pd.to_numeric(first_series(forecast_diag, 'yhat'), errors='coerce')
            diagnostics['forecast_avg_expected'] = float(expected_vals.mean()) if expected_vals.notna().any() else None
            base = diagnostics.get('recent_avg_y') or diagnostics.get('overall_avg_y') or None
            diagnostics['forecast_to_history_ratio'] = float((expected_vals.mean() / base)) if base and expected_vals.notna().any() else None
    diagnostics['display_forecast_source'] = display_forecast_source
    if display_forecast_note:
        diagnostics['display_forecast_note'] = display_forecast_note
    if daily_calendar_shape_note:
        diagnostics['daily_calendar_shape_note'] = daily_calendar_shape_note
    result = {
        'diagnostics': diagnostics,
        'model_activity': model_activity,
        'lightgbm_intelligence': lightgbm_intelligence,
        'settings': {
            'revenue_lag_modeling_enabled': payload.enable_revenue_lag_modeling,
            'revenue_lag_profile': payload.revenue_lag_profile if payload.enable_revenue_lag_modeling else 'off',
            'feature_engineering': 'next_level_v1_gpu',
            'training_window_mode': payload.training_window_mode,
            'training_window_start': payload.training_window_start,
            'training_window_end': payload.training_window_end,
            'training_window_used': training_window_meta,
        },
        'top_models': top_models,
        'best_model': ranked.get('best_model', {}),
        'best_forecast': ranked.get('best_forecast', []),
        'blended_forecast': ranked.get('blended_forecast', []),
        'display_forecast': display_forecast,
        'display_forecast_source': display_forecast_source,
        'ensemble_preview': ensemble_preview[:100],
        'backtest_rows': len(all_metrics),
        'model_errors': model_errors,
        'hardware': {
            'gpu_requested': payload.use_gpu,
            'gpu_available': config.get('use_gpu', False),
            'gpu_used_for': gpu_used_for,
            'gpu_backend': config.get('gpu_backend', 'cpu'),
            'gpu_strategy': config.get('gpu_strategy', 'off'),
            'gpu_name': config.get('gpu_name'),
            'gpu_memory_mb': config.get('gpu_memory_mb'),
            'lightgbm_device': gpu_actual,
            'xgboost_device': xgb_actual,
            'note': hardware_note,
        },
    }
    _emit_progress(progress_callback, 100, 'Completed', 'Forecast results are ready.', model_activity=model_activity)
    return result
