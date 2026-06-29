from __future__ import annotations

import json
import math
import html
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.responses import RedirectResponse

from app.config import BASE_DIR, UPLOADS_DIR
from app.services.file_service import get_sheet_names, load_dataframe
from app.services.forecast_service import ForecastRequest, run_forecast
from app.services.progress_service import write_progress
from app.services.spending_slowdown import build_spending_slowdown
from src.pipeline.data_contracts import (
    dedupe_columns_keep_first,
    first_series,
    normalize_date_series,
    enforce_schema,
)


def safe_number(value):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def json_ready_records(df: pd.DataFrame) -> list[dict]:
    temp = df.copy()
    for col in temp.columns:
        if pd.api.types.is_datetime64_any_dtype(temp[col]):
            temp[col] = temp[col].dt.strftime('%Y-%m-%d')
    return temp.to_dict(orient='records')


def plot_card(chart_id: str, title: str, spec: dict, subtitle: str | None = None) -> dict:
    return {
        'id': chart_id,
        'title': title,
        'subtitle': subtitle or '',
        'spec': json.dumps(json_clean(spec), default=str, allow_nan=False),
    }



def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_clean(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        try:
            return pd.Timestamp(value).isoformat()
        except Exception:
            return None
    if isinstance(value, (np.floating, float)):
        try:
            if not np.isfinite(value):
                return None
        except Exception:
            pass
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value

def safe_divide(numerator, denominator, default=0.0):
    try:
        num = float(numerator)
        den = float(denominator)
        if pd.isna(num) or pd.isna(den) or den == 0:
            return default
        return num / den
    except Exception:
        return default


def compact_category_label(value, max_len: int = 26) -> str:
    text = str(value or '').strip()
    if not text:
        return 'Unknown'

    parts = [part.strip() for part in text.split('|') if part.strip()]
    campaign = None
    ad_name = None
    affiliate = None
    platform = None

    for part in parts:
        lower = part.lower()
        if lower.startswith('campaign='):
            campaign = part.split('=', 1)[1].strip()
        elif lower.startswith('ad=') or lower.startswith('ad name='):
            ad_name = part.split('=', 1)[1].strip()
        elif lower.startswith('affiliate=') or lower.startswith('aff_param='):
            affiliate = part.split('=', 1)[1].strip()
        elif lower.startswith('platform='):
            platform = part.split('=', 1)[1].strip()

    compact_parts = []
    if ad_name:
        compact_parts.append(f"A={ad_name}")
    if campaign:
        compact_parts.append(f"C={campaign}")
    if affiliate:
        compact_parts.append(f"AFF={affiliate.replace('AFF-', '')}")
    if platform:
        compact_parts.append(f"P={platform}")

    if compact_parts:
        label = " | ".join(compact_parts)
    else:
        candidates = [ad_name, campaign, affiliate, platform, text]
        label = next((candidate for candidate in candidates if candidate), text)
    if len(label) <= max_len:
        return label
    return label[: max_len - 1].rstrip() + '…'


DRIVER_ALLOWLIST = {
    'leads', 'cost', 'spend', 'clicks', 'visits', 'registrations', 'impressions',
    'event_flag', 'outage_flag', 'promo_flag', 'holiday_flag'
}

DRIVER_BLOCKLIST = {
    'ds', 'date', 'y', 'target', 'series_id', 'category',
    'revenue_per_lead', 'rev_per_lead', 'roas', 'roas_calc',
    'margin', 'margin_after_cost', 'profit', 'profit_per_lead',
    'forecast', 'expected', 'conservative', 'aggressive', 'lower', 'upper',
    'p10', 'p50', 'p90', 'zscore', 'recent_flag'
}



def diagnostic_frequency_label(settings: dict | None, hist_daily: pd.DataFrame | None = None) -> str:
    freq = str((settings or {}).get('frequency') or '').strip().upper()
    mapping = {
        'D': 'day',
        'DAILY': 'day',
        'W': 'week',
        'WEEKLY': 'week',
        'M': 'month',
        'MONTHLY': 'month',
        'Q': 'quarter',
        'QUARTERLY': 'quarter',
    }
    if freq in mapping:
        return mapping[freq]
    if hist_daily is not None and isinstance(hist_daily, pd.DataFrame) and not hist_daily.empty and 'ds' in hist_daily.columns:
        return 'day'
    return 'period'



def monthly_equivalent_factor(period_unit: str) -> float:
    unit = str(period_unit or '').strip().lower()
    mapping = {
        'day': 30.4375,
        'week': 4.345,
        'month': 1.0,
        'quarter': 1.0 / 3.0,
        'period': 1.0,
    }
    return float(mapping.get(unit, 1.0))


def normalize_frequency_code(settings: dict | None = None, hist_daily: pd.DataFrame | None = None, forecast_df: pd.DataFrame | None = None) -> str:
    freq = str((settings or {}).get('frequency') or '').strip().upper()
    mapping = {
        'D': 'D',
        'DAILY': 'D',
        'W': 'W',
        'WEEKLY': 'W',
        'M': 'M',
        'MONTHLY': 'M',
    }
    if freq in mapping:
        return mapping[freq]

    for frame in (forecast_df, hist_daily):
        if frame is None or frame.empty or 'ds' not in frame.columns:
            continue
        ds = pd.to_datetime(first_series(frame, 'ds'), errors='coerce').dropna()
        if len(ds) >= 2:
            detected = _detect_frequency(ds)
            if detected in {'D', 'W', 'M'}:
                return detected
    return 'D'


def aggregate_series_to_frequency(df: pd.DataFrame | None, freq_code: str, value_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty or 'ds' not in df.columns:
        return pd.DataFrame(columns=['ds'] + value_cols)

    work = dedupe_columns_keep_first(df.copy())
    work['ds'] = pd.to_datetime(first_series(work, 'ds'), errors='coerce')
    work = work.dropna(subset=['ds']).sort_values('ds').reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=['ds'] + value_cols)

    keep = ['ds'] + [col for col in value_cols if col in work.columns]
    work = work[keep].copy()
    for col in keep:
        if col != 'ds':
            work[col] = pd.to_numeric(first_series(work, col), errors='coerce')

    if freq_code == 'D':
        return work

    grouped = work.copy()
    if freq_code == 'W':
        grouped['ds'] = grouped['ds'].dt.to_period('W-SUN').apply(lambda p: p.start_time)
    elif freq_code == 'M':
        grouped['ds'] = grouped['ds'].dt.to_period('M').dt.to_timestamp()

    agg_map = {col: 'sum' for col in keep if col != 'ds'}
    return grouped.groupby('ds', as_index=False).agg(agg_map).sort_values('ds').reset_index(drop=True)


def format_plot_dates(ds: pd.Series, freq_code: str) -> list[str]:
    parsed = pd.to_datetime(ds, errors='coerce')
    if freq_code == 'M':
        return parsed.dt.strftime('%Y-%m').tolist()
    if freq_code == 'W':
        return parsed.dt.strftime('%Y-%m-%d').radd('Week of ').tolist()
    return parsed.dt.strftime('%Y-%m-%d').tolist()


def prepare_risk_band_display(hist_daily: pd.DataFrame | None, forecast_df: pd.DataFrame, freq_code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist_chart = aggregate_series_to_frequency(hist_daily, freq_code, ['y']) if hist_daily is not None and not hist_daily.empty else pd.DataFrame(columns=['ds', 'y'])
    fc_chart = aggregate_series_to_frequency(forecast_df, freq_code, ['conservative', 'expected', 'aggressive'])
    if fc_chart.empty:
        return hist_chart, fc_chart

    for col in ['conservative', 'expected', 'aggressive']:
        if col in fc_chart.columns:
            fc_chart[col] = pd.to_numeric(fc_chart[col], errors='coerce').clip(lower=0)

    if {'conservative', 'expected'}.issubset(fc_chart.columns):
        fc_chart['conservative'] = np.minimum(fc_chart['conservative'], fc_chart['expected'])
    if {'aggressive', 'expected'}.issubset(fc_chart.columns):
        fc_chart['aggressive'] = np.maximum(fc_chart['aggressive'], fc_chart['expected'])

    smooth_window = 1
    if freq_code == 'D':
        smooth_window = 7
    elif freq_code == 'W':
        smooth_window = 2

    if smooth_window > 1:
        for col in ['conservative', 'expected', 'aggressive']:
            if col in fc_chart.columns:
                fc_chart[col] = fc_chart[col].rolling(smooth_window, min_periods=1).mean()

    return hist_chart, fc_chart


def format_history_window_label(hist_daily: pd.DataFrame | None) -> str:
    if hist_daily is None or hist_daily.empty or 'ds' not in hist_daily.columns:
        return 'Full aligned history'
    ds = pd.to_datetime(first_series(hist_daily, 'ds'), errors='coerce').dropna()
    if ds.empty:
        return 'Full aligned history'
    start = ds.min().strftime('%Y-%m-%d')
    end = ds.max().strftime('%Y-%m-%d')
    return f'{start} to {end}'

def build_driver_impact_frame(hist_daily: pd.DataFrame, allowed_cols: set[str] | None = None) -> pd.DataFrame:
    if hist_daily is None or hist_daily.empty or 'y' not in hist_daily.columns:
        return pd.DataFrame()

    work = hist_daily.copy()
    work['y'] = pd.to_numeric(work['y'], errors='coerce')
    recent = work.tail(min(len(work), 90)).copy()
    base_y = float(pd.to_numeric(recent['y'], errors='coerce').dropna().mean() or 0.0)
    if base_y <= 0:
        return pd.DataFrame()

    numeric_candidates = []
    for col in work.columns:
        if col in DRIVER_BLOCKLIST:
            continue
        if allowed_cols is not None and col not in allowed_cols:
            continue
        if allowed_cols is None and col not in DRIVER_ALLOWLIST:
            continue
        series = pd.to_numeric(work[col], errors='coerce')
        if series.notna().sum() < 10:
            continue
        if float(series.fillna(0).abs().sum()) == 0.0:
            continue
        numeric_candidates.append(col)

    rows = []
    for col in numeric_candidates:
        x = pd.to_numeric(work[col], errors='coerce')
        y = pd.to_numeric(work['y'], errors='coerce')
        valid = pd.DataFrame({'x': x, 'y': y}).dropna().copy()
        if len(valid) < 10:
            continue

        x_series = valid['x'].astype(float)
        y_series = valid['y'].astype(float)
        x_std = float(x_series.std(ddof=0) or 0.0)
        y_std = float(y_series.std(ddof=0) or 0.0)
        if x_std == 0.0 or y_std == 0.0:
            continue

        corr = float(x_series.corr(y_series)) if len(valid) > 2 else np.nan
        if pd.isna(corr):
            continue

        positive = valid[(valid['x'] > 0) & (valid['y'] > 0)].copy()
        elasticity = np.nan
        if len(positive) >= 12 and positive['x'].nunique() >= 4:
            try:
                elasticity = float(np.polyfit(np.log1p(positive['x'].astype(float)), np.log1p(positive['y'].astype(float)), 1)[0])
            except Exception:
                elasticity = np.nan

        x_pct = x_series.pct_change().replace([np.inf, -np.inf], np.nan)
        y_pct = y_series.pct_change().replace([np.inf, -np.inf], np.nan)
        pct_valid = pd.DataFrame({'x': x_pct, 'y': y_pct}).dropna()
        pct_beta = np.nan
        if len(pct_valid) >= 8 and pct_valid['x'].std(ddof=0) not in (0, np.nan):
            x_var = float(pct_valid['x'].var(ddof=0) or 0.0)
            if x_var > 0:
                pct_beta = float(pct_valid['x'].cov(pct_valid['y']) / x_var)

        signals = [v for v in [elasticity, pct_beta, corr] if not pd.isna(v)]
        if not signals:
            continue
        effect_score = float(np.median(signals))
        effect_score = float(np.clip(effect_score, -3.0, 3.0))

        recent_x = pd.to_numeric(recent[col], errors='coerce').dropna()
        base_x = float(recent_x.mean() or 0.0) if not recent_x.empty else float(x_series.tail(min(len(x_series), 30)).mean() or 0.0)
        if base_x == 0.0 and col not in {'event_flag', 'outage_flag', 'promo_flag', 'holiday_flag'}:
            continue

        stability = max(0.0, 1.0 - min(1.0, abs(float(x_series.diff().std(ddof=0) or 0.0)) / (abs(base_x) + 1e-9)))
        support = min(1.0, len(valid) / 45.0)
        confidence = float(np.clip((abs(corr) * 0.45) + (support * 0.35) + (stability * 0.20), 0.0, 1.0))
        if confidence < 0.18:
            continue

        impact_10pct = float(base_y * effect_score * 0.10)
        rows.append({
            'driver': col,
            'correlation_to_revenue': corr,
            'elasticity_score': elasticity if not pd.isna(elasticity) else None,
            'pct_change_beta': pct_beta if not pd.isna(pct_beta) else None,
            'effect_score': effect_score,
            'confidence_score': confidence,
            'base_value_recent': base_x,
            'base_revenue_recent': base_y,
            'revenue_change_if_driver_moves_10pct': impact_10pct,
            'suggested_direction': 'increase' if impact_10pct >= 0 else 'decrease',
            'evidence_strength': 'high' if confidence >= 0.75 else 'medium' if confidence >= 0.45 else 'low',
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=['confidence_score', 'revenue_change_if_driver_moves_10pct'],
        key=lambda s: s.abs() if s.name == 'revenue_change_if_driver_moves_10pct' else s,
        ascending=[False, False],
    ).reset_index(drop=True)
    return out


def _mapped_driver_columns(mapping: dict | None, upload_meta: dict | None = None) -> tuple[set[str], list[dict]]:
    mapping = mapping or {}
    upload_meta = upload_meta or {}
    files = upload_meta.get('files', {}) if isinstance(upload_meta, dict) else {}

    sheets: list[dict] = []

    def _clean_cols(cols):
        return [str(c).strip() for c in (cols or []) if str(c).strip()]

    leads_cfg = mapping.get('leads') or {}
    if leads_cfg.get('value_column'):
        sheets.append({'sheet_key': 'leads', 'sheet_label': 'Leads', 'columns': ['leads']})

    cost_cfg = mapping.get('cost') or {}
    if cost_cfg.get('value_column'):
        sheets.append({'sheet_key': 'cost', 'sheet_label': 'Cost / Spend', 'columns': ['cost']})

    events_cfg = mapping.get('events') or {}
    event_cols = []
    if events_cfg.get('event_flag_column'):
        event_cols.append('event_flag')
    if events_cfg.get('outage_flag_column'):
        event_cols.append('outage_flag')
    if event_cols:
        sheets.append({'sheet_key': 'events', 'sheet_label': 'Events / Flags', 'columns': event_cols})

    custom_reg_cfg = mapping.get('custom_regressors') or {}
    custom_reg_cols = _clean_cols(custom_reg_cfg.get('value_columns'))
    if custom_reg_cols:
        sheets.append({'sheet_key': 'custom_regressors', 'sheet_label': 'Custom Regressors', 'columns': custom_reg_cols})

    for key, cfg in mapping.items():
        if not (isinstance(cfg, dict) and key.startswith('custom_')):
            continue
        cols = _clean_cols(cfg.get('value_columns'))
        if not cols:
            continue
        meta = files.get(key, {})
        label = meta.get('label') or key.replace('custom_', 'Custom Sheet ')
        sheets.append({'sheet_key': key, 'sheet_label': label, 'columns': cols})

    allowed = set()
    for item in sheets:
        allowed.update(item['columns'])

    allowed.update({'leads', 'cost', 'spend', 'clicks', 'visits', 'registrations', 'impressions', 'event_flag', 'outage_flag', 'promo_flag', 'holiday_flag'})
    return allowed, sheets


def _build_sheet_impact_frame(hist_daily: pd.DataFrame, impact_df: pd.DataFrame, sheet_defs: list[dict]) -> pd.DataFrame:
    if hist_daily is None or hist_daily.empty or impact_df is None or impact_df.empty or not sheet_defs:
        return pd.DataFrame()

    hist_work = hist_daily.copy()
    hist_work['y'] = pd.to_numeric(hist_work.get('y'), errors='coerce')
    rows = []
    for sheet in sheet_defs:
        cols = [c for c in sheet.get('columns', []) if c in hist_work.columns]
        if not cols:
            continue
        subset = impact_df[impact_df['driver'].isin(cols)].copy()
        if subset.empty:
            continue

        strongest = subset.iloc[subset['revenue_change_if_driver_moves_10pct'].abs().idxmax()]
        signed_impact = float(pd.to_numeric(subset['revenue_change_if_driver_moves_10pct'], errors='coerce').fillna(0).sum())
        abs_impact = float(pd.to_numeric(subset['revenue_change_if_driver_moves_10pct'], errors='coerce').abs().sum())
        avg_conf = float(pd.to_numeric(subset['confidence_score'], errors='coerce').fillna(0).mean())

        corr_values = pd.to_numeric(subset['correlation_to_revenue'], errors='coerce').dropna()
        net_corr = float(corr_values.mean()) if not corr_values.empty else None

        rev_share = None
        try:
            y_valid = hist_work['y'].dropna()
            if len(y_valid) >= 10:
                modeled = pd.Series(0.0, index=hist_work.index, dtype='float64')
                for c in cols:
                    x = pd.to_numeric(hist_work[c], errors='coerce')
                    if x.notna().sum() < 3:
                        continue
                    x_std = float(x.std(ddof=0) or 0.0)
                    if x_std == 0.0:
                        continue
                    z = (x - float(x.mean() or 0.0)) / x_std
                    weight = float(pd.to_numeric(subset.loc[subset['driver'] == c, 'effect_score'], errors='coerce').fillna(0).mean())
                    modeled = modeled.add(z.fillna(0) * weight, fill_value=0)
                modeled_std = float(modeled.std(ddof=0) or 0.0)
                y_std = float(y_valid.std(ddof=0) or 0.0)
                if modeled_std > 0 and y_std > 0:
                    rev_share = float(np.clip(abs(float(modeled.corr(hist_work['y'].fillna(method='ffill').fillna(method='bfill').fillna(0)) or 0.0)), 0.0, 1.0))
        except Exception:
            rev_share = None

        rows.append({
            'sheet': sheet['sheet_label'],
            'active_columns': len(cols),
            'drivers_found': int(len(subset)),
            'strongest_driver': str(strongest['driver']),
            'strongest_driver_impact_10pct': float(strongest['revenue_change_if_driver_moves_10pct']),
            'sheet_net_impact_10pct': signed_impact,
            'sheet_total_absolute_impact': abs_impact,
            'avg_confidence_score': avg_conf,
            'net_correlation_to_revenue': net_corr,
            'estimated_revenue_relationship_share': rev_share,
            'direction': 'tailwind' if signed_impact >= 0 else 'headwind',
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(['sheet_total_absolute_impact', 'avg_confidence_score'], ascending=[False, False]).reset_index(drop=True)
    return out


def _build_quality_visuals(df: pd.DataFrame | None) -> dict:
    visuals = {'charts': [], 'table_preview': []}
    if df is None or df.empty:
        return visuals
    temp = df.copy()
    date_col = 'ds' if 'ds' in temp.columns else None
    value_col = 'y' if 'y' in temp.columns else None
    if not date_col or not value_col:
        return visuals
    temp[date_col] = pd.to_datetime(first_series(temp, date_col), errors='coerce')
    temp[value_col] = pd.to_numeric(first_series(temp, value_col), errors='coerce')
    temp = temp.dropna(subset=[date_col]).copy()
    temp = temp.assign(ds=temp[date_col].dt.normalize())
    agg_map = {'y': 'sum', **{c: 'sum' for c in ['leads', 'cost'] if c in temp.columns}}
    daily = temp.groupby('ds', as_index=False).agg(agg_map)
    daily['ds'] = pd.to_datetime(first_series(daily, 'ds'), errors='coerce')
    daily = daily.dropna(subset=['ds']).sort_values('ds').reset_index(drop=True)
    if 'roas' not in daily.columns and {'y', 'cost'}.issubset(daily.columns):
        daily['roas'] = daily['y'] / daily['cost'].replace(0, pd.NA)
    visuals['table_preview'] = _json_ready_records(daily.tail(30))

    visuals['charts'].append(_plot_card('quality-revenue-trend', 'Revenue Trend', {
        'data': [
            {'type': 'scatter', 'mode': 'lines', 'name': 'Revenue', 'x': daily['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': daily['y'].round(2).tolist()}
        ],
        'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 40, 'r': 20, 't': 30, 'b': 40}}
    }, 'Raw daily revenue after merging mapped datasets.'))

    combo_data = [
        {'type': 'scatter', 'mode': 'lines', 'name': 'Revenue', 'x': daily['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': daily['y'].round(2).tolist(), 'yaxis': 'y'}
    ]
    if 'leads' in daily.columns:
        combo_data.append({'type': 'bar', 'name': 'Leads', 'x': daily['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': daily['leads'].fillna(0).round(2).tolist(), 'yaxis': 'y2', 'opacity': 0.45})
    if 'cost' in daily.columns:
        combo_data.append({'type': 'scatter', 'mode': 'lines', 'name': 'Cost', 'x': daily['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': daily['cost'].fillna(0).round(2).tolist(), 'yaxis': 'y2'})
    visuals['charts'].append(_plot_card('quality-driver-trend', 'Revenue vs Leads / Cost', {
        'data': combo_data,
        'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 40, 'r': 40, 't': 30, 'b': 40}, 'yaxis2': {'overlaying': 'y', 'side': 'right'}}
    }, 'Use this to spot whether driver volume and spend move with revenue.'))

    dow = daily.copy()
    dow['dow'] = pd.to_datetime(dow['ds']).dt.day_name()
    dow['dow_num'] = pd.to_datetime(dow['ds']).dt.dayofweek
    dow_avg = dow.groupby(['dow_num', 'dow'], as_index=False)['y'].mean().sort_values('dow_num')
    visuals['charts'].append(_plot_card('quality-dow', 'Average Revenue by Day of Week', {
        'data': [{'type': 'bar', 'x': dow_avg['dow'].tolist(), 'y': dow_avg['y'].round(2).tolist(), 'name': 'Avg Revenue'}],
        'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 40, 'r': 20, 't': 30, 'b': 40}}
    }, 'This helps reveal weekly seasonality before forecasting.'))

    month_avg = daily.copy()
    month_avg['month'] = pd.to_datetime(month_avg['ds']).dt.month_name().str.slice(0, 3)
    month_avg['month_num'] = pd.to_datetime(month_avg['ds']).dt.month
    month_avg = month_avg.groupby(['month_num', 'month'], as_index=False)['y'].mean().sort_values('month_num')
    visuals['charts'].append(_plot_card('quality-month', 'Average Revenue by Month', {
        'data': [{'type': 'bar', 'x': month_avg['month'].tolist(), 'y': month_avg['y'].round(2).tolist(), 'name': 'Avg Revenue'}],
        'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 40, 'r': 20, 't': 30, 'b': 40}}
    }, 'This helps reveal annual seasonality and peak months.'))
    return visuals





def _parse_cohort_month_series(series: pd.Series) -> pd.Series:
    raw = first_series(pd.DataFrame({"v": series}), 'v')
    try:
        parsed = normalize_date_series(raw)
    except Exception:
        parsed = pd.to_datetime(raw, errors='coerce')
    if parsed.notna().any():
        return parsed.dt.to_period('M').dt.to_timestamp()
    cleaned = raw.astype(str).str.strip().str.replace('/', '-', regex=False)
    for fmt in ('%Y-%b', '%b-%Y', '%Y-%m', '%Y-%m-%d'):
        parsed = pd.to_datetime(cleaned, format=fmt, errors='coerce')
        if parsed.notna().any():
            return parsed.dt.to_period('M').dt.to_timestamp()
    return pd.to_datetime(cleaned, errors='coerce').dt.to_period('M').dt.to_timestamp()



def _extract_cohort_longframe(upload_meta: dict | None, mapping: dict | None) -> tuple[pd.DataFrame | None, dict]:
    cohort_cfg = (mapping or {}).get('cohort_revenue') or {}
    cohort_df = None
    if upload_meta and cohort_cfg.get('file_key'):
        cohort_df, _, _ = _resolve_dataset(upload_meta, cohort_cfg.get('file_key'), cohort_cfg.get('sheet_name'))

    if cohort_df is None or cohort_df.empty or not cohort_cfg.get('lead_month_column') or not cohort_cfg.get('transaction_month_column') or not cohort_cfg.get('value_column'):
        rev_cfg = (mapping or {}).get('revenue') or {}
        if upload_meta and rev_cfg.get('file_key'):
            rev_df, _, _ = _resolve_dataset(upload_meta, rev_cfg.get('file_key'), rev_cfg.get('sheet_name'))
            if rev_df is not None and not rev_df.empty:
                cols_lower = {str(c).strip().lower(): c for c in rev_df.columns}
                def pick(options):
                    for opt in options:
                        if opt in cols_lower:
                            return cols_lower[opt]
                    return None
                auto_lead = pick(['lead month','lead_month','cohort month','cohort_month'])
                auto_txn = pick(['transaction month','transaction_month','txn month','txn_month','revenue month','revenue_month'])
                auto_val = rev_cfg.get('target_column') if rev_cfg.get('target_column') in rev_df.columns else pick(['revenue','value','amount','sales'])
                auto_cat = rev_cfg.get('category_column') if rev_cfg.get('category_column') in rev_df.columns else pick(['platform','source','category','affiliate','channel'])
                if auto_lead and auto_txn and auto_val:
                    cohort_cfg = {
                        'file_key': rev_cfg.get('file_key'),
                        'sheet_name': rev_cfg.get('sheet_name'),
                        'lead_month_column': auto_lead,
                        'transaction_month_column': auto_txn,
                        'value_column': auto_val,
                        'category_column': auto_cat,
                    }
                    cohort_df = rev_df.copy()

    if cohort_df is None or cohort_df.empty:
        return None, cohort_cfg

    lead_col = cohort_cfg.get('lead_month_column')
    txn_col = cohort_cfg.get('transaction_month_column')
    value_col = cohort_cfg.get('value_column')
    category_col = cohort_cfg.get('category_column') or None
    dim_cols = [c for c, _ in _available_dimension_columns(cohort_df, cohort_cfg, include_category=True)]
    if not lead_col or not txn_col or not value_col or not {lead_col, txn_col, value_col}.issubset(cohort_df.columns):
        return None, cohort_cfg

    temp = cohort_df[[c for c in [lead_col, txn_col, value_col, *dim_cols] if c and c in cohort_df.columns]].copy()
    temp['lead_month'] = _parse_cohort_month_series(temp[lead_col])
    temp['transaction_month'] = _parse_cohort_month_series(temp[txn_col])
    temp['revenue'] = pd.to_numeric(first_series(temp, value_col), errors='coerce').fillna(0)
    temp = temp.dropna(subset=['lead_month', 'transaction_month'])
    if temp.empty:
        return None, cohort_cfg
    temp = temp[temp['transaction_month'] >= temp['lead_month']].copy()
    if temp.empty:
        return None, cohort_cfg
    if dim_cols:
        temp['cohort_category'] = _build_dimension_label_series(temp, cohort_cfg, fallback_col=category_col, default='All Sources')
    else:
        temp['cohort_category'] = 'All Sources'
    temp['cohort_age_month'] = ((temp['transaction_month'].dt.year - temp['lead_month'].dt.year) * 12 + (temp['transaction_month'].dt.month - temp['lead_month'].dt.month)).astype(int)
    return temp, cohort_cfg


def _build_cohort_revenue_visuals(upload_meta: dict | None, mapping: dict | None, base_df: pd.DataFrame | None = None) -> dict:
    out = {
        'available': False,
        'summary': {},
        'table': [],
        'platform_table': [],
        'table_grain_label': 'Platform',
        'curve_chart': None,
        'platform_chart': None,
        'heatmap_chart': None,
        'payback_chart': None,
        'payback_table': [],
        'narrative': [],
        'roas_table': [],
        'payback_curve': [],
        'payback_summary': [],
    }
    if not upload_meta or not mapping:
        return out
    temp, cohort_cfg = _extract_cohort_longframe(upload_meta, mapping)
    if temp is None or temp.empty:
        return out

    def _project_months_to_target(
        monthly_revenue: list[float],
        cohort_cost: float | None,
        current_cumulative_revenue: float | None,
        target_roas: float = 1.0,
        max_projection_months: int = 36,
    ) -> float | None:
        try:
            cost_value = float(cohort_cost or 0.0)
            cumulative_value = float(current_cumulative_revenue or 0.0)
        except Exception:
            return None
        if cost_value <= 0:
            return None
        target_revenue = cost_value * float(target_roas)
        if cumulative_value >= target_revenue:
            return 0.0
        revenue_series = [float(v) for v in monthly_revenue if pd.notna(v) and float(v) >= 0]
        if not revenue_series:
            return None
        recent_positive = [v for v in revenue_series if v > 0]
        if not recent_positive:
            return None
        if len(recent_positive) >= 2:
            ratios = []
            for prev, curr in zip(recent_positive[:-1], recent_positive[1:]):
                if prev > 0:
                    ratios.append(curr / prev)
            if ratios:
                decay = float(np.clip(np.median(ratios), 0.55, 0.98))
            else:
                decay = 0.88
        else:
            decay = 0.9
        next_month_revenue = max(recent_positive[-1], 0.0)
        if next_month_revenue <= 0:
            return None
        projected_cumulative = cumulative_value
        for month_ahead in range(1, max_projection_months + 1):
            next_month_revenue = next_month_revenue * decay
            projected_cumulative += next_month_revenue
            if projected_cumulative >= target_revenue:
                return float(month_ahead)
        return None

    category_col = cohort_cfg.get('category_column') or None
    temp = temp.copy()
    temp['cost_match_key'] = 'All Sources'

    # Build cohort spend from the mapped cost dataset first. This is the cleanest source
    # for month-level ROAS because it preserves the original spend grain instead of relying
    # on whichever helper rows survived the modeled history merge.
    cost_rollup = pd.DataFrame()
    cost_month_rollup = pd.DataFrame()
    cost_cfg = (mapping or {}).get('cost') or {}
    shared_cost_keys = []
    seen_shared_columns: set[str] = set()
    for key in ['platform_column', 'category_column', 'affiliate_id_column', 'campaign_id_column', 'ad_id_column']:
        cohort_col = cohort_cfg.get(key)
        cost_col = cost_cfg.get(key)
        if not cohort_col or not cost_col:
            continue
        shared_signature = f'{cohort_col}::{cost_col}'
        if shared_signature in seen_shared_columns:
            continue
        seen_shared_columns.add(shared_signature)
        shared_cost_keys.append(key)
    if shared_cost_keys:
        label_map = {
            'platform_column': 'Platform',
            'category_column': 'Group',
            'affiliate_id_column': 'Affiliate',
            'campaign_id_column': 'Campaign',
            'ad_id_column': 'Ad',
        }
        out['table_grain_label'] = ' + '.join(label_map[key] for key in shared_cost_keys)
    cohort_cost_match_cfg = {key: cohort_cfg.get(key) for key in shared_cost_keys}
    cost_match_cfg = {key: cost_cfg.get(key) for key in shared_cost_keys}
    if cohort_cost_match_cfg:
        temp['cost_match_key'] = _build_dimension_label_series(
            temp,
            cohort_cost_match_cfg,
            fallback_col=cohort_cost_match_cfg.get('category_column'),
            default='All Sources',
        )
    if upload_meta and cost_cfg.get('file_key') and cost_cfg.get('date_column') and cost_cfg.get('value_column'):
        cost_df, _, _ = _resolve_dataset(upload_meta, cost_cfg.get('file_key'), cost_cfg.get('sheet_name'))
        if cost_df is not None and not cost_df.empty:
            cost_date_col = cost_cfg.get('date_column')
            cost_value_col = cost_cfg.get('value_column')
            cost_dim_cols = [c for c, _ in _available_dimension_columns(cost_df, cost_cfg, include_category=True)]
            keep_cols = [c for c in [cost_date_col, cost_value_col, *cost_dim_cols] if c and c in cost_df.columns]
            if cost_date_col in keep_cols and cost_value_col in keep_cols:
                cost_df = cost_df[keep_cols].copy()
                cost_df['lead_month'] = pd.to_datetime(first_series(cost_df, cost_date_col), errors='coerce').dt.to_period('M').dt.to_timestamp()
                cost_df['cost'] = pd.to_numeric(first_series(cost_df, cost_value_col), errors='coerce').fillna(0)
                cost_df = cost_df.dropna(subset=['lead_month'])
                if not cost_df.empty:
                    cost_month_rollup = cost_df.groupby(['lead_month'], as_index=False)['cost'].sum().rename(columns={'cost': 'month_cost'})
                    if cost_match_cfg:
                        cost_df['cost_match_key'] = _build_dimension_label_series(
                            cost_df,
                            cost_match_cfg,
                            fallback_col=cost_match_cfg.get('category_column'),
                            default='All Sources',
                        )
                        cost_rollup = cost_df.groupby(['lead_month', 'cost_match_key'], as_index=False)['cost'].sum()
    if cost_month_rollup.empty and base_df is not None and not base_df.empty and 'cost' in base_df.columns and 'ds' in base_df.columns:
        cost_df = base_df.copy()
        cost_df['lead_month'] = pd.to_datetime(first_series(cost_df, 'ds'), errors='coerce').dt.to_period('M').dt.to_timestamp()
        cost_df['cost'] = pd.to_numeric(first_series(cost_df, 'cost'), errors='coerce').fillna(0)
        cost_month_rollup = cost_df.groupby(['lead_month'], as_index=False)['cost'].sum().rename(columns={'cost': 'month_cost'})
        if shared_cost_keys:
            hist_match_cfg = {}
            for key in shared_cost_keys:
                if key == 'platform_column' and 'platform' in cost_df.columns:
                    hist_match_cfg[key] = 'platform'
                elif key == 'category_column' and 'category' in cost_df.columns:
                    hist_match_cfg[key] = 'category'
                elif mapping.get(key) and mapping.get(key) in cost_df.columns:
                    hist_match_cfg[key] = mapping.get(key)
            if hist_match_cfg:
                cost_df['cost_match_key'] = _build_dimension_label_series(
                    cost_df,
                    hist_match_cfg,
                    fallback_col=hist_match_cfg.get('category_column'),
                    default='All Sources',
                )
                cost_rollup = cost_df.groupby(['lead_month', 'cost_match_key'], as_index=False)['cost'].sum()

    cohort_grp = temp.groupby(['lead_month', 'cohort_category', 'cost_match_key'], as_index=False).agg(
        total_revenue=('revenue', 'sum'),
        active_months=('transaction_month', 'nunique'),
        first_txn_month=('transaction_month', 'min'),
        last_txn_month=('transaction_month', 'max'),
        first_month_revenue=('revenue', lambda s: float(s.iloc[0]) if len(s) else 0.0),
    )
    cohort_grp['cost'] = np.nan
    if not cost_rollup.empty:
        cohort_grp = cohort_grp.merge(cost_rollup.rename(columns={'cost': 'matched_bucket_cost'}), on=['lead_month', 'cost_match_key'], how='left')
        bucket_revenue_total = cohort_grp.groupby(['lead_month', 'cost_match_key'], dropna=False)['total_revenue'].transform('sum')
        bucket_share = np.where(bucket_revenue_total > 0, cohort_grp['total_revenue'] / bucket_revenue_total, np.nan)
        cohort_grp['cost'] = np.where(
            pd.notna(cohort_grp.get('matched_bucket_cost')),
            pd.to_numeric(cohort_grp['matched_bucket_cost'], errors='coerce') * bucket_share,
            np.nan,
        )
    if not cost_month_rollup.empty:
        cohort_grp = cohort_grp.merge(cost_month_rollup, on=['lead_month'], how='left')
        cohort_grp['cost'] = pd.to_numeric(cohort_grp['cost'], errors='coerce')
        matched_month_cost = cohort_grp.groupby('lead_month', dropna=False)['cost'].transform(lambda s: pd.to_numeric(s, errors='coerce').fillna(0).sum())
        remaining_month_cost = (pd.to_numeric(cohort_grp['month_cost'], errors='coerce').fillna(0) - matched_month_cost).clip(lower=0)
        unmatched_revenue = cohort_grp['total_revenue'].where(cohort_grp['cost'].isna(), 0.0)
        unmatched_revenue_total = unmatched_revenue.groupby(cohort_grp['lead_month'], dropna=False).transform('sum')
        unmatched_share = np.where(
            cohort_grp['cost'].isna() & (unmatched_revenue_total > 0),
            cohort_grp['total_revenue'] / unmatched_revenue_total,
            np.nan,
        )
        fallback_cost = np.where(pd.notna(unmatched_share), remaining_month_cost * unmatched_share, np.nan)
        cohort_grp['cost'] = cohort_grp['cost'].where(cohort_grp['cost'].notna(), fallback_cost)
        cohort_grp = cohort_grp.drop(columns=['month_cost'], errors='ignore')
    cohort_grp = cohort_grp.drop(columns=['matched_bucket_cost'], errors='ignore')
    cohort_grp['roas'] = cohort_grp['total_revenue'] / pd.to_numeric(cohort_grp['cost'], errors='coerce').replace(0, pd.NA)
    cohort_grp['revenue_share'] = cohort_grp['total_revenue'] / max(float(cohort_grp['total_revenue'].sum() or 0.0), 1.0)
    cohort_grp['payback_share_m0'] = cohort_grp['first_month_revenue'] / pd.to_numeric(cohort_grp['cost'], errors='coerce').replace(0, pd.NA)

    has_platform_split = bool(category_col)
    if cohort_grp['roas'].notna().any():
        roas_median = float(cohort_grp['roas'].dropna().median())
        def _classify(row):
            roas = row.get('roas')
            share = row.get('revenue_share', 0)
            if has_platform_split:
                if pd.notna(roas):
                    if roas >= max(1.25, roas_median):
                        return 'Scale Platform'
                    if roas < 0.9:
                        return 'Cap / Rework Platform'
                    return 'Watch Platform'
                if share >= 0.08:
                    return 'Verify Recent ROAS'
                if share < 0.03:
                    return 'Cap / Rework Platform'
                return 'Watch Platform'
            if pd.notna(roas):
                if roas >= max(1.25, roas_median):
                    return 'Strong Cohort - replicate newer traffic mix'
                if roas < 0.9:
                    return 'Weak Cohort - investigate source mix'
                return 'Mixed Cohort - monitor payback'
            if share >= 0.08:
                return 'High-value Cohort - study pattern'
            if share < 0.03:
                return 'Low-value Cohort - investigate'
            return 'Monitor Cohort'
    else:
        def _classify(row):
            share = row.get('revenue_share', 0)
            if has_platform_split:
                if share >= 0.08:
                    return 'Verify Recent ROAS'
                if share < 0.03:
                    return 'Cap / Rework Platform'
                return 'Watch Platform'
            if share >= 0.08:
                return 'High-value Cohort - study pattern'
            if share < 0.03:
                return 'Low-value Cohort - investigate'
            return 'Monitor Cohort'
    cohort_grp['action'] = cohort_grp.apply(_classify, axis=1)
    cohort_grp = cohort_grp.sort_values(['lead_month', 'total_revenue', 'roas'], ascending=[False, False, False], na_position='last').reset_index(drop=True)

    out['available'] = True
    out['summary'] = {
        'cohort_count': int(len(cohort_grp)),
        'lead_months': int(cohort_grp['lead_month'].nunique()),
        'sources': int(cohort_grp['cohort_category'].nunique()),
        'cohort_revenue_total': _safe_number(cohort_grp['total_revenue'].sum()),
        'cohort_cost_total': _safe_number(pd.to_numeric(cohort_grp['cost'], errors='coerce').sum()) if cohort_grp['cost'].notna().any() else None,
        'cohort_roas_total': _safe_number(cohort_grp['total_revenue'].sum() / max(float(pd.to_numeric(cohort_grp['cost'], errors='coerce').sum() or 0.0), 1.0)) if cohort_grp['cost'].notna().any() else None,
    }
    table_group_col = 'cost_match_key' if shared_cost_keys else 'cohort_category'
    show = cohort_grp.groupby(['lead_month', table_group_col], as_index=False).agg(
        total_revenue=('total_revenue', 'sum'),
        cost=('cost', 'sum'),
        roas=('roas', lambda s: float(pd.to_numeric(s, errors='coerce').dropna().mean()) if pd.to_numeric(s, errors='coerce').notna().any() else np.nan),
        active_months=('active_months', 'max'),
        revenue_share=('total_revenue', lambda s: float(pd.to_numeric(s, errors='coerce').sum() / max(float(cohort_grp['total_revenue'].sum() or 0.0), 1.0))),
    )
    show = show.rename(columns={table_group_col: 'cohort_category'})
    show = show.sort_values(['lead_month', 'total_revenue', 'roas'], ascending=[False, False, False], na_position='last')
    show['lead_month'] = show['lead_month'].dt.strftime('%Y-%b')
    out['table'] = _json_ready_records(show[['lead_month', 'cohort_category', 'total_revenue', 'cost', 'roas', 'active_months', 'revenue_share']].round(4))

    recent_lead_months = sorted(cohort_grp['lead_month'].dropna().unique())[-12:]
    recent_cohort_grp = cohort_grp[cohort_grp['lead_month'].isin(recent_lead_months)].copy() if len(recent_lead_months) else cohort_grp.copy()

    platform_group_col = table_group_col
    platform_grp = cohort_grp.groupby(platform_group_col, as_index=False).agg(
        total_revenue=('total_revenue', 'sum'),
        total_cost=('cost', 'sum'),
        cohorts=('lead_month', 'nunique'),
    )
    platform_grp['roas'] = platform_grp['total_revenue'] / pd.to_numeric(platform_grp['total_cost'], errors='coerce').replace(0, pd.NA)
    platform_grp['avg_revenue_per_cohort'] = platform_grp['total_revenue'] / platform_grp['cohorts'].replace(0, pd.NA)
    recent_platform_grp = recent_cohort_grp.groupby(platform_group_col, as_index=False).agg(
        recent_revenue=('total_revenue', 'sum'),
        recent_cost=('cost', 'sum'),
        recent_cohorts=('lead_month', 'nunique'),
    ) if not recent_cohort_grp.empty else pd.DataFrame(columns=[platform_group_col,'recent_revenue','recent_cost','recent_cohorts'])
    if not recent_platform_grp.empty:
        recent_platform_grp['recent_roas'] = recent_platform_grp['recent_revenue'] / pd.to_numeric(recent_platform_grp['recent_cost'], errors='coerce').replace(0, pd.NA)
    platform_grp = platform_grp.merge(recent_platform_grp, on=platform_group_col, how='left')
    platform_median_revenue = float(platform_grp['total_revenue'].median()) if not platform_grp.empty else 0.0
    recent_platform_median_revenue = float(pd.to_numeric(platform_grp.get('recent_revenue'), errors='coerce').dropna().median()) if not platform_grp.empty and 'recent_revenue' in platform_grp.columns and pd.to_numeric(platform_grp.get('recent_revenue'), errors='coerce').notna().any() else 0.0
    def _platform_action(r):
        recent_roas = r.get('recent_roas')
        recent_revenue = _safe_number(r.get('recent_revenue')) or 0.0
        if pd.notna(recent_roas):
            if recent_roas >= 1.2 and recent_revenue >= recent_platform_median_revenue:
                return 'Scale'
            if recent_roas < 0.9:
                return 'Remove / Rework'
            return 'Watch'
        if recent_revenue > 0:
            return 'Verify Recent ROAS'
        if r['total_revenue'] >= platform_median_revenue:
            return 'Historical Only - Verify'
        return 'Watch'
    platform_grp['action'] = platform_grp.apply(_platform_action, axis=1)
    platform_grp = platform_grp.sort_values(['recent_revenue', 'total_revenue', 'roas'], ascending=[False, False, False], na_position='last')
    platform_grp = platform_grp.rename(columns={platform_group_col: 'cohort_category'})
    out['platform_table'] = _json_ready_records(platform_grp.round(4))

    cohort_month_rollup = cohort_grp.groupby('lead_month', as_index=False).agg(
        total_revenue=('total_revenue', 'sum'),
        total_cost=('cost', 'sum'),
        cohorts=('cohort_category', 'nunique'),
        active_months=('active_months', 'max'),
    )
    cohort_month_rollup['roas'] = cohort_month_rollup['total_revenue'] / pd.to_numeric(cohort_month_rollup['total_cost'], errors='coerce').replace(0, pd.NA)
    cohort_month_rollup['roas_pct'] = pd.to_numeric(cohort_month_rollup['roas'], errors='coerce') * 100.0
    # ROAS can keep climbing above break-even; payback_pct is the share of
    # acquisition cost recovered toward the 100% break-even threshold.
    cohort_month_rollup['payback_pct'] = (pd.to_numeric(cohort_month_rollup['roas'], errors='coerce').clip(lower=0, upper=1) * 100.0)
    cohort_month_rollup = cohort_month_rollup.sort_values('lead_month').reset_index(drop=True)
    cohort_month_rollup['time_to_100_roas_months'] = pd.NA
    cohort_month_rollup['projected_time_to_100_roas_months'] = pd.NA
    cohort_month_rollup['maturity_status'] = cohort_month_rollup['active_months'].apply(
        lambda x: 'Mature' if pd.notna(x) and x >= 6 else ('Growing' if pd.notna(x) and x >= 3 else 'Early')
    )
    cohort_month_rollup['maturity_weight'] = cohort_month_rollup['active_months'].apply(
        lambda x: 1.0 if pd.notna(x) and x >= 6 else (0.75 if pd.notna(x) and x >= 3 else 0.5)
    )
    cohort_month_rollup['maturity_adjusted_roas'] = cohort_month_rollup['roas'] * cohort_month_rollup['maturity_weight']
    cohort_month_show = cohort_month_rollup.sort_values('lead_month', ascending=False).reset_index(drop=True).copy()
    cohort_month_show['lead_month'] = pd.to_datetime(cohort_month_show['lead_month']).dt.strftime('%Y-%b')
    payback_group_col = 'cost_match_key' if shared_cost_keys else 'cohort_category'
    payback = temp.groupby(['lead_month', payback_group_col, 'cohort_age_month'], as_index=False)['revenue'].sum()
    payback = payback.rename(columns={payback_group_col: 'cohort_category'})
    payback_curve = cohort_month_rollup[['lead_month', 'payback_pct', 'active_months']].copy()
    payback_curve = payback_curve.sort_values('lead_month')
    payback_curve['lead_month'] = payback_curve['lead_month'].dt.strftime('%Y-%b')
    out['payback_curve'] = _json_ready_records(payback_curve.round(4))


    if not payback.empty:
        payback = payback.sort_values(['lead_month', 'cohort_category', 'cohort_age_month']).copy()
        payback['cumulative_revenue'] = payback.groupby(['lead_month', 'cohort_category'])['revenue'].cumsum()
        payback_cost = cohort_grp.groupby(['lead_month', payback_group_col], as_index=False)['cost'].sum()
        payback_cost = payback_cost.rename(columns={payback_group_col: 'cohort_category', 'cost': 'cohort_cost'})
        payback = payback.merge(payback_cost, on=['lead_month', 'cohort_category'], how='left')
        payback['cumulative_roas'] = payback['cumulative_revenue'] / pd.to_numeric(payback['cohort_cost'], errors='coerce').replace(0, pd.NA)
        payback_month_rollup = payback.groupby(['lead_month', 'cohort_age_month'], as_index=False).agg(
            cumulative_revenue=('cumulative_revenue', 'sum'),
            cohort_cost=('cohort_cost', 'sum'),
        )
        payback_month_rollup['cumulative_roas'] = payback_month_rollup['cumulative_revenue'] / pd.to_numeric(payback_month_rollup['cohort_cost'], errors='coerce').replace(0, pd.NA)
        time_to_100 = (
            payback_month_rollup[payback_month_rollup['cumulative_roas'] >= 1]
            .sort_values(['lead_month', 'cohort_age_month'])
            .groupby('lead_month', as_index=False)
            .first()[['lead_month', 'cohort_age_month']]
            .rename(columns={'cohort_age_month': 'time_to_100_roas_months'})
        )
        if not time_to_100.empty:
            cohort_month_rollup = cohort_month_rollup.drop(columns=['time_to_100_roas_months'], errors='ignore').merge(time_to_100, on='lead_month', how='left')
            cohort_month_show = cohort_month_show.drop(columns=['time_to_100_roas_months'], errors='ignore')
            cohort_month_show = cohort_month_show.merge(
                time_to_100.assign(lead_month=pd.to_datetime(time_to_100['lead_month']).dt.strftime('%Y-%b')),
                on='lead_month',
                how='left',
            )
        lead_month_projection_rows = []
        for lead_month, grp in payback_month_rollup.sort_values(['lead_month', 'cohort_age_month']).groupby('lead_month'):
            current_cumulative = pd.to_numeric(grp['cumulative_revenue'], errors='coerce').dropna()
            cohort_cost_series = pd.to_numeric(grp['cohort_cost'], errors='coerce').dropna()
            monthly_increments = pd.to_numeric(grp['cumulative_revenue'], errors='coerce').diff().fillna(pd.to_numeric(grp['cumulative_revenue'], errors='coerce')).tolist()
            projected_months = _project_months_to_target(
                monthly_revenue=monthly_increments,
                cohort_cost=float(cohort_cost_series.iloc[-1]) if not cohort_cost_series.empty else None,
                current_cumulative_revenue=float(current_cumulative.iloc[-1]) if not current_cumulative.empty else None,
                target_roas=1.0,
            )
            lead_month_projection_rows.append({
                'lead_month': lead_month,
                'projected_time_to_100_roas_months': projected_months,
            })
        projection_df = pd.DataFrame(lead_month_projection_rows)
        if not projection_df.empty:
            cohort_month_rollup = cohort_month_rollup.drop(columns=['projected_time_to_100_roas_months'], errors='ignore').merge(projection_df, on='lead_month', how='left')
            cohort_month_show = cohort_month_show.drop(columns=['projected_time_to_100_roas_months'], errors='ignore')
            cohort_month_show = cohort_month_show.merge(
                projection_df.assign(lead_month=pd.to_datetime(projection_df['lead_month']).dt.strftime('%Y-%b')),
                on='lead_month',
                how='left',
            )
        payback_latest = payback.sort_values('cohort_age_month').groupby(['lead_month', 'cohort_category'], as_index=False).tail(1)
        payback_latest = payback_latest.sort_values(['lead_month', 'cumulative_roas', 'cumulative_revenue'], ascending=[False, False, False], na_position='last')
        payback_show = payback_latest.copy()
        if payback_group_col == 'cost_match_key' and payback_show['cohort_category'].nunique(dropna=True) <= 1:
            payback_show['cohort_category'] = 'Company total'
        payback_show['lead_month'] = pd.to_datetime(payback_show['lead_month']).dt.strftime('%Y-%b')
        out['payback_table'] = _json_ready_records(payback_show[['lead_month', 'cohort_category', 'cohort_age_month', 'cumulative_revenue', 'cohort_cost', 'cumulative_roas']].round(4))
        if payback['cumulative_roas'].notna().any():
            latest_curve_months = sorted(payback_latest['lead_month'].dropna().unique())[-18:]
            top_curves = (payback_latest[payback_latest['lead_month'].isin(latest_curve_months)]
                          .sort_values(['lead_month', 'cumulative_revenue', 'cumulative_roas'], ascending=[False, False, False], na_position='last')
                          .groupby('lead_month', as_index=False)
                          .head(1)[['lead_month', 'cohort_category']]
                          .drop_duplicates())
            curve_df = payback.merge(top_curves, on=['lead_month', 'cohort_category'], how='inner')
            curve_df = curve_df[pd.to_numeric(curve_df['cohort_age_month'], errors='coerce') <= 18].copy()
            if payback_group_col == 'cost_match_key' and curve_df['cohort_category'].nunique(dropna=True) <= 1:
                curve_df['cohort_category'] = 'Company total'
            out['payback_chart'] = {
                'data': [
                    {
                        'type': 'scatter', 'mode': 'lines+markers',
                        'name': f"{pd.Timestamp(m).strftime('%Y-%b')} · {cat}",
                        'x': grp['cohort_age_month'].tolist(),
                        'y': pd.to_numeric(grp['cumulative_roas'], errors='coerce').round(3).tolist(),
                    }
                    for (m, cat), grp in curve_df.groupby(['lead_month', 'cohort_category'])
                ],
                'layout': {'paper_bgcolor':'transparent','plot_bgcolor':'transparent','font':{'color':'#edf1f7'},'margin':{'l':40,'r':20,'t':30,'b':40}, 'xaxis': {'title': 'Cohort age (months)', 'range': [0, 18]}, 'yaxis': {'title': 'Cumulative ROAS'}}
            }

    out['roas_table'] = _json_ready_records(
        cohort_month_show[['lead_month', 'total_revenue', 'total_cost', 'roas', 'roas_pct', 'payback_pct', 'maturity_adjusted_roas', 'maturity_status', 'time_to_100_roas_months', 'projected_time_to_100_roas_months', 'cohorts', 'active_months']]
        .round(4)
    )
    for row in out['roas_table']:
        if row.get('time_to_100_roas_months') in (None, '', 'n/a') or pd.isna(row.get('time_to_100_roas_months')):
            row['time_to_100_roas_months'] = 'Not reached'
        if row.get('projected_time_to_100_roas_months') in (None, '', 'n/a') or pd.isna(row.get('projected_time_to_100_roas_months')):
            row['projected_time_to_100_roas_months'] = 'Not projected'

    payback_summary = cohort_month_show[['lead_month', 'roas', 'roas_pct', 'payback_pct', 'maturity_adjusted_roas', 'maturity_status', 'active_months', 'time_to_100_roas_months', 'projected_time_to_100_roas_months']].copy()
    payback_summary['_lead_month_dt'] = pd.to_datetime(payback_summary['lead_month'], format='%Y-%b', errors='coerce')
    latest_12_lead_months = sorted(payback_summary['_lead_month_dt'].dropna().unique())[-12:]
    recent_payback_summary = payback_summary[payback_summary['_lead_month_dt'].isin(latest_12_lead_months)].copy()
    if recent_payback_summary.empty:
        recent_payback_summary = payback_summary.copy()
    fastest_payback = recent_payback_summary[recent_payback_summary['time_to_100_roas_months'].notna()].sort_values(['time_to_100_roas_months', 'payback_pct'], ascending=[True, False]).head(3)
    slowest_payback = recent_payback_summary.sort_values(['projected_time_to_100_roas_months', 'maturity_adjusted_roas', 'payback_pct'], ascending=[False, True, True], na_position='last').head(3)
    summary_records = []
    for _, row in fastest_payback.iterrows():
        summary_records.append({
            'bucket': 'Fastest Payback',
            'lead_month': row['lead_month'],
            'payback_pct': row['payback_pct'],
            'roas_pct': row.get('roas_pct'),
            'maturity_adjusted_roas': row['maturity_adjusted_roas'],
            'maturity_status': row['maturity_status'],
            'active_months': row['active_months'],
            'time_to_100_roas_months': row['time_to_100_roas_months'],
            'projected_time_to_100_roas_months': row.get('projected_time_to_100_roas_months'),
        })
    for _, row in slowest_payback.iterrows():
        summary_records.append({
            'bucket': 'Needs Attention',
            'lead_month': row['lead_month'],
            'payback_pct': row['payback_pct'],
            'roas_pct': row.get('roas_pct'),
            'maturity_adjusted_roas': row['maturity_adjusted_roas'],
            'maturity_status': row['maturity_status'],
            'active_months': row['active_months'],
            'time_to_100_roas_months': row['time_to_100_roas_months'],
            'projected_time_to_100_roas_months': row.get('projected_time_to_100_roas_months'),
        })
    out['payback_summary'] = _json_ready_records(pd.DataFrame(summary_records).round(4)) if summary_records else []

    # cohort curves
    curve = temp.groupby(['lead_month', 'cohort_age_month'], as_index=False)['revenue'].sum()
    if not curve.empty:
        curve['cumulative_revenue'] = curve.sort_values('cohort_age_month').groupby('lead_month')['revenue'].cumsum()
        top_months = sorted(cohort_grp['lead_month'].dropna().unique())[-18:]
        curve = curve[curve['lead_month'].isin(top_months)]
        out['curve_chart'] = {
            'data': [
                {
                    'type': 'scatter', 'mode': 'lines+markers', 'name': pd.Timestamp(m).strftime('%Y-%b'),
                    'x': grp['cohort_age_month'].tolist(),
                    'y': grp['cumulative_revenue'].round(2).tolist(),
                }
                for m, grp in curve.groupby('lead_month')
            ],
            'layout': {'paper_bgcolor':'transparent','plot_bgcolor':'transparent','font':{'color':'#edf1f7'},'margin':{'l':40,'r':20,'t':30,'b':40}, 'xaxis': {'title': 'Cohort age (months)'}, 'yaxis': {'title': 'Cumulative revenue'}}
        }

    top_platforms = platform_grp.sort_values(['recent_revenue', 'total_revenue'], ascending=[False, False], na_position='last').head(10)
    if out.get('roas_table'):
        out['narrative'].append('Each lead-month cohort now carries its own ROAS so you can compare payback by cohort instead of relying on one blended percentage.')
        out['narrative'].append('Cohort maturity status helps distinguish early cohorts from mature cohorts so shorter-running cohorts are not misread as underperformers.')
        out['narrative'].append('Build 028 adds maturity-adjusted performance so cohorts can be ranked more fairly even when they have not had the same amount of time to pay back.')
        out['narrative'].append('Projected months to 100% ROAS uses the recent cohort payback curve to extrapolate forward, so it is an estimate rather than a guaranteed date.')

    if not top_platforms.empty:
        revenue_col = 'recent_revenue' if top_platforms['recent_revenue'].notna().any() else 'total_revenue'
        roas_col = 'recent_roas' if 'recent_roas' in top_platforms.columns else 'roas'
        out['platform_chart'] = {
            'data': [{
                'type': 'bar',
                'x': top_platforms['cohort_category'].tolist(),
                'y': pd.to_numeric(top_platforms[revenue_col], errors='coerce').fillna(0).round(2).tolist(),
                'customdata': [[None if pd.isna(r) else round(float(r), 2)] for r in top_platforms[roas_col].tolist()],
                'hovertemplate': 'Source: %{x}<br>Revenue: $%{y:,.2f}<br>ROAS: %{customdata[0]:,.2f}x<extra></extra>'
            }],
            'layout': {'paper_bgcolor':'transparent','plot_bgcolor':'transparent','font':{'color':'#edf1f7'},'margin':{'l':40,'r':20,'t':30,'b':80}, 'xaxis': {'tickangle': -25}, 'yaxis': {'title': 'Recent cohort revenue (latest 12 lead months)'}}
        }

    # heatmap of lead month x transaction month
    hm = temp.groupby(['lead_month', 'transaction_month'], as_index=False)['revenue'].sum()
    if not hm.empty:
        lead_vals = sorted(hm['lead_month'].dropna().unique())[-24:]
        txn_vals = sorted(hm['transaction_month'].dropna().unique())[-24:]
        hm = hm[hm['lead_month'].isin(lead_vals) & hm['transaction_month'].isin(txn_vals)]
        pivot = hm.pivot(index='lead_month', columns='transaction_month', values='revenue').fillna(0)
        heatmap_height = int(min(560, max(320, 86 + (len(pivot.index) * 34))))
        out['heatmap_chart'] = {
            'data': [{
                'type': 'heatmap',
                'x': [pd.Timestamp(x).strftime('%Y-%b') for x in pivot.columns],
                'y': [pd.Timestamp(y).strftime('%Y-%b') for y in pivot.index],
                'z': pivot.round(2).values.tolist(),
                'xgap': 2,
                'ygap': 2,
                'zsmooth': False,
                'colorscale': [
                    [0.0, '#fbfdff'],
                    [0.18, '#e0f2fe'],
                    [0.38, '#93c5fd'],
                    [0.58, '#60a5fa'],
                    [0.78, '#2563eb'],
                    [1.0, '#1e3a8a'],
                ],
                'hovertemplate': 'Lead Month: %{y}<br>Txn Month: %{x}<br>Revenue: $%{z:,.2f}<extra></extra>',
                'colorbar': {
                    'title': {'text': 'Revenue'},
                    'tickprefix': '$',
                    'tickformat': ',.0f',
                    'len': 0.8,
                    'thickness': 18,
                    'outlinewidth': 0,
                    'y': 0.5,
                }
            }],
            'layout': {
                'paper_bgcolor': 'transparent',
                'plot_bgcolor': 'transparent',
                'font': {'color': '#edf1f7'},
                'height': heatmap_height,
                'margin': {'l': 84, 'r': 54, 't': 18, 'b': 44},
                'xaxis': {'tickangle': -18, 'automargin': True, 'side': 'bottom', 'type': 'category'},
                'yaxis': {'automargin': True, 'autorange': 'reversed', 'type': 'category'},
            }
        }

    if not cohort_grp.empty:
        narrative_base = recent_cohort_grp if not recent_cohort_grp.empty else cohort_grp
        best = narrative_base.sort_values(['total_revenue', 'roas'], ascending=[False, False], na_position='last').iloc[0]
        worst = narrative_base.sort_values(['roas', 'total_revenue'], ascending=[True, False], na_position='last').iloc[0]
        latest_month_txt = pd.Timestamp(max(recent_lead_months)).strftime('%Y-%b') if len(recent_lead_months) else pd.Timestamp(cohort_grp['lead_month'].max()).strftime('%Y-%b')
        out['narrative'] = [
            f"Recent focus window runs through {latest_month_txt}. Top recent cohort: {pd.Timestamp(best['lead_month']).strftime('%Y-%b')} · {best['cohort_category']} generated ${best['total_revenue']:,.0f}" + (f" at {best['roas']:.2f}x ROAS." if pd.notna(best.get('roas')) else '.'),
            f"Recent cohort to review: {pd.Timestamp(worst['lead_month']).strftime('%Y-%b')} · {worst['cohort_category']}" + (f" is only at {worst['roas']:.2f}x ROAS." if pd.notna(worst.get('roas')) else f" contributes just {worst['revenue_share']*100:.1f}% of recent cohort revenue."),
            'Use lead-month cohorts to understand payback patterns and use platform/source actions for actual scale, cap, or remove decisions going forward.',
        ]
    return out





def _build_waterfall_explanation(wf_rows: pd.DataFrame | None) -> dict:
    if wf_rows is None or wf_rows.empty or 'component' not in wf_rows.columns or 'impact' not in wf_rows.columns:
        return {}
    data = wf_rows.copy()
    anchors = data[data['component'].isin(['Starting revenue', 'Current revenue'])]
    deltas = data[~data['component'].isin(['Starting revenue', 'Current revenue'])].copy()
    if anchors.empty or deltas.empty:
        return {}
    start = float(pd.to_numeric(anchors.loc[anchors['component'] == 'Starting revenue', 'impact'], errors='coerce').dropna().iloc[0]) if (anchors['component'] == 'Starting revenue').any() else None
    end = float(pd.to_numeric(anchors.loc[anchors['component'] == 'Current revenue', 'impact'], errors='coerce').dropna().iloc[0]) if (anchors['component'] == 'Current revenue').any() else None
    deltas['impact'] = pd.to_numeric(deltas['impact'], errors='coerce')
    drag_row = deltas.sort_values('impact').iloc[0] if not deltas.empty else None
    lift_row = deltas.sort_values('impact', ascending=False).iloc[0] if not deltas.empty else None
    takeaway = 'Net revenue is roughly flat versus the baseline.'
    if start is not None and end is not None:
        diff = end - start
        if abs(diff) >= max(abs(start) * 0.01, 250):
            takeaway = 'Revenue is improving versus the baseline.' if diff > 0 else 'Revenue is slipping versus the baseline.'
    return {
        'start_vs_end': (f'Baseline revenue starts at ${start:,.0f} and ends at ${end:,.0f}.' if start is not None and end is not None else 'Baseline and current revenue are being compared across the same bridge.'),
        'main_drag': (f"{drag_row['component']} reduced revenue the most." if drag_row is not None and pd.notna(drag_row['impact']) and float(drag_row['impact']) < 0 else 'No major negative driver stood out.'),
        'main_lift': (f"{lift_row['component']} added the most upside." if lift_row is not None and pd.notna(lift_row['impact']) and float(lift_row['impact']) > 0 else 'No major positive driver stood out.'),
        'takeaway': takeaway,
    }


def _build_lag_alignment_chart(daily: pd.DataFrame | None, lag_days: int | None) -> dict | None:
    if daily is None or daily.empty or lag_days is None or lag_days < 0:
        return None
    if not {'ds', 'y', 'cost', 'leads'}.issubset(daily.columns):
        return None
    work = daily.copy()
    work['ds'] = pd.to_datetime(first_series(work, 'ds'), errors='coerce')
    work['y'] = pd.to_numeric(first_series(work, 'y'), errors='coerce')
    work['cost'] = pd.to_numeric(first_series(work, 'cost'), errors='coerce')
    work['leads'] = pd.to_numeric(first_series(work, 'leads'), errors='coerce')
    work = work.dropna(subset=['ds']).sort_values('ds').tail(120).copy()
    if work.empty:
        return None
    work['lagged_revenue'] = work['y'].shift(-int(lag_days))
    overlap = work.dropna(subset=['lagged_revenue']).copy()
    if overlap.empty:
        return None
    work = overlap.reset_index(drop=True)
    x_vals = work['ds'].dt.strftime('%Y-%m-%d').tolist()
    lag_marker_idx = max(0, min(len(work) - 1, len(work) - int(lag_days) - 1)) if len(work) else 0
    lag_marker_date = work['ds'].iloc[lag_marker_idx].strftime('%Y-%m-%d') if len(work) else None
    cost_vals = work['cost'].round(2).tolist()
    lead_vals = work['leads'].round(2).tolist()
    actual_revenue_vals = work['y'].round(2).tolist()
    revenue_vals = work['lagged_revenue'].round(2).astype(object).where(work['lagged_revenue'].notna(), None).tolist()
    return _plot_card('stats-lag-alignment', 'Cost, Leads & Lagged Revenue Alignment', {
        'data': [
            {
                'type': 'bar',
                'name': 'Leads',
                'x': x_vals,
                'y': lead_vals,
                'yaxis': 'y2',
                'marker': {'color': 'rgba(79, 169, 199, 0.38)', 'line': {'color': 'rgba(79, 169, 199, 0.12)', 'width': 1}},
                'hovertemplate': '%{x}<br>Leads: %{y:,.0f}<extra></extra>',
            },
            {
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Cost',
                'x': x_vals,
                'y': cost_vals,
                'line': {'color': '#2f6f7a', 'width': 2.5},
                'hovertemplate': '%{x}<br>Cost: $%{y:,.0f}<extra></extra>',
            },
            {
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Actual Revenue',
                'x': x_vals,
                'y': actual_revenue_vals,
                'line': {'color': 'rgba(95, 124, 255, 0.42)', 'width': 2},
                'hovertemplate': '%{x}<br>Actual revenue: $%{y:,.0f}<extra></extra>',
            },
            {
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Lagged Revenue',
                'x': x_vals,
                'y': revenue_vals,
                'line': {'color': '#5f7cff', 'width': 3.2, 'dash': 'dot'},
                'fill': 'tozeroy',
                'fillcolor': 'rgba(95, 124, 255, 0.08)',
                'connectgaps': False,
                'hovertemplate': '%{x}<br>Lagged revenue: $%{y:,.0f}<extra></extra>',
            },
        ],
        'layout': {
            'paper_bgcolor': 'transparent',
            'plot_bgcolor': 'transparent',
            'font': {'color': '#26313f', 'family': 'Inter, system-ui, sans-serif'},
            'margin': {'l': 64, 'r': 64, 't': 18, 'b': 58},
            'bargap': 0.18,
            'hovermode': 'x unified',
            'legend': {'orientation': 'h', 'x': 0, 'y': 1.12, 'xanchor': 'left', 'font': {'size': 12}},
            'xaxis': {
                'title': '',
                'showgrid': True,
                'gridcolor': 'rgba(31, 41, 55, 0.07)',
                'zeroline': False,
                'tickfont': {'color': '#536174'},
            },
            'yaxis': {
                'title': 'Dollars',
                'tickprefix': '$',
                'separatethousands': True,
                'showgrid': True,
                'gridcolor': 'rgba(31, 41, 55, 0.08)',
                'zerolinecolor': 'rgba(31, 41, 55, 0.16)',
                'tickfont': {'color': '#536174'},
            },
            'yaxis2': {
                'title': 'Leads',
                'overlaying': 'y',
                'side': 'right',
                'showgrid': False,
                'zeroline': False,
                'tickfont': {'color': '#536174'},
            },
            'shapes': ([{
                'type': 'line',
                'xref': 'x',
                'yref': 'paper',
                'x0': lag_marker_date,
                'x1': lag_marker_date,
                'y0': 0,
                'y1': 1,
                'line': {'color': 'rgba(95, 124, 255, 0.28)', 'width': 2, 'dash': 'dash'},
            }] if lag_marker_date else []),
            'annotations': ([{
                'xref': 'x',
                'yref': 'paper',
                'x': lag_marker_date,
                'y': 1.04,
                'text': f'{int(lag_days)}d lag',
                'showarrow': False,
                'font': {'color': '#5f7cff', 'size': 12},
                'bgcolor': 'rgba(95, 124, 255, 0.10)',
                'bordercolor': 'rgba(95, 124, 255, 0.25)',
                'borderpad': 4,
            }] if lag_marker_date else []),
        }
    }, f'Revenue is shifted back by {int(lag_days)} days so spend, lead volume, and the delayed revenue they likely produced can be compared on the same timeline. Actual revenue stays on the chart as a reference, and the view is trimmed to the valid overlap window where shifted revenue exists.')



def _build_leads_lagged_revenue_alignment_chart(daily: pd.DataFrame | None, lag_days: int | None) -> dict | None:
    if daily is None or daily.empty or lag_days is None or lag_days < 0:
        return None
    if not {'ds', 'y', 'leads'}.issubset(daily.columns):
        return None
    work = daily.copy()
    work['ds'] = pd.to_datetime(first_series(work, 'ds'), errors='coerce')
    work['y'] = pd.to_numeric(first_series(work, 'y'), errors='coerce')
    work['leads'] = pd.to_numeric(first_series(work, 'leads'), errors='coerce')
    work = work.dropna(subset=['ds']).sort_values('ds').tail(120).copy()
    if work.empty:
        return None
    work['lagged_revenue'] = work['y'].shift(-int(lag_days))
    work = work.dropna(subset=['lagged_revenue']).dropna(subset=['leads'], how='all').reset_index(drop=True)
    if work.empty:
        return None
    return _plot_card('stats-leads-lagged-revenue', 'Leads & Lagged Revenue Conversion Timing', {
        'data': [
            {'type': 'bar', 'name': 'Leads', 'x': work['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': work['leads'].round(2).tolist(), 'opacity': 0.42, 'yaxis': 'y2', 'hovertemplate': '%{x}<br>Leads: %{y:,.0f}<extra></extra>'},
            {'type': 'scatter', 'mode': 'lines', 'name': 'Lagged Revenue', 'x': work['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': work['lagged_revenue'].round(2).astype(object).where(work['lagged_revenue'].notna(), None).tolist(), 'line': {'width': 3}, 'hovertemplate': '%{x}<br>Lagged revenue: $%{y:,.0f}<extra></extra>'},
        ],
        'layout': {
            'paper_bgcolor':'transparent',
            'plot_bgcolor':'transparent',
            'font':{'color':'#edf1f7'},
            'margin':{'l':55,'r':70,'t':30,'b':60},
            'xaxis':{'title':'Date'},
            'yaxis':{'title':'Lagged revenue', 'tickprefix':'$', 'separatethousands': 1},
            'yaxis2':{'title':'Leads', 'overlaying':'y', 'side':'right', 'showgrid':False, 'zeroline':False},
            'barmode':'overlay',
        }
    }, f'Shows when leads are likely turning into revenue by shifting revenue back {int(lag_days)} days to the lead period that likely produced the purchase. The chart is trimmed to the valid overlap window.')


def _chart_render_state(chart: dict | None) -> dict:
    if not isinstance(chart, dict):
        return {'id': None, 'title': None, 'ok': False}
    try:
        spec = chart.get('spec')
        ok = bool(chart.get('id')) and bool(chart.get('title')) and bool(spec)
    except Exception:
        ok = False
    return {'id': chart.get('id'), 'title': chart.get('title'), 'ok': ok}


def _protect_results_visuals(visuals: dict | None) -> dict:
    visuals = _merge_results_visual_defaults(json_clean(visuals if isinstance(visuals, dict) else {}))
    chart_groups = {
        'main': visuals.get('charts') or [],
        'statistics': visuals.get('statistics_charts') or [],
        'spending_slowdown': visuals.get('spending_slowdown_charts') or [],
    }
    chart_registry = {}
    chart_errors = []
    for group_name, charts in chart_groups.items():
        protected = []
        for chart in charts:
            state = _chart_render_state(chart)
            if state['ok']:
                protected.append(chart)
                chart_registry[state['id']] = {'title': state['title'], 'group': group_name, 'ok': True}
            else:
                chart_errors.append({'group': group_name, 'title': chart.get('title') if isinstance(chart, dict) else 'unknown', 'reason': 'chart payload missing id/title/spec'})
        if group_name == 'main':
            visuals['charts'] = protected
        elif group_name == 'statistics':
            visuals['statistics_charts'] = protected
        elif group_name == 'spending_slowdown':
            visuals['spending_slowdown_charts'] = protected
    visuals['chart_registry'] = chart_registry
    visuals['chart_errors'] = chart_errors[:20]
    visuals['protected_results_layer'] = True
    visuals['results_contract_version'] = 'G056'
    visuals['charts_available'] = bool(visuals.get('charts') or visuals.get('statistics_charts') or visuals.get('spending_slowdown_charts'))
    if chart_errors and not visuals.get('render_warning'):
        visuals['render_warning'] = 'Some chart sections were skipped because their payloads were incomplete, but the rest of the gallery is protected and still renderable.'
    return visuals

def _build_statistics_visuals(hist: pd.DataFrame | None, hist_daily: pd.DataFrame | None, series_summary: pd.DataFrame | None = None) -> dict:
    out = {
        'available': False,
        'cards': {},
        'charts': [],
        'lag_table': [],
        'frontier_table': [],
        'attrition_table': [],
    }
    if hist_daily is None or hist_daily.empty or 'ds' not in hist_daily.columns or 'y' not in hist_daily.columns:
        return out

    daily = hist_daily.copy()
    daily['ds'] = pd.to_datetime(first_series(daily, 'ds'), errors='coerce')
    daily['y'] = pd.to_numeric(first_series(daily, 'y'), errors='coerce')
    daily = daily.dropna(subset=['ds', 'y']).sort_values('ds').reset_index(drop=True)
    if daily.empty:
        return out

    # Monthly rollup used by several statistics. Prefer the row-level modeling
    # table when available so allocated cost/leads keep the same grain as revenue.
    monthly_source = daily.copy()
    if hist is not None and not hist.empty and {'ds', 'y'}.issubset(hist.columns):
        try:
            hist_monthly_source = dedupe_columns_keep_first(hist.copy())
            hist_monthly_source['ds'] = pd.to_datetime(first_series(hist_monthly_source, 'ds'), errors='coerce')
            hist_monthly_source['y'] = pd.to_numeric(first_series(hist_monthly_source, 'y'), errors='coerce')
            for helper_col in ['cost', 'leads']:
                if helper_col in hist_monthly_source.columns:
                    hist_monthly_source[helper_col] = pd.to_numeric(first_series(hist_monthly_source, helper_col), errors='coerce')
            hist_monthly_source = hist_monthly_source.dropna(subset=['ds', 'y'])
            if not hist_monthly_source.empty:
                monthly_source = hist_monthly_source
        except Exception:
            monthly_source = daily.copy()

    monthly = monthly_source.copy()
    monthly['month'] = monthly['ds'].dt.to_period('M').dt.to_timestamp()
    agg = monthly.groupby('month', as_index=False).agg(revenue=('y', 'sum'))
    if 'cost' in monthly.columns:
        agg = agg.merge(monthly.groupby('month', as_index=False)['cost'].sum(), on='month', how='left')
    if 'leads' in monthly.columns:
        agg = agg.merge(monthly.groupby('month', as_index=False)['leads'].sum(), on='month', how='left')
    if 'cost' in agg.columns:
        agg['roas'] = pd.to_numeric(agg['revenue'], errors='coerce') / pd.to_numeric(agg['cost'], errors='coerce').replace(0, pd.NA)
    if 'leads' in agg.columns:
        agg['revenue_per_lead'] = pd.to_numeric(agg['revenue'], errors='coerce') / pd.to_numeric(agg['leads'], errors='coerce').replace(0, pd.NA)

    charts = []

    # Diminishing returns curve (monthly spend vs revenue)
    if {'cost', 'revenue'}.issubset(agg.columns):
        dr = agg[['month', 'cost', 'revenue']].copy()
        dr['cost'] = pd.to_numeric(dr['cost'], errors='coerce')
        dr['revenue'] = pd.to_numeric(dr['revenue'], errors='coerce')
        dr = dr.dropna().query('cost > 0 and revenue >= 0').sort_values('cost')
        if len(dr) >= 4:
            try:
                response_corr = float(dr['cost'].corr(dr['revenue']))
                coef = np.polyfit(np.log1p(dr['cost'].astype(float)), dr['revenue'].astype(float), 1)
                draw_fit = bool(np.isfinite(response_corr) and response_corr >= 0.2 and coef[0] > 0)
                traces = [
                    {
                        'type': 'scatter',
                        'mode': 'markers',
                        'x': dr['cost'].round(2).tolist(),
                        'y': dr['revenue'].round(2).tolist(),
                        'text': [pd.Timestamp(m).strftime('%Y-%b') for m in dr['month']],
                        'name': 'Observed months',
                        'hovertemplate': '%{text}<br>Cost: $%{x:,.0f}<br>Revenue: $%{y:,.0f}<extra></extra>',
                    }
                ]
                description = 'Plots monthly spend against revenue. The fitted response line is shown only when the observed relationship is positive enough to support a directional read.'
                if draw_fit:
                    xline = np.linspace(float(dr['cost'].min()), float(dr['cost'].max()), 40)
                    yline = np.maximum(coef[0] * np.log1p(xline) + coef[1], 0)
                    marginal = np.diff(yline) / np.diff(xline)
                    out['cards']['marginal_return_latest'] = float(marginal[-1]) if len(marginal) else None
                    traces.append({
                        'type': 'scatter',
                        'mode': 'lines',
                        'x': xline.round(2).tolist(),
                        'y': yline.round(2).tolist(),
                        'name': 'Fitted response',
                        'hovertemplate': 'Expected revenue: $%{y:,.0f}<extra></extra>',
                    })
                else:
                    out['cards']['marginal_return_latest'] = None
                charts.append(_plot_card('stats-diminishing-returns', 'Diminishing Returns Curve', {
                    'data': traces,
                    'layout': {'paper_bgcolor':'transparent','plot_bgcolor':'transparent','font':{'color':'#edf1f7'},'margin':{'l':55,'r':20,'t':30,'b':60},'xaxis':{'title':'Monthly cost / spend'},'yaxis':{'title':'Monthly revenue'}}
                }, description))
            except Exception:
                pass

    # Lag analysis (leads leading revenue)
    if 'leads' in daily.columns:
        lag_rows = []
        lead_series = pd.to_numeric(daily['leads'], errors='coerce')
        rev_series = pd.to_numeric(daily['y'], errors='coerce')
        for lag in range(0, 61, 5):
            shifted = lead_series.shift(lag)
            valid = pd.DataFrame({'x': shifted, 'y': rev_series}).dropna()
            if len(valid) < 12 or valid['x'].std(ddof=0) == 0 or valid['y'].std(ddof=0) == 0:
                continue
            corr = float(valid['x'].corr(valid['y']))
            lag_rows.append({'lag_days': lag, 'correlation': corr, 'abs_correlation': abs(corr)})
        if lag_rows:
            lag_df = pd.DataFrame(lag_rows).sort_values('lag_days')
            best = lag_df.sort_values('abs_correlation', ascending=False).iloc[0]
            out['cards']['best_revenue_lag_days'] = int(best['lag_days'])
            out['cards']['best_revenue_lag_corr'] = float(best['correlation'])
            out['lag_table'] = _json_ready_records(lag_df.round(4))
            charts.append(_plot_card('stats-lag-analysis', 'Lag Analysis', {
                'data': [{
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'x': lag_df['lag_days'].tolist(),
                    'y': lag_df['correlation'].round(4).tolist(),
                    'hovertemplate': 'Lag %{x} days<br>Correlation: %{y:.3f}<extra></extra>',
                }],
                'layout': {'paper_bgcolor':'transparent','plot_bgcolor':'transparent','font':{'color':'#edf1f7'},'margin':{'l':55,'r':20,'t':30,'b':55},'xaxis':{'title':'Lead lag in days'},'yaxis':{'title':'Lead / revenue correlation'}}
            }, 'Measures how many days after leads arrive the strongest relationship with revenue tends to appear.'))
            lag_alignment_chart = _build_lag_alignment_chart(daily, int(best['lag_days'])) if 'cost' in daily.columns else None
            if lag_alignment_chart:
                charts.append(lag_alignment_chart)
            leads_lagged_chart = _build_leads_lagged_revenue_alignment_chart(daily, int(best['lag_days']))
            if leads_lagged_chart:
                charts.append(leads_lagged_chart)

    # Efficiency frontier
    frontier = pd.DataFrame()
    if series_summary is not None and not series_summary.empty and {'category', 'total_revenue', 'total_cost'}.issubset(series_summary.columns):
        frontier = series_summary[['category', 'total_revenue', 'total_cost']].copy()
        frontier = frontier.rename(columns={'category': 'segment', 'total_revenue': 'revenue', 'total_cost': 'cost'})
    elif {'cost', 'revenue'}.issubset(agg.columns):
        frontier = agg[['month', 'revenue', 'cost']].copy()
        frontier['segment'] = frontier['month'].dt.strftime('%Y-%b')
    if not frontier.empty:
        frontier['revenue'] = pd.to_numeric(frontier['revenue'], errors='coerce')
        frontier['cost'] = pd.to_numeric(frontier['cost'], errors='coerce')
        frontier = frontier.dropna().query('cost > 0 and revenue >= 0').copy()
        if not frontier.empty:
            frontier['roas'] = frontier['revenue'] / frontier['cost'].replace(0, pd.NA)
            frontier['efficiency_score'] = frontier['roas'].rank(pct=True).fillna(0)
            frontier = frontier.sort_values(['roas', 'revenue'], ascending=[False, False])
            out['frontier_table'] = _json_ready_records(frontier[['segment', 'cost', 'revenue', 'roas']].head(30).round(4))
            label_frontier = frontier.sort_values(['revenue', 'roas'], ascending=[False, False]).head(5)
            charts.append(_plot_card('stats-efficiency-frontier', 'Efficiency Frontier', {
                'data': [
                    {
                        'type': 'scatter',
                        'mode': 'markers',
                        'x': frontier['cost'].round(2).tolist(),
                        'y': frontier['revenue'].round(2).tolist(),
                        'text': frontier['segment'].astype(str).tolist(),
                        'customdata': frontier['roas'].round(3).tolist(),
                        'marker': {'size': 8, 'opacity': 0.82},
                        'name': 'Segments',
                        'hovertemplate': '%{text}<br>Cost: $%{x:,.0f}<br>Revenue: $%{y:,.0f}<br>ROAS: %{customdata:.2f}x<extra></extra>',
                    },
                    {
                        'type': 'scatter',
                        'mode': 'markers+text',
                        'x': label_frontier['cost'].round(2).tolist(),
                        'y': label_frontier['revenue'].round(2).tolist(),
                        'text': label_frontier['segment'].astype(str).str.replace('Platform=', '', regex=False).str.replace(' | Affiliate=', '<br>', regex=False).tolist(),
                        'customdata': label_frontier['roas'].round(3).tolist(),
                        'textposition': 'top center',
                        'textfont': {'size': 10},
                        'marker': {'size': 10, 'line': {'width': 1, 'color': '#edf1f7'}},
                        'name': 'Top revenue segments',
                        'hovertemplate': '%{text}<br>Cost: $%{x:,.0f}<br>Revenue: $%{y:,.0f}<br>ROAS: %{customdata:.2f}x<extra></extra>',
                    },
                ],
                'layout': {'paper_bgcolor':'transparent','plot_bgcolor':'transparent','font':{'color':'#edf1f7'},'margin':{'l':55,'r':120,'t':30,'b':55},'xaxis':{'title':'Cost / spend'},'yaxis':{'title':'Revenue'},'showlegend':True}
            }, 'Compares cost and revenue together so you can spot which segments or months sit on the efficient frontier and which lag behind.'))

    # Attrition rate (monthly revenue retention decay)
    if len(agg) >= 2:
        attr = agg[['month', 'revenue']].copy().sort_values('month')
        attr['prev_revenue'] = attr['revenue'].shift(1)
        attr['retention_rate'] = attr['revenue'] / attr['prev_revenue'].replace(0, pd.NA)
        attr['attrition_rate'] = (1 - attr['retention_rate']).clip(lower=0)
        attr = attr.dropna(subset=['attrition_rate'])
        if not attr.empty:
            out['cards']['avg_attrition_rate'] = float(attr['attrition_rate'].mean())
            out['attrition_table'] = _json_ready_records(attr[['month', 'revenue', 'prev_revenue', 'attrition_rate']].round(4))
            charts.append(_plot_card('stats-attrition-rate', 'Attrition Rate', {
                'data': [{
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'x': [pd.Timestamp(m).strftime('%Y-%b') for m in attr['month']],
                    'y': (attr['attrition_rate'] * 100).round(2).tolist(),
                    'hovertemplate': '%{x}<br>Attrition: %{y:.1f}%<extra></extra>',
                }],
                'layout': {'paper_bgcolor':'transparent','plot_bgcolor':'transparent','font':{'color':'#edf1f7'},'margin':{'l':55,'r':20,'t':30,'b':70},'xaxis':{'tickangle':-25},'yaxis':{'title':'Attrition rate %'}}
            }, 'Shows how much monthly revenue slips versus the prior month, which helps quantify decay and retention pressure.'))

    out['charts'] = charts
    out['available'] = bool(charts)
    return out



def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        hit = lowered.get(str(candidate).lower())
        if hit is not None:
            return hit
    return None


def _raw_value_by_category(
    upload_meta: dict | None,
    mapping: dict | None,
    section_key: str,
    output_name: str,
) -> tuple[dict[str, float], dict[str, float]]:
    cfg = (mapping or {}).get(section_key) or {}
    file_key = cfg.get('file_key')
    value_col = cfg.get('value_column')
    date_col = cfg.get('date_column')
    category_col = (
        cfg.get('category_column')
        or cfg.get('platform_column')
        or cfg.get('affiliate_id_column')
        or cfg.get('campaign_id_column')
        or cfg.get('ad_id_column')
    )
    if not (upload_meta and file_key and value_col and date_col and category_col):
        return {}, {}
    try:
        raw_df, _, _ = _resolve_dataset(upload_meta, file_key, cfg.get('sheet_name'))
    except Exception:
        return {}, {}
    required = [date_col, category_col, value_col]
    if raw_df is None or raw_df.empty or any(col not in raw_df.columns for col in required):
        return {}, {}
    work = raw_df[required].copy()
    work['ds'] = pd.to_datetime(first_series(work, date_col), errors='coerce')
    raw_category = first_series(work, category_col).astype(str).str.strip()
    work['category'] = 'Platform=' + raw_category
    work['raw_category'] = raw_category
    work[output_name] = pd.to_numeric(first_series(work, value_col), errors='coerce').fillna(0)
    work = work.dropna(subset=['ds'])
    work = work[(work['raw_category'] != '') & (work['raw_category'].str.lower() != 'nan')]
    if work.empty:
        return {}, {}

    total_prefixed = work.groupby('category')[output_name].sum().to_dict()
    total_raw = work.groupby('raw_category')[output_name].sum().to_dict()
    total = {**total_raw, **total_prefixed}

    recent_cut = work['ds'].max() - pd.Timedelta(days=29) if work['ds'].notna().any() else None
    if recent_cut is None:
        return total, {}
    recent_work = work[work['ds'] >= recent_cut]
    recent_prefixed = recent_work.groupby('category')[output_name].sum().to_dict()
    recent_raw = recent_work.groupby('raw_category')[output_name].sum().to_dict()
    recent = {**recent_raw, **recent_prefixed}
    return total, recent


def _raw_cost_by_category(upload_meta: dict | None, mapping: dict | None) -> tuple[dict[str, float], dict[str, float]]:
    return _raw_value_by_category(upload_meta, mapping, 'cost', 'cost')


def _raw_leads_by_category(upload_meta: dict | None, mapping: dict | None) -> tuple[dict[str, float], dict[str, float]]:
    return _raw_value_by_category(upload_meta, mapping, 'leads', 'leads')


def _build_dimension_budget_recommendations(hist: pd.DataFrame, mapping: dict | None = None, upload_meta: dict | None = None, top_n: int = 8) -> list[dict]:
    if hist is None or hist.empty or 'ds' not in hist.columns or 'y' not in hist.columns:
        return []
    _, direct_recent_cost_by_category = _raw_cost_by_category(upload_meta, mapping)
    _, direct_recent_leads_by_category = _raw_leads_by_category(upload_meta, mapping)
    dim_candidates: list[tuple[str, str | None]] = []
    revenue_cfg = (mapping or {}).get('revenue', {}) if mapping else {}
    for label, key in [('Campaign', 'campaign_id_column'), ('Ad', 'ad_id_column'), ('Affiliate', 'affiliate_id_column'), ('Platform', 'platform_column'), ('Category', 'category_column')]:
        col = revenue_cfg.get(key) if revenue_cfg else None
        if not col or col not in hist.columns:
            if label == 'Platform':
                col = _first_existing_column(hist, ['platform', 'source', 'channel', 'category'])
            elif label == 'Campaign':
                col = _first_existing_column(hist, ['campaign_id', 'campaign', 'campaign name'])
            elif label == 'Ad':
                col = _first_existing_column(hist, ['ad_id', 'ads_id', 'ad', 'creative_id', 'creative'])
            elif label == 'Affiliate':
                col = _first_existing_column(hist, ['affiliate_id', 'affiliate', 'aff_id'])
            elif label == 'Category':
                col = _first_existing_column(hist, ['category'])
        if col and col in hist.columns and (label, col) not in dim_candidates:
            dim_candidates.append((label, col))
    out: list[dict] = []
    for label, col in dim_candidates[:4]:
        work = hist[['ds', 'y', col] + [c for c in ['cost', 'leads'] if c in hist.columns]].copy()
        work[col] = first_series(work, col).astype(str).str.strip()
        work = work[(work[col] != '') & (work[col].str.lower() != 'nan')]
        if work.empty:
            continue
        if label == 'Category':
            platform_like_share = work[col].str.startswith('Platform=').mean()
            if platform_like_share >= 0.8 and any(existing.get('dimension_type') == 'Platform' for existing in out):
                continue
        latest = pd.to_datetime(work['ds'], errors='coerce').max()
        if pd.isna(latest):
            continue
        recent_cut = latest - pd.Timedelta(days=29)
        prior_cut = latest - pd.Timedelta(days=59)
        recent = work[work['ds'] >= recent_cut]
        prior = work[(work['ds'] >= prior_cut) & (work['ds'] < recent_cut)]
        if recent.empty:
            continue
        rec = recent.groupby(col, as_index=False).agg(recent_revenue=('y', 'sum'))
        if 'cost' in recent.columns:
            rec = rec.merge(recent.groupby(col, as_index=False).agg(recent_cost=('cost', 'sum')), on=col, how='left')
        if 'leads' in recent.columns:
            rec = rec.merge(recent.groupby(col, as_index=False).agg(recent_leads=('leads', 'sum')), on=col, how='left')
        pri = prior.groupby(col, as_index=False).agg(prior_revenue=('y', 'sum')) if not prior.empty else pd.DataFrame(columns=[col, 'prior_revenue'])
        merged = rec.merge(pri, on=col, how='left').fillna({'prior_revenue': 0})
        if direct_recent_cost_by_category:
            direct_cost = merged[col].astype(str).map(direct_recent_cost_by_category)
            if 'recent_cost' in merged.columns:
                merged['recent_cost'] = direct_cost.fillna(merged['recent_cost'])
            else:
                merged['recent_cost'] = direct_cost
        if direct_recent_leads_by_category:
            direct_leads = merged[col].astype(str).map(direct_recent_leads_by_category)
            if 'recent_leads' in merged.columns:
                merged['recent_leads'] = direct_leads.fillna(merged['recent_leads'])
            else:
                merged['recent_leads'] = direct_leads
        merged['delta_pct'] = (((merged['recent_revenue'] - merged['prior_revenue']) / merged['prior_revenue'].replace(0, np.nan)) * 100.0)
        merged['delta_pct'] = merged['delta_pct'].replace([np.inf, -np.inf], np.nan)
        if 'recent_cost' in merged.columns:
            merged['recent_roas'] = merged['recent_revenue'] / pd.to_numeric(merged['recent_cost'], errors='coerce').replace(0, pd.NA)
        if 'recent_leads' in merged.columns:
            merged['recent_rpl'] = merged['recent_revenue'] / pd.to_numeric(merged['recent_leads'], errors='coerce').replace(0, pd.NA)
        merged['share'] = merged['recent_revenue'] / max(float(merged['recent_revenue'].sum() or 0.0), 1.0)
        merged = merged.sort_values(['recent_revenue', 'delta_pct'], ascending=[False, True], na_position='last').head(top_n)
        for _, row in merged.iterrows():
            roas = _safe_number(row.get('recent_roas'))
            delta_pct = _safe_number(row.get('delta_pct'))
            share = _safe_number(row.get('share')) or 0.0
            if roas is None:
                shift = 0
                action = 'Recent ROAS is unavailable — platform may be down, paused, or untracked; verify status before budget changes'
            elif roas < 0.85:
                shift = -15 if share >= 0.08 else -10
                action = 'Reduce or rework; recent ROAS is below break-even'
            elif roas < 1.0:
                shift = -10 if share >= 0.12 else 0
                action = 'Hold or trim; recent ROAS is still below break-even'
            elif delta_pct is not None and delta_pct <= -18:
                shift = -15 if share >= 0.18 else -10
                action = 'Reduce budget and inspect funnel quality'
            elif roas >= 1.25 and (delta_pct is None or delta_pct >= -5):
                shift = 12 if share < 0.3 else 8
                action = 'Scale carefully; this is a healthy pocket'
            elif delta_pct is not None and delta_pct >= 12:
                shift = 8
                action = 'Support momentum with measured spend'
            else:
                shift = 0
                action = 'Hold flat and keep monitoring'
            out.append({
                'dimension_type': label,
                'name': str(row[col]),
                'recent_revenue': float(row.get('recent_revenue') or 0),
                'prior_revenue': float(row.get('prior_revenue') or 0),
                'delta_pct': delta_pct,
                'recent_roas': roas,
                'recent_rpl': _safe_number(row.get('recent_rpl')),
                'revenue_share': share * 100.0,
                'recommended_budget_shift_pct': shift,
                'action': action,
            })
    out = sorted(out, key=lambda r: (abs(float(r.get('recommended_budget_shift_pct') or 0)), float(r.get('recent_revenue') or 0)), reverse=True)
    return out[:top_n]


def _build_whale_watch(hist: pd.DataFrame, top_n: int = 30) -> list[dict]:
    if hist is None or hist.empty or 'ds' not in hist.columns or 'y' not in hist.columns:
        return []
    whale_col = _first_existing_column(hist, ['profile_id', 'member_id', 'customer_id', 'user_id', 'account_id', 'profileid', 'memberid'])
    if not whale_col:
        return []
    source_cols = [c for c in ['platform', 'affiliate_id', 'campaign_id', 'ad_id', 'category'] if c in hist.columns]
    work = hist[['ds', 'y', whale_col] + source_cols + [c for c in ['cost', 'leads', 'last_active', 'last_activity'] if c in hist.columns]].copy()
    work[whale_col] = first_series(work, whale_col).astype(str).str.strip()
    work = work[(work[whale_col] != '') & (work[whale_col].str.lower() != 'nan')]
    if work.empty:
        return []
    latest = pd.to_datetime(work['ds'], errors='coerce').max()
    if pd.isna(latest):
        return []
    trailing_90 = work[work['ds'] >= (latest - pd.Timedelta(days=89))].copy()
    if trailing_90.empty:
        trailing_90 = work.copy()
    top_ids = trailing_90.groupby(whale_col, as_index=False)['y'].sum().sort_values('y', ascending=False).head(top_n)[whale_col].tolist()
    whales = work[work[whale_col].isin(top_ids)].copy()
    recent_cut = latest - pd.Timedelta(days=29)
    prior_cut = latest - pd.Timedelta(days=59)
    total_recent = float(whales[whales['ds'] >= recent_cut]['y'].sum() or 0.0)

    def _whale_tier(total_revenue: float) -> str:
        if total_revenue >= 20000: return 'God Tier'
        if total_revenue >= 10000: return 'Super User'
        if total_revenue >= 5000: return 'Core User'
        if total_revenue >= 2000: return 'High-Value User'
        if total_revenue >= 500: return 'High Value'
        return 'Emerging'

    rows = []
    for key, sub in whales.groupby(whale_col):
        recent_sub = sub[sub['ds'] >= recent_cut]
        prior_sub = sub[(sub['ds'] >= prior_cut) & (sub['ds'] < recent_cut)]
        recent_rev = float(recent_sub['y'].sum() or 0.0)
        prior_rev = float(prior_sub['y'].sum() or 0.0)
        lifetime_rev = float(sub['y'].sum() or 0.0)
        delta_pct = ((recent_rev - prior_rev) / prior_rev * 100.0) if prior_rev > 0 else None
        last_seen = pd.to_datetime(sub['ds'], errors='coerce').max()
        days_since = int((latest - last_seen).days) if pd.notna(last_seen) else None
        primary_source = {}
        if source_cols:
            source_rollup = sub.groupby(source_cols, dropna=False, as_index=False)['y'].sum().sort_values('y', ascending=False)
            if not source_rollup.empty:
                top = source_rollup.iloc[0]
                primary_source = {c: (None if pd.isna(top.get(c)) else str(top.get(c))) for c in source_cols}
        status = 'Stable'
        action = 'Keep high-value user experience stable and monitor package behavior.'
        if delta_pct is not None and delta_pct <= -25:
            status = 'At risk'
            action = 'Add high-value user reactivation treatment and review spend/package mix.'
        if days_since is not None and days_since >= 14:
            status = 'Cooling off'
            action = 'User has gone quiet; trigger retention outreach before budget scaling.'
        if recent_rev <= 0 and prior_rev > 0:
            status = 'Dormant'
            action = 'Recent spend is zero; prioritize win-back before scaling acquisition tied to this user source.'
        rows.append({
            'whale_id': str(key),
            'whale_tier': _whale_tier(lifetime_rev),
            'lifetime_revenue': lifetime_rev,
            'recent_revenue': recent_rev,
            'prior_revenue': prior_rev,
            'delta_pct': delta_pct,
            'revenue_share_pct': (recent_rev / max(total_recent, 1.0)) * 100.0,
            'days_since_last_seen': days_since,
            'status': status,
            'action': action,
            **primary_source,
        })
    rows = sorted(rows, key=lambda r: (float(r.get('recent_revenue') or 0), float(r.get('lifetime_revenue') or 0)), reverse=True)
    return rows[:top_n]



def _build_attribution_table(series_summary: pd.DataFrame | None) -> tuple[list[dict], dict]:
    if series_summary is None or series_summary.empty:
        return [], {}
    use = series_summary.copy()
    keep = [c for c in ['category', 'total_revenue', 'share_of_revenue', 'roas', 'recent_roas', 'rev_per_lead', 'total_cost', 'recent_revenue', 'recent_cost', 'stability_score'] if c in use.columns]
    if not keep:
        return [], {}
    use = use[keep].copy().head(15)
    out = []
    for _, row in use.iterrows():
        total_cost = _safe_number(row.get('total_cost'))
        recent_cost = _safe_number(row.get('recent_cost'))
        out.append({
            'channel': row.get('category'),
            'revenue': _safe_number(row.get('total_revenue')),
            'revenue_share_pct': _safe_number((row.get('share_of_revenue') or 0) * 100.0 if pd.notna(row.get('share_of_revenue')) else None),
            'roas': _safe_number(row.get('roas')) if total_cost and total_cost > 0 else None,
            'recent_roas': _safe_number(row.get('recent_roas')) if recent_cost and recent_cost > 0 else None,
            'rev_per_lead': _safe_number(row.get('rev_per_lead')),
            'total_cost': total_cost,
            'recent_revenue': _safe_number(row.get('recent_revenue')),
            'recent_cost': recent_cost,
            'stability_score': _safe_number(row.get('stability_score')),
        })
    summary = {}
    if out:
        top = max(out, key=lambda r: float(r.get('revenue') or 0))
        summary = {
            'top_channel': top.get('channel'),
            'top_revenue': top.get('revenue'),
            'channel_count': len(out),
        }
    return out, summary


def _build_scenario2_table(scenario_defaults: dict, allocation_rows: list[dict] | None) -> tuple[list[dict], dict]:
    base_revenue = float(scenario_defaults.get('planning_monthly_revenue') or scenario_defaults.get('base_revenue') or 0.0)
    base_cost = float(scenario_defaults.get('planning_monthly_cost') or scenario_defaults.get('base_cost') or 0.0)
    base_leads = float(scenario_defaults.get('planning_monthly_leads') or scenario_defaults.get('base_leads') or 0.0)
    base_rpl = float(scenario_defaults.get('planning_monthly_rpl') or scenario_defaults.get('base_rev_per_lead') or 0.0)
    if base_revenue <= 0 and base_leads > 0 and base_rpl > 0:
        base_revenue = base_leads * base_rpl
    if base_cost <= 0 and allocation_rows:
        try:
            base_cost = float(sum(float(r.get('recent_cost') or 0) for r in allocation_rows))
        except Exception:
            base_cost = 0.0
    scenarios = [
        ('Downside', 0.92, 0.95, 0.97),
        ('Expected', 1.00, 1.00, 1.00),
        ('Scale Efficient Channels', 1.08, 1.05, 1.03),
        ('Lead Quality Lift', 1.00, 1.04, 1.08),
        ('Aggressive Growth', 1.15, 1.12, 1.05),
    ]
    rows = []
    for name, budget_mult, leads_mult, quality_mult in scenarios:
        spend = base_cost * budget_mult
        leads = base_leads * leads_mult
        revenue = max(base_revenue, base_rpl * base_leads) * quality_mult * (0.55 + 0.45 * leads_mult)
        if base_cost > 0:
            revenue *= (0.65 + 0.35 * budget_mult)
        roas = (revenue / spend) if spend > 0 else None
        rows.append({
            'scenario': name,
            'projected_spend': _safe_number(spend),
            'projected_leads': _safe_number(leads),
            'projected_revenue': _safe_number(revenue),
            'projected_roas': _safe_number(roas),
        })
    best = max(rows, key=lambda r: float(r.get('projected_revenue') or 0)) if rows else {}
    cards = {
        'base_revenue': _safe_number(base_revenue),
        'base_cost': _safe_number(base_cost),
        'base_leads': _safe_number(base_leads),
        'best_scenario': best.get('scenario'),
        'best_revenue': best.get('projected_revenue'),
        'best_roas': best.get('projected_roas'),
        'period_basis': 'latest_completed_month',
        'period_label': scenario_defaults.get('planning_month_label') or 'Latest completed month',
        'leads_label': 'Modeled lead volume',
    }
    return rows, cards


def _build_mapped_monthly_rollup(
    upload_meta: dict | None,
    mapping: dict | None,
    section_key: str,
    output_col: str,
) -> pd.DataFrame:
    cfg = (mapping or {}).get(section_key) or {}
    if not upload_meta or not cfg.get('file_key') or not cfg.get('date_column') or not cfg.get('value_column'):
        return pd.DataFrame(columns=['month', output_col])
    try:
        raw_df, _, _ = _resolve_dataset(upload_meta, cfg.get('file_key'), cfg.get('sheet_name'))
    except Exception:
        raw_df = None
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=['month', output_col])
    date_col = cfg.get('date_column')
    value_col = cfg.get('value_column')
    if date_col not in raw_df.columns or value_col not in raw_df.columns:
        return pd.DataFrame(columns=['month', output_col])
    work = raw_df[[date_col, value_col]].copy()
    work['month'] = normalize_date_series(first_series(work, date_col)).dt.to_period('M').dt.to_timestamp()
    work[output_col] = pd.to_numeric(first_series(work, value_col), errors='coerce')
    work = work.dropna(subset=['month', output_col])
    if work.empty:
        return pd.DataFrame(columns=['month', output_col])
    return work.groupby('month', as_index=False)[output_col].sum()


def _build_mapped_daily_rollup(
    upload_meta: dict | None,
    mapping: dict | None,
    section_key: str,
    output_col: str,
) -> pd.DataFrame:
    cfg = (mapping or {}).get(section_key) or {}
    if not upload_meta or not cfg.get('file_key') or not cfg.get('date_column') or not cfg.get('value_column'):
        return pd.DataFrame(columns=['ds', output_col])
    try:
        raw_df, _, _ = _resolve_dataset(upload_meta, cfg.get('file_key'), cfg.get('sheet_name'))
    except Exception:
        raw_df = None
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=['ds', output_col])
    date_col = cfg.get('date_column')
    value_col = cfg.get('value_column')
    if date_col not in raw_df.columns or value_col not in raw_df.columns:
        return pd.DataFrame(columns=['ds', output_col])
    work = raw_df[[date_col, value_col]].copy()
    work['ds'] = normalize_date_series(first_series(work, date_col)).dt.normalize()
    work[output_col] = pd.to_numeric(first_series(work, value_col), errors='coerce')
    work = work.dropna(subset=['ds', output_col])
    if work.empty:
        return pd.DataFrame(columns=['ds', output_col])
    return work.groupby('ds', as_index=False)[output_col].sum()


def _build_stakeholder_report(visuals: dict, result: dict) -> tuple[list[str], str]:
    lines = []
    overview = visuals.get('overview_cards') or {}
    insights = visuals.get('insights') or {}
    missing_recent_roas = [
        row for row in (visuals.get('planning_dimension_recommendations') or [])
        if _safe_number((row or {}).get('recent_roas')) is None
    ]
    latest_rev = overview.get('latest_revenue')
    forecast_rows = overview.get('forecast_rows')
    avg_forecast = (result.get('diagnostics') or {}).get('forecast_avg_expected')
    if latest_rev is not None and avg_forecast is not None:
        delta = ((float(avg_forecast) - float(latest_rev)) / max(abs(float(latest_rev)), 1.0)) * 100.0
        lines.append(f"Latest realized revenue is ${float(latest_rev):,.0f}, while the model's average forecast period is ${float(avg_forecast):,.0f} ({delta:+.1f}% vs. latest actual).")
    top_driver = insights.get('top_driver_name')
    if top_driver:
        lines.append(f"Explainable AI points to {top_driver} as the strongest modeled driver behind the current revenue path.")
    if visuals.get('planning_whale_watch'):
        whale = visuals['planning_whale_watch'][0]
        lines.append(f"User Intelligence 2.0 shows {whale.get('whale_id')} as the largest recent high-value profile, with status marked {whale.get('status', 'Stable')}.")
    if visuals.get('attribution_summary', {}).get('top_channel'):
        top_channel = visuals['attribution_summary']['top_channel']
        top_rev = visuals['attribution_summary'].get('top_revenue') or 0
        lines.append(f"Channel Attribution Engine ranks {top_channel} as the top revenue contributor at roughly ${float(top_rev):,.0f} across the visible history.")
    if visuals.get('cohort_rev_available'):
        cohort_count = (visuals.get('cohort_rev_summary') or {}).get('cohort_count') or 0
        lines.append(f"Cohort Intelligence is active with {int(cohort_count)} lead-month cohorts available for payback and maturity review.")
    if visuals.get('scenario2_cards', {}).get('best_scenario'):
        s2 = visuals['scenario2_cards']
        lines.append(f"Scenario Simulator 2.0 currently favors '{s2.get('best_scenario')}', projecting about ${float(s2.get('best_revenue') or 0):,.0f} in modeled revenue at {float(s2.get('best_roas') or 0):.2f}x ROAS.")
    if missing_recent_roas:
        sample = missing_recent_roas[0]
        lines.append(f"Recent ROAS is unavailable for {sample.get('name', 'at least one platform segment')}, which usually means the platform was taken down, paused, or stopped tracking — verify status before moving budget.")
    if not lines:
        lines.append('This run is ready for stakeholder review, but richer mapped inputs such as leads, cost, categories, and profile identifiers will unlock stronger executive guidance.')
    html_items = ''.join(f'<li>{html.escape(line)}</li>' for line in lines[:6])
    html_block = f"<div class='stakeholder-report-export'><h1>Auto-Report</h1><ul>{html_items}</ul></div>"
    return lines[:6], html_block


def _results_visual_defaults() -> dict:
    return {
        'charts': [],
        'forecast_table': [],
        'overview_cards': {},
        'insights': {},
        'cohort_table': [],
        'series_table': [],
        'optimizer_table': [],
        'time_machine_table': [],
        'drivers_table': [],
        'tracking_table': [],
        'alerts': [],
        'impact_table': [],
        'allocation_table': [],
        'model_ranking': [],
        'diagnostics_rows': [],
        'sheet_impact_table': [],
        'cohort_rev_available': False,
        'cohort_rev_summary': {},
        'cohort_rev_table': [],
        'cohort_rev_platform_table': [],
        'cohort_rev_table_grain_label': 'Platform',
        'cohort_rev_curve_chart': None,
        'cohort_rev_platform_chart': None,
        'cohort_rev_heatmap_chart': None,
        'cohort_rev_payback_chart': None,
        'cohort_rev_payback_table': [],
        'cohort_rev_narrative': [],
        'cohort_rev_roas_table': [],
        'cohort_payback_curve': [],
        'cohort_payback_summary': [],
        'early_warning_table': [],
        'waterfall_table': [],
        'waterfall_explanation': {},
        'waterfall_explanation': {},
        'goal_optimizer_profiles': [],
        'planning_dimension_recommendations': [],
        'planning_whale_watch': [],
        'planning_quarter_horizon_months': 5,
        'statistics_available': False,
        'statistics_cards': {},
        'statistics_charts': [],
        'statistics_lag_table': [],
        'statistics_frontier_table': [],
        'statistics_attrition_table': [],
        'cost_validation_available': False,
        'cost_validation_cards': {},
        'cost_validation_platform_table': [],
        'cost_validation_company_table': [],
        'scenario_compare_table': [],
        'scenario_compare_chart': None,
        'budget_opt_table': [],
        'budget_opt_chart': None,
        'anomaly_table': [],
        'anomaly_chart': None,
        'rolling_update_table': [],
        'rolling_update_chart': None,
        'health_monitor_table': [],
        'health_monitor_cards': {},
        'auto_insights_table': [],
        'spending_slowdown_available': False,
        'spending_slowdown_message': '',
        'spending_slowdown_cards': {},
        'spending_slowdown_charts': [],
        'spending_slowdown_decomposition_table': [],
        'spending_slowdown_returning_table': [],
        'spending_slowdown_cohort_table': [],
        'forecast_start': None,
        'chart_registry': {},
        'chart_errors': [],
        'protected_results_layer': False,
        'results_contract_version': 'legacy',
        'render_warning': None,
        'render_error': None,
        'lightgbm_intelligence_available': False,
        'lightgbm_exec_summary': [],
        'lightgbm_shap_summary': [],
        'lightgbm_local_explanations': [],
        'lightgbm_shap_aggregations': [],
        'lightgbm_ranked_opportunities': [],
        'lightgbm_lead_quality_scores': [],
        'lightgbm_anomaly_rows': [],
        'lightgbm_whale_predictions': [],
        'lightgbm_response_curve': [],
        'lightgbm_scenarios': [],
        'lightgbm_affiliate_quality': [],
        'lightgbm_confidence_rows': [],
    }


def _merge_results_visual_defaults(visuals: dict | None) -> dict:
    merged = _results_visual_defaults()
    if isinstance(visuals, dict):
        merged.update(visuals)
    return merged


def _build_minimal_results_visuals(result: dict) -> dict:
    forecast_rows = result.get('display_forecast') or result.get('blended_forecast') or result.get('best_forecast') or []
    forecast_table: list[dict] = []
    chart_spec = None
    if forecast_rows:
        for row in forecast_rows[:500]:
            if isinstance(row, dict):
                forecast_table.append({k: row.get(k) for k in row.keys()})
        ds = [row.get('ds') for row in forecast_rows if isinstance(row, dict)]
        yhat = [row.get('yhat') for row in forecast_rows if isinstance(row, dict)]
        lower = [row.get('yhat_lower') for row in forecast_rows if isinstance(row, dict)]
        upper = [row.get('yhat_upper') for row in forecast_rows if isinstance(row, dict)]
        chart_spec = {
            'data': [
                {'type': 'scatter', 'mode': 'lines', 'name': 'Forecast', 'x': ds, 'y': yhat},
                {'type': 'scatter', 'mode': 'lines', 'name': 'Lower', 'x': ds, 'y': lower, 'line': {'dash': 'dot'}},
                {'type': 'scatter', 'mode': 'lines', 'name': 'Upper', 'x': ds, 'y': upper, 'line': {'dash': 'dot'}},
            ],
            'layout': {
                'margin': {'l': 48, 'r': 18, 't': 24, 'b': 48},
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'legend': {'orientation': 'h'},
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Forecast'},
            },
        }
    model_ranking = result.get('top_models') or []
    diagnostics = result.get('diagnostics') or {}
    overview_cards = {
        'forecast_start': forecast_rows[0].get('ds') if forecast_rows and isinstance(forecast_rows[0], dict) else 'n/a',
        'latest_actual': diagnostics.get('recent_avg_y'),
        'forecast_avg': diagnostics.get('forecast_avg_expected'),
        'forecast_to_history_ratio': diagnostics.get('forecast_to_history_ratio'),
    }
    visuals = _merge_results_visual_defaults({
        'overview_cards': overview_cards,
        'diagnostics_rows': [{'metric': k.replace('_', ' ').title(), 'value': v} for k, v in diagnostics.items() if v is not None][:12],
        'model_ranking': model_ranking,
        'forecast_table': forecast_table,
        'charts': ([{'id': 'fallback-forecast-chart', 'title': 'Forecast Preview', 'subtitle': 'Fallback chart built from saved forecast output.', 'spec': json.dumps(chart_spec)}] if chart_spec else []),
        'render_warning': 'Detailed results visuals could not be rebuilt from the merged modeling table, so this page is showing a safe fallback view from the saved forecast output.'
    })
    return _protect_results_visuals(visuals)


def _build_results_visuals(df: pd.DataFrame | None, result: dict, mapping: dict | None = None, upload_meta: dict | None = None) -> dict:
    visuals = _merge_results_visual_defaults({
        'charts': [],
        'forecast_table': [],
        'overview_cards': {},
        'insights': {},
        'cohort_table': [],
        'series_table': [],
        'optimizer_table': [],
        'time_machine_table': [],
        'drivers_table': [],
        'tracking_table': [],
        'alerts': [],
        'impact_table': [],
        'allocation_table': [],
        'model_ranking': [],
        'diagnostics_rows': [],
        'sheet_impact_table': [],
        'cohort_rev_available': False,
        'cohort_rev_summary': {},
        'cohort_rev_table': [],
        'cohort_rev_platform_table': [],
        'cohort_rev_table_grain_label': 'Platform',
        'cohort_rev_curve_chart': None,
        'cohort_rev_platform_chart': None,
        'cohort_rev_heatmap_chart': None,
        'cohort_rev_payback_chart': None,
        'cohort_rev_payback_table': [],
        'cohort_rev_narrative': [],
        'cohort_rev_roas_table': [],
        'cohort_payback_curve': [],
        'cohort_payback_summary': [],
        'early_warning_table': [],
        'waterfall_table': [],
        'goal_optimizer_profiles': [],
        'planning_dimension_recommendations': [],
        'planning_whale_watch': [],
        'planning_quarter_horizon_months': 5,
        'statistics_available': False,
        'statistics_cards': {},
        'statistics_charts': [],
        'statistics_lag_table': [],
        'statistics_frontier_table': [],
        'statistics_attrition_table': [],
        'cost_validation_available': False,
        'cost_validation_cards': {},
        'cost_validation_platform_table': [],
        'cost_validation_company_table': [],
        'scenario_compare_table': [],
        'scenario_compare_chart': None,
        'budget_opt_table': [],
        'budget_opt_chart': None,
        'anomaly_table': [],
        'anomaly_chart': None,
        'rolling_update_table': [],
        'rolling_update_chart': None,
        'health_monitor_table': [],
        'health_monitor_cards': {},
        'auto_insights_table': [],
        'spending_slowdown_available': False,
        'spending_slowdown_message': '',
        'spending_slowdown_cards': {},
        'spending_slowdown_charts': [],
        'spending_slowdown_decomposition_table': [],
        'spending_slowdown_returning_table': [],
        'spending_slowdown_cohort_table': [],
        'whale_export_table': [],
        'explainability_summary': [],
        'scenario2_table': [],
        'scenario2_cards': {},
        'attribution_table': [],
        'attribution_summary': {},
        'stakeholder_report_html': '',
        'stakeholder_report_lines': [],
        'lightgbm_intelligence_available': False,
        'lightgbm_exec_summary': [],
        'lightgbm_shap_summary': [],
        'lightgbm_local_explanations': [],
        'lightgbm_shap_aggregations': [],
        'lightgbm_ranked_opportunities': [],
        'lightgbm_lead_quality_scores': [],
        'lightgbm_anomaly_rows': [],
        'lightgbm_whale_predictions': [],
        'lightgbm_response_curve': [],
        'lightgbm_scenarios': [],
        'lightgbm_affiliate_quality': [],
        'lightgbm_confidence_rows': [],
    })

    hist = df.copy() if df is not None else pd.DataFrame()
    hist_daily = pd.DataFrame(columns=['ds', 'y'])
    series_summary = pd.DataFrame()
    allowed_driver_cols, mapped_sheet_defs = _mapped_driver_columns(mapping, upload_meta)

    def _json(df_like: pd.DataFrame, limit: int = 20) -> list[dict]:
        if df_like is None or df_like.empty:
            return []
        return _json_ready_records(df_like.head(limit).round(4))

    def _safe_pct(curr, prev):
        curr = float(curr) if pd.notna(curr) else None
        prev = float(prev) if pd.notna(prev) else None
        if curr is None or prev in (None, 0):
            return None
        return ((curr - prev) / prev) * 100.0

    if hist is not None and not hist.empty:
        hist = dedupe_columns_keep_first(hist.copy())
        ds_source = 'ds' if 'ds' in hist.columns else (hist.columns[0] if len(hist.columns) > 0 else None)
        y_source = 'y' if 'y' in hist.columns else (hist.columns[1] if len(hist.columns) > 1 else None)
        if ds_source is not None and y_source is not None:
            hist['ds'] = pd.to_datetime(first_series(hist, ds_source), errors='coerce')
            hist['y'] = pd.to_numeric(first_series(hist, y_source), errors='coerce')
            for c in hist.columns:
                if c not in {'ds', 'y', 'series_id', 'category'}:
                    try:
                        hist[c] = pd.to_numeric(first_series(hist, c))
                    except Exception:
                        hist[c] = first_series(hist, c)
            if 'series_id' not in hist.columns:
                hist['series_id'] = 'company_total'
            if 'category' not in hist.columns:
                hist['category'] = hist['series_id'].astype(str)
            hist = hist.dropna(subset=['ds']).copy()
            hist = hist.assign(ds=hist['ds'].dt.normalize())

            agg_map = {'y': 'sum'}
            candidate_sum_cols = ['clicks', 'visits', 'registrations', 'spend', 'impressions', 'event_flag', 'outage_flag', 'promo_flag', 'holiday_flag']
            for c in candidate_sum_cols:
                if c in hist.columns:
                    agg_map[c] = 'sum' if c not in {'event_flag', 'outage_flag', 'promo_flag', 'holiday_flag'} else 'max'
            hist_daily = hist.groupby('ds', as_index=False).agg(agg_map)

            def _helper_daily_rollup(value_col: str, helper_cfg: dict | None) -> pd.DataFrame | None:
                if value_col not in hist.columns:
                    return None
                cfg = helper_cfg or {}
                helper_keys = []
                if value_col == 'cost' and '__cost_bucket' in hist.columns:
                    helper_keys = ['__cost_bucket']
                else:
                    for key_col in [
                        cfg.get('platform_column'),
                        cfg.get('category_column'),
                        cfg.get('affiliate_id_column'),
                        cfg.get('campaign_id_column'),
                        cfg.get('ad_id_column'),
                        'category',
                        'series_id',
                    ]:
                        if key_col and key_col in hist.columns and key_col not in helper_keys:
                            helper_keys.append(key_col)
                helper_frame = hist[['ds', value_col] + helper_keys].copy()
                helper_frame[value_col] = pd.to_numeric(helper_frame[value_col], errors='coerce')
                helper_frame = helper_frame.dropna(subset=['ds', value_col])
                if helper_frame.empty:
                    return None
                dedupe_keys = ['ds'] + helper_keys if helper_keys else ['ds']
                helper_frame = helper_frame.groupby(dedupe_keys, as_index=False)[value_col].max()
                return helper_frame.groupby('ds', as_index=False)[value_col].sum()

            leads_daily = _helper_daily_rollup('leads', (mapping or {}).get('leads'))
            cost_daily = _helper_daily_rollup('cost', (mapping or {}).get('cost'))
            for helper_daily in [leads_daily, cost_daily]:
                if helper_daily is not None and not helper_daily.empty:
                    hist_daily = hist_daily.merge(helper_daily, on='ds', how='left')

            hist_daily = dedupe_columns_keep_first(hist_daily)
            direct_total_cost_by_category, direct_recent_cost_by_category = _raw_cost_by_category(upload_meta, mapping)
            direct_total_leads_by_category, direct_recent_leads_by_category = _raw_leads_by_category(upload_meta, mapping)
            hist_daily['ds'] = pd.to_datetime(first_series(hist_daily, 'ds'), errors='coerce')
            hist_daily['y'] = pd.to_numeric(first_series(hist_daily, 'y'), errors='coerce')
            if {'y', 'cost'}.issubset(hist_daily.columns):
                hist_daily['roas'] = (hist_daily['y'].clip(lower=0) / pd.to_numeric(hist_daily['cost'], errors='coerce').replace(0, pd.NA)).clip(lower=0)
            if {'y', 'leads'}.issubset(hist_daily.columns):
                hist_daily['revenue_per_lead'] = (hist_daily['y'].clip(lower=0) / pd.to_numeric(hist_daily['leads'], errors='coerce').replace(0, pd.NA)).clip(lower=0)
            hist_daily = hist_daily.dropna(subset=['ds']).sort_values('ds').reset_index(drop=True)

            if 'category' in hist.columns and hist['category'].nunique(dropna=True) > 1:
                ser = hist.copy()
                recent_cut = ser['ds'].max() - pd.Timedelta(days=29) if ser['ds'].notna().any() else None
                ser['recent_flag'] = (ser['ds'] >= recent_cut).astype(int) if recent_cut is not None else 0

                grp = ser.groupby('category', as_index=False).agg(
                    total_revenue=('y', 'sum'),
                    periods=('ds', 'nunique'),
                    avg_daily=('y', 'mean'),
                    volatility=('y', lambda s: float(pd.to_numeric(s, errors='coerce').std(ddof=0) or 0.0)),
                )
                for src_col, out_col in [('leads', 'total_leads'), ('cost', 'total_cost'), ('clicks', 'total_clicks'), ('visits', 'total_visits')]:
                    if src_col in ser.columns:
                        temp = ser.groupby('category', as_index=False)[src_col].sum().rename(columns={src_col: out_col})
                        grp = grp.merge(temp, on='category', how='left')

                recent_grp = ser[ser['recent_flag'] == 1].groupby('category', as_index=False).agg(
                    recent_revenue=('y', 'sum'),
                    recent_days=('ds', 'nunique'),
                    recent_avg_daily=('y', 'mean'),
                )
                if 'cost' in ser.columns:
                    recent_cost = ser[ser['recent_flag'] == 1].groupby('category', as_index=False)['cost'].sum().rename(columns={'cost': 'recent_cost'})
                    recent_grp = recent_grp.merge(recent_cost, on='category', how='left')
                if 'leads' in ser.columns:
                    recent_leads = ser[ser['recent_flag'] == 1].groupby('category', as_index=False)['leads'].sum().rename(columns={'leads': 'recent_leads'})
                    recent_grp = recent_grp.merge(recent_leads, on='category', how='left')
                grp = grp.merge(recent_grp, on='category', how='left')

                if direct_total_cost_by_category:
                    direct_cost = grp['category'].map(direct_total_cost_by_category)
                    grp['total_cost'] = direct_cost.combine_first(grp['total_cost']) if 'total_cost' in grp.columns else direct_cost.fillna(0.0)
                if direct_recent_cost_by_category:
                    direct_cost = grp['category'].map(direct_recent_cost_by_category)
                    grp['recent_cost'] = direct_cost.combine_first(grp['recent_cost']) if 'recent_cost' in grp.columns else direct_cost.fillna(0.0)
                if direct_total_leads_by_category:
                    direct_leads = grp['category'].map(direct_total_leads_by_category)
                    grp['total_leads'] = direct_leads.combine_first(grp['total_leads']) if 'total_leads' in grp.columns else direct_leads.fillna(0.0)
                if direct_recent_leads_by_category:
                    direct_leads = grp['category'].map(direct_recent_leads_by_category)
                    grp['recent_leads'] = direct_leads.combine_first(grp['recent_leads']) if 'recent_leads' in grp.columns else direct_leads.fillna(0.0)

                if 'total_leads' in grp.columns:
                    grp['rev_per_lead'] = (grp['total_revenue'].clip(lower=0) / pd.to_numeric(grp['total_leads'], errors='coerce').replace(0, pd.NA)).clip(lower=0)
                if 'total_cost' in grp.columns:
                    grp['total_cost'] = pd.to_numeric(grp['total_cost'], errors='coerce')
                    grp['roas'] = grp['total_revenue'] / grp['total_cost'].replace(0, pd.NA)
                    grp.loc[grp['total_cost'].fillna(0) <= 0, 'roas'] = np.nan
                if 'recent_cost' in grp.columns:
                    grp['recent_cost'] = pd.to_numeric(grp['recent_cost'], errors='coerce')
                    grp['recent_roas'] = grp['recent_revenue'] / grp['recent_cost'].replace(0, pd.NA)
                    grp.loc[grp['recent_cost'].fillna(0) <= 0, 'recent_roas'] = np.nan

                grp['share_of_revenue'] = grp['total_revenue'] / max(float(grp['total_revenue'].sum() or 0.0), 1.0)
                grp['stability_score'] = pd.to_numeric(
                    grp['avg_daily'].fillna(0) / grp['volatility'].replace(0, pd.NA),
                    errors='coerce',
                ).fillna(0)
                grp = grp.sort_values('total_revenue', ascending=False).reset_index(drop=True)
                series_summary = grp
                visuals['cohort_table'] = _json(grp, 12)
                keep_cols = ['category', 'share_of_revenue', 'stability_score'] + [c for c in ['roas', 'recent_roas', 'rev_per_lead'] if c in grp.columns]

                visuals['series_table'] = _json(grp[keep_cols], 12)

            if 'cost' in hist.columns:
                raw_cost_base = pd.DataFrame()
                cost_cfg = (mapping or {}).get('cost') or {}
                cost_file_key = cost_cfg.get('file_key')
                cost_value_col = cost_cfg.get('value_column')
                cost_date_col = cost_cfg.get('date_column')
                cost_category_col = (
                    cost_cfg.get('category_column')
                    or cost_cfg.get('platform_column')
                    or cost_cfg.get('affiliate_id_column')
                    or cost_cfg.get('campaign_id_column')
                    or cost_cfg.get('ad_id_column')
                )
                if upload_meta and cost_file_key and cost_value_col and cost_date_col and cost_category_col:
                    try:
                        raw_cost_df, _, _ = _resolve_dataset(upload_meta, cost_file_key, cost_cfg.get('sheet_name'))
                        if raw_cost_df is not None and not raw_cost_df.empty and all(c in raw_cost_df.columns for c in [cost_date_col, cost_category_col, cost_value_col]):
                            raw_cost_base = raw_cost_df[[cost_date_col, cost_category_col, cost_value_col]].copy()
                            raw_cost_base['ds'] = pd.to_datetime(first_series(raw_cost_base, cost_date_col), errors='coerce')
                            raw_cost_base['cost_bucket'] = first_series(raw_cost_base, cost_category_col).astype(str).str.strip()
                            raw_cost_base['cost'] = pd.to_numeric(first_series(raw_cost_base, cost_value_col), errors='coerce').fillna(0)
                            raw_cost_base = raw_cost_base.dropna(subset=['ds'])
                            raw_cost_base = raw_cost_base[(raw_cost_base['cost_bucket'] != '') & (raw_cost_base['cost_bucket'].str.lower() != 'nan')]
                    except Exception:
                        raw_cost_base = pd.DataFrame()

                if raw_cost_base.empty:
                    cost_base = hist.copy()
                    if '__cost_bucket' in cost_base.columns:
                        cost_base['cost_bucket'] = first_series(cost_base, '__cost_bucket')
                    elif 'platform' in cost_base.columns:
                        cost_base['cost_bucket'] = first_series(cost_base, 'platform')
                    elif 'category' in cost_base.columns:
                        cost_base['cost_bucket'] = first_series(cost_base, 'category')
                    else:
                        cost_base['cost_bucket'] = 'company_total'
                    cost_base = cost_base.dropna(subset=['ds']).copy()
                    cost_base['cost_bucket'] = cost_base['cost_bucket'].fillna('company_total').astype(str)
                    cost_base['month'] = pd.to_datetime(cost_base['ds']).dt.to_period('M').dt.to_timestamp()
                    daily_bucket_cost = cost_base.groupby(['ds', 'cost_bucket'], as_index=False).agg(daily_platform_cost=('cost', 'max'))
                    daily_bucket_cost['month'] = pd.to_datetime(daily_bucket_cost['ds']).dt.to_period('M').dt.to_timestamp()
                else:
                    cost_base = raw_cost_base.copy()
                    cost_base['month'] = pd.to_datetime(cost_base['ds']).dt.to_period('M').dt.to_timestamp()
                    monthly_raw = cost_base.groupby(['month', 'cost_bucket'], as_index=False).agg(monthly_platform_cost=('cost', 'sum'))
                    monthly_raw['days_in_month'] = pd.to_datetime(monthly_raw['month']).dt.daysinmonth
                    monthly_raw['daily_platform_cost'] = pd.to_numeric(monthly_raw['monthly_platform_cost'], errors='coerce') / pd.to_numeric(monthly_raw['days_in_month'], errors='coerce').replace(0, pd.NA)
                    daily_bucket_cost = monthly_raw.rename(columns={'daily_platform_cost': 'daily_platform_cost'})[['month', 'cost_bucket', 'daily_platform_cost']].copy()

                rev_base = hist.copy()
                rev_base['month'] = pd.to_datetime(rev_base['ds']).dt.to_period('M').dt.to_timestamp()
                if 'platform' in rev_base.columns:
                    rev_base['cost_bucket'] = first_series(rev_base, 'platform').astype(str).str.strip()
                elif 'category' in rev_base.columns:
                    rev_base['cost_bucket'] = first_series(rev_base, 'category').astype(str).str.replace(r'^Platform=', '', regex=True).str.strip()
                else:
                    rev_base['cost_bucket'] = 'company_total'
                month_bucket_revenue = rev_base.groupby(['month', 'cost_bucket'], as_index=False).agg(monthly_platform_revenue=('y', 'sum')) if 'y' in rev_base.columns else pd.DataFrame(columns=['month','cost_bucket','monthly_platform_revenue'])

                if raw_cost_base.empty:
                    monthly_platform = daily_bucket_cost.groupby(['month', 'cost_bucket'], as_index=False).agg(
                        monthly_platform_cost=('daily_platform_cost', 'sum'),
                        avg_daily_platform_cost=('daily_platform_cost', 'mean')
                    )
                else:
                    monthly_platform = cost_base.groupby(['month', 'cost_bucket'], as_index=False).agg(monthly_platform_cost=('cost', 'sum'))
                    monthly_platform['avg_daily_platform_cost'] = pd.to_numeric(monthly_platform['monthly_platform_cost'], errors='coerce') / pd.to_datetime(monthly_platform['month']).dt.daysinmonth

                if not monthly_platform.empty:
                    monthly_platform['days_in_month'] = pd.to_datetime(monthly_platform['month']).dt.daysinmonth
                    monthly_platform['daily_platform_cost'] = pd.to_numeric(monthly_platform['monthly_platform_cost'], errors='coerce') / pd.to_numeric(monthly_platform['days_in_month'], errors='coerce').replace(0, pd.NA)
                    monthly_platform['recomputed_monthly_cost'] = pd.to_numeric(monthly_platform['daily_platform_cost'], errors='coerce') * pd.to_numeric(monthly_platform['days_in_month'], errors='coerce')
                    if not month_bucket_revenue.empty:
                        monthly_platform = monthly_platform.merge(month_bucket_revenue, on=['month', 'cost_bucket'], how='left')
                        monthly_platform['platform_roas'] = pd.to_numeric(monthly_platform['monthly_platform_revenue'], errors='coerce') / pd.to_numeric(monthly_platform['monthly_platform_cost'], errors='coerce').replace(0, pd.NA)
                    monthly_platform = monthly_platform.sort_values(['month', 'cost_bucket']).reset_index(drop=True)
                    monthly_company = monthly_platform.groupby('month', as_index=False).agg(monthly_company_cost=('monthly_platform_cost', 'sum'))
                    monthly_company['days_in_month'] = pd.to_datetime(monthly_company['month']).dt.daysinmonth
                    monthly_company['company_daily_cost'] = pd.to_numeric(monthly_company['monthly_company_cost'], errors='coerce') / pd.to_numeric(monthly_company['days_in_month'], errors='coerce').replace(0, pd.NA)
                    monthly_company['recomputed_monthly_cost'] = pd.to_numeric(monthly_company['company_daily_cost'], errors='coerce') * pd.to_numeric(monthly_company['days_in_month'], errors='coerce')
                    if 'y' in rev_base.columns:
                        monthly_revenue = rev_base.groupby('month', as_index=False).agg(monthly_company_revenue=('y', 'sum'))
                        monthly_company = monthly_company.merge(monthly_revenue, on='month', how='left')
                        monthly_company['company_roas'] = pd.to_numeric(monthly_company['monthly_company_revenue'], errors='coerce') / pd.to_numeric(monthly_company['monthly_company_cost'], errors='coerce').replace(0, pd.NA)
                    monthly_platform_show = monthly_platform.rename(columns={'cost_bucket':'platform'}).copy()
                    monthly_platform_show['month'] = pd.to_datetime(monthly_platform_show['month']).dt.strftime('%Y-%m')
                    monthly_company_show = monthly_company.copy()
                    monthly_company_show['month'] = pd.to_datetime(monthly_company_show['month']).dt.strftime('%Y-%m')
                    visuals['cost_validation_available'] = True
                    visuals['cost_validation_cards'] = {
                        'months': int(monthly_company['month'].nunique()),
                        'platforms': int(monthly_platform['cost_bucket'].nunique()),
                        'latest_month': monthly_company_show['month'].iloc[-1] if not monthly_company_show.empty else None,
                        'latest_month_cost': _safe_number(monthly_company['monthly_company_cost'].iloc[-1]) if not monthly_company.empty else None,
                        'latest_daily_cost': _safe_number(monthly_company['company_daily_cost'].iloc[-1]) if not monthly_company.empty else None,
                    }
                    visuals['cost_validation_platform_table'] = _json_ready_records(monthly_platform_show.round(4))
                    visuals['cost_validation_company_table'] = _json_ready_records(monthly_company_show.round(4))

    stats_visuals = _build_statistics_visuals(hist, hist_daily, series_summary)
    visuals['statistics_available'] = stats_visuals.get('available', False)
    visuals['statistics_cards'] = stats_visuals.get('cards', {})
    visuals['statistics_charts'] = stats_visuals.get('charts', [])
    visuals['statistics_lag_table'] = stats_visuals.get('lag_table', [])
    visuals['statistics_frontier_table'] = stats_visuals.get('frontier_table', [])
    visuals['statistics_attrition_table'] = stats_visuals.get('attrition_table', [])

    cohort_visuals = _build_cohort_revenue_visuals(upload_meta, mapping, base_df=hist)
    visuals['cohort_rev_available'] = cohort_visuals.get('available', False)
    visuals['cohort_rev_summary'] = cohort_visuals.get('summary', {})
    visuals['cohort_rev_table'] = cohort_visuals.get('table', [])
    visuals['cohort_rev_platform_table'] = cohort_visuals.get('platform_table', [])
    visuals['cohort_rev_table_grain_label'] = cohort_visuals.get('table_grain_label', 'Platform')
    visuals['cohort_rev_curve_chart'] = cohort_visuals.get('curve_chart')
    visuals['cohort_rev_platform_chart'] = cohort_visuals.get('platform_chart')
    visuals['cohort_rev_heatmap_chart'] = cohort_visuals.get('heatmap_chart')
    visuals['cohort_rev_payback_chart'] = cohort_visuals.get('payback_chart')
    visuals['cohort_rev_payback_table'] = cohort_visuals.get('payback_table', [])
    visuals['cohort_rev_narrative'] = cohort_visuals.get('narrative', [])
    visuals['cohort_rev_roas_table'] = cohort_visuals.get('roas_table', [])
    visuals['cohort_payback_curve'] = cohort_visuals.get('payback_curve', [])
    visuals['cohort_payback_summary'] = cohort_visuals.get('payback_summary', [])

    forecast_df = pd.DataFrame(result.get('display_forecast') or result.get('blended_forecast') or result.get('ensemble_preview') or [])
    if not forecast_df.empty:
        forecast_df = dedupe_columns_keep_first(forecast_df)
        if 'ds' in forecast_df.columns:
            forecast_df['ds'] = pd.to_datetime(first_series(forecast_df, 'ds'), errors='coerce')
        for col in ['yhat', 'expected', 'conservative', 'aggressive', 'lower', 'upper', 'yhat_lower', 'yhat_upper']:
            if col in forecast_df.columns:
                forecast_df[col] = pd.to_numeric(first_series(forecast_df, col), errors='coerce')
        if 'expected' not in forecast_df.columns:
            if 'yhat' in forecast_df.columns:
                forecast_df['expected'] = forecast_df['yhat']
            elif 'mean' in forecast_df.columns:
                forecast_df['expected'] = forecast_df['mean']
        if 'conservative' not in forecast_df.columns:
            if 'lower' in forecast_df.columns:
                forecast_df['conservative'] = forecast_df['lower']
            elif 'yhat_lower' in forecast_df.columns:
                forecast_df['conservative'] = forecast_df['yhat_lower']
            elif 'expected' in forecast_df.columns:
                forecast_df['conservative'] = forecast_df['expected'] * 0.92
        if 'aggressive' not in forecast_df.columns:
            if 'upper' in forecast_df.columns:
                forecast_df['aggressive'] = forecast_df['upper']
            elif 'yhat_upper' in forecast_df.columns:
                forecast_df['aggressive'] = forecast_df['yhat_upper']
            elif 'expected' in forecast_df.columns:
                forecast_df['aggressive'] = forecast_df['expected'] * 1.08
        forecast_df['p10'] = forecast_df['conservative']
        forecast_df['p50'] = forecast_df['expected']
        forecast_df['p90'] = forecast_df['aggressive']
        visuals['forecast_table'] = _json(forecast_df, 80)
    else:
        forecast_df = pd.DataFrame(columns=['ds', 'expected', 'conservative', 'aggressive'])

    visuals['overview_cards'] = {
        'history_rows': int(len(hist_daily)),
        'forecast_rows': int(len(forecast_df)),
        'latest_revenue': _safe_number(hist_daily['y'].iloc[-1]) if not hist_daily.empty else None,
        'forecast_start': forecast_df['ds'].min().strftime('%Y-%m-%d') if not forecast_df.empty and forecast_df['ds'].notna().any() else None,
    }

    expected_total = lower_total = upper_total = band_width_pct = None
    if not forecast_df.empty and 'expected' in forecast_df.columns:
        expected_total = float(pd.to_numeric(forecast_df['expected'], errors='coerce').sum())
        lower_total = float(pd.to_numeric(forecast_df['conservative'], errors='coerce').sum())
        upper_total = float(pd.to_numeric(forecast_df['aggressive'], errors='coerce').sum())
        band_width_pct = ((upper_total - lower_total) / expected_total * 100.0) if expected_total else None
        visuals['overview_cards']['forecast_total'] = expected_total
        visuals['overview_cards']['locked_in_revenue'] = None

    if not forecast_df.empty:
        forecast_avg = float(pd.to_numeric(forecast_df['expected'], errors='coerce').mean() or 0.0)
        lower_avg = float(pd.to_numeric(forecast_df['conservative'], errors='coerce').mean() or 0.0)
        upper_avg = float(pd.to_numeric(forecast_df['aggressive'], errors='coerce').mean() or 0.0)
        current_runrate = float(pd.to_numeric(hist_daily['y'], errors='coerce').tail(30).mean() or 0.0) if not hist_daily.empty else 0.0
        scenario_compare_rows = [
            {'scenario': 'Current run-rate', 'avg_period_revenue': current_runrate, 'total_revenue': current_runrate * max(len(forecast_df), 1), 'vs_expected_pct': _safe_pct(current_runrate, forecast_avg)},
            {'scenario': 'Conservative (P10)', 'avg_period_revenue': lower_avg, 'total_revenue': lower_total, 'vs_expected_pct': _safe_pct(lower_avg, forecast_avg)},
            {'scenario': 'Expected (P50)', 'avg_period_revenue': forecast_avg, 'total_revenue': expected_total, 'vs_expected_pct': 0.0},
            {'scenario': 'Aggressive (P90)', 'avg_period_revenue': upper_avg, 'total_revenue': upper_total, 'vs_expected_pct': _safe_pct(upper_avg, forecast_avg)},
        ]
    elif not hist_daily.empty:
        horizon = max(1, min(12, int(max(round(len(hist_daily.tail(90)) / 30), 1))))
        current_runrate = float(pd.to_numeric(hist_daily['y'], errors='coerce').tail(30).mean() or 0.0)
        total_base = current_runrate * horizon
        scenario_compare_rows = [
            {'scenario': f'Current run-rate ({horizon} mo)', 'avg_period_revenue': current_runrate, 'total_revenue': total_base, 'vs_expected_pct': 0.0},
            {'scenario': 'Conservative', 'avg_period_revenue': current_runrate * 0.92, 'total_revenue': total_base * 0.92, 'vs_expected_pct': -8.0},
            {'scenario': 'Expected', 'avg_period_revenue': current_runrate, 'total_revenue': total_base, 'vs_expected_pct': 0.0},
            {'scenario': 'Aggressive', 'avg_period_revenue': current_runrate * 1.08, 'total_revenue': total_base * 1.08, 'vs_expected_pct': 8.0},
        ]
    if scenario_compare_rows:
        visuals['scenario_compare_table'] = _json_ready_records(pd.DataFrame(scenario_compare_rows).round(4))
        visuals['scenario_compare_chart'] = _plot_card('results-scenario-compare', 'Scenario Comparison Engine', {
            'data': [{
                'type': 'bar',
                'x': [r['scenario'] for r in scenario_compare_rows],
                'y': [round(float(r['total_revenue'] or 0), 2) for r in scenario_compare_rows],
                'hovertemplate': '%{x}<br>Total revenue: $%{y:,.0f}<extra></extra>',
            }],
            'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 45, 'r': 20, 't': 30, 'b': 60}, 'yaxis': {'title': 'Projected total revenue'}}
        }, 'Compares the base forecast against downside, upside, and current run-rate so planning decisions can be made side by side.')

    alerts = []
    root_cause = []
    patterns = []
    optimizer_rows = []
    tracking_rows = []
    driver_rows = []
    time_machine_rows = []
    impact_rows = []
    allocation_rows = []
    scenario_defaults = {}

    # Build 035: keep a revenue-only planning baseline even when leads/cost are not mapped
    try:
        if not scenario_defaults.get('base_revenue') and 'y' in hist_daily.columns:
            _rev_series = pd.to_numeric(hist_daily['y'], errors='coerce').dropna()
            if not _rev_series.empty:
                scenario_defaults['base_revenue'] = float(_rev_series.tail(30).mean())
                scenario_defaults['base_revenue_only'] = True
        if not scenario_defaults.get('base_cost'):
            scenario_defaults['base_cost'] = float(scenario_defaults.get('base_cost') or 0)
        if not scenario_defaults.get('base_leads'):
            scenario_defaults['base_leads'] = float(scenario_defaults.get('base_leads') or 0)
        if not scenario_defaults.get('base_rev_per_lead'):
            if scenario_defaults.get('base_leads', 0) > 0 and scenario_defaults.get('base_revenue', 0) > 0:
                scenario_defaults['base_rev_per_lead'] = float(scenario_defaults['base_revenue']) / max(float(scenario_defaults['base_leads']), 1.0)
            else:
                scenario_defaults['base_rev_per_lead'] = 0.0
        if not scenario_defaults.get('base_roas'):
            if scenario_defaults.get('base_cost', 0) > 0 and scenario_defaults.get('base_revenue', 0) > 0:
                scenario_defaults['base_roas'] = float(scenario_defaults['base_revenue']) / max(float(scenario_defaults['base_cost']), 1.0)
            else:
                scenario_defaults['base_roas'] = 0.0
    except Exception:
        pass
    confidence_score = None
    top_driver_name = None
    monthly_optimizer_profiles = []
    anomaly_rows = []
    rolling_update_rows = []
    scenario_compare_rows = []
    health_rows = []

    if not hist_daily.empty:
        recent_7 = hist_daily.tail(7)
        prior_28 = hist_daily.iloc[:-7].tail(28) if len(hist_daily) > 7 else pd.DataFrame()
        recent_30 = hist_daily.tail(30)
        recent_base_revenue = float(recent_30['y'].mean()) if not recent_30.empty else float(hist_daily['y'].mean())
        recent_target_revenue = float(pd.to_numeric(forecast_df['expected'], errors='coerce').mean()) if not forecast_df.empty else recent_base_revenue

        monthly_hist = hist_daily.copy()
        monthly_hist['month'] = monthly_hist['ds'].dt.to_period('M').dt.to_timestamp()
        monthly_roll = monthly_hist.groupby('month', as_index=False).agg(revenue=('y', 'sum'))
        raw_leads_monthly = _build_mapped_monthly_rollup(upload_meta, mapping, 'leads', 'leads')
        raw_cost_monthly = _build_mapped_monthly_rollup(upload_meta, mapping, 'cost', 'cost')
        if 'leads' in monthly_hist.columns:
            monthly_roll = monthly_roll.merge(monthly_hist.groupby('month', as_index=False)['leads'].sum(), on='month', how='left')
        if not raw_leads_monthly.empty:
            monthly_roll = monthly_roll.drop(columns=['leads'], errors='ignore').merge(raw_leads_monthly, on='month', how='left')
        if 'cost' in monthly_hist.columns:
            monthly_roll = monthly_roll.merge(monthly_hist.groupby('month', as_index=False)['cost'].sum(), on='month', how='left')
        if not raw_cost_monthly.empty:
            monthly_roll = monthly_roll.drop(columns=['cost'], errors='ignore').merge(raw_cost_monthly, on='month', how='left')
        if 'leads' in monthly_roll.columns:
            monthly_roll['rpl'] = monthly_roll['revenue'] / pd.to_numeric(monthly_roll['leads'], errors='coerce').replace(0, pd.NA)
        if 'cost' in monthly_roll.columns:
            monthly_roll['roas'] = monthly_roll['revenue'] / pd.to_numeric(monthly_roll['cost'], errors='coerce').replace(0, pd.NA)
        if {'leads', 'cost'}.issubset(monthly_roll.columns):
            monthly_roll['leads_per_dollar'] = pd.to_numeric(monthly_roll['leads'], errors='coerce') / pd.to_numeric(monthly_roll['cost'], errors='coerce').replace(0, pd.NA)
            monthly_roll['cost_per_lead'] = pd.to_numeric(monthly_roll['cost'], errors='coerce') / pd.to_numeric(monthly_roll['leads'], errors='coerce').replace(0, pd.NA)
        monthly_roll = monthly_roll.sort_values('month').tail(18)
        if not monthly_roll.empty:
            monthly_profiles_show = monthly_roll.copy()
            monthly_profiles_show['month'] = monthly_profiles_show['month'].dt.strftime('%Y-%m')
            monthly_optimizer_profiles = _json_ready_records(monthly_profiles_show.round(4))
            complete_monthly = monthly_roll.copy()
            latest_complete = complete_monthly.sort_values('month').iloc[-1]
            scenario_defaults['planning_month_label'] = pd.to_datetime(latest_complete['month']).strftime('%Y-%m')
            scenario_defaults['planning_monthly_revenue'] = float(pd.to_numeric(pd.Series([latest_complete.get('revenue')]), errors='coerce').iloc[0] or 0.0)
            if 'cost' in latest_complete.index and pd.notna(latest_complete.get('cost')):
                scenario_defaults['planning_monthly_cost'] = float(latest_complete.get('cost') or 0.0)
            if 'leads' in latest_complete.index and pd.notna(latest_complete.get('leads')):
                scenario_defaults['planning_monthly_leads'] = float(latest_complete.get('leads') or 0.0)
            if 'rpl' in latest_complete.index and pd.notna(latest_complete.get('rpl')):
                scenario_defaults['planning_monthly_rpl'] = float(latest_complete.get('rpl') or 0.0)
            elif scenario_defaults.get('planning_monthly_leads'):
                scenario_defaults['planning_monthly_rpl'] = scenario_defaults['planning_monthly_revenue'] / max(float(scenario_defaults['planning_monthly_leads']), 1.0)
            if 'roas' in latest_complete.index and pd.notna(latest_complete.get('roas')):
                scenario_defaults['planning_monthly_roas'] = float(latest_complete.get('roas') or 0.0)
            elif scenario_defaults.get('planning_monthly_cost'):
                scenario_defaults['planning_monthly_roas'] = scenario_defaults['planning_monthly_revenue'] / max(float(scenario_defaults['planning_monthly_cost']), 1.0)

        visuals['planning_dimension_recommendations'] = _build_dimension_budget_recommendations(hist, mapping, upload_meta)
        visuals['planning_whale_watch'] = _build_whale_watch(hist)

        recent_7_avg = float(recent_7['y'].mean()) if not recent_7.empty else None
        prior_28_avg = float(prior_28['y'].mean()) if not prior_28.empty else None
        trend_pct = _safe_pct(recent_7_avg, prior_28_avg)
        if trend_pct is not None:
            direction = 'up' if trend_pct > 0 else 'down'
            root_cause.append(f'Recent 7-day revenue is {abs(trend_pct):.1f}% {direction} versus the prior 28-day baseline.')
            if abs(trend_pct) >= 12:
                alerts.append({'level': 'warning' if trend_pct < 0 else 'info', 'title': 'Revenue shift detected', 'detail': root_cause[-1]})

        for col, label in [('leads', 'Lead volume'), ('cost', 'Spend'), ('roas', 'ROAS'), ('revenue_per_lead', 'Revenue per lead')]:
            if col in hist_daily.columns:
                curr = float(recent_7[col].mean()) if not recent_7.empty else None
                prev = float(prior_28[col].mean()) if not prior_28.empty else None
                pct = _safe_pct(curr, prev)
                tracking_rows.append({'metric': label, 'recent_7_avg': curr, 'prior_28_avg': prev, 'delta_pct': pct})
                if pct is not None and abs(pct) >= 10:
                    root_cause.append(f'{label} moved {pct:+.1f}% over the same comparison window.')

        early_warning_rows = []
        def _push_warning(level, signal, metric, detail, action, severity):
            early_warning_rows.append({
                'level': level,
                'signal': signal,
                'metric': metric,
                'detail': detail,
                'recommended_action': action,
                'severity_score': round(float(severity), 1),
            })

        if trend_pct is not None and trend_pct <= -10:
            _push_warning('warning', 'Revenue downshift', 'Revenue', f'Recent 7-day revenue is {abs(trend_pct):.1f}% below the prior 28-day baseline.', 'Check lead volume, platform mix, and recent site friction first.', min(100, abs(trend_pct) * 2.2))
        if trend_pct is not None and trend_pct >= 12:
            _push_warning('info', 'Revenue acceleration', 'Revenue', f'Recent 7-day revenue is {trend_pct:.1f}% above the prior 28-day baseline.', 'Validate that the lift is real and see which platforms deserve more budget.', min(100, abs(trend_pct) * 1.7))

        tracking_by_metric = {str(r.get('metric')): r for r in tracking_rows}
        lead_delta = tracking_by_metric.get('Lead volume', {}).get('delta_pct')
        if lead_delta is not None and lead_delta <= -12:
            _push_warning('warning', 'Lead volume compression', 'Leads', f'Lead flow is down {abs(lead_delta):.1f}% versus baseline.', 'Investigate traffic source health, caps, and funnel availability before changing monetization targets.', min(100, abs(lead_delta) * 2.0))
        rpl_delta = tracking_by_metric.get('Revenue per lead', {}).get('delta_pct')
        if rpl_delta is not None and rpl_delta <= -8:
            _push_warning('warning', 'Lead quality / monetization softness', 'Revenue per lead', f'Revenue per lead is down {abs(rpl_delta):.1f}% versus baseline.', 'Review cohort payback, platform mix, and pricing / conversion quality.', min(100, abs(rpl_delta) * 2.1))
        roas_delta = tracking_by_metric.get('ROAS', {}).get('delta_pct')
        if roas_delta is not None and roas_delta <= -8:
            _push_warning('warning', 'Efficiency pressure', 'ROAS', f'ROAS is down {abs(roas_delta):.1f}% versus baseline.', 'Tighten budget allocation and redirect spend toward stronger cohorts or platforms.', min(100, abs(roas_delta) * 2.1))
        cost_delta = tracking_by_metric.get('Spend', {}).get('delta_pct')
        if cost_delta is not None and cost_delta >= 12 and trend_pct is not None and trend_pct < 6:
            _push_warning('warning', 'Spend rising faster than revenue', 'Spend', f'Spend is up {cost_delta:.1f}% while revenue changed {trend_pct:.1f}%.', 'Protect budget and require proof of payback before scaling more spend.', min(100, (cost_delta - max(trend_pct, 0)) * 2.0))

        for rec in visuals.get('planning_dimension_recommendations', [])[:4]:
            delta_pct = _safe_number(rec.get('delta_pct'))
            if delta_pct is None or delta_pct > -15:
                continue
            metric = rec.get('dimension_type') or 'Segment'
            name = rec.get('name') or 'Unknown'
            share = _safe_number(rec.get('revenue_share')) or 0.0
            _push_warning(
                'warning',
                f'{metric} pressure: {name}',
                metric,
                f'{metric} {name} is down {abs(delta_pct):.1f}% versus the prior window and still carries about {share:.1f}% of recent revenue.',
                'Reduce budget, inspect conversion quality, and confirm whether seasonality or creative fatigue is driving the drop.',
                min(100, abs(delta_pct) * (1.4 if share >= 10 else 1.0))
            )

        for whale in visuals.get('planning_whale_watch', [])[:3]:
            delta_pct = _safe_number(whale.get('delta_pct'))
            if delta_pct is None or delta_pct > -20:
                continue
            _push_warning(
                'warning',
                f'User softening: {whale.get("whale_id")}',
                'Users',
                f'User {whale.get("whale_id")} is down {abs(delta_pct):.1f}% versus the prior 30-day window.',
                'Protect revenue by running retention or reactivation actions before pushing broader budget increases.',
                min(100, abs(delta_pct) * 1.8)
            )

        if early_warning_rows:
            warn_df = pd.DataFrame(early_warning_rows).sort_values(['severity_score', 'level'], ascending=[False, True])
            visuals['early_warning_table'] = _json_ready_records(warn_df.head(12))
            visuals['charts'].append(_plot_card('results-early-warnings', 'Early Warning Monitor', {
                'data': [{
                    'type': 'bar',
                    'x': warn_df['signal'].astype(str).tolist()[:8],
                    'y': pd.to_numeric(warn_df['severity_score'], errors='coerce').round(1).tolist()[:8],
                    'text': warn_df['metric'].astype(str).tolist()[:8],
                    'hovertemplate': '%{x}<br>Severity: %{y:.1f}<br>Metric: %{text}<extra></extra>',
                }],
                'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 40, 'r': 20, 't': 30, 'b': 80}, 'yaxis': {'title': 'Severity score'}}
            }, 'Flags the main conditions that deserve attention now so the team can act before the forecast drifts further.'))

        cohort_temp, _cohort_cfg = _extract_cohort_longframe(upload_meta, mapping) if upload_meta and mapping else (None, {})
        if cohort_temp is not None and not cohort_temp.empty:
            monthly_cohort = cohort_temp.groupby('transaction_month', as_index=False).agg(
                total_revenue=('revenue', 'sum'),
                new_cohort_revenue=('revenue', lambda s: float(s[cohort_temp.loc[s.index, 'cohort_age_month'] == 0].sum())),
                matured_revenue=('revenue', lambda s: float(s[cohort_temp.loc[s.index, 'cohort_age_month'] > 0].sum())),
            ).sort_values('transaction_month').reset_index(drop=True)
            if len(monthly_cohort) >= 2:
                recent_month = monthly_cohort.tail(1).copy()
                prior_months = monthly_cohort.iloc[max(0, len(monthly_cohort) - 4):-1].copy()
                if not prior_months.empty:
                    start_rev = float(prior_months['total_revenue'].mean() or 0.0)
                    end_rev = float(recent_month['total_revenue'].iloc[0] or 0.0)
                    lead_impact = float(recent_month['new_cohort_revenue'].iloc[0] - prior_months['new_cohort_revenue'].mean())
                    maturity_impact = float(recent_month['matured_revenue'].iloc[0] - prior_months['matured_revenue'].mean())
                    mix_impact = 0.0
                    if 'cohort_category' in cohort_temp.columns and cohort_temp['cohort_category'].nunique(dropna=True) > 1:
                        recent_txn = cohort_temp[cohort_temp['transaction_month'].isin(recent_month['transaction_month'])].groupby('cohort_category', as_index=False)['revenue'].sum()
                        prior_txn = cohort_temp[cohort_temp['transaction_month'].isin(prior_months['transaction_month'])].groupby('cohort_category', as_index=False)['revenue'].mean().rename(columns={'revenue':'prior_avg_revenue'})
                        mix_df = recent_txn.merge(prior_txn, on='cohort_category', how='outer').fillna(0)
                        total_recent = float(mix_df['revenue'].sum() or 0.0)
                        total_prior = float(mix_df['prior_avg_revenue'].sum() or 0.0)
                        if total_recent > 0 and total_prior > 0:
                            mix_df['recent_share'] = mix_df['revenue'] / total_recent
                            mix_df['prior_share'] = mix_df['prior_avg_revenue'] / total_prior
                            mix_impact = float(((mix_df['recent_share'] - mix_df['prior_share']) * total_recent).sum())
                    residual = end_rev - start_rev - lead_impact - maturity_impact - mix_impact
                    wf_rows = pd.DataFrame([
                        {'component': 'Starting revenue', 'impact': start_rev, 'kind': 'anchor'},
                        {'component': 'New cohort impact', 'impact': lead_impact, 'kind': 'delta'},
                        {'component': 'Older cohort maturation', 'impact': maturity_impact, 'kind': 'delta'},
                        {'component': 'Platform mix impact', 'impact': mix_impact, 'kind': 'delta'},
                        {'component': 'Other / unexplained', 'impact': residual, 'kind': 'delta'},
                        {'component': 'Current revenue', 'impact': end_rev, 'kind': 'anchor'},
                    ])
                    visuals['waterfall_table'] = _json_ready_records(wf_rows.round(4))
                    visuals['charts'].append(_plot_card('results-waterfall', 'Revenue Change Waterfall', {
                        'data': [{
                            'type': 'waterfall',
                            'x': wf_rows['component'].tolist(),
                            'y': wf_rows['impact'].round(2).tolist(),
                            'measure': ['absolute', 'relative', 'relative', 'relative', 'relative', 'total'],
                            'text': [f"${v:,.0f}" for v in wf_rows['impact'].tolist()],
                            'textposition': 'outside',
                            'connector': {'line': {'color': 'rgba(255,255,255,0.25)'}},
                        }],
                        'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 50, 'r': 20, 't': 30, 'b': 80}, 'yaxis': {'title': 'Avg revenue per month'}}
                    }, 'Baseline revenue uses the most recent 30-day average before the forecast window, then shows how lead volume, monetization quality, and mix shift the outcome from that baseline.'))
        elif 'leads' in hist_daily.columns and 'revenue_per_lead' in hist_daily.columns:
            prev_leads = float(prior_28['leads'].mean()) if not prior_28.empty else None
            curr_leads = float(recent_7['leads'].mean()) if not recent_7.empty else None
            prev_rpl = float(prior_28['revenue_per_lead'].mean()) if not prior_28.empty else None
            curr_rpl = float(recent_7['revenue_per_lead'].mean()) if not recent_7.empty else None
            if prev_leads is not None and curr_leads is not None and prev_rpl is not None and curr_rpl is not None:
                start_rev = float(prior_28_avg or 0.0)
                end_rev = float(recent_7_avg or 0.0)
                lead_impact = (curr_leads - prev_leads) * prev_rpl
                quality_impact = curr_leads * (curr_rpl - prev_rpl)
                mix_impact = 0.0
                if not series_summary.empty and {'category', 'recent_revenue', 'recent_cost', 'share_of_revenue'}.issubset(series_summary.columns):
                    s = series_summary.copy()
                    s['overall_rev_share'] = s['share_of_revenue'].fillna(0)
                    total_recent_revenue = float(pd.to_numeric(s['recent_revenue'], errors='coerce').fillna(0).sum() or 0.0)
                    if total_recent_revenue > 0 and 'recent_roas' in s.columns and s['recent_roas'].notna().any():
                        s['recent_share'] = pd.to_numeric(s['recent_revenue'], errors='coerce').fillna(0) / total_recent_revenue
                        ref = float(pd.to_numeric(s['recent_roas'], errors='coerce').dropna().median() or 0.0)
                        mix_impact = float((((s['recent_share'] - s['overall_rev_share']) * (pd.to_numeric(s['recent_roas'], errors='coerce').fillna(ref) - ref)).fillna(0)).sum() * max(start_rev, 0))
                residual = end_rev - start_rev - lead_impact - quality_impact - mix_impact
                wf_rows = pd.DataFrame([
                    {'component': 'Starting revenue', 'impact': start_rev, 'kind': 'anchor'},
                    {'component': 'Lead volume impact', 'impact': lead_impact, 'kind': 'delta'},
                    {'component': 'Revenue per lead impact', 'impact': quality_impact, 'kind': 'delta'},
                    {'component': 'Platform mix impact', 'impact': mix_impact, 'kind': 'delta'},
                    {'component': 'Other / unexplained', 'impact': residual, 'kind': 'delta'},
                    {'component': 'Current revenue', 'impact': end_rev, 'kind': 'anchor'},
                ])
                visuals['waterfall_table'] = _json_ready_records(wf_rows.round(4))
                visuals['waterfall_explanation'] = _build_waterfall_explanation(wf_rows)
                visuals['charts'].append(_plot_card('results-waterfall', 'Revenue Change Waterfall', {
                    'data': [{
                        'type': 'waterfall',
                        'x': wf_rows['component'].tolist(),
                        'y': wf_rows['impact'].round(2).tolist(),
                        'measure': ['absolute', 'relative', 'relative', 'relative', 'relative', 'total'],
                        'text': [f"${v:,.0f}" for v in wf_rows['impact'].tolist()],
                        'textposition': 'outside',
                        'connector': {'line': {'color': 'rgba(255,255,255,0.25)'}},
                    }],
                    'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 50, 'r': 20, 't': 30, 'b': 80}, 'yaxis': {'title': 'Avg revenue per period'}}
                }, 'Baseline revenue uses the most recent 30-day average before the forecast window. The bridge then shows how lead volume, monetization quality, mix, and residual effects move you away from that starting point.'))

        win = min(28, max(5, len(hist_daily)))
        min_p = 3 if len(hist_daily) < 10 else min(7, win)
        rolling_mean = hist_daily['y'].rolling(win, min_periods=min_p).mean().shift(1)
        rolling_std = hist_daily['y'].rolling(win, min_periods=min_p).std(ddof=0).shift(1)
        if rolling_mean.isna().all():
            rolling_mean = hist_daily['y'].expanding(min_periods=2).mean().shift(1)
            rolling_std = hist_daily['y'].expanding(min_periods=2).std(ddof=0).shift(1)
        z = (hist_daily['y'] - rolling_mean) / rolling_std.replace(0, pd.NA)
        anomaly_all = hist_daily.assign(rolling_mean=rolling_mean, zscore=z).dropna(subset=['rolling_mean'])
        anomaly_hits = anomaly_all[anomaly_all['zscore'].abs() >= 2.0].tail(6)
        for _, row in anomaly_hits.iterrows():
            level = 'warning' if row['zscore'] < 0 else 'info'
            alerts.append({
                'level': level,
                'title': f"Anomaly on {row['ds'].strftime('%Y-%m-%d')}",
                'detail': f"Revenue printed {row['zscore']:+.2f} standard deviations from its recent 28-period trend."
            })
            anomaly_rows.append({
                'date': row['ds'].strftime('%Y-%m-%d'),
                'revenue': float(row['y']),
                'zscore': float(row['zscore']),
                'severity': 'High' if abs(float(row['zscore'])) >= 3 else 'Medium',
                'direction': 'Below trend' if float(row['zscore']) < 0 else 'Above trend',
            })
        anomaly_chart_df = anomaly_all.tail(90).copy()
        if not anomaly_chart_df.empty:
            anomaly_chart_df['flagged'] = anomaly_chart_df['zscore'].abs() >= 2.0
            visuals['anomaly_chart'] = _plot_card('results-anomaly-monitor', 'Revenue Anomaly Monitor', {
                'data': [
                    {
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': 'Revenue',
                        'x': anomaly_chart_df['ds'].dt.strftime('%Y-%m-%d').tolist(),
                        'y': anomaly_chart_df['y'].round(2).tolist(),
                        'hovertemplate': '%{x}<br>Revenue: $%{y:,.0f}<extra></extra>',
                    },
                    {
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': 'Rolling baseline',
                        'x': anomaly_chart_df['ds'].dt.strftime('%Y-%m-%d').tolist(),
                        'y': anomaly_chart_df['rolling_mean'].round(2).astype(object).where(anomaly_chart_df['rolling_mean'].notna(), None).tolist(),
                        'line': {'dash': 'dot'},
                        'hovertemplate': '%{x}<br>Baseline: $%{y:,.0f}<extra></extra>',
                    },
                    {
                        'type': 'scatter',
                        'mode': 'markers',
                        'name': 'Flagged anomalies',
                        'x': anomaly_chart_df.loc[anomaly_chart_df['flagged'], 'ds'].dt.strftime('%Y-%m-%d').tolist(),
                        'y': anomaly_chart_df.loc[anomaly_chart_df['flagged'], 'y'].round(2).tolist(),
                        'customdata': anomaly_chart_df.loc[anomaly_chart_df['flagged'], 'zscore'].round(2).tolist(),
                        'hovertemplate': '%{x}<br>Revenue: $%{y:,.0f}<br>Z-score: %{customdata:.2f}<extra></extra>',
                        'marker': {'size': 9, 'symbol': 'diamond'},
                    },
                ],
                'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 45, 'r': 20, 't': 30, 'b': 60}, 'yaxis': {'title': 'Revenue'}}
            }, 'Shows the last 90 periods of revenue against its rolling baseline and highlights any flagged anomalies.')
        if anomaly_rows:
            anomaly_df2 = pd.DataFrame(anomaly_rows)
            visuals['anomaly_table'] = _json_ready_records(anomaly_df2.round(4))
        elif not anomaly_chart_df.empty:
            anomaly_recent = anomaly_chart_df.tail(12).copy()
            anomaly_recent['severity'] = anomaly_recent['zscore'].abs().apply(lambda v: 'High' if pd.notna(v) and abs(float(v)) >= 3 else 'Normal')
            anomaly_recent['direction'] = anomaly_recent['zscore'].apply(lambda v: 'Below trend' if pd.notna(v) and float(v) < 0 else 'Above / inline')
            anomaly_recent_df = pd.DataFrame({
                'date': anomaly_recent['ds'].dt.strftime('%Y-%m-%d'),
                'revenue': anomaly_recent['y'].round(2),
                'zscore': pd.to_numeric(anomaly_recent['zscore'], errors='coerce').round(3),
                'severity': anomaly_recent['severity'],
                'direction': anomaly_recent['direction'],
            }).fillna('')
            visuals['anomaly_table'] = _json_ready_records(anomaly_recent_df)

        if 'leads' in hist_daily.columns:
            lead_mean = float(recent_30['leads'].mean()) if not recent_30.empty else 0.0
            scenario_defaults['base_leads'] = lead_mean
            if 'cost' in hist_daily.columns:
                cost_mean = float(recent_30['cost'].mean()) if not recent_30.empty else 0.0
                scenario_defaults['base_cost'] = cost_mean
                scenario_defaults['base_cpl'] = (cost_mean / lead_mean) if lead_mean else None
            if 'leads' in recent_30.columns:
                recent_rev = pd.to_numeric(recent_30['y'], errors='coerce').clip(lower=0).sum()
                recent_leads = pd.to_numeric(recent_30['leads'], errors='coerce').clip(lower=0).sum()
                rpl = _safe_divide(recent_rev, recent_leads, default=0.0)
                rpl = rpl if rpl > 0 else None
                scenario_defaults['base_rev_per_lead'] = rpl
                locked_in_30 = lead_mean * 30 * (rpl or 0) * 0.65
                locked_in_60 = lead_mean * 60 * (rpl or 0) * 0.90
                visuals['overview_cards']['locked_in_revenue'] = locked_in_60
                root_cause.append(
                    f'At the current lead quality, roughly ${locked_in_30:,.0f} of the next 30 days and ${locked_in_60:,.0f} of the next 60 days already looks supported by recent lead flow.'
                )

        if 'cost' in hist_daily.columns:
            cost_sum_recent = pd.to_numeric(recent_30['cost'], errors='coerce').replace(0, pd.NA).sum()
            cost_sum_all = pd.to_numeric(hist_daily['cost'], errors='coerce').replace(0, pd.NA).sum()
            recent_roas = float(recent_30['y'].sum() / cost_sum_recent) if cost_sum_recent else None
            overall_roas = float(hist_daily['y'].sum() / cost_sum_all) if cost_sum_all else None
            if recent_roas is not None:
                scenario_defaults['current_roas'] = recent_roas
            if overall_roas is not None:
                scenario_defaults['realized_roas'] = overall_roas
            avg_daily_cost = float(recent_30['cost'].mean()) if not recent_30.empty else 0.0
            action = 'Hold'
            if recent_roas is not None and overall_roas is not None:
                if recent_roas > overall_roas * 1.05:
                    action = 'Increase budget 10%'
                elif recent_roas < overall_roas * 0.95:
                    action = 'Trim budget 10%'
            optimizer_rows.append({
                'segment': 'Overall portfolio',
                'recent_roas': recent_roas,
                'overall_roas': overall_roas,
                'avg_daily_spend': avg_daily_cost,
                'suggested_monthly_budget': avg_daily_cost * 30 * (1.10 if 'Increase' in action else 0.90 if 'Trim' in action else 1.0),
                'action': action,
            })

        # Driver sensitivity / impact analysis
        impact_df = _build_driver_impact_frame(hist_daily, allowed_cols=allowed_driver_cols)
        if not impact_df.empty:
            impact_rows = impact_df.to_dict(orient='records')
            visuals['impact_table'] = _json(impact_df, 20)
            top_impact = impact_df.iloc[0]
            confidence_label = str(top_impact.get('evidence_strength') or 'medium')
            root_cause.append(
                f"{top_impact['driver']} currently shows the strongest modeled pull on revenue. A 10% move in that driver maps to about ${float(top_impact['revenue_change_if_driver_moves_10pct']):,.0f} in average-period revenue, with {confidence_label} confidence from recent history."
            )
            visuals['charts'].append(_plot_card('results-driver-impact', 'Revenue Driver Impact', {
                'data': [
                    {
                        'type': 'bar',
                        'x': impact_df['driver'].astype(str).tolist()[:10],
                        'y': impact_df['revenue_change_if_driver_moves_10pct'].round(2).tolist()[:10],
                        'name': '10% driver move impact',
                        'customdata': impact_df['confidence_score'].round(2).tolist()[:10],
                        'hovertemplate': '%{x}<br>Revenue impact: $%{y:,.0f}<br>Confidence: %{customdata:.2f}<extra></extra>',
                    }
                ],
                'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 50, 'r': 20, 't': 30, 'b': 80}}
            }, 'Shows the approximate revenue lift or drag from changing each true business driver by 10%, ranked by signal strength and confidence.'))

        weekday = hist_daily.copy()
        weekday['weekday'] = weekday['ds'].dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday['weekday'] = pd.Categorical(weekday['weekday'], categories=weekday_order, ordered=True)
        weekday_summary = weekday.groupby('weekday', observed=False)['y'].mean().dropna().sort_values(ascending=False)
        if not weekday_summary.empty:
            patterns.append(
                f'Best weekday: {weekday_summary.index[0]} averages ${weekday_summary.iloc[0]:,.0f}; weakest weekday: {weekday_summary.index[-1]} averages ${weekday_summary.iloc[-1]:,.0f}.'
            )
            visuals['charts'].append(_plot_card('results-weekday-pattern', 'Weekday Revenue Pattern', {
                'data': [{'type': 'bar', 'x': weekday_summary.index.astype(str).tolist(), 'y': weekday_summary.round(2).tolist(), 'name': 'Avg revenue'}],
                'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 40, 'r': 20, 't': 30, 'b': 60}}
            }, 'A quick pattern-detection view showing which weekdays tend to produce the strongest revenue.'))

        monthly = hist_daily.copy()
        monthly['month'] = monthly['ds'].dt.to_period('M').dt.to_timestamp()
        month_table = monthly.groupby('month', as_index=False).agg(revenue=('y', 'sum'))
        if 'leads' in monthly.columns:
            month_table['leads'] = monthly.groupby('month')['leads'].sum().values
        if 'cost' in monthly.columns:
            month_table['cost'] = monthly.groupby('month')['cost'].sum().values
            month_table['roas'] = month_table['revenue'] / pd.to_numeric(month_table['cost'], errors='coerce').replace(0, pd.NA)
        month_table['mom_pct'] = month_table['revenue'].pct_change() * 100.0
        time_machine_rows = _json(month_table.tail(12), 12)
        if not month_table.empty:
            best_month = month_table.sort_values('revenue', ascending=False).iloc[0]
            patterns.append(f"Strongest month in history: {pd.to_datetime(best_month['month']).strftime('%Y-%m')} with ${best_month['revenue']:,.0f}.")

    run_settings = result.get('settings', {}) if isinstance(result, dict) else {}
    forecast_freq_code = normalize_frequency_code(run_settings, hist_daily, forecast_df)
    if not forecast_df.empty:
        hist_for_chart, forecast_for_chart = prepare_risk_band_display(hist_daily, forecast_df, forecast_freq_code)
        trend_data = []
        if not hist_for_chart.empty:
            trend_data.append({
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Historical Revenue',
                'x': format_plot_dates(hist_for_chart['ds'], forecast_freq_code),
                'y': hist_for_chart['y'].round(2).tolist(),
                'line': {'color': '#6f8fff', 'width': 2.4},
                'hovertemplate': '%{x}<br>Historical revenue: $%{y:,.0f}<extra></extra>',
            })
        trend_data.append({
            'type': 'scatter',
            'mode': 'lines',
            'name': 'P90 / Aggressive',
            'x': format_plot_dates(forecast_for_chart['ds'], forecast_freq_code),
            'y': forecast_for_chart['aggressive'].round(2).astype(object).where(forecast_for_chart['aggressive'].notna(), None).tolist(),
            'line': {'dash': 'dot', 'color': '#f28cc8', 'width': 1.8},
            'hovertemplate': '%{x}<br>P90: $%{y:,.0f}<extra></extra>',
        })
        trend_data.append({
            'type': 'scatter',
            'mode': 'lines',
            'name': 'P10 / Conservative',
            'x': format_plot_dates(forecast_for_chart['ds'], forecast_freq_code),
            'y': forecast_for_chart['conservative'].round(2).astype(object).where(forecast_for_chart['conservative'].notna(), None).tolist(),
            'line': {'dash': 'dot', 'color': '#7aa8ff', 'width': 1.8},
            'fill': 'tonexty',
            'fillcolor': 'rgba(122, 168, 255, 0.14)',
            'hovertemplate': '%{x}<br>P10: $%{y:,.0f}<extra></extra>',
        })
        trend_data.append({
            'type': 'scatter',
            'mode': 'lines',
            'name': 'P50 / Expected',
            'x': format_plot_dates(forecast_for_chart['ds'], forecast_freq_code),
            'y': forecast_for_chart['expected'].round(2).astype(object).where(forecast_for_chart['expected'].notna(), None).tolist(),
            'line': {'color': '#5f78ff', 'width': 3},
            'hovertemplate': '%{x}<br>P50: $%{y:,.0f}<extra></extra>',
        })
        risk_band_axis_title = {'D': 'Date', 'W': 'Week', 'M': 'Month'}.get(forecast_freq_code, 'Date')
        risk_band_cadence_label = {'D': 'daily', 'W': 'weekly', 'M': 'monthly'}.get(forecast_freq_code, 'daily')
        visuals['charts'].append(_plot_card('results-main-forecast', 'Historical + Forecast Risk Bands', {
            'data': trend_data,
            'layout': {
                'paper_bgcolor': 'transparent',
                'plot_bgcolor': 'transparent',
                'font': {'color': '#edf1f7'},
                'margin': {'l': 46, 'r': 20, 't': 30, 'b': 52},
                'hovermode': 'x unified',
                'xaxis': {'title': risk_band_axis_title},
                'yaxis': {'title': 'Revenue'},
            }
        }, f'This view upgrades the forecast to a P10 / P50 / P90 style range so you can plan against downside and upside risk, not just one line. It automatically rolls the history and forecast to the run cadence ({risk_band_cadence_label}).'))

    ranking = pd.DataFrame(result.get('top_models') or [])
    if not ranking.empty:
        visuals['model_ranking'] = _json(ranking, 12)
        model_names, scores = [], []
        for _, row in ranking.head(8).iterrows():
            model_names.append(row.get('model') or row.get('model_name') or 'model')
            scores.append(_safe_number(row.get('final_rank_score') or row.get('final_score') or row.get('score')))
        visuals['charts'].append(_plot_card('results-ranking', 'Top Model Scores', {
            'data': [{'type': 'bar', 'x': model_names, 'y': scores, 'name': 'Score'}],
            'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 40, 'r': 20, 't': 30, 'b': 60}}
        }, 'Compare the model competition and see which models ranked highest.'))

    raw_fi = ((result.get('best_model') or {}).get('feature_importance') or {})
    fi = raw_fi if isinstance(raw_fi, dict) else {}
    if fi:
        items = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
        total_importance = sum(float(v) for _, v in items) or 1.0
        explain_rows = []
        for k, v in items:
            share_pct = (float(v) / total_importance) * 100.0
            explain_rows.append({'feature': k, 'importance': float(v), 'share_pct': share_pct})
            driver_rows.append({'driver': k, 'importance': float(v), 'share_pct': share_pct})
        visuals['drivers_table'] = explain_rows
        visuals['charts'].append(_plot_card('results-features', 'Feature Importance Share', {
            'data': [{'type': 'bar', 'orientation': 'h', 'x': [round(r['share_pct'], 2) for r in explain_rows][::-1], 'y': [r['feature'] for r in explain_rows][::-1], 'name': 'Share %'}],
            'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 180, 'r': 20, 't': 30, 'b': 40}}
        }, "This normalizes the winning model's feature importance into easy-to-read shares, so the forecast becomes more explainable."))

    diagnostics = result.get('diagnostics', {}) if isinstance(result, dict) else {}
    diagnostics = dict(diagnostics)
    diag_rows = []
    period_unit = _diagnostic_frequency_label(run_settings, hist_daily)
    history_window_label = _format_history_window_label(hist_daily)
    monthly_factor = _monthly_equivalent_factor(period_unit)
    raw_leads_daily = _build_mapped_daily_rollup(upload_meta, mapping, 'leads', 'leads')
    if not raw_leads_daily.empty:
        lead_values = pd.to_numeric(raw_leads_daily['leads'], errors='coerce').dropna()
        if not lead_values.empty:
            diagnostics['avg_leads'] = float(lead_values.mean())
            diagnostics['max_leads'] = float(lead_values.max())
    diagnostics_meta = {
        'period_unit': period_unit,
        'history_window_label': history_window_label,
        'history_basis_label': f'Full aligned history, per {period_unit}',
        'recent_basis_label': f'Last 30 historical {period_unit}s',
        'monthly_factor': monthly_factor,
    }
    diag_label_map = {
        'recent_avg_y': f'Recent Avg Revenue (last 30 {period_unit}s)',
        'recent_median_y': f'Recent Median Revenue (last 30 {period_unit}s)',
        'recent_max_y': f'Recent Max Revenue (last 30 {period_unit}s)',
        'forecast_avg_expected': f'Forecast Avg Revenue (per forecast {period_unit})',
        'forecast_to_history_ratio': 'Forecast / Recent History Ratio',
        'avg_leads': f'Avg Leads (daily total from Leads sheet)',
        'max_leads': f'Max Leads (single-day total from Leads sheet)',
        'avg_cost': f'Avg Cost (full history, per {period_unit})',
        'max_cost': f'Max Cost (full history, per {period_unit})',
        'duplicate_ds_rows': 'Duplicate Date Rows',
    }
    diag_monthly_label_map = {
        'recent_avg_y': 'Recent Avg Revenue (monthly equivalent)',
        'recent_median_y': 'Recent Median Revenue (monthly equivalent)',
        'recent_max_y': 'Recent Max Revenue (monthly equivalent)',
        'forecast_avg_expected': 'Forecast Avg Revenue (monthly equivalent)',
        'forecast_to_history_ratio': 'Forecast / Recent History Ratio',
        'avg_leads': 'Avg Leads (monthly equivalent)',
        'max_leads': 'Max Leads (monthly equivalent)',
        'avg_cost': 'Avg Cost (monthly equivalent)',
        'max_cost': 'Max Cost (monthly equivalent)',
        'duplicate_ds_rows': 'Duplicate Date Rows',
    }
    diag_tooltip_map = {
        'recent_avg_y': f'Average revenue across the last 30 historical {period_unit}s in the aligned modeling history.',
        'recent_median_y': f'Median revenue across the last 30 historical {period_unit}s in the aligned modeling history.',
        'recent_max_y': f'Highest revenue observed in the last 30 historical {period_unit}s in the aligned modeling history.',
        'forecast_avg_expected': f'Average predicted revenue per forecast {period_unit}.',
        'forecast_to_history_ratio': 'Average forecast scale divided by recent historical revenue scale.',
        'avg_leads': 'Average of daily lead totals from the mapped Leads sheet. Daily lead rows are summed by date first, then averaged.',
        'max_leads': 'Highest single-day lead total from the mapped Leads sheet after summing all lead rows by date.',
        'avg_cost': f'Average cost across the full aligned history, measured per historical {period_unit}.',
        'max_cost': f'Highest cost observed in a single historical {period_unit} across the full aligned history.',
        'duplicate_ds_rows': 'Count of duplicate date rows detected after alignment.',
    }
    scalable_keys = {'recent_avg_y', 'recent_median_y', 'recent_max_y', 'forecast_avg_expected', 'avg_leads', 'max_leads', 'avg_cost', 'max_cost'}
    for key in ['recent_avg_y', 'recent_median_y', 'recent_max_y', 'forecast_avg_expected', 'forecast_to_history_ratio', 'avg_leads', 'max_leads', 'avg_cost', 'max_cost', 'duplicate_ds_rows']:
        if key in diagnostics and diagnostics.get(key) is not None:
            try:
                value = float(diagnostics.get(key))
                monthly_value = float(value * monthly_factor) if key in scalable_keys else float(value)
                diag_rows.append({
                    'metric': diag_label_map.get(key, key.replace('_', ' ').title()),
                    'monthly_metric': diag_monthly_label_map.get(key, key.replace('_', ' ').title()),
                    'value': value,
                    'monthly_value': monthly_value,
                    'tooltip': diag_tooltip_map.get(key, ''),
                    'raw_key': key,
                    'scalable': key in scalable_keys,
                })
            except Exception:
                pass
    if diag_rows:
        primary_diag_rows = [r for r in diag_rows if r.get('raw_key') != 'forecast_to_history_ratio']
        ratio_diag_rows = [r for r in diag_rows if r.get('raw_key') == 'forecast_to_history_ratio']
        diagnostic_traces = [{
            'type': 'bar',
            'x': [r['metric'] for r in primary_diag_rows],
            'y': [float(r['value']) for r in primary_diag_rows],
            'name': 'Value metrics'
        }]
        if ratio_diag_rows:
            diagnostic_traces.append({
                'type': 'scatter',
                'mode': 'markers+text',
                'x': [r['metric'] for r in ratio_diag_rows],
                'y': [float(r['value']) for r in ratio_diag_rows],
                'name': 'Forecast / recent ratio',
                'yaxis': 'y2',
                'text': [f"{float(r['value']):.2f}x" for r in ratio_diag_rows],
                'textposition': 'top center',
                'marker': {'size': 12}
            })
        visuals['diagnostics_rows'] = diag_rows
        visuals['diagnostics_meta'] = diagnostics_meta
        visuals['charts'].append(_plot_card('results-diagnostics', 'Forecast Diagnostics', {
            'data': diagnostic_traces,
            'layout': {
                'paper_bgcolor': 'transparent',
                'plot_bgcolor': 'transparent',
                'font': {'color': '#edf1f7'},
                'margin': {'l': 40, 'r': 70, 't': 30, 'b': 170},
                'yaxis': {'title': 'Value metrics'},
                'yaxis2': {'title': 'Ratio', 'overlaying': 'y', 'side': 'right', 'showgrid': False, 'rangemode': 'tozero'}
            }
        }, f'History window: {history_window_label}. Recent revenue averages use the last 30 historical {period_unit}s. Lead averages use daily totals from the mapped Leads sheet. Cost averages use the full aligned history.'))

    if not hist_daily.empty and {'y', 'leads'}.issubset(hist_daily.columns):
        combo = hist_daily[['ds', 'y', 'leads']].copy()
        visuals['charts'].append(_plot_card('results-rev-leads', 'Revenue vs Leads', {
            'data': [
                {'type': 'scatter', 'mode': 'lines', 'name': 'Revenue', 'x': combo['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': combo['y'].round(2).tolist(), 'yaxis': 'y'},
                {'type': 'bar', 'name': 'Leads', 'x': combo['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': combo['leads'].fillna(0).round(2).tolist(), 'yaxis': 'y2', 'opacity': 0.35},
            ],
            'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 40, 'r': 40, 't': 30, 'b': 40}, 'yaxis2': {'overlaying': 'y', 'side': 'right'}}
        }, 'Use this to visually compare lead volume against realized revenue.'))

    if not hist_daily.empty:
        rolling = hist_daily[['ds', 'y']].copy()
        rolling['roll7'] = rolling['y'].rolling(7, min_periods=1).mean()
        rolling['roll30'] = rolling['y'].rolling(30, min_periods=1).mean()
        visuals['charts'].append(_plot_card('results-rolling', 'Revenue Smoothing', {
            'data': [
                {'type': 'scatter', 'mode': 'lines', 'name': 'Daily Revenue', 'x': rolling['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': rolling['y'].round(2).tolist(), 'line': {'width': 1}},
                {'type': 'scatter', 'mode': 'lines', 'name': '7-day Avg', 'x': rolling['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': rolling['roll7'].round(2).tolist()},
                {'type': 'scatter', 'mode': 'lines', 'name': '30-day Avg', 'x': rolling['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': rolling['roll30'].round(2).tolist()},
            ],
            'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 40, 'r': 20, 't': 30, 'b': 40}}
        }, 'Smooth the raw series so you can inspect trend direction more easily.'))

        monthly = hist_daily.copy()
        monthly['month'] = monthly['ds'].dt.to_period('M').dt.to_timestamp()
        monthly_totals = monthly.groupby('month', as_index=False)['y'].sum()
        if not monthly_totals.empty:
            monthly_totals['prev_month'] = monthly_totals['y'].shift(1)
            monthly_totals['mom_pct'] = monthly_totals['y'].pct_change() * 100.0
            forecast_monthly = pd.DataFrame()
            if not forecast_df.empty and 'ds' in forecast_df.columns and 'expected' in forecast_df.columns:
                forecast_monthly = forecast_df.copy()
                forecast_monthly['month'] = forecast_monthly['ds'].dt.to_period('M').dt.to_timestamp()
                forecast_monthly = forecast_monthly.groupby('month', as_index=False)['expected'].sum().rename(columns={'expected': 'forecast_revenue'})
            rolling_update = monthly_totals.tail(12).copy()
            if not forecast_monthly.empty:
                rolling_update = rolling_update.merge(forecast_monthly, on='month', how='left')
            else:
                rolling_update['forecast_revenue'] = pd.NA
            rolling_update_rows = _json_ready_records(rolling_update.round(4))
            visuals['rolling_update_table'] = rolling_update_rows
            rolling_data = [
                    {'type': 'bar', 'name': 'Actual monthly revenue', 'x': rolling_update['month'].dt.strftime('%Y-%m').tolist(), 'y': rolling_update['y'].round(2).tolist()},
            ]
            forecast_vals = pd.to_numeric(rolling_update['forecast_revenue'], errors='coerce')
            if forecast_vals.notna().any():
                rolling_data.append({'type': 'scatter', 'mode': 'lines+markers', 'name': 'Forecast monthly revenue', 'x': rolling_update['month'].dt.strftime('%Y-%m').tolist(), 'y': forecast_vals.round(2).astype(object).where(forecast_vals.notna(), None).tolist()})
            if 'prev_month' in rolling_update.columns and rolling_update['prev_month'].notna().any():
                prev_vals = pd.to_numeric(rolling_update['prev_month'], errors='coerce')
                rolling_data.append({'type': 'scatter', 'mode': 'lines', 'name': 'Previous month baseline', 'x': rolling_update['month'].dt.strftime('%Y-%m').tolist(), 'y': prev_vals.round(2).astype(object).where(prev_vals.notna(), None).tolist(), 'line': {'dash': 'dot'}})
            visuals['rolling_update_chart'] = _plot_card('results-rolling-updates', 'Rolling Forecast Updates', {
                'data': rolling_data,
                'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 45, 'r': 20, 't': 30, 'b': 60}, 'yaxis': {'title': 'Revenue'}}
            }, 'Tracks recent monthly actuals against forecast and baseline context so teams can spot drift and decide when to rerun models.')
        visuals['charts'].append(_plot_card('results-monthly', 'Monthly Revenue Totals', {
            'data': [{'type': 'bar', 'name': 'Monthly Revenue', 'x': monthly_totals['month'].dt.strftime('%Y-%m').tolist(), 'y': monthly_totals['y'].round(2).tolist()}],
            'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 40, 'r': 20, 't': 30, 'b': 60}}
        }, 'This is useful for quickly spotting seasonality and month-over-month changes.'))
        if len(monthly_totals) >= 24:
            yoy = monthly.copy()
            yoy['year'] = yoy['ds'].dt.year.astype(str)
            yoy['month_name'] = yoy['ds'].dt.strftime('%b')
            yoy_month = yoy.groupby(['year', 'month_name'], as_index=False)['y'].sum()
            month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            yoy_month['month_name'] = pd.Categorical(yoy_month['month_name'], categories=month_order, ordered=True)
            yoy_month = yoy_month.sort_values(['year', 'month_name'])
            traces = []
            for yr, g in yoy_month.groupby('year'):
                traces.append({'type': 'scatter', 'mode': 'lines+markers', 'name': str(yr), 'x': g['month_name'].astype(str).tolist(), 'y': g['y'].round(2).tolist()})
            visuals['charts'].append(_plot_card('results-yoy', 'Year-over-Year Monthly Revenue', {
                'data': traces,
                'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 40, 'r': 20, 't': 30, 'b': 40}}
            }, 'Compare the same months across recent years to spot seasonal trends.'))

    if not series_summary.empty:
        top_series = series_summary.head(10)
        top_series_revenue = pd.to_numeric(top_series['total_revenue'], errors='coerce').fillna(0)
        top_series_stability = pd.to_numeric(top_series['stability_score'], errors='coerce').fillna(0)
        top_series_labels = top_series['category'].astype(str).map(compact_category_label)
        top_series_hover = top_series['category'].astype(str).tolist()
        visuals['charts'].append(_plot_card('results-series-intel', 'Series / Cohort Intelligence', {
            'data': [
                {
                    'type': 'bar',
                    'x': top_series_labels.tolist(),
                    'y': top_series_revenue.round(2).tolist(),
                    'name': 'Revenue',
                    'customdata': top_series_hover,
                    'hovertemplate': '%{customdata}<br>Revenue: $%{y:,.0f}<extra></extra>',
                },
                {
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'x': top_series_labels.tolist(),
                    'y': top_series_stability.round(2).tolist(),
                    'name': 'Stability',
                    'yaxis': 'y2',
                    'customdata': top_series_hover,
                    'hovertemplate': '%{customdata}<br>Stability: %{y:.2f}<extra></extra>',
                },
            ],
            'layout': {
                'paper_bgcolor': 'transparent',
                'plot_bgcolor': 'transparent',
                'font': {'color': '#edf1f7'},
                'margin': {'l': 46, 'r': 52, 't': 30, 'b': 130},
                'xaxis': {'tickangle': -24, 'automargin': True},
                'yaxis2': {'overlaying': 'y', 'side': 'right', 'range': [0, 1], 'automargin': True},
            }
        }, 'This acts like an affiliate or platform intelligence layer by ranking categories on revenue and stability.'))

        if 'recent_roas' in series_summary.columns:
            alloc = series_summary.copy()
            if 'recent_cost' in alloc.columns:
                alloc['recent_cost'] = pd.to_numeric(alloc['recent_cost'], errors='coerce')
                alloc.loc[alloc['recent_cost'].fillna(0) <= 0, 'recent_roas'] = np.nan
            alloc['extra_budget_10pct'] = pd.to_numeric(alloc['recent_cost'], errors='coerce').fillna(0) * 0.10 if 'recent_cost' in alloc.columns else 0
            alloc['projected_revenue_from_extra_budget'] = alloc['extra_budget_10pct'] * pd.to_numeric(alloc['recent_roas'], errors='coerce').fillna(0)
            alloc['projected_total_revenue_after_shift'] = pd.to_numeric(alloc['recent_revenue'], errors='coerce').fillna(0) + alloc['projected_revenue_from_extra_budget']
            recent_roas_series = pd.to_numeric(alloc['recent_roas'], errors='coerce')
            recent_roas_median = recent_roas_series.median()
            recent_revenue_series = pd.to_numeric(alloc.get('recent_revenue', 0), errors='coerce').fillna(0)
            recent_revenue_median = recent_revenue_series[recent_revenue_series > 0].median()
            if pd.isna(recent_revenue_median):
                recent_revenue_median = 0

            def _allocation_action(row):
                roas = _safe_number(row.get('recent_roas'))
                revenue = _safe_number(row.get('recent_revenue')) or 0.0
                cost = _safe_number(row.get('recent_cost')) or 0.0
                if cost <= 0:
                    return 'Verify cost tracking'
                if roas is None:
                    return 'Verify Recent ROAS'
                if roas >= max(1.2, float(recent_roas_median or 0)) and revenue >= float(recent_revenue_median or 0):
                    return 'Add budget'
                if roas < 1:
                    return 'Hold / rework'
                return 'Hold / review'

            alloc['budget_action'] = alloc.apply(_allocation_action, axis=1)
            allocation_rows = _json(alloc[['category', 'recent_revenue', 'recent_cost', 'recent_roas', 'extra_budget_10pct', 'projected_revenue_from_extra_budget', 'projected_total_revenue_after_shift', 'budget_action']].sort_values('projected_revenue_from_extra_budget', ascending=False), 15)
            visuals['allocation_table'] = allocation_rows
            visuals['budget_opt_table'] = allocation_rows
            top_budget = alloc.sort_values('projected_revenue_from_extra_budget', ascending=False).head(10)
            has_positive_spend = pd.to_numeric(top_budget.get('recent_cost', 0), errors='coerce').fillna(0).gt(0).any()
            has_positive_addon = pd.to_numeric(top_budget.get('extra_budget_10pct', 0), errors='coerce').fillna(0).gt(0).any()
            if not top_budget.empty and (has_positive_spend or has_positive_addon):
                top_budget_labels = top_budget['category'].astype(str).map(compact_category_label)
                top_budget_hover = top_budget['category'].astype(str).tolist()
                visuals['budget_opt_chart'] = _plot_card('results-budget-optimizer', 'Budget Optimization Engine', {
                    'data': [
                        {'type': 'bar', 'name': 'Current spend', 'x': top_budget_labels.tolist(), 'y': pd.to_numeric(top_budget['recent_cost'], errors='coerce').fillna(0).round(2).tolist(), 'customdata': top_budget_hover, 'hovertemplate': '%{customdata}<br>Current spend: $%{y:,.0f}<extra></extra>'},
                        {'type': 'bar', 'name': 'Suggested add-on budget', 'x': top_budget_labels.tolist(), 'y': pd.to_numeric(top_budget['extra_budget_10pct'], errors='coerce').fillna(0).round(2).tolist(), 'customdata': top_budget_hover, 'hovertemplate': '%{customdata}<br>Suggested add-on budget: $%{y:,.0f}<extra></extra>'},
                        {'type': 'scatter', 'mode': 'lines+markers', 'name': 'Projected added revenue', 'x': top_budget_labels.tolist(), 'y': pd.to_numeric(top_budget['projected_revenue_from_extra_budget'], errors='coerce').fillna(0).round(2).tolist(), 'yaxis': 'y2', 'customdata': top_budget_hover, 'hovertemplate': '%{customdata}<br>Projected added revenue: $%{y:,.0f}<extra></extra>'},
                    ],
                    'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 45, 'r': 45, 't': 30, 'b': 130}, 'barmode': 'group', 'xaxis': {'tickangle': -24, 'automargin': True}, 'yaxis2': {'overlaying': 'y', 'side': 'right', 'title': 'Projected added revenue', 'automargin': True}}
                }, 'Highlights where incremental budget is most likely to produce more revenue based on recent cost and efficiency behavior.')
            optimizer_rows.extend(allocation_rows[:10])
            projected_added_revenue = pd.to_numeric(alloc['projected_revenue_from_extra_budget'], errors='coerce').fillna(0)
            top_alloc = (
                alloc.assign(projected_revenue_from_extra_budget=projected_added_revenue)
                .loc[projected_added_revenue > 0]
                .sort_values('projected_revenue_from_extra_budget', ascending=False)
                .head(10)
            )
            if not top_alloc.empty:
                best_cat = top_alloc.iloc[0]
                top_alloc_labels = top_alloc['category'].astype(str).map(compact_category_label)
                top_alloc_hover = top_alloc['category'].astype(str).tolist()
                root_cause.append(
                    f"{best_cat['category']} is the strongest budget-expansion candidate right now. A 10% increase to its recent spend projects about ${best_cat['projected_revenue_from_extra_budget']:,.0f} in additional revenue at its recent efficiency."
                )
                visuals['charts'].append(_plot_card('results-budget-allocation', 'Budget Allocation Opportunity by Category', {
                    'data': [{
                        'type': 'bar',
                        'x': top_alloc_labels.tolist(),
                        'y': pd.to_numeric(top_alloc['projected_revenue_from_extra_budget'], errors='coerce').fillna(0).round(2).tolist(),
                        'name': 'Projected added revenue',
                        'customdata': top_alloc_hover,
                        'hovertemplate': '%{customdata}<br>Projected added revenue: $%{y:,.0f}<extra></extra>',
                    }],
                    'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 50, 'r': 20, 't': 30, 'b': 130}, 'xaxis': {'tickangle': -24, 'automargin': True}}
                }, 'If a category or affiliate is driving a lot of revenue efficiently, this estimates the added revenue from giving it 10% more recent budget.'))
            elif allocation_rows:
                root_cause.append(
                    'Budget allocation opportunity is waiting on usable cost data. The current category rows have no positive recent cost, so the app can rank revenue activity but cannot estimate incremental budget lift yet.'
                )

    recalibration_note = 'Recalibrate weekly.'
    if not hist_daily.empty:
        span_days = int((hist_daily['ds'].max() - hist_daily['ds'].min()).days) if len(hist_daily) > 1 else 0
        if span_days >= 365:
            recalibration_note = 'Recalibrate weekly and refresh seasonality monthly because the app has a full-year pattern history.'
        elif span_days >= 90:
            recalibration_note = 'Recalibrate weekly because the series has enough history to drift without fully changing regime.'
        else:
            recalibration_note = 'Recalibrate every few days because the history window is still short.'

    if impact_rows:
        impact_df = pd.DataFrame(impact_rows).sort_values('revenue_change_if_driver_moves_10pct', key=lambda s: s.abs(), ascending=False)
        top_driver_name = impact_df.iloc[0]['driver']
    else:
        top_driver_name = None

    if not hist_daily.empty:
        if confidence_score is None:
            revenue_cv = float(pd.to_numeric(hist_daily['y'], errors='coerce').std(ddof=0) or 0.0)
            revenue_mean = float(pd.to_numeric(hist_daily['y'], errors='coerce').mean() or 0.0)
            revenue_ratio = (revenue_cv / revenue_mean) if revenue_mean else 1.0
            depth_score = min(len(hist_daily) / 180.0, 1.0)
            stability_score = max(0.0, 1.0 - min(revenue_ratio, 1.5) / 1.5)
            confidence_score = round((depth_score * 0.55 + stability_score * 0.45) * 100.0, 1)
            visuals['confidence_table'] = [
                {'factor': 'History depth', 'score_pct': round(depth_score * 100, 1)},
                {'factor': 'Series stability', 'score_pct': round(stability_score * 100, 1)},
                {'factor': 'Overall confidence', 'score_pct': confidence_score},
            ]
        if not top_driver_name:
            top_driver_name = 'Revenue trend'
    # Data health monitoring
    duplicate_rows = int(pd.to_datetime(first_series(hist, 'ds'), errors='coerce').duplicated().sum()) if ('ds' in hist.columns and not hist.empty) else 0
    missing_ds = int(pd.to_datetime(first_series(hist, 'ds'), errors='coerce').isna().sum()) if ('ds' in hist.columns and not hist.empty) else len(hist) if not hist.empty else 0
    missing_y = int(pd.to_numeric(first_series(hist, 'y'), errors='coerce').isna().sum()) if ('y' in hist.columns and not hist.empty) else len(hist) if not hist.empty else 0
    history_rows = int(len(hist_daily))
    coverage_days = int((hist_daily['ds'].max() - hist_daily['ds'].min()).days) if not hist_daily.empty and len(hist_daily) > 1 else 0
    health_rows = [
        {'check': 'History depth', 'status': 'Healthy' if history_rows >= 90 else 'Watch', 'detail': f'{history_rows} daily rows available'},
        {'check': 'Date coverage', 'status': 'Healthy' if coverage_days >= 60 else 'Watch', 'detail': f'{coverage_days} days covered'},
        {'check': 'Missing dates', 'status': 'Healthy' if missing_ds == 0 else 'Issue', 'detail': f'{missing_ds} rows missing a usable date'},
        {'check': 'Missing target values', 'status': 'Healthy' if missing_y == 0 else 'Issue', 'detail': f'{missing_y} rows missing revenue / target'},
        {'check': 'Duplicate dates', 'status': 'Healthy' if duplicate_rows == 0 else 'Watch', 'detail': f'{duplicate_rows} duplicate date rows found'},
        {'check': 'Forecast band width', 'status': 'Healthy' if (band_width_pct or 0) <= 25 else 'Watch', 'detail': f'{band_width_pct:.1f}% width' if band_width_pct is not None else 'n/a'},
    ]
    visuals['health_monitor_table'] = health_rows
    visuals['health_monitor_cards'] = {
        'issues': sum(1 for r in health_rows if r['status'] == 'Issue'),
        'watch_items': sum(1 for r in health_rows if r['status'] == 'Watch'),
        'healthy_items': sum(1 for r in health_rows if r['status'] == 'Healthy'),
    }

    auto_insights = []
    if trend_pct is not None:
        auto_insights.append(f"Revenue trend: the recent 7-day average is {'up' if trend_pct > 0 else 'down'} {abs(trend_pct):.1f}% versus the prior 28-day baseline.")
    if top_driver_name:
        auto_insights.append(f"Primary modeled lever: {top_driver_name} is the strongest driver behind revenue movement in this run.")
    if scenario_compare_rows:
        best_scn = max(scenario_compare_rows, key=lambda r: float(r.get('total_revenue') or 0))
        auto_insights.append(f"Scenario view: {best_scn['scenario']} is the highest-output path at about ${float(best_scn['total_revenue'] or 0):,.0f} for the visible horizon.")
    if allocation_rows:
        try:
            best_alloc = allocation_rows[0]
            auto_insights.append(f"Budget opportunity: {best_alloc['category']} has the strongest projected lift from incremental spend in the current allocation model.")
        except Exception:
            pass
    if anomaly_rows:
        auto_insights.append(f"Anomaly watch: {len(anomaly_rows)} unusual revenue prints were found against the rolling baseline and should be reviewed before acting on trend alone.")
    if visuals.get('cohort_rev_platform_table'):
        auto_insights.append('Cohort lens: use platform/source rollups to decide what to scale, cap, or remove; lead-month cohorts are diagnostic, not actionable by themselves.')
    if visuals['health_monitor_cards']['issues']:
        auto_insights.append('Data health: there are issues in the input health checks, so planning outputs should be interpreted with caution until the data is cleaned.')
    visuals['auto_insights_table'] = [{'insight': x} for x in auto_insights[:10]]

    visuals['insights'] = {
        'expected_total': expected_total,
        'lower_total': lower_total,
        'upper_total': upper_total,
        'band_width_pct': band_width_pct,
        'root_cause': root_cause[:8],
        'patterns': patterns[:6],
        'recalibration_note': recalibration_note,
        'scenario_defaults': scenario_defaults,
        'alerts': alerts[:8],
        'top_driver_name': top_driver_name,
        'auto_insights': auto_insights[:10],
        'local_time_hint': 'Times are shown in your browser local time.',
    }
    visuals['alerts'] = alerts[:8]
    visuals['optimizer_table'] = optimizer_rows[:20]
    visuals['goal_optimizer_profiles'] = monthly_optimizer_profiles[:18]
    visuals['time_machine_table'] = time_machine_rows[:12]
    visuals['drivers_table'] = driver_rows[:20] if driver_rows else visuals.get('drivers_table', [])
    visuals['tracking_table'] = tracking_rows[:12]

    spending = build_spending_slowdown(df)
    visuals['spending_slowdown_available'] = bool(spending.get('available'))
    visuals['spending_slowdown_message'] = spending.get('message', '')
    visuals['spending_slowdown_cards'] = spending.get('cards', {})
    visuals['spending_slowdown_decomposition_table'] = spending.get('decomposition_table', [])
    visuals['spending_slowdown_returning_table'] = spending.get('returning_table', [])
    visuals['spending_slowdown_cohort_table'] = spending.get('cohort_table', [])
    spending_charts = []
    for idx, chart in enumerate(spending.get('charts', []), start=1):
        chart_type = chart.get('type', 'scatter')
        spec = {
            'data': [{
                'type': chart_type,
                'mode': 'lines+markers' if chart_type == 'scatter' else None,
                'x': chart.get('x', []),
                'y': chart.get('y', []),
                'hovertemplate': chart.get('hover', '%{x}<br>%{y}<extra></extra>'),
            }],
            'layout': {
                'paper_bgcolor': 'transparent',
                'plot_bgcolor': 'transparent',
                'font': {'color': '#edf1f7'},
                'margin': {'l': 55, 'r': 20, 't': 30, 'b': 60},
                'xaxis': {'title': ''},
                'yaxis': {'title': chart.get('y_title', '')},
            },
        }
        if chart_type == 'bar':
            spec['data'][0].pop('mode', None)
        spending_charts.append(_plot_card(
            f'results-spending-slowdown-{idx}',
            chart.get('title', f'Spending Chart {idx}'),
            spec,
            chart.get('subtitle'),
        ))
    visuals['spending_slowdown_charts'] = spending_charts

    visuals['whale_export_table'] = visuals.get('planning_whale_watch', [])
    explain_rows = []
    for row in (visuals.get('drivers_table') or [])[:10]:
        explain_rows.append({
            'feature': row.get('feature') or row.get('driver'),
            'importance': row.get('importance'),
            'share_pct': row.get('share_pct'),
        })
    impact_rows = visuals.get('impact_table') or []
    if impact_rows:
        top_impact = impact_rows[0]
        explain_rows.append({
            'feature': top_impact.get('driver'),
            'importance': top_impact.get('confidence_score'),
            'share_pct': top_impact.get('revenue_change_if_driver_moves_10pct'),
        })

    lgbm_intel = result.get('lightgbm_intelligence') if isinstance(result, dict) else {}
    if isinstance(lgbm_intel, dict):
        visuals['lightgbm_intelligence_available'] = bool(lgbm_intel.get('available'))
        visuals['lightgbm_exec_summary'] = lgbm_intel.get('executive_summary') or []
        shap_layer = lgbm_intel.get('shap') or {}
        ranking_layer = lgbm_intel.get('ranking') or {}
        lead_quality_layer = lgbm_intel.get('lead_quality') or {}
        anomaly_layer = lgbm_intel.get('residual_anomalies') or {}
        whale_layer = lgbm_intel.get('whale_prediction') or {}
        returns_layer = lgbm_intel.get('diminishing_returns') or {}
        scenario_layer = lgbm_intel.get('scenario_simulation') or {}
        affiliate_layer = lgbm_intel.get('affiliate_quality') or {}
        confidence_layer = lgbm_intel.get('confidence') or {}
        visuals['lightgbm_shap_summary'] = shap_layer.get('global_summary') or []
        visuals['lightgbm_local_explanations'] = shap_layer.get('local_explanations') or []
        visuals['lightgbm_shap_aggregations'] = shap_layer.get('aggregations') or []
        visuals['lightgbm_ranked_opportunities'] = ranking_layer.get('ranked_opportunities') or []
        visuals['lightgbm_lead_quality_scores'] = lead_quality_layer.get('scores') or []
        visuals['lightgbm_anomaly_rows'] = anomaly_layer.get('anomaly_rows') or []
        visuals['lightgbm_whale_predictions'] = whale_layer.get('scores') or []
        visuals['lightgbm_response_curve'] = returns_layer.get('response_curve') or []
        visuals['lightgbm_scenarios'] = scenario_layer.get('scenarios') or []
        visuals['lightgbm_affiliate_quality'] = affiliate_layer.get('scores') or []
        visuals['lightgbm_confidence_rows'] = confidence_layer.get('risk_rows') or []
        for row in visuals['lightgbm_shap_summary'][:8]:
            explain_rows.append({
                'feature': row.get('driver') or row.get('feature'),
                'importance': row.get('mean_abs_shap'),
                'share_pct': row.get('share_pct'),
            })
        if visuals['lightgbm_ranked_opportunities']:
            existing = visuals.get('planning_dimension_recommendations') or []
            seen = {
                (str(r.get('dimension_type') or ''), str(r.get('dimension') or r.get('label') or ''))
                for r in existing
                if isinstance(r, dict)
            }
            additions = []
            for row in visuals['lightgbm_ranked_opportunities'][:20]:
                key = (str(row.get('dimension_type') or ''), str(row.get('dimension') or ''))
                if key in seen:
                    continue
                additions.append({
                    'dimension_type': row.get('dimension_type'),
                    'dimension': row.get('dimension'),
                    'label': row.get('dimension'),
                    'recommendation': row.get('recommendation'),
                    'ranking_score': row.get('ranking_score'),
                    'reason': row.get('business_reason'),
                    'source': 'lightgbm_ranker',
                })
            if additions:
                visuals['planning_dimension_recommendations'] = (existing + additions)[:40]
        if visuals['lightgbm_anomaly_rows']:
            existing_anomalies = visuals.get('anomaly_table') or []
            model_anomalies = []
            for row in visuals['lightgbm_anomaly_rows'][:20]:
                model_anomalies.append({
                    'title': f"LightGBM residual watch {row.get('date')}",
                    'date': row.get('date'),
                    'actual_revenue': row.get('actual_revenue'),
                    'predicted_revenue': row.get('predicted_revenue'),
                    'anomaly_score': row.get('anomaly_score'),
                    'severity': row.get('severity') or row.get('risk_label'),
                    'likely_drivers': row.get('likely_drivers'),
                    'recommended_action': row.get('recommended_action'),
                    'source': 'lightgbm_residual',
                })
            visuals['anomaly_table'] = (existing_anomalies + model_anomalies)[:60]
        if visuals['lightgbm_whale_predictions']:
            whale_existing = visuals.get('whale_export_table') or visuals.get('planning_whale_watch') or []
            prediction_by_id = {str(r.get('profile_id')): r for r in visuals['lightgbm_whale_predictions'] if r.get('profile_id')}
            merged_whales = []
            for row in whale_existing:
                merged = dict(row)
                pred = prediction_by_id.get(str(row.get('whale_id') or row.get('profile_id') or ''))
                if pred:
                    merged.update({
                        'behavior_label': pred.get('behavior_label'),
                        'cooling_risk': pred.get('cooling_risk'),
                        'reactivation_likelihood': pred.get('reactivation_likelihood'),
                        'revenue_drop_risk': pred.get('revenue_drop_risk'),
                        'ml_recommended_action': pred.get('recommended_action'),
                    })
                merged_whales.append(merged)
            if merged_whales:
                visuals['whale_export_table'] = merged_whales
            else:
                visuals['whale_export_table'] = visuals['lightgbm_whale_predictions']
            visuals['planning_whale_watch'] = visuals.get('planning_whale_watch') or visuals['whale_export_table']
        if visuals['lightgbm_response_curve']:
            existing_budget = visuals.get('budget_opt_table') or []
            response_rows = []
            for row in visuals['lightgbm_response_curve']:
                response_rows.append({
                    'scenario': f"ML spend {row.get('spend_change_pct')}%",
                    'projected_revenue': row.get('projected_revenue'),
                    'projected_roas': row.get('projected_roas'),
                    'marginal_roas': row.get('marginal_roas'),
                    'zone': row.get('zone'),
                    'business_label': row.get('business_label'),
                    'source': 'lightgbm_response_curve',
                })
            visuals['budget_opt_table'] = (existing_budget + response_rows)[:60]
        if visuals['lightgbm_scenarios']:
            existing_scenarios = visuals.get('scenario2_table') or []
            scenario_additions = []
            for row in visuals['lightgbm_scenarios']:
                scenario_additions.append({
                    'scenario': row.get('scenario'),
                    'projected_revenue': row.get('projected_revenue'),
                    'projected_roas': row.get('projected_roas'),
                    'risk_band_low': row.get('risk_band_low'),
                    'risk_band_high': row.get('risk_band_high'),
                    'interpretation': row.get('interpretation'),
                    'source': 'lightgbm_simulation',
                })
            visuals['scenario2_table'] = (existing_scenarios + scenario_additions)[:60]
        if visuals['lightgbm_affiliate_quality']:
            existing_attr = visuals.get('attribution_table') or []
            affiliate_additions = []
            for row in visuals['lightgbm_affiliate_quality']:
                affiliate_additions.append({
                    'channel': row.get('affiliate'),
                    'quality_score': row.get('quality_score'),
                    'quality_tier': row.get('quality_tier'),
                    'recommendation': row.get('recommendation'),
                    'short_term_strong_long_term_weak': row.get('short_term_strong_long_term_weak'),
                    'source': 'lightgbm_affiliate_quality',
                })
            visuals['attribution_table'] = (existing_attr + affiliate_additions)[:80]
        if visuals['lightgbm_confidence_rows']:
            existing_conf = visuals.get('confidence_table') or []
            visuals['confidence_table'] = (existing_conf + visuals['lightgbm_confidence_rows'])[:60]
    visuals['explainability_summary'] = explain_rows[:12]

    attribution_table, attribution_summary = _build_attribution_table(series_summary)
    visuals['attribution_table'] = attribution_table
    visuals['attribution_summary'] = attribution_summary

    scenario2_table, scenario2_cards = _build_scenario2_table(scenario_defaults, visuals.get('allocation_table') or [])
    visuals['scenario2_table'] = scenario2_table
    visuals['scenario2_cards'] = scenario2_cards

    report_lines, report_html = _build_stakeholder_report(visuals, result)
    visuals['stakeholder_report_lines'] = report_lines
    visuals['stakeholder_report_html'] = report_html
    return visuals

def _forecast_settings_path(project_id: int, run_id: int) -> Path:
    return _project_dir(project_id) / f'run_{run_id}_forecast_settings.json'


def _visuals_snapshot_path(project_id: int, run_id: int) -> Path:
    return _project_dir(project_id) / f'run_{run_id}_visuals.json'


def _write_visuals_snapshot(project_id: int, run_id: int, visuals: dict) -> None:
    try:
        path = _visuals_snapshot_path(project_id, run_id)
        path.write_text(json.dumps(json_clean(_merge_results_visual_defaults(visuals)), indent=2, default=str), encoding='utf-8')
    except Exception:
        pass


def _load_visuals_snapshot(project_id: int, run_id: int) -> dict:
    path = _visuals_snapshot_path(project_id, run_id)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        return _protect_results_visuals(payload if isinstance(payload, dict) else {})
    except Exception:
        return {}


def _load_forecast_settings(project_id: int, run_id: int) -> dict:
    path = _forecast_settings_path(project_id, run_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.write_bytes(src.read_bytes())


def _clone_run_artifacts(project_id: int, source_run_id: int, new_run_id: int) -> None:
    _project_dir(project_id).mkdir(parents=True, exist_ok=True)
    _copy_if_exists(_upload_meta_path(project_id, source_run_id), _upload_meta_path(project_id, new_run_id))
    _copy_if_exists(_mapping_meta_path(project_id, source_run_id), _mapping_meta_path(project_id, new_run_id))
    _copy_if_exists(_forecast_settings_path(project_id, source_run_id), _forecast_settings_path(project_id, new_run_id))


def _run_forecast_job(project_id: int, run_id: int, payload: ForecastRequest) -> None:
    try:
        write_progress(project_id, run_id, status='running', percent=2, step='Starting', detail='Forecast engine is starting up and validating the saved run settings before model training begins.')
        try:
            artifact_dir = UPLOADS_DIR / f'project_{project_id}' / f'run_{run_id}_lightgbm_artifacts'
            artifact_dir.mkdir(parents=True, exist_ok=True)
            payload.lightgbm_artifact_dir = str(artifact_dir)
        except Exception:
            pass

        def progress_callback(percent: int, step: str, detail: str, model_activity: dict | None = None) -> None:
            payload = {'status': 'running', 'percent': percent, 'step': step, 'detail': detail}
            if model_activity is not None:
                payload['model_activity'] = model_activity
            write_progress(project_id, run_id, **payload)

        result = run_forecast(payload, progress_callback=progress_callback)
        result_path = UPLOADS_DIR / f'project_{project_id}' / f'run_{run_id}_result.json'
        result_path.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
        try:
            _write_visuals_snapshot(project_id, run_id, _build_minimal_results_visuals(result))
        except Exception:
            pass
        write_progress(
            project_id,
            run_id,
            status='completed',
            percent=100,
            step='Completed',
            detail='Forecast results are ready. SignalForge finished training, ranking, formatting, and saving the final dashboard payload.',
            result_url=f'/projects/{project_id}/results?run_id={run_id}',
            model_activity=(result or {}).get('model_activity', {}),
        )
    except Exception as exc:
        write_progress(project_id, run_id, status='failed', percent=100, step='Failed', detail=f'The forecast run stopped before completion. Root cause: {exc}', error=str(exc))





def _load_result_payload(project_id: int, run_id: int) -> dict:
    result_path = UPLOADS_DIR / f'project_{project_id}' / f'run_{run_id}_result.json'
    if not result_path.exists():
        return {}
    try:
        return json.loads(result_path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _safe_run_settings(project_id: int, run_id: int) -> dict:
    settings = _load_forecast_settings(project_id, run_id)
    return settings if isinstance(settings, dict) else {}


def _build_run_summary(project: object | None, run: object | None, result: dict, settings: dict) -> dict:
    top_models = result.get('top_models') or []
    best = top_models[0] if top_models else (result.get('best_model') or {})
    def _get(obj, *keys):
        if isinstance(obj, dict):
            for key in keys:
                if key in obj and obj.get(key) is not None:
                    return obj.get(key)
        else:
            for key in keys:
                if hasattr(obj, key):
                    val = getattr(obj, key)
                    if val is not None:
                        return val
        return None

    return {
        'project_name': getattr(project, 'name', None),
        'run_id': getattr(run, 'id', None),
        'run_label': getattr(run, 'label', None),
        'status': getattr(run, 'status', None),
        'best_model': _get(best, 'model', 'model_name') or 'n/a',
        'best_score': _safe_number(_get(best, 'final_rank_score', 'final_score', 'score')),
        'forecast_rows': len(result.get('display_forecast') or result.get('blended_forecast') or result.get('ensemble_preview') or []),
        'backtest_rows': result.get('backtest_rows'),
        'horizon': settings.get('horizon'),
        'frequency': settings.get('frequency'),
        'output_mode': settings.get('output_mode'),
        'scoring_metric': settings.get('scoring_metric'),
        'models_count': len(settings.get('models') or []),
        'gpu_requested': settings.get('use_gpu', False),
    }


def _build_run_history_df(project_id: int, run_id: int) -> pd.DataFrame:
    upload_meta = _load_upload_meta(project_id, run_id)
    mapping = _load_mapping_meta(project_id, run_id)
    if not upload_meta or not mapping:
        return pd.DataFrame(columns=['ds', 'y'])
    df, _ = _friendly_build_model_dataframe(upload_meta, mapping)
    if df is None or df.empty:
        return pd.DataFrame(columns=['ds', 'y'])
    hist = dedupe_columns_keep_first(df.copy())
    ds_source = 'ds' if 'ds' in hist.columns else (hist.columns[0] if len(hist.columns) > 0 else None)
    y_source = 'y' if 'y' in hist.columns else (hist.columns[1] if len(hist.columns) > 1 else None)
    if ds_source is None or y_source is None:
        return pd.DataFrame(columns=['ds', 'y'])
    hist['ds'] = pd.to_datetime(first_series(hist, ds_source), errors='coerce')
    hist['y'] = pd.to_numeric(first_series(hist, y_source), errors='coerce')
    for c in ['leads', 'cost']:
        if c in hist.columns:
            hist[c] = pd.to_numeric(first_series(hist, c), errors='coerce')
    hist = hist.dropna(subset=['ds'])
    hist = hist.assign(ds=hist['ds'].dt.normalize())
    agg_map = {'y': 'sum'}
    for c in ['leads', 'cost']:
        if c in hist.columns:
            agg_map[c] = 'sum'
    hist_daily = hist.groupby('ds', as_index=False).agg(agg_map)
    hist_daily = dedupe_columns_keep_first(hist_daily)
    hist_daily['ds'] = pd.to_datetime(first_series(hist_daily, 'ds'), errors='coerce')
    hist_daily['y'] = pd.to_numeric(first_series(hist_daily, 'y'), errors='coerce')
    return hist_daily.dropna(subset=['ds']).sort_values('ds').reset_index(drop=True)


def _build_compare_visuals(left_summary: dict, right_summary: dict, left_hist: pd.DataFrame, right_hist: pd.DataFrame, left_result: dict, right_result: dict) -> dict:
    visuals = {'charts': [], 'comparison_table': [], 'settings_rows': []}
    left_hist = left_hist.copy() if left_hist is not None else pd.DataFrame(columns=['ds', 'y'])
    right_hist = right_hist.copy() if right_hist is not None else pd.DataFrame(columns=['ds', 'y'])

    for hist in [left_hist, right_hist]:
        if not hist.empty:
            hist['ds'] = pd.to_datetime(first_series(hist, 'ds'), errors='coerce')
            hist['y'] = pd.to_numeric(first_series(hist, 'y'), errors='coerce')

    left_fc = pd.DataFrame(left_result.get('display_forecast') or left_result.get('blended_forecast') or left_result.get('ensemble_preview') or [])
    right_fc = pd.DataFrame(right_result.get('display_forecast') or right_result.get('blended_forecast') or right_result.get('ensemble_preview') or [])

    def _prep_forecast(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=['ds', 'expected'])
        out = dedupe_columns_keep_first(df.copy())
        if 'ds' in out.columns:
            out['ds'] = pd.to_datetime(first_series(out, 'ds'), errors='coerce')
        for col in ['yhat', 'expected', 'conservative', 'aggressive', 'lower', 'upper']:
            if col in out.columns:
                out[col] = pd.to_numeric(first_series(out, col), errors='coerce')
        if 'expected' not in out.columns:
            if 'yhat' in out.columns:
                out['expected'] = out['yhat']
            elif 'mean' in out.columns:
                out['expected'] = pd.to_numeric(first_series(out, 'mean'), errors='coerce')
        return out.dropna(subset=['ds']).sort_values('ds').reset_index(drop=True)

    left_fc = _prep_forecast(left_fc)
    right_fc = _prep_forecast(right_fc)

    def _series_to_plot(s: pd.Series) -> list:
        return s.round(2).astype(object).where(s.notna(), None).tolist()

    overlay_data = []
    if not left_hist.empty:
        overlay_data.append({'type': 'scatter', 'mode': 'lines', 'name': f"{left_summary.get('run_label') or 'Left'} history", 'x': left_hist['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': _series_to_plot(left_hist['y'])})
    if not right_hist.empty:
        overlay_data.append({'type': 'scatter', 'mode': 'lines', 'name': f"{right_summary.get('run_label') or 'Right'} history", 'x': right_hist['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': _series_to_plot(right_hist['y'])})
    if not left_fc.empty and 'expected' in left_fc.columns:
        overlay_data.append({'type': 'scatter', 'mode': 'lines', 'name': f"{left_summary.get('run_label') or 'Left'} forecast", 'x': left_fc['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': _series_to_plot(left_fc['expected']), 'line': {'dash': 'dot'}})
    if not right_fc.empty and 'expected' in right_fc.columns:
        overlay_data.append({'type': 'scatter', 'mode': 'lines', 'name': f"{right_summary.get('run_label') or 'Right'} forecast", 'x': right_fc['ds'].dt.strftime('%Y-%m-%d').tolist(), 'y': _series_to_plot(right_fc['expected']), 'line': {'dash': 'dot'}})

    visuals['charts'].append(_plot_card('compare-forecast-overlay', 'Run Comparison Overlay', {
        'data': overlay_data,
        'layout': {'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 40, 'r': 20, 't': 30, 'b': 40}}
    }, 'Compare historical revenue and expected forecast lines from both runs on one chart.'))

    metric_names = ['best_score', 'forecast_rows', 'backtest_rows', 'models_count']
    metric_labels = {'best_score': 'Best score', 'forecast_rows': 'Forecast rows', 'backtest_rows': 'Backtest rows', 'models_count': 'Models used'}
    left_vals = [left_summary.get(k) for k in metric_names]
    right_vals = [right_summary.get(k) for k in metric_names]
    visuals['charts'].append(_plot_card('compare-summary-bars', 'Run Metrics Side by Side', {
        'data': [
            {'type': 'bar', 'name': left_summary.get('run_label') or 'Left', 'x': [metric_labels[k] for k in metric_names], 'y': left_vals},
            {'type': 'bar', 'name': right_summary.get('run_label') or 'Right', 'x': [metric_labels[k] for k in metric_names], 'y': right_vals},
        ],
        'layout': {'barmode': 'group', 'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 40, 'r': 20, 't': 30, 'b': 60}}
    }, 'Quickly compare run size, backtest depth, and top-level ranking output.'))

    left_models = pd.DataFrame(left_result.get('top_models') or [])
    right_models = pd.DataFrame(right_result.get('top_models') or [])
    def _prep_models(dfm: pd.DataFrame, label: str) -> pd.DataFrame:
        if dfm is None or dfm.empty:
            return pd.DataFrame(columns=['model','score','run_label'])
        dfm = dedupe_columns_keep_first(dfm.copy())
        model_col = 'model' if 'model' in dfm.columns else ('model_name' if 'model_name' in dfm.columns else None)
        score_col = 'final_rank_score' if 'final_rank_score' in dfm.columns else ('final_score' if 'final_score' in dfm.columns else None)
        if model_col is None or score_col is None:
            return pd.DataFrame(columns=['model','score','run_label'])
        out = pd.DataFrame({
            'model': first_series(dfm, model_col),
            'score': pd.to_numeric(first_series(dfm, score_col), errors='coerce'),
            'run_label': label,
        }).dropna(subset=['model']).head(10)
        return out
    left_models = _prep_models(left_models, left_summary.get('run_label') or 'Left')
    right_models = _prep_models(right_models, right_summary.get('run_label') or 'Right')
    models_cat = pd.concat([left_models, right_models], ignore_index=True)
    if not models_cat.empty:
        visuals['charts'].append(_plot_card('compare-top-models', 'Top Model Score Comparison', {
            'data': [
                {'type': 'bar', 'name': label, 'x': grp['model'].tolist(), 'y': grp['score'].round(4).tolist()}
                for label, grp in models_cat.groupby('run_label')
            ],
            'layout': {'barmode': 'group', 'paper_bgcolor': 'transparent', 'plot_bgcolor': 'transparent', 'font': {'color': '#edf1f7'}, 'margin': {'l': 40, 'r': 20, 't': 30, 'b': 80}}
        }, 'See how the strongest models changed across the two runs.'))

    left_table = left_fc[['ds', 'expected']].copy() if {'ds', 'expected'}.issubset(left_fc.columns) else pd.DataFrame(columns=['ds','expected'])
    right_table = right_fc[['ds', 'expected']].copy() if {'ds', 'expected'}.issubset(right_fc.columns) else pd.DataFrame(columns=['ds','expected'])
    if not left_table.empty:
        left_table = left_table.rename(columns={'expected': 'left_expected'})
    if not right_table.empty:
        right_table = right_table.rename(columns={'expected': 'right_expected'})
    if not left_table.empty or not right_table.empty:
        compare_table = pd.merge(left_table, right_table, on='ds', how='outer').sort_values('ds').reset_index(drop=True)
        if 'left_expected' in compare_table.columns and 'right_expected' in compare_table.columns:
            compare_table['delta'] = pd.to_numeric(compare_table['right_expected'], errors='coerce') - pd.to_numeric(compare_table['left_expected'], errors='coerce')
        visuals['comparison_table'] = _json_ready_records(compare_table.head(60))

    all_setting_keys = ['business_profile', 'frequency', 'horizon', 'scoring_metric', 'output_mode', 'use_gpu', 'enable_revenue_lag_modeling', 'revenue_lag_profile']
    left_settings = left_result.get('settings') or {}
    right_settings = right_result.get('settings') or {}
    for key in all_setting_keys:
        visuals['settings_rows'].append({
            'setting': key,
            'left': left_settings.get(key),
            'right': right_settings.get(key),
            'different': left_settings.get(key) != right_settings.get(key),
        })
    return visuals

def redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url=url, status_code=303)



def _project_dir(project_id: int) -> Path:
    return UPLOADS_DIR / f'project_{project_id}'


def _upload_meta_path(project_id: int, run_id: int) -> Path:
    return _project_dir(project_id) / f'run_{run_id}_upload.json'


def _mapping_meta_path(project_id: int, run_id: int) -> Path:
    return _project_dir(project_id) / f'run_{run_id}_mapping.json'


def _load_upload_meta(project_id: int, run_id: int) -> dict:
    path = _upload_meta_path(project_id, run_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _load_mapping_meta(project_id: int, run_id: int) -> dict:
    path = _mapping_meta_path(project_id, run_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))



def _auto_select(columns: list[str], patterns: list[list[str]], exclude: list[list[str]] | None = None) -> str | None:
    """
    Return the first column whose lowercased name contains ALL words in any
    pattern group, and contains NONE of the words in any exclude group.
    """
    cols_lower = [(c, c.lower().replace('_', ' ').replace('-', ' ')) for c in columns]
    for pat in patterns:
        for col, low in cols_lower:
            if all(p in low for p in pat):
                if exclude and any(all(e in low for e in eg) for eg in exclude):
                    continue
                return col
    return None


def _dataset_file_options(upload_meta: dict) -> list[tuple[str, str]]:
    files = upload_meta.get('files', {})
    options = []
    for key, meta in files.items():
        label = meta.get('label') or meta.get('filename', key)
        options.append((key, label))
    return options


def _build_regressor_columns(mapping: dict) -> list[str]:
    """Derive the full list of regressor column names from a saved mapping dict."""
    import re as _re
    cols: list[str] = []
    if mapping.get('leads', {}).get('value_column'):
        cols.append('leads')
    if mapping.get('cost', {}).get('value_column'):
        cols.append('cost')
    if mapping.get('roas_column'):
        cols.append(mapping['roas_column'])
    if mapping.get('events', {}).get('event_flag_column'):
        cols.append('event_flag')
    if mapping.get('events', {}).get('outage_flag_column'):
        cols.append('outage_flag')
    # Columns typed into the revenue-sheet custom regressor box
    for col in mapping.get('custom_regressor_columns', []):
        if col and col not in cols:
            cols.append(col)
    # Columns from the dedicated custom_regressors section
    for col in (mapping.get('custom_regressors') or {}).get('value_columns', []):
        if col and col not in cols:
            cols.append(col)
    # Columns from dynamic custom_N sheets
    for key, val in mapping.items():
        if _re.match(r'^custom_\d+$', key) and isinstance(val, dict):
            for col in val.get('value_columns', []):
                if col and col not in cols:
                    cols.append(col)
    return cols


def _cleaned_source_key(file_key: str | None, sheet_name: str | None) -> str:
    return f"{file_key or ''}|{sheet_name or '__default__'}"


def _metadata_path(value: str | Path | None) -> Path:
    path = Path(str(value or ''))
    return path if path.is_absolute() else BASE_DIR / path


def _resolve_dataset(upload_meta: dict, file_key: str | None, sheet_name: str | None) -> tuple[pd.DataFrame | None, list[str], str | None]:
    if not file_key:
        return None, [], None
    cleaned_meta = (upload_meta.get('cleaned_sources') or {}).get(_cleaned_source_key(file_key, sheet_name))
    if cleaned_meta and cleaned_meta.get('path'):
        cleaned_path = _metadata_path(cleaned_meta['path'])
        if cleaned_path.exists():
            df = load_dataframe(cleaned_path, sheet_name=None)
            return df, [], sheet_name
    meta = upload_meta.get('files', {}).get(file_key)
    if not meta:
        return None, [], None
    path = _metadata_path(meta['path'])
    sheet_names = get_sheet_names(path)
    chosen_sheet = sheet_name or (sheet_names[0] if sheet_names else None)
    df = load_dataframe(path, sheet_name=chosen_sheet)
    return df, sheet_names, chosen_sheet




def _coerce_datetime_column(df: pd.DataFrame, col: str | None) -> pd.DataFrame:
    df = dedupe_columns_keep_first(df)
    if not col or col not in df.columns:
        return df
    df = df.copy()
    df[col] = normalize_date_series(first_series(df, col))
    return df


def _normalize_dimension_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col and col in out.columns:
            ser = first_series(out, col).astype(str).str.strip()
            ser = ser.replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, 'NaT': pd.NA})
            out[col] = ser
    return out


def _detail_dimension_specs(mapping_cfg: dict | None) -> list[tuple[str, str]]:
    cfg = mapping_cfg or {}
    return [
        ('platform_column', 'Platform'),
        ('affiliate_id_column', 'Affiliate'),
        ('campaign_id_column', 'Campaign'),
        ('ad_id_column', 'Ad'),
    ]


def _available_dimension_columns(df: pd.DataFrame, mapping_cfg: dict | None, include_category: bool = True) -> list[tuple[str, str]]:
    cols: list[tuple[str, str]] = []
    cfg = mapping_cfg or {}
    for key, label in _detail_dimension_specs(cfg):
        col = cfg.get(key)
        if col and col in df.columns:
            cols.append((col, label))
    if include_category:
        cat = cfg.get('category_column')
        if cat and cat in df.columns and all(col != cat for col, _ in cols):
            cols.append((cat, 'Group'))
    return cols


def _build_dimension_label_series(df: pd.DataFrame, mapping_cfg: dict | None, fallback_col: str | None = None, default: str = 'All Sources') -> pd.Series:
    work = df.copy()
    specs = _available_dimension_columns(work, mapping_cfg, include_category=True)
    if not specs and fallback_col and fallback_col in work.columns:
        specs = [(fallback_col, 'Group')]
    if not specs:
        return pd.Series([default] * len(work), index=work.index, dtype='object')
    work = _normalize_dimension_columns(work, [col for col, _ in specs])
    frames = []
    for col, label in specs:
        ser = first_series(work, col)
        frames.append(label + '=' + ser.fillna('Unknown').astype(str))
    if len(frames) == 1:
        return frames[0]
    combined = frames[0]
    for part in frames[1:]:
        combined = combined + ' | ' + part
    return combined


def _target_looks_like_timestamp(y_series: pd.Series, ds_series: pd.Series) -> bool:
    y_num = pd.to_numeric(y_series, errors='coerce')
    if y_num.notna().sum() == 0:
        return False
    median_val = float(y_num.dropna().median())
    if median_val < 1e14:
        return False
    try:
        ds_num = pd.to_numeric(pd.to_datetime(ds_series, errors='coerce').astype('int64'), errors='coerce')
    except Exception:
        ds_num = pd.Series(index=y_num.index, dtype='float64')
    aligned = pd.concat([y_num.rename('y'), ds_num.rename('dsn')], axis=1).dropna()
    if aligned.empty:
        return True
    match_ratio = float((aligned['y'] == aligned['dsn']).mean())
    return match_ratio >= 0.75 or median_val > 1e17


def _find_likely_target_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = set(exclude or [])
    candidates: list[tuple[str, float]] = []
    for col in df.columns:
        if col in exclude:
            continue
        s = pd.to_numeric(first_series(df, col), errors='coerce')
        valid_ratio = float(s.notna().mean()) if len(s) else 0.0
        if valid_ratio < 0.8:
            continue
        median_val = float(s.dropna().median()) if s.notna().any() else 0.0
        if median_val <= 0 or median_val > 1e14:
            continue
        score = valid_ratio * 1000 - abs(median_val)
        candidates.append((col, score))
    return [col for col, _ in sorted(candidates, key=lambda x: x[1], reverse=True)[:5]]


def _build_revenue_only_dataframe(upload_meta: dict, mapping: dict) -> pd.DataFrame:
    revenue_cfg = mapping.get('revenue', {})
    revenue_df, _, revenue_sheet = _resolve_dataset(upload_meta, revenue_cfg.get('file_key', 'revenue'), revenue_cfg.get('sheet_name'))
    if revenue_df is None:
        raise ValueError('Revenue dataset is not configured.')
    revenue_cfg['sheet_name'] = revenue_sheet
    date_col = revenue_cfg.get('date_column')
    target_col = revenue_cfg.get('target_column')
    if not date_col or not target_col:
        raise ValueError('Revenue date and target columns are required.')
    base = dedupe_columns_keep_first(revenue_df.copy())
    base = _coerce_datetime_column(base, date_col)
    if date_col not in base.columns or base[date_col].isna().all():
        raise ValueError(f"The revenue date column '{date_col}' could not be interpreted as dates.")
    if target_col not in base.columns:
        raise ValueError(f"The revenue target column '{target_col}' was not found in the revenue dataset.")
    base['ds'] = base[date_col]
    base['y'] = pd.to_numeric(first_series(base, target_col), errors='coerce')
    detail_cols = [c for c, _ in _available_dimension_columns(base, revenue_cfg, include_category=True)]
    if detail_cols:
        base = _normalize_dimension_columns(base, detail_cols)
        base['category'] = _build_dimension_label_series(base, revenue_cfg, fallback_col=revenue_cfg.get('category_column'), default='company_total')
        base['series_id'] = base['category'].astype(str)
    else:
        base['series_id'] = 'company_total'
        base['category'] = 'company_total'
    base = enforce_schema(base)
    return dedupe_columns_keep_first(base)


def _friendly_build_model_dataframe(upload_meta: dict, mapping: dict) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = _build_model_dataframe(upload_meta, mapping)
        if df is None or df.empty:
            fallback = _build_revenue_only_dataframe(upload_meta, mapping)
            return fallback, 'Merged helper datasets could not be aligned cleanly, so SignalForge fell back to the revenue-only modeling table.' if fallback is not None and not fallback.empty else None
        return df, None
    except Exception as exc:
        try:
            fallback = _build_revenue_only_dataframe(upload_meta, mapping)
            if fallback is not None and not fallback.empty:
                return fallback, f'Merged helper datasets failed to build, so SignalForge fell back to the revenue-only modeling table. Details: {exc}'
        except Exception:
            pass
        return None, str(exc)

def _choose_merge_keys(revenue_df: pd.DataFrame, revenue_cfg: dict | None, helper_df: pd.DataFrame, helper_cfg: dict | None) -> tuple[list[str], list[str]]:
    left_on = ['__merge_ds']
    right_on = ['__merge_ds']
    revenue_cfg = revenue_cfg or {}
    helper_cfg = helper_cfg or {}
    dim_keys = ['platform_column', 'affiliate_id_column', 'campaign_id_column', 'ad_id_column', 'category_column']
    for key in dim_keys:
        left_col = revenue_cfg.get(key)
        right_col = helper_cfg.get(key)
        if left_col and right_col and left_col in revenue_df.columns and right_col in helper_df.columns:
            left_on.append(left_col)
            right_on.append(right_col)
    return left_on, right_on


def _aggregate_helper_dataset(temp: pd.DataFrame, group_cols: list[str], value_cols: list[str], mode: str) -> pd.DataFrame:
    # Reduce helper sheets to one row per merge key so joins do not duplicate revenue rows.
    agg: dict[str, str] = {}
    for col in value_cols:
        if col not in temp.columns:
            continue
        agg[col] = 'max' if mode == 'flag' else 'sum'
    if not agg:
        return temp.drop_duplicates(subset=group_cols)
    return temp.groupby(group_cols, as_index=False).agg(agg)


def _detect_frequency(dates: pd.Series) -> str:
    """Infer the granularity of a date series: D, W, or M.

    Use UNIQUE normalized dates before measuring gaps. Helper sheets like
    monthly platform cost often contain many rows stamped with the same month
    anchor date (one per platform). Looking at raw row-to-row gaps makes the
    median gap zero and incorrectly classifies the helper as daily, which skips
    monthly-to-daily distribution entirely.
    """
    raw = dates if isinstance(dates, pd.Series) else pd.Series(dates)
    clean = pd.to_datetime(raw, format='%Y-%m-%d', errors='coerce')
    if clean.notna().sum() < max(2, int(len(raw) * 0.6)):
        clean = pd.to_datetime(raw, errors='coerce')
    clean = clean.dropna()
    if len(clean) < 2:
        return 'D'
    clean = pd.Series(pd.Index(clean.dt.normalize().unique())).sort_values()
    if len(clean) < 2:
        return 'M'
    diffs = clean.diff().dropna().dt.days
    if diffs.empty:
        return 'M'
    median_gap = float(diffs.median())
    if median_gap <= 1.5:
        return 'D'
    if median_gap <= 10:
        return 'W'
    return 'M'


def _resample_helper_to_revenue(
    helper_df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    revenue_dates: pd.Series,
    helper_frequency: str,
    fill_method: str,
    category_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Expand a coarser helper dataset (weekly/monthly) to match the daily revenue
    date index, applying the chosen fill strategy.

    fill_method:
      'distribute' — split the period total evenly across each day in that period
      'avg_monthly'— for monthly helper data, intelligently expand each month total
                     to daily rows at the finest available grain (ad/campaign/platform/total)
      'ffill'      — forward-fill the period value onto every day
      'interpolate'— linearly interpolate between period midpoints
    """
    helper_df = helper_df.copy()
    helper_df[date_col] = pd.to_datetime(helper_df[date_col], errors='coerce')
    helper_df = helper_df.dropna(subset=[date_col])

    # If helper is already daily or finer, no expansion needed
    if helper_frequency == 'D':
        return helper_df

    revenue_dates = pd.to_datetime(revenue_dates, errors='coerce').dropna()
    rev_min, rev_max = revenue_dates.min(), revenue_dates.max()
    all_days = pd.date_range(rev_min, rev_max, freq='D')

    category_cols = [c for c in (category_cols or []) if c and c in helper_df.columns]
    if category_cols:
        grouped_items = list(helper_df.groupby(category_cols, dropna=False))
    else:
        grouped_items = [(None, helper_df.copy())]
    result_frames = []

    for grp, subset in grouped_items:
        subset = subset.copy()

        subset = subset.set_index(date_col).sort_index()
        # Keep only value columns
        subset = subset[[c for c in value_cols if c in subset.columns]]
        # Numeric coerce
        for col in subset.columns:
            subset[col] = pd.to_numeric(subset[col], errors='coerce')

        if fill_method in {'distribute', 'avg_monthly'}:
            # Expand each period row across all days it covers, dividing evenly.
            # For monthly helper data, the value represents the FULL MONTH total,
            # so we spread it across every calendar day in that month regardless
            # of whether the source row is stamped on the 1st, last day, or any
            # other day inside the month.
            rows = []
            for period_anchor, row in subset.iterrows():
                period_anchor = pd.Timestamp(period_anchor)
                use_month_logic = helper_frequency == 'M' and fill_method == 'avg_monthly'
                if helper_frequency == 'M':
                    period_start = period_anchor.to_period('M').start_time.normalize()
                    period_end = period_anchor.to_period('M').end_time.normalize()
                elif helper_frequency == 'W':
                    # Treat the helper row as a weekly bucket anchored on the
                    # observed date and spread that weekly total across 7 days.
                    period_start = period_anchor.normalize()
                    period_end = (period_anchor + pd.Timedelta(days=6)).normalize()
                else:
                    period_start = period_anchor.normalize()
                    period_end = period_anchor.normalize()
                period_days = pd.date_range(period_start, period_end, freq='D')
                n_days = len(period_days)
                for day in period_days:
                    r = {'__expand_ds': day}
                    for col in row.index:
                        # avg_monthly only changes monthly behavior; daily data is never resampled
                        # because this function only runs for coarser-than-daily helpers.
                        r[col] = row[col] / n_days if pd.notna(row[col]) else float('nan')
                    rows.append(r)
            if not rows:
                continue
            expanded = pd.DataFrame(rows).set_index('__expand_ds')
            # Aggregate in case of overlapping periods or mixed monthly rows for the same grain.
            expanded = expanded.groupby(level=0).sum()
            # Reindex to all revenue days — days not covered by any period stay NaN
            # (not 0) so the model treats them as truly missing.
            expanded = expanded.reindex(all_days)

        elif fill_method == 'ffill':
            expanded = subset.reindex(all_days).ffill()

        elif fill_method == 'interpolate':
            expanded = subset.reindex(all_days)
            for col in expanded.columns:
                expanded[col] = expanded[col].interpolate(method='time')

        else:
            expanded = subset.reindex(all_days).ffill()

        expanded = expanded.reset_index().rename(columns={'index': date_col})
        if category_cols:
            grp_vals = grp if isinstance(grp, tuple) else (grp,)
            for col_name, col_val in zip(category_cols, grp_vals):
                expanded[col_name] = col_val
        result_frames.append(expanded)

    if not result_frames:
        return helper_df

    return pd.concat(result_frames, ignore_index=True)


def _build_model_dataframe(upload_meta: dict, mapping: dict) -> pd.DataFrame:
    revenue_cfg = mapping.get('revenue', {})
    revenue_df, _, revenue_sheet = _resolve_dataset(upload_meta, revenue_cfg.get('file_key', 'revenue'), revenue_cfg.get('sheet_name'))
    if revenue_df is None:
        raise ValueError('Revenue dataset is not configured.')
    revenue_cfg['sheet_name'] = revenue_sheet
    date_col = revenue_cfg.get('date_column')
    target_col = revenue_cfg.get('target_column')
    if not date_col or not target_col:
        raise ValueError('Revenue date and target columns are required.')
    df = dedupe_columns_keep_first(revenue_df.copy())
    df = _coerce_datetime_column(df, date_col)
    if df[date_col].isna().all():
        raise ValueError(f"The revenue date column '{date_col}' could not be interpreted as dates. Check whether the sheet contains Excel serial dates or text values that need cleaning.")
    df['__merge_ds'] = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
    # Create internal normalized columns early so downstream forecasting does not depend on sheet names.
    df['ds'] = df[date_col]

    if target_col == date_col:
        suggestions = _find_likely_target_columns(df, exclude=[date_col])
        hint = f" Try one of these instead: {', '.join(suggestions)}." if suggestions else ''
        raise ValueError(f"The Target Column cannot be the same as the Date Column ('{date_col}').{hint}")

    if target_col in df.columns:
        target_value = first_series(df, target_col)
        df[target_col] = pd.to_numeric(target_value, errors='coerce')
        df['y'] = df[target_col]

        if df['y'].notna().sum() == 0:
            suggestions = _find_likely_target_columns(df, exclude=[date_col, target_col])
            hint = f" Try one of these instead: {', '.join(suggestions)}." if suggestions else ''
            raise ValueError(f"The mapped Target Column '{target_col}' could not be interpreted as numeric revenue.{hint}")

        if _target_looks_like_timestamp(df['y'], df[date_col]):
            suggestions = _find_likely_target_columns(df, exclude=[date_col, target_col])
            hint = f" Likely revenue columns: {', '.join(suggestions)}." if suggestions else ''
            raise ValueError(
                f"The mapped Target Column '{target_col}' appears to contain datetime/timestamp values instead of revenue.{hint}"
            )
    for base_numeric in [mapping.get('roas_column'), *mapping.get('custom_regressor_columns', [])]:
        if base_numeric and base_numeric in df.columns:
            base_value = df[base_numeric]
            if isinstance(base_value, pd.DataFrame):
                base_value = base_value.iloc[:, 0]
            df[base_numeric] = pd.to_numeric(base_value, errors='coerce')

    profile_id_col = revenue_cfg.get('profile_id_column') or mapping.get('profile_id_column')
    if profile_id_col and profile_id_col in df.columns:
        profile_series = first_series(df, profile_id_col).astype(str).str.strip()
        profile_series = profile_series.replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, 'none': pd.NA})
        df['profile_id'] = profile_series

    for helper_name in ['leads', 'cost']:
        helper_cfg = mapping.get(helper_name, {})
        file_key = helper_cfg.get('file_key')
        value_col = helper_cfg.get('value_column')
        if not file_key or not value_col:
            continue
        helper_df, _, helper_sheet = _resolve_dataset(upload_meta, file_key, helper_cfg.get('sheet_name'))
        if helper_df is None:
            continue
        helper_cfg['sheet_name'] = helper_sheet
        helper_date_col = helper_cfg.get('date_column') or date_col
        helper_category_col = helper_cfg.get('category_column')
        if helper_date_col not in helper_df.columns:
            raise ValueError(f"The {helper_name} dataset date column '{helper_date_col}' was not found in the selected file or sheet.")
        if value_col == helper_date_col:
            suggestions = _find_likely_target_columns(helper_df, exclude=[helper_date_col])
            hint = f" Try one of these instead: {', '.join(suggestions)}." if suggestions else ''
            raise ValueError(
                f"The {helper_name} Value Column cannot be the same as the Date Column ('{helper_date_col}').{hint}"
            )
        helper_dim_cols = [
            helper_cfg.get('platform_column'),
            helper_cfg.get('affiliate_id_column'),
            helper_cfg.get('campaign_id_column'),
            helper_cfg.get('ad_id_column'),
            helper_category_col,
        ]
        keep_cols = []
        for c in [helper_date_col, *helper_dim_cols, value_col]:
            if c and c in helper_df.columns and c not in keep_cols:
                keep_cols.append(c)
        if not keep_cols:
            continue
        temp = helper_df[keep_cols].copy()
        temp = _coerce_datetime_column(temp, helper_date_col)
        if temp[helper_date_col].isna().all():
            raise ValueError(f"The {helper_name} date column '{helper_date_col}' could not be interpreted as dates.")

        # ── Frequency mismatch handling ───────────────────────────────────────
        helper_freq_cfg = helper_cfg.get('helper_frequency', 'auto')
        fill_method = helper_cfg.get('fill_method', 'distribute')
        if helper_freq_cfg == 'auto':
            helper_freq_cfg = _detect_frequency(temp[helper_date_col])
        revenue_freq = _detect_frequency(df[date_col])
        # Only resample when helper is coarser than revenue
        freq_order = {'D': 0, 'W': 1, 'M': 2}
        if freq_order.get(helper_freq_cfg, 0) > freq_order.get(revenue_freq, 0):
            temp[value_col] = pd.to_numeric(temp[value_col], errors='coerce')
            temp = _resample_helper_to_revenue(
                temp, helper_date_col, [value_col],
                df[date_col], helper_freq_cfg, fill_method,
                category_cols=[c for c in [helper_cfg.get('platform_column'), helper_category_col, helper_cfg.get('affiliate_id_column'), helper_cfg.get('campaign_id_column'), helper_cfg.get('ad_id_column')] if c],
            )

        temp['__merge_ds'] = pd.to_datetime(temp[helper_date_col], errors='coerce').dt.normalize()
        standardized_value_col = 'leads' if helper_name == 'leads' else 'cost'
        temp[value_col] = pd.to_numeric(temp[value_col], errors='coerce')
        if _target_looks_like_timestamp(temp[value_col], temp[helper_date_col]):
            suggestions = _find_likely_target_columns(helper_df, exclude=[helper_date_col, value_col])
            hint = f" Try one of these instead: {', '.join(suggestions)}." if suggestions else ''
            raise ValueError(
                f"The mapped {helper_name} Value Column '{value_col}' appears to contain datetime/timestamp values instead of {helper_name}.{hint}"
            )
        left_on, right_on = _choose_merge_keys(df, revenue_cfg, temp, helper_cfg)
        rename_map = {value_col: standardized_value_col}
        temp = temp.rename(columns=rename_map)

        if helper_name == 'cost':
            # Cost must live at a broad analytical grain first, ideally date + platform.
            # Joining spend to affiliate/campaign/ad rows duplicates the same spend across
            # many revenue rows and explodes company totals. So for cost we intentionally
            # build a clean lookup at the widest stable bucket and only then allocate it
            # down across detailed revenue rows that share that bucket.
            cost_bucket_pairs = [
                (revenue_cfg.get('platform_column'), helper_cfg.get('platform_column')),
                (revenue_cfg.get('category_column'), helper_cfg.get('category_column')),
                (revenue_cfg.get('affiliate_id_column'), helper_cfg.get('affiliate_id_column')),
            ]
            cost_left_on = ['__merge_ds']
            cost_right_on = ['__merge_ds']
            chosen_bucket_left = None
            chosen_bucket_right = None
            for left_candidate, right_candidate in cost_bucket_pairs:
                if left_candidate and right_candidate and left_candidate in df.columns and right_candidate in temp.columns:
                    chosen_bucket_left = left_candidate
                    chosen_bucket_right = right_candidate
                    cost_left_on.append(left_candidate)
                    cost_right_on.append(right_candidate)
                    break

            right_keep = [c for c in cost_right_on if c in temp.columns] + ([standardized_value_col] if standardized_value_col in temp.columns else [])
            temp = temp[right_keep]
            temp = _aggregate_helper_dataset(temp, cost_right_on, [standardized_value_col], mode='sum')

            # Keep both a bucket-level cost merge (date + platform/category/affiliate)
            # and a date-only company fallback. The bucket merge preserves the correct
            # platform grain when mappings line up. The date-only fallback prevents cost
            # from disappearing entirely when platform labels do not match exactly.
            bucket_temp = temp.copy()
            company_temp = temp.groupby(['__merge_ds'], as_index=False).agg(**{standardized_value_col: (standardized_value_col, 'sum')})
            bucket_cost_col = f'__{standardized_value_col}_bucket_total'
            company_cost_col = f'__{standardized_value_col}_company_total'
            bucket_temp = bucket_temp.rename(columns={standardized_value_col: bucket_cost_col})
            company_temp = company_temp.rename(columns={standardized_value_col: company_cost_col})

            df = df.merge(bucket_temp, how='left', left_on=cost_left_on, right_on=cost_right_on)
            df = df.merge(company_temp, how='left', on='__merge_ds')

            alloc_group = list(cost_left_on)
            bucket_row_counts = df.groupby(alloc_group, dropna=False)[bucket_cost_col].transform('size') if bucket_cost_col in df.columns else pd.Series(index=df.index, dtype='float64')
            bucket_row_counts = pd.to_numeric(bucket_row_counts, errors='coerce').replace(0, pd.NA)
            company_row_counts = df.groupby(['__merge_ds'], dropna=False)[company_cost_col].transform('size') if company_cost_col in df.columns else pd.Series(index=df.index, dtype='float64')
            company_row_counts = pd.to_numeric(company_row_counts, errors='coerce').replace(0, pd.NA)

            bucket_vals = pd.to_numeric(df[bucket_cost_col], errors='coerce') if bucket_cost_col in df.columns else pd.Series(index=df.index, dtype='float64')
            company_vals = pd.to_numeric(df[company_cost_col], errors='coerce') if company_cost_col in df.columns else pd.Series(index=df.index, dtype='float64')
            bucket_alloc = np.where(bucket_vals.notna() & bucket_row_counts.notna(), bucket_vals / bucket_row_counts, np.nan)
            company_alloc = np.where(company_vals.notna() & company_row_counts.notna(), company_vals / company_row_counts, np.nan)
            df[standardized_value_col] = np.where(pd.notna(bucket_alloc), bucket_alloc, company_alloc)

            # Store the analytical spend bucket used for validation/rollups.
            if chosen_bucket_left and chosen_bucket_left in df.columns:
                cost_bucket_series = first_series(df, chosen_bucket_left)
                cost_bucket_series = cost_bucket_series.where(cost_bucket_series.notna(), 'company_total')
                df['__cost_bucket'] = np.where(pd.notna(bucket_vals), cost_bucket_series, 'company_total')
            else:
                df['__cost_bucket'] = 'company_total'

            drop_cols = [c for c in [bucket_cost_col, company_cost_col] if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)
        else:
            right_keep = [c for c in right_on if c in temp.columns] + ([standardized_value_col] if standardized_value_col in temp.columns else [])
            temp = temp[right_keep]
            temp = _aggregate_helper_dataset(temp, right_on, [standardized_value_col], mode='sum')
            df = df.merge(temp, how='left', left_on=left_on, right_on=right_on)

            # When the helper dataset is less granular than revenue (for example,
            # monthly or daily company spend merged onto affiliate-level revenue
            # rows), a direct merge duplicates the helper value across every matched
            # revenue row. That inflates summed daily/monthly totals. Split the
            # helper total evenly across the matched revenue rows whenever we joined
            # on fewer dimensions than exist on the revenue side.
            revenue_detail_keys = [
                revenue_cfg.get('platform_column'),
                revenue_cfg.get('affiliate_id_column'),
                revenue_cfg.get('campaign_id_column'),
                revenue_cfg.get('ad_id_column'),
                revenue_cfg.get('category_column'),
            ]
            revenue_detail_keys = [c for c in revenue_detail_keys if c and c in df.columns]
            joined_detail_keys = [c for c in left_on if c != '__merge_ds']
            needs_allocation = bool(revenue_detail_keys) and len(joined_detail_keys) < len(revenue_detail_keys)
            if needs_allocation and standardized_value_col in df.columns:
                match_counts = df.groupby(left_on, dropna=False)[standardized_value_col].transform('size')
                match_counts = pd.to_numeric(match_counts, errors='coerce').replace(0, pd.NA)
                helper_vals = pd.to_numeric(df[standardized_value_col], errors='coerce')
                df[standardized_value_col] = np.where(
                    helper_vals.notna() & match_counts.notna(),
                    helper_vals / match_counts,
                    helper_vals,
                )

    event_cfg = mapping.get('events', {})
    if event_cfg.get('file_key'):
        helper_df, _, helper_sheet = _resolve_dataset(upload_meta, event_cfg.get('file_key'), event_cfg.get('sheet_name'))
        if helper_df is not None:
            event_cfg['sheet_name'] = helper_sheet
            helper_date_col = event_cfg.get('date_column') or date_col
            helper_category_col = event_cfg.get('category_column')
            event_col = event_cfg.get('event_flag_column')
            outage_col = event_cfg.get('outage_flag_column')
            if helper_date_col and helper_date_col not in helper_df.columns:
                raise ValueError(f"The events dataset date column '{helper_date_col}' was not found in the selected file or sheet.")
            keep_cols = [c for c in [helper_date_col, helper_category_col, event_col, outage_col] if c and c in helper_df.columns]
            if keep_cols:
                temp = helper_df[keep_cols].copy()
                temp = _coerce_datetime_column(temp, helper_date_col)
                if temp[helper_date_col].isna().all():
                    raise ValueError(f"The events date column '{helper_date_col}' could not be interpreted as dates.")
                temp['__merge_ds'] = pd.to_datetime(temp[helper_date_col], errors='coerce').dt.normalize()
                left_on, right_on = _choose_merge_keys(df, revenue_cfg, temp, helper_cfg)
                rename_map = {}
                if event_col:
                    rename_map[event_col] = 'event_flag'
                if outage_col:
                    rename_map[outage_col] = 'outage_flag'
                temp = temp.rename(columns=rename_map)
                right_keep = [c for c in right_on if c in temp.columns] + [c for c in ['event_flag', 'outage_flag'] if c in temp.columns]
                temp = temp[right_keep]
                temp = _aggregate_helper_dataset(temp, right_on, [c for c in ['event_flag', 'outage_flag'] if c in temp.columns], mode='flag')
                df = df.merge(temp, how='left', left_on=left_on, right_on=right_on)

    # ── Custom regressors external dataset ───────────────────────────────────
    cr_cfg = mapping.get('custom_regressors', {})
    if cr_cfg.get('file_key') and cr_cfg.get('value_columns'):
        cr_df, _, cr_sheet = _resolve_dataset(upload_meta, cr_cfg.get('file_key'), cr_cfg.get('sheet_name'))
        if cr_df is not None:
            cr_cfg['sheet_name'] = cr_sheet
            cr_date_col = cr_cfg.get('date_column') or date_col
            cr_category_col = cr_cfg.get('category_column')
            cr_value_cols = [c for c in cr_cfg.get('value_columns', []) if c and c in cr_df.columns]
            if cr_date_col in cr_df.columns and cr_value_cols:
                keep = [c for c in [cr_date_col, cr_category_col] + cr_value_cols if c and c in cr_df.columns]
                temp = cr_df[keep].copy()
                temp = _coerce_datetime_column(temp, cr_date_col)
                if temp[cr_date_col].notna().any():
                    cr_freq_cfg = cr_cfg.get('helper_frequency', 'auto')
                    cr_fill = cr_cfg.get('fill_method', 'distribute')
                    if cr_freq_cfg == 'auto':
                        cr_freq_cfg = _detect_frequency(temp[cr_date_col])
                    revenue_freq = _detect_frequency(df[date_col])
                    freq_order = {'D': 0, 'W': 1, 'M': 2}
                    if freq_order.get(cr_freq_cfg, 0) > freq_order.get(revenue_freq, 0):
                        for col in cr_value_cols:
                            temp[col] = pd.to_numeric(temp[col], errors='coerce')
                        temp = _resample_helper_to_revenue(
                            temp, cr_date_col, cr_value_cols,
                            df[date_col], cr_freq_cfg, cr_fill,
                            category_cols=[c for c in [cr_category_col] if c],
                        )
                    temp['__merge_ds'] = pd.to_datetime(temp[cr_date_col], errors='coerce').dt.normalize()
                    for col in cr_value_cols:
                        temp[col] = pd.to_numeric(temp[col], errors='coerce')
                    left_on, right_on = _choose_merge_keys(df, revenue_cfg, temp, cr_cfg)
                    right_keep = [c for c in right_on if c in temp.columns] + cr_value_cols
                    temp = temp[right_keep]
                    temp = _aggregate_helper_dataset(temp, right_on, cr_value_cols, mode='sum')
                    df = df.merge(temp, how='left', left_on=left_on, right_on=right_on)

    # ── Dynamic custom_N sheets ───────────────────────────────────────────────
    import re as _re2
    for mapping_key in sorted(mapping.keys()):
        if not _re2.match(r'^custom_\d+$', mapping_key):
            continue
        cs_cfg = mapping.get(mapping_key, {})
        cs_file_key = cs_cfg.get('file_key') or mapping_key
        cs_value_cols_raw = cs_cfg.get('value_columns', [])
        if not cs_file_key or not cs_value_cols_raw:
            continue
        cs_df, _, cs_sheet = _resolve_dataset(upload_meta, cs_file_key, cs_cfg.get('sheet_name'))
        if cs_df is None:
            continue
        cs_date_col = cs_cfg.get('date_column') or date_col
        cs_category_col = cs_cfg.get('category_column')
        cs_value_cols = [c for c in cs_value_cols_raw if c and c in cs_df.columns]
        if cs_date_col in cs_df.columns and cs_value_cols:
            keep = [c for c in [cs_date_col, cs_category_col] + cs_value_cols if c and c in cs_df.columns]
            temp = cs_df[keep].copy()
            temp = _coerce_datetime_column(temp, cs_date_col)
            if temp[cs_date_col].notna().any():
                cs_freq_cfg = cs_cfg.get('helper_frequency', 'auto')
                cs_fill = cs_cfg.get('fill_method', 'distribute')
                if cs_freq_cfg == 'auto':
                    cs_freq_cfg = _detect_frequency(temp[cs_date_col])
                revenue_freq = _detect_frequency(df[date_col])
                freq_order = {'D': 0, 'W': 1, 'M': 2}
                if freq_order.get(cs_freq_cfg, 0) > freq_order.get(revenue_freq, 0):
                    for col in cs_value_cols:
                        temp[col] = pd.to_numeric(temp[col], errors='coerce')
                    temp = _resample_helper_to_revenue(
                        temp, cs_date_col, cs_value_cols,
                        df[date_col], cs_freq_cfg, cs_fill,
                        category_cols=[c for c in [cs_category_col] if c],
                    )
                temp['__merge_ds'] = pd.to_datetime(temp[cs_date_col], errors='coerce').dt.normalize()
                for col in cs_value_cols:
                    temp[col] = pd.to_numeric(temp[col], errors='coerce')
                left_on, right_on = _choose_merge_keys(df, revenue_cfg, temp, cs_cfg)
                right_keep = [c for c in right_on if c in temp.columns] + cs_value_cols
                temp = temp[right_keep]
                temp = _aggregate_helper_dataset(temp, right_on, cs_value_cols, mode='sum')
                df = df.merge(temp, how='left', left_on=left_on, right_on=right_on)

    # Clean merge helper columns and prefer the internal normalized fields.
    drop_cols = [c for c in df.columns if c.startswith('__merge_ds') or (c.endswith('_y') and c[:-2] in df.columns)]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    if 'ds' not in df.columns:
        df['ds'] = df[date_col]
    if 'y' not in df.columns:
        df['y'] = pd.to_numeric(df[target_col], errors='coerce')
    detail_cols = [c for c, _ in _available_dimension_columns(df, revenue_cfg, include_category=True)]
    if detail_cols:
        df = _normalize_dimension_columns(df, detail_cols)
        df['category'] = _build_dimension_label_series(df, revenue_cfg, fallback_col=revenue_cfg.get('category_column'), default='company_total')
        df['series_id'] = df['category'].astype(str)
    else:
        df['series_id'] = 'company_total'
        df['category'] = 'company_total'
    df = enforce_schema(df)
    return dedupe_columns_keep_first(df)



# Consolidation stability aliases
_safe_number = safe_number
_json_ready_records = json_ready_records
_plot_card = plot_card
_safe_divide = safe_divide
_diagnostic_frequency_label = diagnostic_frequency_label
_monthly_equivalent_factor = monthly_equivalent_factor
_format_history_window_label = format_history_window_label
_build_driver_impact_frame = build_driver_impact_frame
