from __future__ import annotations

import pandas as pd


def _has_mapping(mapping: dict | None, key: str) -> bool:
    if not mapping:
        return False
    val = mapping.get(key)
    if isinstance(val, dict):
        return bool(val.get("file_key") or val.get("value_column") or val.get("value_columns"))
    return bool(val)


def build_quality_guidance(mapping: dict | None, dataset_info: list[dict] | None = None, training_window_context: dict | None = None, manual_adjustments: dict | None = None, saved_settings: dict | None = None) -> list[dict]:
    dataset_info = dataset_info or []
    training_window_context = training_window_context or {}
    manual_adjustments = manual_adjustments or {}
    saved_settings = saved_settings or {}
    items: list[dict] = []

    if _has_mapping(mapping, 'cost') and _has_mapping(mapping, 'leads') and _has_mapping(mapping, 'target_column') and not saved_settings.get('enable_revenue_lag_modeling'):
        items.append({
            'level': 'info',
            'title': 'Lag-aware modeling is currently off',
            'body': 'You have the ingredients for cost, leads, and revenue pairing, but revenue lag modeling is not enabled in this run setup.',
            'action': 'Recommended action: Turn on Lag Intelligence in Forecast Setup if revenue usually trails spend and leads.',
        })

    expanded = [d for d in dataset_info if d.get('resampled')]
    if expanded:
        labels = ', '.join(d.get('label', 'Helper data') for d in expanded[:3])
        items.append({
            'level': 'info',
            'title': 'Helper datasets are being expanded',
            'body': f'{labels} are coarser than the revenue series, so SignalForge will expand them to the revenue grain before modeling.',
            'action': 'Recommended action: Double-check the fill method if the helper files represent monthly or weekly totals.',
        })

    actual_start = training_window_context.get('actual_data_start')
    actual_end = training_window_context.get('actual_data_end')
    if actual_start and actual_end:
        try:
            span = (pd.to_datetime(actual_end) - pd.to_datetime(actual_start)).days + 1
        except Exception:
            span = None
        if span is not None and span < 90:
            items.append({
                'level': 'warning',
                'title': 'Training history is short',
                'body': f'This run only has about {span} days of actual history, which can make pattern detection less stable.',
                'action': 'Recommended action: Use all history or a lighter model package if the recent window is intentionally short.',
            })

    has_adjustments = any(manual_adjustments.get(k) for k in ('excluded_ranges', 'outages', 'promos', 'forced_holidays'))
    if not has_adjustments:
        items.append({
            'level': 'info',
            'title': 'No manual adjustments are tagged',
            'body': 'Manual adjustments help you annotate outages, promos, and forced holiday effects before the forecast is trained.',
            'action': 'Recommended action: Add adjustments only when you know the data includes meaningful one-off events.',
        })

    return items


def build_forecast_guidance(mapping: dict | None, training_window_context: dict | None = None, forecast_settings: dict | None = None) -> list[dict]:
    training_window_context = training_window_context or {}
    forecast_settings = forecast_settings or {}
    items: list[dict] = []

    if _has_mapping(mapping, 'cost') and _has_mapping(mapping, 'leads') and _has_mapping(mapping, 'target_column'):
        if forecast_settings.get('enable_revenue_lag_modeling'):
            items.append({
                'level': 'good',
                'title': 'Lag Intelligence is enabled',
                'body': 'This run will build extra revenue memory features so delayed monetization can be compared against recent spend and lead flow.',
                'action': 'Tip: Keep the Standard profile unless you already know revenue settles much faster or slower than normal.',
            })
        else:
            items.append({
                'level': 'warning',
                'title': 'Lag Intelligence is recommended',
                'body': 'Cost, leads, and revenue are all mapped for this run, so lag-aware modeling can improve pairing when revenue lands after acquisition.',
                'action': 'Recommended action: Turn on Enable lag-aware revenue modeling and keep the Standard profile to start.',
            })

    actual_start = training_window_context.get('actual_data_start')
    actual_end = training_window_context.get('actual_data_end')
    if actual_start and actual_end:
        items.append({
            'level': 'info',
            'title': 'Preset windows anchor to actual data end',
            'body': f'Preset windows will end on {actual_end}, which is the last day of uploaded actual data rather than today\'s date.',
            'action': 'Recommended action: Use Last 90 / 180 / 365 when you want a recent but data-aligned training window.',
        })

    return items
