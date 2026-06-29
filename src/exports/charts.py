from __future__ import annotations

import pandas as pd
import plotly.express as px


def make_forecast_chart(df: pd.DataFrame):
    return px.line(df, x="ds", y="yhat")
