from __future__ import annotations

from io import BytesIO

import pandas as pd


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = uploaded_file.name.lower().split(".")[-1]
    raw_bytes = uploaded_file.getvalue()
    buffer = BytesIO(raw_bytes)

    if suffix == "csv":
        return pd.read_csv(buffer)
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(buffer)
    if suffix == "parquet":
        return pd.read_parquet(buffer)
    raise ValueError(f"Unsupported file type: {suffix}")
