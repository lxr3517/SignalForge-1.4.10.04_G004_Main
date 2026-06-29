from __future__ import annotations

from io import BytesIO

import pandas as pd


def bundle_to_excel_bytes(bundle: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame(bundle.get("all_metrics", [])).to_excel(writer, sheet_name="metrics", index=False)
        pd.DataFrame(bundle["decision"].get("best_forecast", [])).to_excel(writer, sheet_name="best_forecast", index=False)
        pd.DataFrame(bundle["decision"].get("blended_forecast", [])).to_excel(writer, sheet_name="blended_forecast", index=False)
        pd.DataFrame(bundle.get("scenarios", [])).to_excel(writer, sheet_name="scenarios", index=False)
    return output.getvalue()
