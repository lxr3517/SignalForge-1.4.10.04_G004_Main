from __future__ import annotations


def frequency_family(frequency: str | None) -> str:
    freq = str(frequency or "D").upper()
    if freq.startswith("M"):
        return "M"
    if freq.startswith("W"):
        return "W"
    return "D"


def parse_horizon_to_periods(horizon: str, frequency: str) -> int:
    family = frequency_family(frequency)
    if horizon.endswith("D"):
        days = int(horizon[:-1])
        if family == "D":
            return days
        if family == "W":
            return max(1, days // 7)
        if family == "M":
            return max(1, days // 30)
    if horizon.endswith("M"):
        months = int(horizon[:-1])
        if family == "M":
            return months
        if family == "W":
            return months * 4
        if family == "D":
            return months * 30
    return 30
