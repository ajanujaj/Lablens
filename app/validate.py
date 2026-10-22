"""
Step 4: sanity-check the model's JSON with plain code.

A 1.5B model can invent tests or numbers, mangle units or repeat itself. Every result is
checked; anything wrong is reported in `warnings`. In strict mode (default) a result whose
reported value does not appear anywhere in the report text is treated as made up and moved
to `rejected_results` rather than shown as a real value.
"""
import os
import re
from datetime import date

from .gate import strip_dates
from .prompts import STANDARD_UNITS
from .units import to_standard

FLAGS = {"H", "L", "N"}
STRICT = os.getenv("LABLENS_STRICT", "1") == "1"


def _numbers_in(text: str) -> list[float]:
    text = strip_dates(text)                             # "11" in 10/11/2023 is not a result
    text = re.sub(r"(?<=\d),(?=\d)", "", text)          # 1,50,000 -> 150000
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def _appears(value, numbers: list[float]) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return any(abs(v - n) <= 1e-9 + 1e-6 * abs(n) for n in numbers)


def validate(result: dict | None, source_text: str) -> tuple[dict | None, list[str]]:
    if result is None:
        return None, ["Model did not return valid JSON. See raw_outputs."]
    warnings = []
    if result.get("_salvaged"):
        warnings.append("Model output was cut off or malformed; only complete results were kept.")

    d = result.get("sample_date")
    if d:
        try:
            date.fromisoformat(str(d))
        except ValueError:
            warnings.append(f"sample_date '{d}' is not a valid YYYY-MM-DD date.")
    else:
        warnings.append("No sample date found.")

    numbers = _numbers_in(source_text)
    clean, unknown, rejected, seen = [], [], [], set()
    for r in result.get("results") or []:
        if not isinstance(r, dict) or "test" not in r:
            continue
        test = r["test"]
        if test in seen:
            continue                                     # model repeated itself
        seen.add(test)
        if test not in STANDARD_UNITS:
            unknown.append(r)
            warnings.append(f"'{test}' is not one of the 20 supported tests; moved to unrecognised_results.")
            continue
        if not _appears(r.get("reported_value"), numbers):
            if STRICT:
                rejected.append(r)
                warnings.append(f"{test}: value {r.get('reported_value')} is not in the report - "
                                "rejected as likely made up.")
                continue
            warnings.append(f"{test}: reported value {r.get('reported_value')} not found in the report text "
                            "- please double-check.")
        # Unit conversion is done by code: the model reads the number + unit, code does the maths.
        std = to_standard(test, r.get("reported_value"), r.get("reported_unit"))
        if std is not None:
            try:
                model_v = float(r.get("value"))
            except (TypeError, ValueError):
                model_v = None
            if model_v is None or abs(model_v - std) > max(0.01 * abs(std), 1e-9):
                warnings.append(f"{test}: model converted {r.get('reported_value')} {r.get('reported_unit')} to "
                                f"{r.get('value')}; corrected to {std} {STANDARD_UNITS[test]}.")
                r = {**r, "value": std, "value_corrected_from": r.get("value")}
            r = {**r, "unit": STANDARD_UNITS[test]}
        elif r.get("reported_unit"):
            warnings.append(f"{test}: unknown unit '{r.get('reported_unit')}', value not verified.")
        if not isinstance(r.get("value"), (int, float)):
            warnings.append(f"{test}: value '{r.get('value')}' is not a number.")
        if r.get("unit") != STANDARD_UNITS[test]:
            warnings.append(f"{test}: unit '{r.get('unit')}' should be '{STANDARD_UNITS[test]}'.")
        if r.get("flag") not in FLAGS:
            warnings.append(f"{test}: flag '{r.get('flag')}' should be H, L or N.")
        clean.append(r)

    out = {"sample_date": result.get("sample_date"), "patient": result.get("patient"), "results": clean}
    if unknown:
        out["unrecognised_results"] = unknown
    if rejected:
        out["rejected_results"] = rejected
    if not clean:
        warnings.append("No reliable test results were found in this report.")
    return out, warnings
