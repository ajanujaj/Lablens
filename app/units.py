"""
Unit conversion done by code, not by the model.

LLMs are unreliable at arithmetic (e.g. 2.95 nmol/L T3 -> it wrote 129 ng/dL, correct is 192).
The model is still useful for *reading* the report (which test, which number, which unit);
this module recomputes the standard-unit value from `reported_value` + `reported_unit`.

Factors are the same as data/synth_data.py:  reported = standard * scale + offset
"""
import re

# test -> (decimals of standard value, {normalised unit: (scale, offset)})
_STD = (1, 0)
CONVERSIONS = {
    "hemoglobin":        (1, {"g/dl": _STD, "gm/dl": _STD, "gm%": _STD, "g%": _STD, "g/l": (10, 0)}),
    "wbc":               (2, {"10^3/ul": _STD, "x10^3/ul": _STD, "10^3/cumm": _STD, "thou/ul": _STD,
                              "k/ul": _STD, "10^9/l": _STD, "x10^9/l": _STD,
                              "/cumm": (1000, 0), "cells/cumm": (1000, 0), "/ul": (1000, 0), "/mm3": (1000, 0)}),
    "platelets":         (0, {"10^3/ul": _STD, "x10^3/ul": _STD, "10^3/cumm": _STD, "thou/ul": _STD,
                              "k/ul": _STD, "10^9/l": _STD, "x10^9/l": _STD,
                              "lakh/cumm": (0.01, 0), "lakhs/cumm": (0.01, 0), "lakh/ul": (0.01, 0),
                              "/cumm": (1000, 0), "/ul": (1000, 0), "/mm3": (1000, 0)}),
    "glucose_fasting":   (0, {"mg/dl": _STD, "mmol/l": (1 / 18.016, 0)}),
    "hba1c":             (1, {"%": _STD, "mmol/mol": (10.929, -23.5)}),
    "cholesterol_total": (0, {"mg/dl": _STD, "mmol/l": (1 / 38.67, 0)}),
    "ldl":               (0, {"mg/dl": _STD, "mmol/l": (1 / 38.67, 0)}),
    "hdl":               (0, {"mg/dl": _STD, "mmol/l": (1 / 38.67, 0)}),
    "triglycerides":     (0, {"mg/dl": _STD, "mmol/l": (1 / 88.57, 0)}),
    "tsh":               (2, {"uiu/ml": _STD, "miu/l": _STD, "mu/l": _STD, "uu/ml": _STD,
                              "piu/ml": _STD, "plu/ml": _STD, "ulu/ml": _STD}),      # last 3: common OCR misreads
    "t3_total":          (0, {"ng/dl": _STD, "nmol/l": (0.01536, 0)}),
    "t4_total":          (1, {"ug/dl": _STD, "mcg/dl": _STD, "nmol/l": (12.87, 0)}),
    "creatinine":        (2, {"mg/dl": _STD, "umol/l": (88.42, 0)}),
    "urea":              (0, {"mg/dl": _STD, "mmol/l": (0.1665, 0)}),
    "uric_acid":         (1, {"mg/dl": _STD, "umol/l": (59.48, 0)}),
    "alt_sgpt":          (0, {"u/l": _STD, "iu/l": _STD}),
    "ast_sgot":          (0, {"u/l": _STD, "iu/l": _STD}),
    "bilirubin_total":   (2, {"mg/dl": _STD, "umol/l": (17.1, 0)}),
    "vitamin_d":         (1, {"ng/ml": _STD, "nmol/l": (2.496, 0)}),
    "vitamin_b12":       (0, {"pg/ml": _STD, "pmol/l": (0.7378, 0)}),
}


def normalise_unit(u) -> str:
    if not isinstance(u, str):
        return ""
    u = u.lower().replace("µ", "u").replace("μ", "u").replace("³", "^3").replace("⁹", "^9")
    u = u.replace("×", "x").replace("cu mm", "cumm").replace("cu.mm", "cumm")
    return re.sub(r"[^a-z0-9/^%.]", "", u)               # drops spaces, OCR junk like "•"


def to_standard(test: str, reported_value, reported_unit):
    """Standard-unit value computed by code, or None if the unit isn't known."""
    if test not in CONVERSIONS:
        return None
    dec, table = CONVERSIONS[test]
    factor = table.get(normalise_unit(reported_unit))
    try:
        v = float(reported_value)
    except (TypeError, ValueError):
        return None
    if factor is None:
        return None
    scale, offset = factor
    out = round((v - offset) / scale, dec)
    return int(out) if dec == 0 else out
