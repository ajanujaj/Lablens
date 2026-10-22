"""
Gate: does the extracted text actually contain lab results?

The model was trained only on reports that HAVE results, so if you give it a discharge summary,
prescription or "tests done" list it will invent values. This cheap check runs first and the
model is skipped when no line looks like "<known test name> ... <number>".
"""
import re

# Common names for the 20 supported tests (as they appear on Indian lab reports)
TEST_PATTERNS = {
    "hemoglobin": r"h(a)?emoglobin|\bhb\b|\bhgb\b",
    "wbc": r"\bwbc\b|\btlc\b|leu[ck]ocyte",
    "platelets": r"platelet|\bplt\b",
    "glucose_fasting": r"glucose|blood sugar|\bfbs\b|\bfbg\b",
    "hba1c": r"hba1c|\ba1c\b|glyc(at|osyl)ated",
    "cholesterol_total": r"cholesterol|\bchol\b",
    "ldl": r"\bldl\b",
    "hdl": r"\bhdl\b",
    "triglycerides": r"triglyceride|\btg\b",
    "tsh": r"\btsh\b|thyroid stimulating",
    "t3_total": r"\bt3\b|triiodothyronine",
    "t4_total": r"\bt4\b|thyroxine",
    "creatinine": r"creatinine",
    "urea": r"\burea\b|\bbun\b",
    "uric_acid": r"uric acid",
    "alt_sgpt": r"\bsgpt\b|\balt\b|alanine",
    "ast_sgot": r"\bsgot\b|\bast\b|aspartate",
    "bilirubin_total": r"bilirubin",
    "vitamin_d": r"vitamin\s*d\b|25[- ]?(oh|hydroxy)",
    "vitamin_b12": r"vitamin\s*b\s*12|\bb12\b|cobalamin",
}
_TESTS = {k: re.compile(v, re.I) for k, v in TEST_PATTERNS.items()}
# dd/mm/yyyy, dd-mm-yy, yyyy-mm-dd ... (same separator twice, so "13.0-17.0" ranges are kept)
_DATE = re.compile(r"\b(?:\d{1,2}([/.-])\d{1,2}\1(?:\d{4}|\d{2})|\d{4}([/.-])\d{1,2}\2\d{1,2})\b")


def strip_dates(text: str) -> str:
    return _DATE.sub(" ", text)
# A result value: a number that is not a medicine dose ("1000mcg", "500 mg", "2 tablets")
# or a time period ("3 months").  "mg/dL" style units are still allowed.
_VALUE = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?(?![\d.])"
    r"(?!\s*(?:mg|mcg|g|iu|tab|tablet|cap|capsule|ml|day|week|month|year|hour|time)s?\b(?!\s*/))", re.I)
LOOKAHEAD_CHARS = 80      # value must follow the test name within this many characters (may wrap lines)


def find_lab_tests(text: str) -> list[str]:
    """Supported tests whose name is followed closely by a result value."""
    lines = [strip_dates(l) for l in text.splitlines()]
    found = []
    for i, line in enumerate(lines):
        for key, pat in _TESTS.items():
            if key in found:
                continue
            for m in pat.finditer(line):
                after = " ".join([line[m.end():]] + lines[i + 1:i + 3])[:LOOKAHEAD_CHARS]
                if _VALUE.search(pat.sub(" ", after)):
                    found.append(key)
                    break
    return found
