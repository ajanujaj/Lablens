#!/usr/bin/env python3
"""
LabLens - Week 1: synthetic Indian lab-report generator.

Creates fake blood reports from 6 FICTIONAL labs (different layouts, units,
test names, date formats, OCR-style noise) with exact ground truth, plus
ready-to-train chat datasets:

  reports.jsonl            every report: text + full ground truth + metadata
  extract_{train,val,test}.jsonl   SFT task 1: report text -> normalized JSON
  drift_{train,val,test}.jsonl     SFT/RL task 2: multi-year history -> trend labels
  samples/*.txt            one example report per layout, to eyeball

The "test" split for extraction uses a lab layout the model never sees in
training (default: kaveri) - that is your generalization benchmark.

score_extraction() at the bottom is your eval metric now and your GRPO
reward function later.

Pure Python standard library. Runs on a laptop, Kaggle or Colab.
    python lablens_synth.py --patients 2000 --out data --seed 42
"""
import argparse
import hashlib
import json
import os
import random
import re
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# 1. Test catalog
#    unit = canonical/standard unit. alt = list of (unit, scale, offset, dec)
#    where  alt_value = canonical_value * scale + offset
#    ref  = (low, high) or {"M": (...), "F": (...)}; None = open-ended
# ---------------------------------------------------------------------------
TESTS = {
    "hemoglobin": dict(panel="CBC", unit="g/dL", dec=1, sd=1.2,
        mean={"M": 14.8, "F": 12.9}, ref={"M": (13.0, 17.0), "F": (12.0, 15.5)},
        aliases=["Hemoglobin", "Haemoglobin", "Hb", "HGB", "Haemoglobin (Hb)"],
        alt=[("g/L", 10, 0, 0)]),
    "wbc": dict(panel="CBC", unit="10^3/µL", dec=2, sd=1.6, mean=7.2, ref=(4.0, 11.0),
        aliases=["Total Leucocyte Count", "TLC", "WBC Count", "Total WBC Count"],
        alt=[("/cumm", 1000, 0, 0), ("10^9/L", 1, 0, 2)]),
    "platelets": dict(panel="CBC", unit="10^3/µL", dec=0, sd=60, mean=260, ref=(150, 410),
        aliases=["Platelet Count", "Platelets", "PLT", "Platelet Count (PLT)"],
        alt=[("lakh/cumm", 0.01, 0, 2), ("/cumm", 1000, 0, 0)]),
    "glucose_fasting": dict(panel="DIABETES", unit="mg/dL", dec=0, sd=10, mean=92, ref=(70, 100),
        aliases=["Glucose Fasting", "FBS", "Fasting Blood Sugar", "Plasma Glucose (F)", "Blood Sugar - Fasting"],
        alt=[("mmol/L", 1 / 18.016, 0, 1)]),
    "hba1c": dict(panel="DIABETES", unit="%", dec=1, sd=0.35, mean=5.4, ref=(4.0, 5.6),
        aliases=["HbA1c", "Glycated Haemoglobin", "Glycosylated Hemoglobin (HbA1c)", "A1c"],
        alt=[("mmol/mol", 10.929, -23.5, 0)]),
    "cholesterol_total": dict(panel="LIPID", unit="mg/dL", dec=0, sd=30, mean=185, ref=(None, 200),
        aliases=["Total Cholesterol", "Cholesterol, Total", "S. Cholesterol", "CHOL"],
        alt=[("mmol/L", 1 / 38.67, 0, 2)]),
    "ldl": dict(panel="LIPID", unit="mg/dL", dec=0, sd=28, mean=110, ref=(None, 100),
        aliases=["LDL Cholesterol", "LDL-C", "Cholesterol - LDL", "LDL (Direct)"],
        alt=[("mmol/L", 1 / 38.67, 0, 2)]),
    "hdl": dict(panel="LIPID", unit="mg/dL", dec=0, sd=9, mean={"M": 44, "F": 52},
        ref={"M": (40, None), "F": (50, None)},
        aliases=["HDL Cholesterol", "HDL-C", "Cholesterol - HDL", "HDL (Direct)"],
        alt=[("mmol/L", 1 / 38.67, 0, 2)]),
    "triglycerides": dict(panel="LIPID", unit="mg/dL", dec=0, sd=45, mean=140, ref=(None, 150),
        aliases=["Triglycerides", "TG", "S. Triglycerides", "Triglyceride"],
        alt=[("mmol/L", 1 / 88.57, 0, 2)]),
    "tsh": dict(panel="THYROID", unit="µIU/mL", dec=2, sd=1.0, mean=2.3, ref=(0.4, 4.5),
        aliases=["TSH", "Thyroid Stimulating Hormone", "TSH (Ultrasensitive)", "S. TSH"],
        alt=[("mIU/L", 1, 0, 2)]),
    "t3_total": dict(panel="THYROID", unit="ng/dL", dec=0, sd=20, mean=120, ref=(80, 200),
        aliases=["T3, Total", "Total T3", "Triiodothyronine (T3)", "T3"],
        alt=[("nmol/L", 0.01536, 0, 2)]),
    "t4_total": dict(panel="THYROID", unit="µg/dL", dec=1, sd=1.5, mean=8.0, ref=(5.1, 14.1),
        aliases=["T4, Total", "Total T4", "Thyroxine (T4)", "T4"],
        alt=[("nmol/L", 12.87, 0, 1)]),
    "creatinine": dict(panel="KFT", unit="mg/dL", dec=2, sd=0.15, mean={"M": 0.95, "F": 0.75},
        ref={"M": (0.7, 1.3), "F": (0.5, 1.1)},
        aliases=["Creatinine", "S. Creatinine", "Serum Creatinine", "Creatinine, Serum"],
        alt=[("µmol/L", 88.42, 0, 0)]),
    "urea": dict(panel="KFT", unit="mg/dL", dec=0, sd=6, mean=26, ref=(15, 40),
        aliases=["Urea", "Blood Urea", "S. Urea", "Urea, Serum"],
        alt=[("mmol/L", 0.1665, 0, 1)]),
    "uric_acid": dict(panel="KFT", unit="mg/dL", dec=1, sd=1.0, mean={"M": 5.8, "F": 4.6},
        ref={"M": (3.5, 7.2), "F": (2.6, 6.0)},
        aliases=["Uric Acid", "S. Uric Acid", "Uric Acid, Serum"],
        alt=[("µmol/L", 59.48, 0, 0)]),
    "alt_sgpt": dict(panel="LFT", unit="U/L", dec=0, sd=10, mean=28, ref=(7, 45),
        aliases=["SGPT", "ALT (SGPT)", "Alanine Aminotransferase", "SGPT/ALT"],
        alt=[("IU/L", 1, 0, 0)]),
    "ast_sgot": dict(panel="LFT", unit="U/L", dec=0, sd=8, mean=25, ref=(8, 40),
        aliases=["SGOT", "AST (SGOT)", "Aspartate Aminotransferase", "SGOT/AST"],
        alt=[("IU/L", 1, 0, 0)]),
    "bilirubin_total": dict(panel="LFT", unit="mg/dL", dec=2, sd=0.25, mean=0.7, ref=(0.2, 1.2),
        aliases=["Bilirubin Total", "Total Bilirubin", "S. Bilirubin (Total)", "T. Bilirubin"],
        alt=[("µmol/L", 17.1, 0, 1)]),
    "vitamin_d": dict(panel="VITAMINS", unit="ng/mL", dec=1, sd=9, mean=22, ref=(30, 100),
        aliases=["Vitamin D (25-OH)", "25-Hydroxy Vitamin D", "Vit D3", "Vitamin D Total"],
        alt=[("nmol/L", 2.496, 0, 1)]),
    "vitamin_b12": dict(panel="VITAMINS", unit="pg/mL", dec=0, sd=120, mean=330, ref=(211, 911),
        aliases=["Vitamin B12", "Vit B12", "Cyanocobalamin", "Vitamin B-12"],
        alt=[("pmol/L", 0.7378, 0, 0)]),
}

PANEL_TITLES = {
    "CBC": "Complete Blood Count", "DIABETES": "Diabetes Profile", "LIPID": "Lipid Profile",
    "THYROID": "Thyroid Profile (Total)", "KFT": "Kidney Function Test",
    "LFT": "Liver Function Test", "VITAMINS": "Vitamins",
}
PANELS = {}
for _k, _t in TESTS.items():
    PANELS.setdefault(_t["panel"], []).append(_k)


def per_sex(x, sex):
    return x[sex] if isinstance(x, dict) else x


# ---------------------------------------------------------------------------
# 2. Number / unit / range helpers
# ---------------------------------------------------------------------------
def fmt(v, dec):
    return f"{v:.{dec}f}" if dec > 0 else str(int(round(v)))


def to_unit(v, u):            # u = (unit, scale, offset, dec)
    return round(v * u[1] + u[2], u[3])


def from_unit(v, u):
    return (v - u[2]) / u[1]


def fmt_range(lo, hi, dec, sep):
    if lo is None:
        return f"< {fmt(hi, dec)}"
    if hi is None:
        return f"> {fmt(lo, dec)}"
    return f"{fmt(lo, dec)}{sep}{fmt(hi, dec)}"


def flag_for(v, lo, hi):
    if lo is not None and v < lo:
        return "L"
    if hi is not None and v > hi:
        return "H"
    return "N"


def stable_pct(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 100


# ---------------------------------------------------------------------------
# 3. Fictional people
# ---------------------------------------------------------------------------
FIRST_M = ["Rahul", "Amit", "Suresh", "Vikram", "Arjun", "Rohan", "Sanjay", "Manoj",
           "Karthik", "Imran", "Harpreet", "Deepak", "Anil", "Rajesh", "Nikhil"]
FIRST_F = ["Priya", "Anjali", "Sunita", "Kavya", "Neha", "Pooja", "Lakshmi", "Fatima",
           "Meera", "Divya", "Rekha", "Sneha", "Asha", "Nandini", "Simran"]
LAST = ["Sharma", "Verma", "Iyer", "Reddy", "Nair", "Patel", "Gupta", "Khan", "Singh",
        "Das", "Joshi", "Kulkarni", "Menon", "Banerjee", "Rao", "Jain"]
DOCTORS = ["Dr. A. Mehta", "Dr. S. Krishnan", "Dr. R. Bose", "Dr. P. Deshpande",
           "Self", "Dr. N. Qureshi", "Dr. K. Pillai"]


# ---------------------------------------------------------------------------
# 4. Six fictional lab layouts
#    each renderer gets header dict h and sections [(title, rows)]
#    row = dict(name, val, unit, rng, flag)
# ---------------------------------------------------------------------------
def r_sunrise(h, sections):
    L = ["SUNRISE DIAGNOSTICS",
         f"Quality Healthcare Since 1998 | {h['address']} | Ph: {h['phone']}",
         "-" * 80,
         f"Patient Name : {h['title']} {h['pname']:<22} Age/Sex : {h['age']} Y / {h['sex']}",
         f"Ref. By      : {h['refdoc']:<25} Lab No  : {h['lab_id']}",
         f"Collected    : {h['collected']:<25} Reported: {h['reported']}",
         "-" * 80,
         f"{'Test Description':<36}{'Result':<12}{'Unit':<12}Bio. Ref. Interval",
         "-" * 80]
    for title, rows in sections:
        L += ["", title.upper()]
        for r in rows:
            v = r["val"] + ("" if r["flag"] == "N" else " " + r["flag"])
            L.append(f"{r['name']:<36}{v:<12}{r['unit']:<12}{r['rng']}")
    L += ["", "*** End of Report ***",
          "This is a computer generated report and does not require a signature."]
    return "\n".join(L)


def r_medpath(h, sections):
    word = {"H": "  HIGH", "L": "  LOW", "N": ""}
    L = ["MedPath Labs - Your health, our priority", h["address"], "",
         f"Name: {h['pname']} | {h['age']} Yrs | {'Male' if h['sex'] == 'M' else 'Female'}",
         f"Sample Date: {h['collected']}    Report ID: {h['lab_id']}",
         f"Referred by: {h['refdoc']}", ""]
    for title, rows in sections:
        L.append(f"[{title}]")
        for r in rows:
            L.append(f"{r['name']} : {r['val']} {r['unit']}  (Ref: {r['rng']}){word[r['flag']]}")
        L.append("")
    L.append("Results relate only to the sample tested. Please correlate clinically.")
    return "\n".join(L)


def r_citycare(h, sections):
    L = ["CITYCARE HOSPITAL - DEPARTMENT OF LABORATORY MEDICINE", h["address"], "",
         f"UHID: {h['lab_id']}   Patient: {h['title']} {h['pname']}   Age: {h['age']}   Gender: {h['sex']}",
         f"Consultant: {h['refdoc']}   Sample Received: {h['collected']}   Report Date: {h['reported']}", ""]
    for title, rows in sections:
        L.append(f"## {title}")
        L.append("| Investigation | Result | Flag | Units | Reference Range |")
        L.append("|---|---|---|---|---|")
        for r in rows:
            f = "" if r["flag"] == "N" else r["flag"]
            L.append(f"| {r['name']} | {r['val']} | {f} | {r['unit']} | {r['rng']} |")
        L.append("")
    L.append("Verified by: Consultant Pathologist (electronically signed)")
    return "\n".join(L)


def r_apex(h, sections):
    word = {"H": " (High)", "L": " (Low)", "N": ""}
    L = ["Apex Pathology Centre", f"{h['address']}  Tel {h['phone']}", "=" * 50,
         f"Patient     {h['title']} {h['pname']}",
         f"Age / Sex   {h['age']} years / {'Male' if h['sex'] == 'M' else 'Female'}",
         f"Collected   {h['collected']}",
         f"Ref Doctor  {h['refdoc']}", "=" * 50]
    for title, rows in sections:
        L += ["", title.upper(), "~" * len(title)]
        for r in rows:
            L.append(r["name"])
            L.append(f"    Result : {r['val']} {r['unit']}{word[r['flag']]}")
            L.append(f"    Normal : {r['rng']}")
    L += ["", "-- End of report --"]
    return "\n".join(L)


def r_nirmal(h, sections):      # no flags at all: model must compute them
    L = ["NIRMAL CLINICAL LAB", f"PATIENT;{h['pname']};AGE;{h['age']};SEX;{h['sex']}",
         f"DATE;{h['collected']};ID;{h['lab_id']};REF;{h['refdoc']}", "",
         "PARAMETER;VALUE;UNIT;REFERENCE"]
    for title, rows in sections:
        L.append(f"#{title.upper()}")
        for r in rows:
            L.append(f"{r['name']};{r['val']};{r['unit']};{r['rng']}")
    return "\n".join(L)


def r_kaveri(h, sections):      # HOLDOUT layout: dot leaders, '*' = abnormal, unit inside range
    L = ["KAVERI PATH LAB & DIAGNOSTIC CENTRE", h["address"], "",
         f"Pt. Name: {h['title']} {h['pname']}    Age: {h['age']}Y    Sex: {h['sex']}",
         f"Dt. of Collection: {h['collected']}    Lab Ref: {h['lab_id']}", ""]
    for title, rows in sections:
        L.append(f"<< {title} >>")
        for r in rows:
            star = " *" if r["flag"] != "N" else ""
            L.append(f"{r['name']} {'.' * max(3, 30 - len(r['name']))} {r['val']}{star}   ({r['rng']} {r['unit']})")
        L.append("")
    L.append("* Value outside biological reference interval")
    return "\n".join(L)


LAYOUTS = {
    #           renderer     date format      p_alt  ascii-units  range-sep
    "sunrise":  (r_sunrise,  "%d/%m/%Y",      0.05,  False,       " - "),
    "medpath":  (r_medpath,  "%d-%b-%Y",      0.15,  True,        "-"),
    "citycare": (r_citycare, "%Y-%m-%d",      0.40,  False,       " - "),
    "apex":     (r_apex,     "%d %B %Y",      0.10,  True,        " to "),
    "nirmal":   (r_nirmal,   "%d.%m.%Y",      0.60,  True,        "-"),
    "kaveri":   (r_kaveri,   "%d/%m/%y",      0.30,  True,        "-"),
}
CITIES = ["Pune", "Jaipur", "Lucknow", "Kochi", "Indore", "Nagpur", "Bhopal", "Coimbatore"]


# ---------------------------------------------------------------------------
# 5. Noise (only on alphabetic tokens - numbers are never corrupted)
# ---------------------------------------------------------------------------
OCR_SWAPS = [("rn", "m"), ("m", "rn"), ("o", "0"), ("O", "0"), ("l", "1"), ("I", "l"),
             ("S", "5"), ("e", "c"), ("B", "8"), ("i", "l"), ("µ", "u"), ("cl", "d")]
JUNK = ["Page 1 of 2", "Scan QR code to verify this report", "-" * 40,
        "CONFIDENTIAL MEDICAL REPORT", "Page 2 of 2", "....continued"]


def ocr_noise(text, level, rng):
    if level == 0:
        return text
    p_tok = 0.04 if level == 1 else 0.10
    p_squash = 0.10 if level == 1 else 0.30
    out = []
    for line in text.split("\n"):
        parts = re.split(r"(\s+)", line)
        for i, tok in enumerate(parts):
            if tok.strip() and not any(c.isdigit() for c in tok) and rng.random() < p_tok:
                a, b = rng.choice(OCR_SWAPS)
                if a in tok:
                    parts[i] = tok.replace(a, b, 1)
        line = "".join(parts)
        if rng.random() < p_squash:                 # PDF text extraction loses alignment
            line = re.sub(r" {2,}", " ", line)
        out.append(line)
    if level == 2:
        for _ in range(rng.randint(1, 3)):
            out.insert(rng.randint(1, len(out)), rng.choice(JUNK))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 6. Patients with multi-year histories (and hidden drift)
# ---------------------------------------------------------------------------
def make_patient(rng, pid):
    sex = rng.choice("MF")
    first = rng.choice(FIRST_M if sex == "M" else FIRST_F)
    n = rng.choices([1, 2, 3, 4, 5], [0.15, 0.20, 0.25, 0.25, 0.15])[0]
    d0 = date(rng.randint(2018, 2022), rng.randint(1, 12), rng.randint(1, 28))
    dates = [d0]
    for _ in range(n - 1):
        dates.append(dates[-1] + timedelta(days=rng.randint(180, 540)))
    span = (dates[-1] - dates[0]).days / 365.25

    base = {k: per_sex(t["mean"], sex) + rng.gauss(0, 0.8) * t["sd"] for k, t in TESTS.items()}
    drift = {}   # test -> slope per year (canonical units)
    if n >= 3 and span >= 2.0:
        for k in rng.sample(list(TESTS), rng.randint(1, 3)):
            drift[k] = rng.choice([-1, 1]) * rng.uniform(0.4, 0.75) * TESTS[k]["sd"]
    return dict(pid=pid, sex=sex, age0=rng.randint(25, 72),
                name=f"{first} {rng.choice(LAST)}",
                title="Mr." if sex == "M" else rng.choice(["Ms.", "Mrs."]),
                dates=dates, base=base, drift=drift)


def true_value(p, k, d, rng):
    t = TESTS[k]
    yrs = (d - p["dates"][0]).days / 365.25
    v = p["base"][k] + p["drift"].get(k, 0) * yrs + rng.gauss(0, 0.15) * t["sd"]
    return max(v, 0.08 * per_sex(t["mean"], p["sex"]))


# ---------------------------------------------------------------------------
# 7. Build one report
# ---------------------------------------------------------------------------
def build_report(p, visit_idx, rng, noise_level, layout_names):
    d = p["dates"][visit_idx]
    lay = rng.choice(layout_names)
    render, dfmt, p_alt, ascii_units, sep = LAYOUTS[lay]
    age = p["age0"] + (d - p["dates"][0]).days // 365
    sex = p["sex"]

    panels = {TESTS[k]["panel"] for k in p["drift"]}
    others = [x for x in PANELS if x not in panels]
    panels |= set(rng.sample(others, min(len(others), rng.randint(1, 3))))
    panels = [x for x in PANELS if x in panels]        # stable order

    sections, gt_rows = [], []
    for pan in panels:
        rows = []
        for k in PANELS[pan]:
            t = TESTS[k]
            canon = (t["unit"], 1, 0, t["dec"])
            use_alt = t["alt"] and (stable_pct(f"{lay}:{k}") < p_alt * 100) != (rng.random() < 0.1)
            u = rng.choice(t["alt"]) if use_alt else canon
            rep_v = to_unit(true_value(p, k, d, rng), u)
            std_v = round(from_unit(rep_v, u), t["dec"])
            lo, hi = per_sex(t["ref"], sex)
            rlo = None if lo is None else to_unit(lo, u)
            rhi = None if hi is None else to_unit(hi, u)
            flag = flag_for(rep_v, rlo, rhi)
            unit_str = u[0].replace("µ", "u") if ascii_units else u[0]
            name = rng.choice(t["aliases"])
            rows.append(dict(name=name, val=fmt(rep_v, u[3]), unit=unit_str,
                             rng=fmt_range(rlo, rhi, u[3], sep), flag=flag))
            gt_rows.append(dict(test=k, value=std_v, unit=t["unit"],
                                reported_name=name, reported_value=rep_v,
                                reported_unit=unit_str, flag=flag))
        sections.append((PANEL_TITLES[pan], rows))

    h = dict(pname=p["name"], title=p["title"], age=age, sex=sex,
             refdoc=rng.choice(DOCTORS), lab_id=f"{lay[:2].upper()}{rng.randint(10**7, 10**8 - 1)}",
             collected=d.strftime(dfmt), reported=(d + timedelta(days=1)).strftime(dfmt),
             address=f"{rng.randint(1, 250)}, Main Road, {rng.choice(CITIES)}",
             phone="000-0000000")
    text = ocr_noise(render(h, sections), noise_level, rng)
    gt = dict(sample_date=d.isoformat(), patient=dict(age=age, sex=sex), results=gt_rows)
    return lay, text, gt


# ---------------------------------------------------------------------------
# 8. Chat-format tasks
# ---------------------------------------------------------------------------
SYSTEM_EXTRACT = (
    "You extract blood test results from Indian lab reports. Return only JSON with keys "
    "sample_date (YYYY-MM-DD), patient {age, sex}, and results: a list of "
    "{test, value, unit, reported_value, reported_unit, flag}. Use only the standard test keys "
    "and convert value to the standard unit. flag is H, L or N against the report's reference range. "
    "Standard units: " + "; ".join(f"{k}={t['unit']}" for k, t in TESTS.items()))

SYSTEM_DRIFT = (
    "You review a patient's lab history (standard units). For every test, classify the trend as "
    "rising, falling or stable. Set in_range_drift true when a test is rising or falling while every "
    "value is still inside the reference range. Return only JSON: "
    '{"trends": [{"test", "direction", "in_range_drift"}]}')


def extraction_target(gt):
    return dict(sample_date=gt["sample_date"], patient=gt["patient"],
                results=[{k: r[k] for k in ("test", "value", "unit", "reported_value",
                                            "reported_unit", "flag")} for r in gt["results"]])


def drift_example(p, history):
    """history: test -> list of (date_iso, std_value). Returns (prompt, target) or None."""
    series = {k: v for k, v in history.items() if len(v) >= 3}
    if not series:
        return None
    lines = [f"Patient: {p['sex']}, age {p['age0']} at first test", ""]
    trends = []
    for k, pts in series.items():
        t = TESTS[k]
        lo, hi = per_sex(t["ref"], p["sex"])
        ref = fmt_range(lo, hi, t["dec"], "-")
        lines.append(f"{k} ({t['unit']}, ref {ref}): " +
                     " | ".join(f"{d}: {fmt(v, t['dec'])}" for d, v in pts))
        slope = p["drift"].get(k, 0)
        direction = "stable" if slope == 0 else ("rising" if slope > 0 else "falling")
        in_range = all(flag_for(v, lo, hi) == "N" for _, v in pts)
        trends.append(dict(test=k, direction=direction,
                           in_range_drift=direction != "stable" and in_range))
    return "\n".join(lines), dict(trends=trends)


def chat(system, user, assistant):
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)}]}


# ---------------------------------------------------------------------------
# 9. Metric / reward
# ---------------------------------------------------------------------------
def score_extraction(pred, gt):
    """Score in [0, 1]. pred = model output (JSON string or dict), gt = ground truth dict.
    Per expected test: 0.6 value correct (in standard unit), 0.2 unit, 0.2 flag.
    Penalty for tests the model invented that are not in the report."""
    if isinstance(pred, str):
        m = re.search(r"\{.*\}", pred, re.S)
        try:
            pred = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            pred = None
    if not isinstance(pred, dict) or not isinstance(pred.get("results"), list):
        return 0.0
    want = {r["test"]: r for r in gt["results"]}
    got = {r["test"]: r for r in pred["results"] if isinstance(r, dict) and "test" in r}
    score = 0.0
    for k, g in want.items():
        r = got.get(k)
        if not r:
            continue
        try:
            ok_v = abs(float(r.get("value")) - g["value"]) <= max(0.01 * abs(g["value"]),
                                                                   10 ** -TESTS[k]["dec"])
        except (TypeError, ValueError):
            ok_v = False
        score += 0.6 * ok_v + 0.2 * (r.get("unit") == g["unit"]) + 0.2 * (r.get("flag") == g["flag"])
    recall = score / len(want)
    invented = sum(1 for k in got if k not in want) / max(len(got), 1)
    return round(max(0.0, recall - 0.5 * invented), 4)


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", type=int, default=2000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-layout", default="kaveri", choices=list(LAYOUTS))
    ap.add_argument("--noise", type=int, choices=[0, 1, 2], default=None,
                    help="fix noise level (default: mix of 0/1/2)")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    os.makedirs(os.path.join(a.out, "samples"), exist_ok=True)
    files = {n: open(os.path.join(a.out, n + ".jsonl"), "w", encoding="utf-8") for n in
             ["reports", "extract_train", "extract_val", "extract_test",
              "drift_train", "drift_val", "drift_test"]}
    layouts = list(LAYOUTS)
    seen_layout, counts = set(), {}

    for i in range(a.patients):
        p = make_patient(rng, f"P{i:05d}")
        pr = rng.random()
        psplit = "test" if pr < 0.1 else "val" if pr < 0.2 else "train"
        history = {}
        for v in range(len(p["dates"])):
            noise = a.noise if a.noise is not None else rng.choices([0, 1, 2], [0.3, 0.5, 0.2])[0]
            lay, text, gt = build_report(p, v, rng, noise, layouts)
            rid = f"{p['pid']}-R{v + 1}"
            # extraction split: unseen layout -> test; else train/val by patient
            esplit = "test" if lay == a.holdout_layout else ("val" if psplit != "train" else "train")
            files["reports"].write(json.dumps(dict(report_id=rid, patient_id=p["pid"], layout=lay,
                                                   noise=noise, split=esplit, text=text,
                                                   ground_truth=gt), ensure_ascii=False) + "\n")
            files[f"extract_{esplit}"].write(json.dumps(
                chat(SYSTEM_EXTRACT, text, extraction_target(gt)), ensure_ascii=False) + "\n")
            counts[f"extract_{esplit}"] = counts.get(f"extract_{esplit}", 0) + 1
            for r in gt["results"]:
                history.setdefault(r["test"], []).append((gt["sample_date"], r["value"]))
            if lay not in seen_layout:
                seen_layout.add(lay)
                with open(os.path.join(a.out, "samples", f"{lay}.txt"), "w", encoding="utf-8") as f:
                    f.write(text)
        ex = drift_example(p, history)
        if ex:
            files[f"drift_{psplit}"].write(json.dumps(chat(SYSTEM_DRIFT, *ex), ensure_ascii=False) + "\n")
            counts[f"drift_{psplit}"] = counts.get(f"drift_{psplit}", 0) + 1

    for f in files.values():
        f.close()
    print(f"Done -> {a.out}/")
    for k in sorted(counts):
        print(f"  {k:<15} {counts[k]:>6}")


if __name__ == "__main__":
    main()