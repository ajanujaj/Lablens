"""Prompts - copied verbatim from training data (data/synth_data.py SYSTEM_EXTRACT).
The model works best when it sees exactly the prompt it was trained with."""

SYSTEM_EXTRACT = "You extract blood test results from Indian lab reports. Return only JSON with keys sample_date (YYYY-MM-DD), patient {age, sex}, and results: a list of {test, value, unit, reported_value, reported_unit, flag}. Use only the standard test keys and convert value to the standard unit. flag is H, L or N against the report's reference range. Standard units: hemoglobin=g/dL; wbc=10^3/µL; platelets=10^3/µL; glucose_fasting=mg/dL; hba1c=%; cholesterol_total=mg/dL; ldl=mg/dL; hdl=mg/dL; triglycerides=mg/dL; tsh=µIU/mL; t3_total=ng/dL; t4_total=µg/dL; creatinine=mg/dL; urea=mg/dL; uric_acid=mg/dL; alt_sgpt=U/L; ast_sgot=U/L; bilirubin_total=mg/dL; vitamin_d=ng/mL; vitamin_b12=pg/mL"

# test key -> standard unit (parsed from the prompt above)
STANDARD_UNITS = {
    "hemoglobin": "g/dL",
    "wbc": "10^3/µL",
    "platelets": "10^3/µL",
    "glucose_fasting": "mg/dL",
    "hba1c": "%",
    "cholesterol_total": "mg/dL",
    "ldl": "mg/dL",
    "hdl": "mg/dL",
    "triglycerides": "mg/dL",
    "tsh": "µIU/mL",
    "t3_total": "ng/dL",
    "t4_total": "µg/dL",
    "creatinine": "mg/dL",
    "urea": "mg/dL",
    "uric_acid": "mg/dL",
    "alt_sgpt": "U/L",
    "ast_sgot": "U/L",
    "bilirubin_total": "mg/dL",
    "vitamin_d": "ng/mL",
    "vitamin_b12": "pg/mL"
}
