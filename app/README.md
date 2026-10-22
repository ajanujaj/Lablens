# LabLens API

```
upload ─► detect type ─► Docling ─► gate ─► fine-tuned Qwen + LoRA ─► validate ─► JSON
         (magic bytes)  (per type)  (any lab   (model_weights/)        (plain code)
                                     values?)
```

**Gate** (`gate.py`): the model was trained only on reports that contain results, so given a
summary / prescription / "tests done" list it invents values. The gate looks for a supported
test name followed by a result value (not a date, dose or "3 months") and skips the model if
there are none.

| Uploaded file | Route | What Docling does |
|---|---|---|
| PDF with a text layer | `digital_pdf` | Reads the PDF's own text + table structure, **no OCR** (fast, exact) |
| PDF of scanned pages | `scanned_pdf` | Full-page OCR + table structure |
| JPG / PNG / TIFF / WEBP / BMP | `image_ocr` | Full-page OCR + table structure |
| HEIC (iPhone photo) | `image_ocr` | Converted to PNG first (needs `pillow-heif`) |
| DOCX / HTML | `docx` / `html` | Native converter, no OCR |
| TXT | `text` | Used as-is (Docling skipped) |

The type is detected from the file's bytes, not its name. A PDF counts as "digital" only if
every page has at least 50 extractable characters; otherwise the whole PDF is OCR'd.

## Setup (Mac)

```bash
cd ~/Desktop/Lablens/Lablens
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-app.txt
uvicorn app.main:app --port 8000
```

First start downloads the base model (~3 GB) and Docling's layout/table models; the first
OCR request also downloads OCR models. After that, everything runs offline.

Open **http://localhost:8000** for an upload page, or **http://localhost:8000/docs** for the API.

```bash
curl -F "file=@report.pdf" http://localhost:8000/analyze           # full pipeline
curl -F "file=@report.pdf" http://localhost:8000/extract-text      # Docling output only
```

## Test files

`data/test_files/` has 8 sample reports (all fictional) with exact expected answers in
`ground_truth.json`: clean digital PDFs, SI units to convert, a 3-page full panel, a scanned
(image-only) PDF, a phone photo, a report with no flag column plus unsupported tests, Indian count
units (lakh/cumm), and a prescription that must be rejected.

```bash
python scripts/evaluate.py pipeline       # server must be running; scores each file
python data/make_test_files.py            # regenerate / edit cases in CASES
```

## Response

```jsonc
{
  "file":       {"detected_type": "pdf", "route": "scanned_pdf", "pages": 2, ...},
  "extraction": {"engine": "docling:scanned_pdf+ocr(auto)", "seconds": 6.1, "text": "..."},
  "analysis":   {"sample_date": "...", "patient": {...}, "results": [...], "unrecognised_results": [...]},
  "warnings":   ["tsh: reported value 9.99 not found in the report text - please double-check."],
  "model":      {"chunks": 1, "seconds": 4.2, "raw_outputs": null}
}
```

`warnings` come from plain-code checks: unsupported test names, wrong standard unit, bad flag,
invalid date, repeated tests. **Unit conversion is done by code** (`units.py`): the model reads the
number and unit, then code recomputes the standard value and corrects the model if its maths was
wrong (`value_corrected_from` shows the model's number). **Strict mode** (default): a result whose `reported_value` does not
appear anywhere in the report text (dates ignored) is moved to `rejected_results` as likely made
up. If the model's output is cut off, every complete result is still recovered.

## Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `LABLENS_OCR_ENGINE` | `auto` | `ocrmac` (macOS built-in, often best on Mac), `easyocr`, `rapidocr`, `tesseract` |
| `LABLENS_OCR_LANGS` | engine default | e.g. `en-US` for ocrmac, `en` for easyocr |
| `LABLENS_BASE` | `unsloth/Qwen2.5-1.5B-Instruct` | base model |
| `LABLENS_ADAPTER` | `model_weights/` | LoRA adapter folder |
| `LABLENS_GATE` | `1` | `0` = always run the model, even if no lab values are found |
| `LABLENS_STRICT` | `1` | `0` = only warn (don't reject) values not found in the report |
| `LABLENS_LOAD_MODEL` | `1` | `0` = Docling only, for testing extraction |
| `LABLENS_CHUNK_TOKENS` | `1200` | long reports are split into chunks this size and merged |
| `LABLENS_MAX_UPLOAD_MB` / `LABLENS_MAX_PAGES` | `20` / `20` | limits |

## Important

The model was trained on raw-text layouts from `data/synth_data.py`, but Docling outputs
**Markdown** (tables become `| Test | Result | Unit | Range |`). Check accuracy on Docling
output, and if it drops, add a Markdown-table layout to the synthetic data and retrain.
