# LabLens

**Turn Indian blood-test reports (PDFs, scans, phone photos) into clean, structured data - with a
fine-tuned 1.5B model that runs locally, and plain-code safety checks around it.**

Indian lab reports have no common format: every lab uses its own layout, test names
("Hb", "Haemoglobin", "HGB"), units (mg/dL vs mmol/L, lakh/cumm vs 10³/µL) and date formats, and
many arrive as scans or phone photos. LabLens reads a report and returns one standard JSON per
report - test, value in a standard unit, H/L/N flag - so results from any lab can be compared over time.

> ⚠️ Portfolio / research project. All training and test data is synthetic. Not a medical device
> and not medical advice.

---

## How it works

```
 upload ──► detect type ──► Docling ──────────► gate ──────► fine-tuned ──────► code checks ──► JSON
 (PDF /     (file bytes,    digital PDF: text   any "test    Qwen2.5-1.5B       • unit conversion
  photo /    digital vs     scanned / photo:    + value"     + LoRA adapter       done by code
  DOCX)      scanned PDF)   OCR + tables        lines? no →  (reads names,      • values not in the
                                                stop         numbers, units)      report rejected
                                                                                • unsupported tests
                                                                                  set aside
```

| Stage | What it does | Code |
|---|---|---|
| File detection | Reads magic bytes (not the extension); checks PDFs for a text layer so OCR only runs on scans | `app/filetype.py` |
| Text extraction | [Docling](https://github.com/docling-project/docling): layout + table structure; OCR for scans/photos | `app/extraction.py` |
| Gate | Skips the model when the document has no lab values (prescriptions, summaries) | `app/gate.py` |
| Extraction model | Qwen2.5-1.5B-Instruct + LoRA (r=16), fine-tuned on 1,500 synthetic reports | `model_weights/`, `app/model.py` |
| Code checks | Recomputes unit conversions, rejects numbers that aren't in the report, recovers cut-off output | `app/units.py`, `app/validate.py` |
| API | FastAPI: `/analyze`, `/extract-text`, `/health`, upload page | `app/main.py` |

## Data and training

- **Synthetic data** (`data/synth_data.py`): 2,000 fictional patients, 6,120 reports from 6 fictional
  labs with different layouts, test-name aliases, units, date formats and OCR-style noise, each with
  an exact correct answer. 20 common tests (CBC, diabetes, lipids, thyroid, kidney, liver, vitamins D/B12).
- **Honest test split:** the test set uses a lab layout (`kaveri`) that never appears in training, so
  scores measure generalisation to an unseen format, not memorisation.
- **Fine-tuning** (`lablens-notebook-fixed.ipynb`, Kaggle T4): Unsloth + TRL SFT, LoRA r=16 / α=32 on all
  attention and MLP projections, 1 epoch, loss on the assistant's answer only.
- **Metric** (same for training and evaluation): per expected test 0.6 × value correct (±1%) + 0.2 × unit
  + 0.2 × flag, minus a penalty for invented tests.

## Results

Run `python scripts/evaluate.py all` (see [Evaluation](#evaluation)) and paste the tables from
`results/eval_<timestamp>.md` here.

### Model: base vs fine-tuned

<!-- paste "Model: extraction on synthetic reports" table here -->
_TBD_

### End-to-end: sample files through the full pipeline

<!-- paste "End-to-end pipeline" table here -->
_TBD_

## What went wrong, and what I changed

Testing on real-looking documents surfaced failures the synthetic test set never showed:

1. **The model invented a whole report.** Given a patient summary with no test values, it produced ~20
   plausible results (and listed aspirin as a lab test), because every training example had results.
   → Added a **gate** that only runs the model when the text contains a test name followed by a value
   (ignoring dates, doses like "500 mg" and phrases like "in 3 months"), and **strict mode**: any value
   that doesn't appear in the report text is rejected. Zero of 9,879 correct test-set answers are
   wrongly rejected.
2. **It got stuck repeating itself** until the token limit, leaving broken JSON.
   → Output is **salvaged**: every complete result object is recovered and duplicates are dropped.
3. **It read a scanned report perfectly but did the maths wrong:** T3 2.95 nmol/L → 129 ng/dL
   (correct: 192). → **Unit conversion moved to code.** The model reads the number and unit; code
   computes the standard value and records any correction. Verified exact on all 9,879 test answers.

The general lesson: use the model for what is fuzzy (reading messy layouts), and plain code for anything
with an exact answer (arithmetic, range checks, "is this number actually in the document?").

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-app.txt
python -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000 to upload a report, or http://localhost:8000/docs for the API. The first
start downloads the base model (~3 GB) and Docling's models. On a Mac, `LABLENS_OCR_ENGINE=ocrmac`
uses the built-in macOS OCR. All settings: [`app/README.md`](app/README.md).

Single report from the command line (no server):

```bash
python finetuning/infer.py data/samples/apex.txt
```

## Evaluation

```bash
python data/synth_data.py --patients 2000 --out data --seed 42   # recreate the synthetic data (not in git)
python scripts/evaluate.py model          # base vs fine-tuned, unseen + seen layouts (50 reports each)
python -m uvicorn app.main:app --port 8000 &
python scripts/evaluate.py pipeline       # 8 sample files through the API
python scripts/evaluate.py all            # both
```

Results are written to `results/eval_<timestamp>.md` (tables) and `.json` (every prediction).
The 8 pipeline test files in `data/test_files/` cover: clean digital PDFs, SI units to convert, a
3-page report, a scanned image-only PDF, a phone photo, a report without H/L flags plus unsupported
tests, Indian count units (lakh/cumm), and a prescription that must be rejected. Regenerate or add
cases with `python data/make_test_files.py`.

## Project layout

```
app/                    FastAPI service (pipeline, gate, unit conversion, checks)
data/synth_data.py      synthetic report generator (training + test data)
data/make_test_files.py sample PDFs / scan / photo with exact expected answers
data/samples/           one example report per synthetic lab layout
data/test_files/        the 8 pipeline test files + ground_truth.json
finetuning/infer.py     command-line inference
lablens-notebook-fixed.ipynb   fine-tuning notebook (Kaggle)
model_weights/          LoRA adapter + tokenizer
scripts/evaluate.py     evaluation (model + pipeline suites)
```

## Limitations

- **Synthetic data only.** Real reports will have layouts and quirks the generator never produced; the
  pipeline needs a set of real, anonymised reports to be properly measured.
- **20 tests.** Other tests (electrolytes, iron studies, urine, CRP ...) are set aside as unrecognised.
- **"No results" documents** are handled by the gate, not the model - the model itself was never shown a
  report without results. Next step: add such examples (and Markdown-table layouts that match Docling's
  output) to the training data.
- **Speed.** On an Apple-silicon Mac with PyTorch (MPS), a one-page report takes about a minute for the
  model plus OCR time. MLX would be the obvious next optimisation.
- Flags are read from the report or judged by the model against the printed range; they are not
  independently recomputed in code.
