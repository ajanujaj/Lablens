"""
LabLens API

  upload -> detect file type -> Docling (route per type) -> fine-tuned model -> validated JSON

Run from the Lablens/ project folder:
    uvicorn app.main:app --port 8000
Then open http://localhost:8000 (simple upload page) or http://localhost:8000/docs (API docs).
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from . import config
from .extraction import extract_text, warm_up
from .filetype import UnsupportedFile, detect
from .gate import find_lab_tests
from .validate import validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("lablens")

state = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.LOAD_MODEL:
        from .model import LabModel
        state["model"] = await run_in_threadpool(LabModel)
        log.info("Model ready.")
    else:
        log.warning("LABLENS_LOAD_MODEL=0: model not loaded, /analyze will return extracted text only.")
    try:
        await run_in_threadpool(warm_up)      # pre-load Docling's PDF models (downloads on first run)
        log.info("Docling ready.")
    except Exception as e:                    # e.g. offline: retry lazily on the first request
        log.warning("Docling warm-up failed (%s); will retry on first PDF.", e)
    yield


app = FastAPI(title="LabLens", version="0.1", lifespan=lifespan)


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_MB} MB.")
    return data


async def _detect_and_extract(file: UploadFile):
    data = await _read_upload(file)
    try:
        info = await run_in_threadpool(detect, data)
        extracted = await run_in_threadpool(extract_text, data, info)
    except UnsupportedFile as e:
        raise HTTPException(415, str(e))
    except Exception as e:                     # Docling failure (corrupt file, OCR engine missing, ...)
        log.exception("Extraction failed")
        raise HTTPException(422, f"Could not read the document: {e}")
    if not extracted.text:
        raise HTTPException(422, "No text could be extracted. If this is a photo, try a sharper, well-lit image.")
    file_block = {"name": file.filename, "detected_type": info.kind, "route": info.route,
                  "pages": info.pages, "notes": info.notes}
    extraction_block = {"engine": extracted.engine, "seconds": round(extracted.seconds, 2),
                        "characters": len(extracted.text), "text": extracted.text}
    return file_block, extraction_block


@app.get("/health")
def health():
    m = state["model"]
    return {"status": "ok", "model_loaded": m is not None, "device": getattr(m, "device", None),
            "base_model": config.BASE_MODEL, "adapter": str(config.ADAPTER_DIR),
            "ocr_engine": config.OCR_ENGINE}


@app.post("/extract-text")
async def extract_only(file: UploadFile = File(...)):
    """Docling step only - useful to see what text the model will receive."""
    file_block, extraction_block = await _detect_and_extract(file)
    return {"file": file_block, "extraction": extraction_block}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), include_text: bool = True):
    """Full pipeline: file -> Docling -> fine-tuned model -> validated JSON."""
    file_block, extraction_block = await _detect_and_extract(file)
    text = extraction_block["text"] if include_text else extraction_block.pop("text")

    model = state["model"]
    if model is None:
        return {"file": file_block, "extraction": extraction_block, "analysis": None,
                "warnings": ["Model not loaded (LABLENS_LOAD_MODEL=0)."]}

    tests_seen = find_lab_tests(text)
    if config.GATE and not tests_seen:
        return {"file": file_block, "extraction": extraction_block, "analysis": None,
                "warnings": ["No lab test results found in this document (it may be a summary, prescription "
                             "or list of tests done, or a test type LabLens doesn't support yet). "
                             "Model was not run."],
                "model": None}

    out = await run_in_threadpool(model.analyze, text)
    result, warnings = validate(out["result"], text)
    return {
        "file": file_block,
        "extraction": extraction_block,
        "analysis": result,
        "warnings": warnings,
        "model": {"device": out["device"], "chunks": out["chunks"], "seconds": out["seconds"],
                  "tests_spotted_by_gate": tests_seen, "raw_outputs": out["raw_outputs"]},
        "disclaimer": "Automated extraction for information only - not medical advice. Verify against the original report.",
    }


@app.get("/", response_class=HTMLResponse)
def upload_page():
    return """<!doctype html><html><head><meta charset="utf-8"><title>LabLens</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font-family:system-ui,sans-serif;max-width:860px;margin:40px auto;padding:0 16px;color:#222}
pre{background:#f5f5f5;padding:12px;border-radius:8px;overflow:auto;max-height:70vh;font-size:13px}
button{padding:8px 16px;font-size:15px}</style></head><body>
<h1>LabLens</h1><p>Upload a lab report (PDF, photo, DOCX or TXT).</p>
<form id="f"><input type="file" name="file" required>
<label><input type="checkbox" id="t"> show extracted text</label> <button>Analyze</button></form>
<p id="s"></p><pre id="o"></pre>
<script>
f.onsubmit=async e=>{e.preventDefault();s.textContent="Working... (first run loads models, may take a minute)";o.textContent="";
const t0=Date.now();const r=await fetch("/analyze?include_text="+t.checked,{method:"POST",body:new FormData(f)});
const j=await r.json();s.textContent=(r.ok?"Done":"Error "+r.status)+" in "+((Date.now()-t0)/1000).toFixed(1)+"s";
o.textContent=JSON.stringify(j,null,2)};
</script></body></html>"""
