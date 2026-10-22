"""All settings in one place. Override any of them with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the Lablens/ project folder

# Fine-tuned model
ADAPTER_DIR = Path(os.getenv("LABLENS_ADAPTER", ROOT / "model_weights"))
BASE_MODEL = os.getenv("LABLENS_BASE", "unsloth/Qwen2.5-1.5B-Instruct")   # same as training
LOAD_MODEL = os.getenv("LABLENS_LOAD_MODEL", "1") == "1"   # 0 = run Docling only (handy for testing)
MAX_NEW_TOKENS = int(os.getenv("LABLENS_MAX_NEW_TOKENS", "1024"))
# Model was trained with 2048-token sequences. Long reports are split into chunks of this many
# tokens (report text only, system prompt not counted) and the results are merged.
CHUNK_TOKENS = int(os.getenv("LABLENS_CHUNK_TOKENS", "1200"))

# Skip the model when the text has no "<test name> ... <value>" lines (see gate.py). 0 = always run.
GATE = os.getenv("LABLENS_GATE", "1") == "1"

# Docling
# auto | ocrmac (macOS built-in) | easyocr | rapidocr | tesseract
OCR_ENGINE = os.getenv("LABLENS_OCR_ENGINE", "auto")
OCR_LANGS = [l for l in os.getenv("LABLENS_OCR_LANGS", "").split(",") if l]
# A PDF page with fewer extractable characters than this is treated as scanned -> OCR
MIN_CHARS_PER_PAGE = int(os.getenv("LABLENS_MIN_CHARS_PER_PAGE", "50"))
MAX_PAGES = int(os.getenv("LABLENS_MAX_PAGES", "20"))

# Uploads
MAX_UPLOAD_MB = int(os.getenv("LABLENS_MAX_UPLOAD_MB", "20"))
