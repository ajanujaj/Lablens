"""
Step 2: turn the file into text with the right Docling setup for its type.

  digital_pdf  -> Docling, no OCR (reads the PDF's own text; fast and exact) + table structure
  scanned_pdf  -> Docling with full-page OCR + table structure
  image_ocr    -> Docling with full-page OCR (HEIC is converted to PNG first)
  docx / html  -> Docling's native converters (no OCR needed)
  text         -> used as-is, Docling skipped

Docling loads its layout/table/OCR models the first time each route is used (first request
for that route is slow; later ones are fast). Converters are cached per route.
"""
import inspect
import io
import logging
import threading
import time
from dataclasses import dataclass

from . import config
from .filetype import FileInfo, UnsupportedFile

log = logging.getLogger("lablens.extraction")

_converters: dict = {}
_lock = threading.Lock()


@dataclass
class Extracted:
    text: str
    engine: str          # what produced the text
    seconds: float


def _ocr_options():
    from docling.datamodel import pipeline_options as po

    engines = {
        "auto": getattr(po, "OcrAutoOptions", None) or po.EasyOcrOptions,
        "ocrmac": po.OcrMacOptions,          # macOS Vision framework: pip install ocrmac
        "easyocr": po.EasyOcrOptions,
        "rapidocr": po.RapidOcrOptions,
        "tesseract": po.TesseractCliOcrOptions,
    }
    if config.OCR_ENGINE not in engines:
        raise ValueError(f"LABLENS_OCR_ENGINE must be one of {list(engines)}")
    opts = engines[config.OCR_ENGINE]()
    if config.OCR_LANGS:
        opts.lang = config.OCR_LANGS
    # Lab reports are small; OCR the whole page rather than only detected regions.
    # (Newer Docling uses `mode`, older versions use `force_full_page_ocr`.)
    fields = type(opts).model_fields
    if "mode" in fields and hasattr(po, "OcrMode"):
        opts.mode = po.OcrMode.FULL_PAGE
    elif "force_full_page_ocr" in fields:
        opts.force_full_page_ocr = True
    return opts


def _pdf_options(ocr: bool):
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.do_table_structure = True                 # lab reports are mostly tables
    opts.table_structure_options.do_cell_matching = True
    opts.do_ocr = ocr
    if ocr:
        opts.ocr_options = _ocr_options()
    return opts


def _build_converter(route: str):
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    if route == "digital_pdf":
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_options(ocr=False))})
    if route == "scanned_pdf":
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_pdf_options(ocr=True))})
    if route == "image_ocr":
        return DocumentConverter(
            allowed_formats=[InputFormat.IMAGE],
            format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=_pdf_options(ocr=True))})
    if route in ("docx", "html"):
        return DocumentConverter(allowed_formats=[InputFormat.DOCX, InputFormat.HTML])
    raise ValueError(route)


def get_converter(route: str):
    with _lock:
        if route not in _converters:
            log.info("Building Docling converter for %s (first use loads models)...", route)
            _converters[route] = _build_converter(route)
        return _converters[route]


def warm_up(routes=("digital_pdf",)):
    for r in routes:
        get_converter(r)


def _heic_to_png(data: bytes) -> bytes:
    try:
        import pillow_heif
        from PIL import Image
    except ImportError:
        raise UnsupportedFile("HEIC (iPhone) photos need `pip install pillow-heif`, or upload a JPG/PNG.")
    pillow_heif.register_heif_opener()
    buf = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _to_markdown(doc) -> str:
    kwargs = {"escape_underscores": False, "escape_html": False}
    allowed = inspect.signature(doc.export_to_markdown).parameters
    return doc.export_to_markdown(**{k: v for k, v in kwargs.items() if k in allowed})


def extract_text(data: bytes, info: FileInfo) -> Extracted:
    t0 = time.perf_counter()

    if info.route == "text":
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        return Extracted(text.strip(), "plain_text", time.perf_counter() - t0)

    from docling.datamodel.base_models import DocumentStream

    ext = info.ext
    if info.kind == "heic":
        data, ext = _heic_to_png(data), "png"

    stream = DocumentStream(name=f"upload.{ext}", stream=io.BytesIO(data))
    result = get_converter(info.route).convert(stream, max_num_pages=config.MAX_PAGES)
    text = _to_markdown(result.document).strip()
    engine = f"docling:{info.route}" + (f"+ocr({config.OCR_ENGINE})" if info.route in ("scanned_pdf", "image_ocr") else "")
    return Extracted(text, engine, time.perf_counter() - t0)
