"""
Step 1: work out what the uploaded file really is.

We look at the file's first bytes ("magic numbers"), not its name, because users rename
files and phones send odd extensions. PDFs are further split into digital (has a text
layer) vs scanned (just images of pages) so we only run slow OCR when it's needed.
"""
import io
import zipfile
from dataclasses import dataclass, field

from . import config


class UnsupportedFile(Exception):
    pass


@dataclass
class FileInfo:
    kind: str                 # pdf | image | heic | docx | html | text
    route: str                # digital_pdf | scanned_pdf | image_ocr | docx | html | text
    ext: str                  # extension we hand to Docling
    pages: int | None = None
    notes: list[str] = field(default_factory=list)


def _sniff(data: bytes) -> tuple[str, str]:
    head = data[:16]
    if head.startswith(b"%PDF"):
        return "pdf", "pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image", "jpg"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image", "tiff"
    if head.startswith(b"BM"):
        return "image", "bmp"
    if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image", "webp"
    if data[4:8] == b"ftyp" and data[8:12] in (b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1", b"msf1"):
        return "heic", "heic"                          # iPhone photos
    if head.startswith(b"PK\x03\x04"):
        try:
            names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        except zipfile.BadZipFile:
            raise UnsupportedFile("Corrupted ZIP-based file.")
        if "word/document.xml" in names:
            return "docx", "docx"
        raise UnsupportedFile("ZIP/Office file that isn't a Word document.")
    # Text-like?
    if b"\x00" not in data[:4096]:
        try:
            txt = data[:4096].decode("utf-8")
        except UnicodeDecodeError:
            try:
                txt = data[:4096].decode("latin-1")
            except UnicodeDecodeError:
                txt = None
        if txt is not None:
            low = txt.lstrip().lower()
            if low.startswith("<!doctype html") or low.startswith("<html"):
                return "html", "html"
            return "text", "txt"
    raise UnsupportedFile("Unrecognised file type. Upload a PDF, image (JPG/PNG/HEIC/TIFF/WEBP), DOCX or TXT.")


def _pdf_text_stats(data: bytes) -> tuple[int, int]:
    """Returns (page_count, pages_with_real_text)."""
    import pypdfium2 as pdfium                          # installed with Docling
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as e:
        raise UnsupportedFile(f"Could not open PDF (password-protected or damaged?): {e}")
    n = len(pdf)
    with_text = 0
    for i in range(min(n, config.MAX_PAGES)):
        page = pdf[i]
        tp = page.get_textpage()
        if len(tp.get_text_range().strip()) >= config.MIN_CHARS_PER_PAGE:
            with_text += 1
        tp.close()
        page.close()
    pdf.close()
    return n, with_text


def detect(data: bytes) -> FileInfo:
    if not data:
        raise UnsupportedFile("Empty file.")
    kind, ext = _sniff(data)

    if kind == "pdf":
        pages, with_text = _pdf_text_stats(data)
        if pages > config.MAX_PAGES:
            raise UnsupportedFile(f"PDF has {pages} pages; limit is {config.MAX_PAGES}.")
        checked = min(pages, config.MAX_PAGES)
        if with_text == checked:
            return FileInfo(kind, "digital_pdf", ext, pages)
        info = FileInfo(kind, "scanned_pdf", ext, pages)
        if with_text:
            info.notes.append(f"{with_text}/{checked} pages have a text layer; OCR-ing the whole PDF to be safe.")
        return info
    if kind in ("image", "heic"):
        return FileInfo(kind, "image_ocr", ext, 1)
    if kind == "docx":
        return FileInfo(kind, "docx", ext)
    if kind == "html":
        return FileInfo(kind, "html", ext)
    return FileInfo(kind, "text", ext)
