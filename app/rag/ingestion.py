from __future__ import annotations

import logging
from io import BytesIO

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, bool]:
    """Extract text from a PDF, falling back to OCR for scanned pages (#6).

    Returns:
        (text, ocr_used) — text extracted, and whether OCR fallback was triggered.

    Strategy:
      1. Try pypdf first (fast, no extra deps).
      2. If pypdf returns no text (scanned/image PDF), fall back to
         pdf2image + pytesseract (tesseract-ocr must be on PATH).
    """
    # ── Primary path: digital PDF ──────────────────────────────────────────
    reader = PdfReader(BytesIO(pdf_bytes))
    parts: list[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            parts.append(txt)

    if parts:
        return "\n".join(parts).strip(), False   # ocr_used=False

    # ── OCR fallback: scanned / image-only PDF ─────────────────────────────
    logger.info("ocr_fallback_triggered", extra={"reason": "pypdf_extracted_no_text"})
    try:
        import pytesseract
        from pdf2image import convert_from_bytes

        images = convert_from_bytes(pdf_bytes, dpi=300)
        ocr_parts: list[str] = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img)
            if text.strip():
                ocr_parts.append(text)
            logger.debug("ocr_page_done", extra={"page": i + 1, "chars": len(text)})

        result = "\n".join(ocr_parts).strip()
        logger.info("ocr_fallback_complete", extra={"pages": len(images), "chars": len(result)})
        return result, True   # ocr_used=True

    except ImportError:
        logger.warning(
            "ocr_unavailable",
            extra={"hint": "Install pytesseract + pdf2image; ensure tesseract is on PATH"},
        )
        return "", False
