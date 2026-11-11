from pathlib import Path

import fitz


def extract_text_from_pdf(path: str | Path) -> str:
    """Extract plain text from a PDF, page by page."""
    document = fitz.open(str(path))
    try:
        if document.is_encrypted:
            raise ValueError("This PDF is password-protected and cannot be read.")
        pages = []
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"[Page {page_number}]\n{text.strip()}")
        return "\n\n".join(pages)
    finally:
        document.close()
