import re
from io import BytesIO
from pathlib import Path

import fitz
from docx import Document
from pptx import Presentation
from striprtf.striprtf import rtf_to_text

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".txt", ".md", ".markdown", ".csv",
    ".json", ".log", ".rtf", ".tsv",
}

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".log", ".tsv"}

PDF_BOILERPLATE = (
    "provided proper attribution is provided",
    "scholarly works",
    "permission to make digital or hard copies",
    "all rights reserved",
    "copyright",
    "equal contribution",
    "equal contributions",
    "corresponding author",
    "arxiv:",
)


def supported_extensions() -> list[str]:
    return sorted(SUPPORTED_EXTENSIONS)


def _block_text(block: str) -> str:
    """Turn one PDF text block into a single readable paragraph."""
    text = block.replace("\x00", " ").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "-", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    text = " ".join(lines).strip()
    # Page breaks sometimes leave a sentence fragment glued to the end of a
    # block. Remove it so the next block starts on a clean sentence boundary.
    text = re.sub(
        r"(?<=[.!?])\s+[A-Z][^.!?]{0,55}\b(?:we|our|the|of|and|to|in|with|for|that|is|are|was|were|by|on|at|as)\s*$",
        "",
        text,
    )
    return text.strip()


def _is_noise_block(text: str, page_number: int) -> bool:
    lower = text.lower().strip()
    if not lower or lower == str(page_number):
        return True
    if any(marker in lower for marker in PDF_BOILERPLATE) and len(text.split()) < 45:
        return True
    if len(text.split()) <= 2 and re.fullmatch(r"[\d\s.·•-]+", text):
        return True
    if re.match(r"^(figure|table|algorithm|equation)\s*\d+\s*[:.]", lower):
        return True
    if re.match(r"^[\u2217*\u2020\u2021\u00a7]+", text):
        return True
    words = text.split()
    if len(words) <= 8 and lower.endswith((" we", " the", " of", " and", " to", " in", " with", " for", " that", " is", " are", " was", " were", " by", " on", " at", " as")):
        return True
    if len(words) >= 8:
        numeric = sum(1 for word in words if re.fullmatch(r"[-+()\d.,%×/·=<>≈]+", word))
        symbolic = sum(1 for char in text if char in "=+×·∑√∫≈≤≥<>")
        if numeric / len(words) > 0.34 or symbolic / max(1, len(text)) > 0.035:
            return True
        # Table rows are usually dense with short numeric cells.
        short_tokens = sum(1 for word in words if len(word) <= 5)
        if numeric >= 4 and short_tokens / len(words) > 0.55:
            return True
    return False


def _extract_pdf_text(document: fitz.Document) -> str:
    paragraphs: list[str] = []
    for page_number, page in enumerate(document, start=1):
        try:
            blocks = page.get_text("blocks", sort=True)
        except Exception:
            continue
        for block in blocks:
            if len(block) < 5 or block[6] != 0:
                continue
            text = _block_text(block[4])
            if len(text) < 2 or _is_noise_block(text, page_number):
                continue
            paragraphs.append(text)
    return "\n\n".join(paragraphs).strip()


def extract_text_from_bytes(content: bytes, filename: str) -> str:
    """Extract plain text from a supported document without writing to disk."""
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Supported: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )

    if suffix == ".pdf":
        document = fitz.open(stream=content, filetype="pdf")
        try:
            if document.is_encrypted:
                raise ValueError("This PDF is password-protected and cannot be read.")
            text = _extract_pdf_text(document)
            if not text:
                raise ValueError(
                    "No selectable text was found in this PDF. It may be a scanned image; try pasting the text instead."
                )
            return text
        finally:
            document.close()

    if suffix == ".docx":
        document = Document(BytesIO(content))
        parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        return "\n\n".join(parts)

    if suffix == ".pptx":
        presentation = Presentation(BytesIO(content))
        slides = []
        for slide in presentation.slides:
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
            if texts:
                slides.append("\n".join(texts))
        return "\n\n".join(slides)

    if suffix == ".rtf":
        return rtf_to_text(content.decode("utf-8", errors="replace"))

    # Plain-text-like formats.
    return content.decode("utf-8", errors="replace")
