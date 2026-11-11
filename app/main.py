import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import MAX_UPLOAD_MB, MIN_TEXT_CHARS, UPLOAD_DIR
from .services.file_parser import extract_text_from_bytes, supported_extensions
from .services.summarizer import answer_question, summarize_text
from .services.text_cleaner import clean_text

app = FastAPI(
    title="Scholia — Research Paper Summarizer",
    description="Upload or paste a research paper and get an extractive, section-aware summary with a Research Gap segment, plus question answering.",
    version="0.4.0",
)

public_dir = Path(__file__).resolve().parents[1] / "public"

DOCUMENTS: dict[str, str] = {}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(public_dir / "index.html")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "documents_in_memory": len(DOCUMENTS)}


@app.get("/api/formats")
def formats() -> dict:
    return {"supported_extensions": supported_extensions()}


@app.post("/api/summarize")
async def summarize(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
    sentences: int = Form(default=12),
) -> dict:
    if file is None and not text:
        raise HTTPException(status_code=400, detail="Provide either a supported file or pasted text.")

    sentences = max(3, min(16, sentences))

    if file is not None:
        if not file.filename:
            raise HTTPException(status_code=400, detail="The uploaded file has no name.")
        content = await file.read()
        if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB.")
        try:
            raw_text = extract_text_from_bytes(content, file.filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="We could not read that file. It may be corrupted or password-protected; try another copy or paste the text.",
            ) from exc
        filename = file.filename
    else:
        raw_text = text or ""
        filename = "pasted-text"

    cleaned = clean_text(raw_text)
    if len(cleaned) < MIN_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail="We could not find enough readable text in that document. Try a PDF with selectable text or paste the paper text directly.",
        )

    try:
        result = summarize_text(cleaned, sentences)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document_id = str(uuid.uuid4())
    DOCUMENTS[document_id] = cleaned
    return {
        "document_id": document_id,
        "filename": filename,
        "character_count": len(cleaned),
        "text": cleaned,
        **result,
    }


@app.post("/api/ask")
def ask(
    document_id: str = Form(default=""),
    question: str = Form(...),
    text: str = Form(default=""),
) -> dict:
    stored = DOCUMENTS.get(document_id) if document_id else None
    if stored is not None:
        text = stored
    if not text:
        raise HTTPException(status_code=404, detail="Document not found. Summarize it first.")
    try:
        return answer_question(text, question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# Local development only: serve the public/ assets when not on Vercel.
# Registered after the API routes so specific routes keep precedence.
if not os.environ.get("VERCEL") and public_dir.is_dir():
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="public")
