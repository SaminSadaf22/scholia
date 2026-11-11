# Scholia — Research Paper Summarizer

A local FastAPI application that turns research papers into concise, section-aware summaries and answers questions about them. Scholia (from the classical *scholium*, an explanatory note on a text) is designed as a portfolio project for applied NLP and machine-learning engineering.

## Features

- One unified upload flow for PDF, DOCX, PPTX, TXT, MD, CSV, JSON, RTF, and similar text files, plus paste-text input
- Extract and clean document text without relying on a GPU
- Generate an extractive summary with LexRank graph centrality, section cues, and Maximal Marginal Relevance (MMR) for redundancy-aware selection
- Organize summaries into thesis-style segments: Author, Institution, Introduction, Literature Review, Research Gap, Methodology, Results and Discussion, and Conclusion
- Add a Research Gap segment from explicit gap language, scarcity statements, or the paper's novelty claim, even when no such heading exists
- Fall back to paragraph- and cue-based section inference when a PDF's headings are missing, so Introduction, Methodology, Results and Discussion, and Conclusion always appear
- Recognise headings that share a line with their first sentence, such as `Introduction: This paper…`
- Extract top keywords for the whole document and each segment
- Answer questions by retrieving the most relevant sentence
- Responsive browser UI with drag-and-drop upload, inline errors, live progress, section cards, copy, download, and suggested follow-up questions

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## Run

```bash
.venv/Scripts/python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Open:

```text
http://127.0.0.1:8001/
```

## API

### Health

```http
GET /health
```

### Summarize

```http
POST /api/summarize
Content-Type: multipart/form-data
```

Fields:

- `file`: supported document file, or
- `text`: pasted paper text
- `sentences`: number of summary sentences (3-16)

### Ask

```http
POST /api/ask
Content-Type: application/x-www-form-urlencoded
```

Fields:

- `document_id`
- `question`

## Architecture

```text
app/
  main.py              FastAPI routes and in-memory document store
  config.py            paths and limits
  services/
    file_parser.py     PDF, DOCX, PPTX, RTF, and text extraction
    text_cleaner.py    normalization, heading detection, and sentence splitting
    summarizer.py      LexRank + MMR selection, section and gap inference, retrieval QA
  static/              browser UI
tests/                 unit tests
```

## Notes

The current implementation uses extractive summarization, so it selects and orders important sentences from the original paper. Sentences are ranked with LexRank over a TF-IDF similarity graph and chosen with MMR so a section covers different points instead of repeating one. This keeps the project lightweight, explainable, and runnable without a GPU. A future extension can add abstractive summarization with a transformer model.

The default depth is 12 sentences (Brief 6, Balanced 12, Detailed 16). The Research Gap segment prefers explicit gap or scarcity statements, then novelty claims such as "to the best of our knowledge", and only then limitations or future-work remarks.
```

## Tests

```bash
.venv/Scripts/python -m unittest discover -s tests
```
