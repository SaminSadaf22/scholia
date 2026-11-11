from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
import os

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    UPLOAD_DIR = Path("/tmp") / "scholia" / "uploads"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 25
DEFAULT_SUMMARY_SENTENCES = 12
MAX_SUMMARY_SENTENCES = 16
MIN_TEXT_CHARS = 40
