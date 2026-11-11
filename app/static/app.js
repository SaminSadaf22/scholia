const form = document.getElementById("summarize-form");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileName = document.getElementById("file-name");
const fileHint = document.getElementById("file-hint");
const fileClear = document.getElementById("file-clear");
const textarea = document.getElementById("paper-text");
const charCount = document.getElementById("char-count");
const slider = document.getElementById("sentence-slider");
const submitButton = document.getElementById("summarize-button");
const submitLabel = submitButton.querySelector(".button-label");
const progressPanel = document.getElementById("progress-panel");
const progressTitle = document.getElementById("progress-title");
const progressDetail = document.getElementById("progress-detail");
const progressBar = document.getElementById("progress-bar");
const progressSteps = [...document.querySelectorAll("#progress-steps li")];
const errorBanner = document.getElementById("error-banner");
const errorText = document.getElementById("error-text");
const errorClose = document.getElementById("error-close");
const emptyState = document.getElementById("empty-state");
const result = document.getElementById("result");
const resultTitle = document.getElementById("result-title");
const resultMeta = document.getElementById("result-meta");
const keywordsEl = document.getElementById("keywords");
const sectionsEl = document.getElementById("sections");
const summaryEl = document.getElementById("summary");
const statSentences = document.getElementById("stat-sentences");
const statReduction = document.getElementById("stat-reduction");
const statRead = document.getElementById("stat-read");
const copyButton = document.getElementById("copy-button");
const downloadButton = document.getElementById("download-button");
const newButton = document.getElementById("new-button");
const askForm = document.getElementById("ask-form");
const answerEl = document.getElementById("answer");
const suggestedQuestions = document.getElementById("suggested-questions");

const SAMPLE_TEXT = `Author
Amina Rahman and Daniel Cho

Institution
Department of Computer Science, Northfield University

Introduction
Reading research papers is a bottleneck for students and researchers. This paper proposes a lightweight, section-aware summarizer for scientific documents. The goal is to produce concise summaries without inventing facts or requiring a GPU.

Literature Review
Earlier work relies on keyword extraction and frequency-based ranking. Related systems use transformer models to generate abstractive summaries. These approaches are accurate but expensive to run and difficult to inspect.

Research Gap
Prior studies rarely preserve the paper's section structure. Existing tools also do not report which parts of a paper they omit, which makes it hard to trust a summary.

Methodology
The system extracts text from PDF, DOCX, and PPTX files in memory. It detects section headings and maps them to a consistent thesis structure. Sentences are ranked with LexRank over a TF-IDF similarity graph and chosen with MMR to avoid redundancy.

Results and Discussion
Experiments on 120 papers show that summaries retain the central claim in most cases. The average processing time is under two seconds on a laptop CPU. Discussion notes that scanned PDFs and highly technical tables remain difficult.

Conclusion
Extractive summarization offers a practical balance between speed, transparency, and factual grounding. Future work will add table extraction and optional abstractive polishing.`;

let mode = "upload";
let documentId = null;
let currentResult = null;
let progressTimer = null;

const PROGRESS_STAGES = [
  { title: "Reading your paper", detail: "Extracting text and finding section headings…", percent: 18, step: 0 },
  { title: "Mapping the structure", detail: "Matching headings to author, methodology, results, and more…", percent: 42, step: 1 },
  { title: "Ranking key sentences", detail: "Scoring sentences for relevance and removing repetition…", percent: 68, step: 2 },
  { title: "Writing your summary", detail: "Assembling each section into a compact, faithful read…", percent: 90, step: 3 },
];

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showError(message) {
  errorText.textContent = message;
  errorBanner.hidden = false;
}

function clearError() {
  errorBanner.hidden = true;
  errorText.textContent = "";
}

function setMode(nextMode) {
  mode = nextMode;
  document.querySelectorAll(".mode-tab").forEach((tab) => {
    const active = tab.dataset.mode === nextMode;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  document.getElementById("pane-upload").hidden = nextMode !== "upload";
  document.getElementById("pane-paste").hidden = nextMode !== "paste";
  document.getElementById("pane-upload").classList.toggle("is-active", nextMode === "upload");
  document.getElementById("pane-paste").classList.toggle("is-active", nextMode === "paste");
  clearError();
}

function updateFileLabel(file) {
  if (!file) {
    fileName.textContent = "Drop your paper here";
    fileHint.textContent = "or click to browse — PDF, DOCX, PPTX, TXT, MD, RTF";
    dropzone.classList.remove("has-file");
    fileClear.hidden = true;
    return;
  }
  fileName.textContent = file.name;
  fileHint.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · ready to summarize`;
  dropzone.classList.add("has-file");
  fileClear.hidden = false;
}

function clearFile() {
  fileInput.value = "";
  updateFileLabel(null);
}

function updateCharCount() {
  charCount.textContent = `${textarea.value.length.toLocaleString()} characters`;
}

function setDepth(value) {
  slider.value = value;
  document.querySelectorAll(".depth-option").forEach((option) => {
    option.classList.toggle("is-active", Number(option.dataset.depth) === Number(value));
  });
}

function startProgress() {
  progressPanel.hidden = false;
  let stageIndex = 0;
  const render = () => {
    const stage = PROGRESS_STAGES[stageIndex % PROGRESS_STAGES.length];
    progressTitle.textContent = stage.title;
    progressDetail.textContent = stage.detail;
    progressBar.style.width = `${stage.percent}%`;
    progressSteps.forEach((step, index) => {
      step.classList.toggle("is-active", index === stage.step);
      step.classList.toggle("is-done", index < stage.step);
    });
    stageIndex += 1;
  };
  render();
  clearInterval(progressTimer);
  progressTimer = setInterval(render, 1200);
}

function stopProgress(success) {
  clearInterval(progressTimer);
  progressTimer = null;
  if (success) {
    progressBar.style.width = "100%";
    progressSteps.forEach((step) => {
      step.classList.remove("is-active");
      step.classList.add("is-done");
    });
    setTimeout(() => {
      progressPanel.hidden = true;
      progressBar.style.width = "8%";
    }, 320);
  } else {
    progressPanel.hidden = true;
    progressBar.style.width = "8%";
  }
}

function formatNumber(value) {
  return Number(value || 0).toLocaleString();
}

function renderResult(data) {
  currentResult = data;
  documentId = data.document_id;

  resultTitle.textContent = data.filename || "Summary";
  const inferredNote = data.structure_inferred ? " · sections inferred (no headings found)" : "";
  resultMeta.textContent = `${data.sections.length} sections · ${data.selected_sentences} of ${data.total_sentences} sentences kept · ${formatNumber(data.character_count)} characters processed${inferredNote}`;

  keywordsEl.innerHTML = (data.keywords || [])
    .map((word) => `<span class="keyword">${escapeHtml(word)}</span>`)
    .join("");

  statSentences.textContent = data.selected_sentences;
  statReduction.textContent = `${Math.max(0, Math.round((1 - (data.compression_ratio || 0)) * 100))}%`;
  const summaryWords = (data.summary || "").split(/\s+/).filter(Boolean).length;
  statRead.textContent = Math.max(1, Math.round(summaryWords / 220));

  sectionsEl.innerHTML = (data.sections || [])
    .map((section, index) => `
      <article class="section-card" style="--i: ${index}">
        <span class="section-index">${String(index + 1).padStart(2, "0")}</span>
        <div class="section-body">
          <h3>${escapeHtml(section.title)}</h3>
          <p>${escapeHtml(section.summary)}</p>
        </div>
        <span class="section-meta">${section.selected_sentences} of ${section.total_sentences} sentences</span>
      </article>
    `)
    .join("");

  summaryEl.textContent = data.summary || "";
  answerEl.innerHTML = "";

  emptyState.hidden = true;
  result.hidden = false;
  result.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function submitForm() {
  clearError();

  if (mode === "upload" && !fileInput.files[0]) {
    showError("Choose a PDF, Word document, slide deck, or text file first.");
    dropzone.focus?.();
    return;
  }

  if (mode === "paste" && textarea.value.trim().length < 40) {
    showError("Paste a little more text so there is enough to summarize.");
    textarea.focus();
    return;
  }

  const payload = new FormData();
  if (mode === "upload") {
    payload.append("file", fileInput.files[0]);
  } else {
    payload.append("text", textarea.value.trim());
  }
  payload.append("sentences", slider.value);

  submitButton.disabled = true;
  submitLabel.textContent = "Summarizing…";
  startProgress();

  try {
    const response = await fetch("/api/summarize", { method: "POST", body: payload });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Summarization failed. Please try again." }));
      throw new Error(error.detail || "Summarization failed. Please try again.");
    }
    const data = await response.json();
    stopProgress(true);
    renderResult(data);
  } catch (error) {
    stopProgress(false);
    showError(error.message || "Something went wrong while summarizing. Please try again.");
  } finally {
    submitButton.disabled = false;
    submitLabel.textContent = "Summarize paper";
  }
}

function resetWorkspace() {
  currentResult = null;
  documentId = null;
  clearFile();
  textarea.value = "";
  updateCharCount();
  setDepth(6);
  clearError();
  answerEl.innerHTML = "";
  result.hidden = true;
  emptyState.hidden = false;
  setMode("upload");
  document.getElementById("summarize").scrollIntoView({ behavior: "smooth", block: "start" });
}

function loadSample(autoSubmit = false) {
  setMode("paste");
  textarea.value = SAMPLE_TEXT;
  updateCharCount();
  clearError();
  if (autoSubmit) {
    form.requestSubmit();
  } else {
    textarea.focus();
  }
}

/* ---------- Input events ---------- */

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitForm();
});

document.querySelectorAll(".mode-tab").forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
});

textarea.addEventListener("input", updateCharCount);

fileInput.addEventListener("change", () => {
  updateFileLabel(fileInput.files[0] || null);
  clearError();
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  updateFileLabel(file);
  clearError();
});

fileClear.addEventListener("click", (event) => {
  event.preventDefault();
  clearFile();
});

document.querySelectorAll(".depth-option").forEach((option) => {
  option.addEventListener("click", () => setDepth(option.dataset.depth));
});

slider.addEventListener("input", () => setDepth(slider.value));

errorClose.addEventListener("click", clearError);

document.getElementById("load-sample").addEventListener("click", () => loadSample(false));
document.getElementById("hero-sample").addEventListener("click", () => loadSample(true));

/* ---------- Result actions ---------- */

copyButton.addEventListener("click", async () => {
  if (!currentResult) return;
  try {
    await navigator.clipboard.writeText(currentResult.summary || "");
    const label = copyButton.querySelector("span");
    const original = label.textContent;
    label.textContent = "Copied";
    setTimeout(() => {
      label.textContent = original;
    }, 1600);
  } catch {
    showError("Clipboard access is unavailable in this browser. You can select the summary text and copy it manually.");
  }
});

downloadButton.addEventListener("click", () => {
  if (!currentResult) return;
  const blob = new Blob([currentResult.summary || ""], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const base = (currentResult.filename || "paper-summary").replace(/\.[^.]+$/, "");
  link.href = url;
  link.download = `${base}-summary.txt`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
});

newButton.addEventListener("click", resetWorkspace);

/* ---------- Question answering ---------- */

async function askQuestion(question) {
  if (!documentId) {
    showError("Summarize a document before asking a question.");
    return;
  }
  const trimmed = question.trim();
  if (!trimmed) return;

  answerEl.innerHTML = `<div class="answer-card"><strong>Searching</strong><p>Looking for the most relevant sentence…</p></div>`;

  const payload = new FormData();
  payload.append("document_id", documentId);
  payload.append("question", trimmed);

  try {
    const response = await fetch("/api/ask", { method: "POST", body: payload });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "The question could not be answered." }));
      throw new Error(error.detail || "The question could not be answered.");
    }
    const data = await response.json();
    const confidence = Math.round((data.confidence || 0) * 100);
    const found = data.source_sentence ? `${confidence}% match to a source sentence` : "No strong match found in the document";
    answerEl.innerHTML = `
      <div class="answer-card">
        <strong>Answer</strong>
        <p>${escapeHtml(data.answer)}</p>
        <small>${escapeHtml(found)}</small>
      </div>
    `;
  } catch (error) {
    answerEl.innerHTML = `
      <div class="answer-card is-error">
        <strong>Couldn’t answer</strong>
        <p>${escapeHtml(error.message)}</p>
      </div>
    `;
  }
}

askForm.addEventListener("submit", (event) => {
  event.preventDefault();
  askQuestion(askForm.question.value);
});

suggestedQuestions.querySelectorAll("button").forEach((button) => {
  button.addEventListener("click", () => {
    askForm.question.value = button.textContent;
    askForm.requestSubmit();
  });
});
