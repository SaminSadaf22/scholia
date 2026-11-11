import unittest

from app.services.summarizer import answer_question, summarize_text
from app.services.text_cleaner import split_sentences


SAMPLE = """
This paper proposes a lightweight method for summarizing scientific documents.
The method extracts sentences using TF-IDF scoring and ranks them by importance.
Experiments show that the approach is fast and requires no GPU.
The conclusion states that extractive summaries preserve factual content.
"""


from app.services.text_cleaner import segment_document


HEADING_WITH_GAP = """
Introduction
Reading research papers is slow and expensive for many students.
This paper studies how section structure affects automatic summaries.

Research Gap
Prior work has not examined whether section-aware allocation improves summary coverage.
Existing tools also rarely report which parts of a paper they omit.

Methodology
We compare a structure-aware selector with a uniform selector on 120 papers.
Both systems use the same sentence encoder and the same input text.

Results
The structure-aware system improves coverage of the conclusion by 18 percent.
Human readers preferred its summaries in most of the evaluated cases.

Conclusion
Section-aware allocation is a practical default for research reading tools.
Future work will extend the evaluation to non-English papers.
"""


HEADINGLESS = """
Reading research papers is a bottleneck for students and researchers.
This paper proposes a lightweight summarizer for scientific documents.
The goal is to produce concise summaries without inventing facts or requiring a GPU.
Papers are increasingly long and dense, which makes skimming them difficult.
Existing tools either summarize too aggressively or require expensive cloud models.

Earlier work relies on keyword extraction and frequency-based ranking.
Related systems use transformer models to generate abstractive summaries.
These approaches are accurate but expensive to run and difficult to inspect.
Prior studies rarely preserve the paper's section structure.
We build on extractive ranking while adding structural cues.

The system extracts text from PDF, DOCX, and PPTX files in memory.
It detects section headings and maps them to a consistent thesis structure.
Sentences are ranked with TF-IDF, position priors, and a redundancy penalty before selection.
We use cosine similarity to remove near-duplicate sentences from each section.
The pipeline runs entirely on a laptop CPU.

Experiments on 120 papers show that summaries retain the central claim in most cases.
The average processing time is under two seconds on a laptop CPU.
Our method outperforms a frequency-only baseline by 14 percent on a human relevance score.
Results also show that section-aware allocation improves coverage of the conclusion.
Discussion notes that scanned PDFs and highly technical tables remain difficult.

Extractive summarization offers a practical balance between speed, transparency, and factual grounding.
Future work will add table extraction and optional abstractive polishing.
We conclude that structure-aware ranking is a strong default for research reading tools.
"""


class SummarizerTests(unittest.TestCase):
    def test_split_sentences(self):
        sentences = split_sentences(SAMPLE)
        self.assertGreaterEqual(len(sentences), 3)

    def test_summary_has_requested_length(self):
        result = summarize_text(SAMPLE, sentence_count=2)
        self.assertEqual(result["selected_sentences"], 2)
        self.assertIn("TF-IDF", result["summary"])

    def test_section_segmentation(self):
        sections = segment_document(SAMPLE.replace("This paper proposes", "Introduction\nThis paper proposes"))
        # The fallback sample has no real headings, so it stays as one document.
        self.assertEqual(len(sections), 1)

    def test_headingless_document_gets_core_sections(self):
        result = summarize_text(HEADINGLESS, sentence_count=9)
        titles = [section["title"] for section in result["sections"]]
        for title in ("Introduction", "Methodology", "Results and Discussion", "Conclusion"):
            self.assertIn(title, titles)
        self.assertIn("Research Gap", titles)
        gap = next(section for section in result["sections"] if section["title"] == "Research Gap")
        self.assertIn("rarely", gap["summary"])
        self.assertGreaterEqual(result["selected_sentences"], 4)

    def test_explicit_research_gap_heading_is_preserved(self):
        result = summarize_text(HEADING_WITH_GAP, sentence_count=12)
        titles = [section["title"] for section in result["sections"]]
        self.assertIn("Research Gap", titles)
        gap = next(section for section in result["sections"] if section["title"] == "Research Gap")
        self.assertIn("Prior work", gap["summary"])

    def test_question_returns_relevant_sentence(self):
        result = answer_question(SAMPLE, "Which scoring method is used?")
        self.assertIsNotNone(result["source_sentence"])
        self.assertIn("TF-IDF", result["answer"])


if __name__ == "__main__":
    unittest.main()
