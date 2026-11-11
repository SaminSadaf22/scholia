import re

SECTION_ALIASES = {
    "author": "Author",
    "authors": "Author",
    "author name": "Author",
    "affiliation": "Institution",
    "affiliations": "Institution",
    "institution": "Institution",
    "institutions": "Institution",
    "university": "Institution",
    "department": "Institution",
    "organization": "Institution",
    "organisation": "Institution",
    "introduction": "Introduction",
    "intro": "Introduction",
    "abstract": "Introduction",
    "overview": "Introduction",
    "motivation": "Introduction",
    "problem statement": "Introduction",
    "objective": "Introduction",
    "objectives": "Introduction",
    "goal": "Introduction",
    "goals": "Introduction",
    "research objective": "Introduction",
    "research objectives": "Introduction",
    "literature review": "Literature Review",
    "literature": "Literature Review",
    "related work": "Literature Review",
    "related works": "Literature Review",
    "background": "Literature Review",
    "background and related work": "Literature Review",
    "previous work": "Literature Review",
    "prior work": "Literature Review",
    "prior art": "Literature Review",
    "research gap": "Research Gap",
    "research gaps": "Research Gap",
    "knowledge gap": "Research Gap",
    "knowledge gaps": "Research Gap",
    "literature gap": "Research Gap",
    "gap analysis": "Research Gap",
    "identified gap": "Research Gap",
    "open problems": "Research Gap",
    "open questions": "Research Gap",
    "open challenges": "Research Gap",
    "unresolved problems": "Research Gap",
    "methodology": "Methodology",
    "method": "Methodology",
    "methods": "Methodology",
    "materials and methods": "Methodology",
    "experimental setup": "Methodology",
    "experiment setup": "Methodology",
    "approach": "Methodology",
    "proposed method": "Methodology",
    "proposed approach": "Methodology",
    "our approach": "Methodology",
    "our method": "Methodology",
    "model architecture": "Methodology",
    "architecture": "Methodology",
    "system design": "Methodology",
    "framework": "Methodology",
    "training": "Methodology",
    "experiments": "Methodology",
    "implementation": "Methodology",
    "implementation details": "Methodology",
    "data collection": "Methodology",
    "dataset": "Methodology",
    "datasets": "Methodology",
    "materials": "Methodology",
    "participants": "Methodology",
    "research design": "Methodology",
    "procedure": "Methodology",
    "results": "Results and Discussion",
    "result": "Results and Discussion",
    "findings": "Results and Discussion",
    "output": "Results and Discussion",
    "outputs": "Results and Discussion",
    "discussion": "Results and Discussion",
    "results and discussion": "Results and Discussion",
    "experimental results": "Results and Discussion",
    "evaluation": "Results and Discussion",
    "evaluation results": "Results and Discussion",
    "performance": "Results and Discussion",
    "ablation study": "Results and Discussion",
    "ablation studies": "Results and Discussion",
    "analysis": "Results and Discussion",
    "comparison": "Results and Discussion",
    "case study": "Results and Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "concluding remarks": "Conclusion",
    "future work": "Conclusion",
    "future directions": "Conclusion",
    "limitations": "Conclusion",
    "discussion and conclusion": "Conclusion",
    "summary and conclusion": "Conclusion",
    "summary": "Conclusion",
}

# Bare "Gap" is a common heading but too ambiguous for prefix matching, which
# would misread a sentence like "Gap between train and test accuracy".
EXACT_SECTION_ALIASES = {"gap": "Research Gap", "gaps": "Research Gap"}

STOP_SECTION_ALIASES = (
    "references",
    "reference",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "appendices",
    "supplementary material",
    "supplementary materials",
    "funding",
    "conflict of interest",
    "conflicts of interest",
    "data availability",
    "author contributions",
    "about the authors",
)

INSTITUTION_WORDS = {
    "university", "institute", "institution", "college", "school", "department",
    "faculty", "laboratory", "lab", "research", "centre", "center", "google",
    "microsoft", "meta", "amazon", "openai", "deepmind", "brain", "inc", "ltd",
}

NAME_PATTERN = re.compile(
    r"\b([A-ZŁ][a-z]+(?:\s+[A-ZŁ]\.?)?\s+(?:[A-ZŁ][a-z]+|[A-ZŁ]\.)\b)"
)

FRONT_MATTER_JUNK = (
    "provided proper attribution",
    "scholarly works",
    "permission to make digital or hard copies",
    "all rights reserved",
    "copyright",
    "equal contribution",
    "corresponding author",
    "conference on neural information processing",
    "arxiv:",
    "proceedings of",
)

INSTITUTION_PATTERNS = (
    r"(?:Department|School|Faculty|College|Institute|Laboratory|Lab)\s+(?:of\s+)?[A-Z][^,;]{2,80}",
    r"University\s+of\s+[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*",
    r"[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,2}\s+(?:University|Institute|College|Laboratory|Center|Centre)",
    r"(?:Google|Microsoft|Meta|Amazon|OpenAI|DeepMind)\s+(?:Brain|Research)",
)


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a lightweight, dependency-free splitter."""
    text = clean_text(text)
    if not text:
        return []

    protected = text
    for abbreviation in ("Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "et al.", "e.g.", "i.e.", "Fig.", "Eq."):
        protected = protected.replace(abbreviation, abbreviation.replace(".", "<DOT>"))

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[])", protected)
    sentences = []
    for part in parts:
        sentence = part.replace("<DOT>", ".").strip()
        if len(sentence) >= 20:
            sentences.append(sentence)
    return sentences


def top_terms(text: str, limit: int = 12) -> list[str]:
    """Return the most frequent meaningful terms in a document."""
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was",
        "were", "has", "have", "had", "not", "but", "can", "may", "also",
        "such", "their", "these", "those", "into", "using", "used", "our",
        "we", "they", "them", "its", "it", "which", "than", "more", "most",
        "paper", "study", "results", "method", "methods", "proposed", "based",
    }
    counts: dict[str, int] = {}
    for word in words:
        if word in stop_words:
            continue
        counts[word] = counts.get(word, 0) + 1

    return [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _strip_heading_prefix(title: str) -> str:
    title = title.strip()
    title = re.sub(
        r"^(?:(?:chapter|section|part)\s+)?(?:\d+(?:\.\d+)*|[IVXLC]+)[.):]?\s+",
        "",
        title,
        flags=re.IGNORECASE,
    )
    return title.strip()


def _canonical_section(title: str) -> str | None:
    title = _strip_heading_prefix(title)
    normalized = re.sub(r"[^a-z\s]", "", title.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized in EXACT_SECTION_ALIASES:
        return EXACT_SECTION_ALIASES[normalized]
    for key, canonical in sorted(SECTION_ALIASES.items(), key=lambda item: -len(item[0])):
        if normalized == key or normalized.startswith(key + " "):
            return canonical
    return None


def _is_stop_heading(line: str) -> bool:
    core = _strip_heading_prefix(line)
    normalized = re.sub(r"[^a-z\s]", "", core.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return any(normalized == key or normalized.startswith(key + " ") for key in STOP_SECTION_ALIASES)


def _clean_institution(value: str) -> str:
    value = re.sub(r"\S+@\S+", "", value)
    value = re.sub(r"[\u2217*\u2020\u2021\u00a7]+", "", value)
    value = re.sub(r"\s+", " ", value).strip(" ,;.")
    return value[:140].strip()


def split_front_matter(text: str) -> tuple[str, str, str]:
    """Separate title/author/affiliation lines from the paper body.

    Returns ``(author, institution, body)``. If the front matter is ambiguous,
    the original text is returned unchanged so no content is lost.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_heading = None
    for index, line in enumerate(lines[:100]):
        canonical = _canonical_section(line)
        if canonical in ("Author", "Institution") and is_heading(line):
            return "", "", text
        if canonical == "Introduction" and is_heading(line):
            first_heading = index
            break
    if first_heading is None or first_heading < 2:
        return "", "", text

    body = "\n\n".join(lines[first_heading:]).strip()
    if not body:
        return "", "", text

    front: list[str] = []
    for line in lines[:first_heading]:
        lower = line.lower()
        if any(marker in lower for marker in FRONT_MATTER_JUNK):
            continue
        line = re.sub(r"\S+@\S+", "", line).strip()
        if line:
            front.append(line)
    if not front:
        return "", "", text

    # The first substantial line is almost always the paper title.
    if len(front) > 1:
        first = front[0]
        is_title = (
            ":" in first
            or (len(first.split()) >= 4 and " and " not in first)
        )
        if is_title:
            front = front[1:]

    authors: list[str] = []
    institutions: list[str] = []
    for line in front:
        lower = line.lower()
        line_institutions = []
        for pattern in INSTITUTION_PATTERNS:
            for match in re.finditer(pattern, line):
                institution = _clean_institution(match.group(0))
                if institution and institution.lower() not in {value.lower() for value in institutions}:
                    institutions.append(institution)
                line_institutions.append(match)

        # Author names usually sit before the affiliation on the same line.
        author_part = line
        if line_institutions:
            author_part = line[: min(match.start() for match in line_institutions)].strip()
        if "@" in author_part or any(word in author_part.lower().split() for word in INSTITUTION_WORDS):
            continue
        for name in NAME_PATTERN.findall(author_part):
            name = name.strip(" .,;*\u2020\u2021\u00a7")
            parts = {part.lower().strip(".") for part in name.split()}
            if not parts or parts & INSTITUTION_WORDS:
                continue
            if name and name.lower() not in {value.lower() for value in authors}:
                authors.append(name)

    if not authors and not institutions:
        return "", "", text

    return ", ".join(authors), "; ".join(institutions), body


def _looks_like_numbered_heading(line: str) -> bool:
    core = line.strip()
    if not re.match(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+\S", core, flags=re.IGNORECASE):
        return False
    words = _strip_heading_prefix(core).split()
    if not 1 <= len(words) <= 10:
        return False
    if core.endswith((".", "?", "!", ";", ",")):
        return False
    if any(character in core for character in "@:"):
        return False
    alpha_words = [word for word in words if any(character.isalpha() for character in word)]
    capitalised = sum(1 for word in alpha_words if word[0].isupper())
    return capitalised >= max(1, int(len(alpha_words) * 0.6))


def is_heading(paragraph: str) -> bool:
    paragraph = paragraph.strip()
    if not paragraph or len(paragraph) > 100:
        return False
    if paragraph.endswith(("!", "?", ";", ",")):
        return False

    core = _strip_heading_prefix(paragraph)
    if not core or len(core.split()) > 8:
        return False

    canonical = _canonical_section(core)
    if canonical is None:
        return False
    # A trailing full stop is allowed for short headings such as "Results.".
    if paragraph.endswith(".") and len(core.split()) > 6:
        return False
    return True


def _inline_heading(line: str) -> tuple[str, str] | None:
    """Recognise headings that share a line with their first sentence."""
    match = re.match(
        r"^(?:\d+(?:\.\d+)*[.)]?\s*)?([A-Za-z][A-Za-z\s&/-]{0,60}?)\s*[:\-\u2013\u2014]\s+(.+)$",
        line.strip(),
    )
    if not match:
        return None
    candidate, remainder = match.group(1).strip(), match.group(2).strip()
    canonical = _canonical_section(candidate)
    if canonical is None or not is_heading(candidate):
        return None
    return canonical, remainder


def segment_document(text: str) -> list[tuple[str, list[str]]]:
    """Split a document into the requested thesis-like sections when headings are present."""
    lines = [line.strip() for line in text.splitlines()]
    sections: list[tuple[str, list[str]]] = []
    current_title = "Introduction"
    current_lines: list[str] = []
    found_heading = False

    def flush() -> None:
        nonlocal current_lines
        joined = " ".join(line for line in current_lines if line).strip()
        if joined:
            sections.append((current_title, [joined]))
        current_lines = []

    for line in lines:
        if not line:
            if current_lines:
                flush()
            continue

        if _is_stop_heading(line):
            if current_lines:
                flush()
            break

        inline = _inline_heading(line)
        if inline is not None:
            canonical, remainder = inline
            if current_lines:
                flush()
            current_title = canonical
            found_heading = True
            if remainder:
                current_lines.append(remainder)
            continue

        canonical = _canonical_section(line)
        if canonical is not None and is_heading(line):
            if current_lines:
                flush()
            current_title = canonical
            found_heading = True
        elif _looks_like_numbered_heading(line):
            # Subheadings such as "3.1 Encoder and Decoder Stacks" belong to
            # the current section; do not let them appear inside the summary.
            continue
        else:
            current_lines.append(line)
    if current_lines:
        flush()

    if not found_heading:
        joined = " ".join(line for line in lines if line).strip()
        return [("Introduction", [joined])] if joined else []

    merged: dict[str, list[str]] = {}
    order: list[str] = []
    for title, paragraphs in sections:
        if title in merged:
            merged[title].extend(paragraphs)
        else:
            merged[title] = list(paragraphs)
            order.append(title)

    return [(title, merged[title]) for title in order if merged[title]]
