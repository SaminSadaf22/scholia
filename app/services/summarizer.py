import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_cleaner import (
    clean_text,
    segment_document,
    split_front_matter,
    split_sentences,
    top_terms,
)

SECTION_CUES = {
    "Introduction": (
        "we propose", "we present", "we introduce", "this paper", "this work",
        "aim", "goal", "objective", "contribution", "problem", "challenge",
        "we study", "we investigate", "in this article",
    ),
    "Literature Review": (
        "prior", "previous", "related", "existing", "literature", "earlier",
        "et al", "however", "gap", "limitation", "has been", "studies",
        "state of the art", "in contrast",
    ),
    "Methodology": (
        "we propose", "we use", "we train", "we evaluate", "dataset", "data",
        "model", "method", "approach", "algorithm", "architecture", "experiment",
        "implementation", "participants", "procedure", "measure", "sample",
        "analysis", "encoder", "decoder", "network", "layer", "attention",
    ),
    "Results and Discussion": (
        "result", "show", "shows", "found", "indicate", "significant", "increase",
        "decrease", "improve", "outperform", "accuracy", "precision", "recall",
        "f1", "bleu", "score", "compared", "table", "figure", "%", "p <", "p=",
        "we observe", "suggest", "better", "higher", "lower",
    ),
    "Conclusion": (
        "conclude", "conclusion", "in this work", "we presented", "contribution",
        "future", "limitation", "recommend", "suggest", "overall", "summary",
    ),
    "Research Gap": (
        "gap", "gaps", "remains unclear", "has not been", "have not been",
        "little is known", "few studies", "unexplored", "open question",
        "open problem", "lack of", "limited", "rarely", "future work",
        "future research", "we address", "this paper fills", "however",
    ),
}

# Language that signals the shortcoming a paper is responding to. Strong cues
# identify an explicit gap; medium cues identify scarcity; weak cues cover
# limitations and future work when the paper never says "gap" outright.
GAP_STRONG_CUES = (
    "research gap", "knowledge gap", "gap in the literature", "gap remains",
    "remains unclear", "remains unknown", "remains unresolved",
    "has not been studied", "has not been investigated", "has not been explored",
    "has not been addressed", "have not been studied", "have not been investigated",
    "no prior work", "no existing work", "few studies", "little is known",
    "little attention", "under-explored", "underexplored", "unexplored",
    "open question", "open problem", "not well understood", "lack of research",
    "limited research", "scarce research", "we address this gap",
    "to fill this gap", "fills this gap", "addresses this gap", "this gap",
    "these gaps", "however, existing", "however, prior", "however, most existing",
    "however, current", "remains a challenge", "remains challenging",
)
GAP_MEDIUM_CUES = (
    "rarely", "seldom", "few works", "few attempts", "little research",
    "little work", "lack of studies", "lack of evidence", "lack of understanding",
    "limited studies", "limited evidence", "limited attention", "no study",
    "no studies",
)
GAP_WEAK_CUES = (
    "limitation", "future work", "future research", "further research",
    "should be investigated", "warrants", "calls for", "challenge",
    "remains", "difficult",
)
GAP_CONTRAST_CUES = (
    "however", "although", "despite", "yet", "whereas", "in contrast",
    "on the other hand",
)
GAP_NOVELTY_CUES = (
    "to the best of our knowledge", "first to", "the first ", "unlike prior",
    "unlike existing", "no prior", "novel",
)

BOILERPLATE_PATTERNS = (
    r"^in this (section|paper),? we (describe|outline|organize|present the rest)",
    r"^the rest of (this|the) (paper|section)",
    r"^the remainder of (this|the) paper",
    r"^we (present|show|report|summari[sz]e) (these|our|the|all) results (in|on|at) (table|figure)",
    r"^(these|our|the) results are (presented|shown|reported|summari[sz]ed) in (table|figure)",
    r"^(table|figure)\s+\d+\s+(summari[sz]es|shows|presents|reports) (these|our|the)?\s*results",
    r"^(figure|table|algorithm|equation)\s+\d+",
    r"^(copyright|all rights reserved)",
    r"^(received|accepted|published)\s",
    r"^we (would like to )?thank",
    r"^this work was supported",
    r"^[\[\d\]]+\s+[A-Z][a-z]+,",
)


def _repair_split_hyphenation(text: str) -> str:
    """Join words torn across PDF lines when the paper spells them closed elsewhere."""
    hyphenated = Counter(re.findall(r"\b[a-z]{3,}-[a-z]{3,}\b", text.lower()))
    words = Counter(re.findall(r"\b[a-z]{6,}\b", text.lower()))

    def replace(match: re.Match) -> str:
        token = match.group(0)
        joined = token.replace("-", "").lower()
        if words[joined] >= 2 and hyphenated[token.lower()] * 2 <= words[joined]:
            return token.replace("-", "")
        return token

    return re.sub(r"\b[a-z]{3,}-[a-z]{3,}\b", replace, text)


def _clean_sentence(sentence: str) -> str:
    for source, target in {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "\u00ad": ""}.items():
        sentence = sentence.replace(source, target)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = re.sub(r"^[\u2022\u25cf\u25aa\-\u2013\u2014]\s*", "", sentence)
    # Inline citations add noise to a summary sentence without changing what
    # it says, so drop them before scoring.
    sentence = re.sub(r"\s*\[\d+(?:\s*[-,;]\s*\d+)*\]", "", sentence)
    sentence = re.sub(r"\s*\((?:[A-Z][A-Za-z\-]+ et al\.?,?\s*)\d{4}[a-z]?\)", "", sentence)
    sentence = re.sub(r"\s+([,.;:])", r"\1", sentence)
    # PDF bullet glyphs often survive mid-sentence after extraction.
    sentence = re.sub(r"\s*[\u2022\u25cf\u25aa\u25e6]\s*", " ", sentence)
    sentence = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "-", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence.replace(" .", ".").strip()


def _informative(sentence: str, title: str) -> bool:
    lower = sentence.lower()
    words = sentence.split()
    if not sentence or len(words) < 4:
        return False
    if re.search(r"\S+@\S+|https?://|doi\.org", lower):
        return False
    if any(re.match(pattern, lower) for pattern in BOILERPLATE_PATTERNS):
        return False
    if len(words) >= 8:
        numeric = sum(1 for word in words if re.fullmatch(r"[-+()\d.,%×/·=<>≈]+", word))
        if numeric / len(words) > 0.42:
            return False
    if title in ("Author", "Institution"):
        return True
    # Reference-list fragments and table rows are rarely useful in a summary.
    if re.match(r"^\[\d+\]", sentence) and len(words) < 30:
        return False
    return True


def _sentence_matrix(sentences: list[str]):
    """Vectorise sentences, falling back to character n-grams for tiny inputs."""
    for kwargs in (
        {"stop_words": "english", "max_features": 8000, "sublinear_tf": True, "ngram_range": (1, 2)},
        {"analyzer": "char_wb", "ngram_range": (3, 5), "sublinear_tf": True},
    ):
        try:
            return TfidfVectorizer(**kwargs).fit_transform(sentences)
        except ValueError:
            continue
    return None


def _lexrank_centrality(matrix, threshold: float = 0.08, damping: float = 0.85) -> np.ndarray:
    """Score sentences by graph centrality (LexRank) instead of centroid similarity.

    Each sentence is a node; edges are cosine similarities above a noise
    threshold. A sentence ranks highly when it is similar to other important
    sentences, not merely when it is close to the section's average wording.
    """
    count = matrix.shape[0]
    if count <= 1:
        return np.ones(count, dtype=float)

    graph = cosine_similarity(matrix)
    np.fill_diagonal(graph, 0.0)
    graph[graph < threshold] = 0.0

    row_sums = graph.sum(axis=1)
    for index in range(count):
        if row_sums[index] > 0:
            graph[index] /= row_sums[index]
        else:
            graph[index] = 1.0 / (count - 1)
            graph[index, index] = 0.0

    scores = np.full(count, 1.0 / count)
    for _ in range(100):
        updated = (1.0 - damping) / count + damping * graph.T.dot(scores)
        if np.abs(updated - scores).sum() < 1e-7:
            scores = updated
            break
        scores = updated
    return scores


def _section_scores(sentences: list[str], title: str) -> tuple[np.ndarray, np.ndarray | None]:
    matrix = _sentence_matrix(sentences)
    if matrix is None:
        centrality = np.ones(len(sentences), dtype=float)
    else:
        centrality = _lexrank_centrality(matrix)
        spread = float(centrality.max() - centrality.min())
        centrality = (centrality - centrality.min()) / spread if spread > 1e-9 else np.ones(len(sentences), dtype=float)

    document_terms = top_terms(" ".join(sentences), 12)
    cues = SECTION_CUES.get(title, ())
    scores = []
    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        word_count = len(sentence.split())
        coverage = (
            sum(1 for term in document_terms if term in lower) / max(1, len(document_terms))
            if document_terms
            else 0.0
        )
        cue_hits = sum(1 for cue in cues if cue in lower)
        cue_score = min(1.0, cue_hits / 2.0)

        position = 1.0
        if index == 0:
            position = 1.14
        elif index == 1:
            position = 1.06
        elif index == len(sentences) - 1:
            position = 1.04

        length = 1.0
        if word_count < 7:
            length = 0.72
        elif word_count > 55:
            length = 0.86
        elif 12 <= word_count <= 38:
            length = 1.08

        score = (0.50 * float(centrality[index]) + 0.22 * coverage + 0.28 * cue_score) * position * length

        if title == "Results and Discussion":
            numeric = sum(1 for word in sentence.split() if re.search(r"\d", word))
            score *= 1.0 + min(0.3, numeric * 0.05)
            if any(marker in lower for marker in ("significant", "outperform", "improve", "better", "higher", "lower", "p <", "p=")):
                score *= 1.12

        if title == "Conclusion" and any(marker in lower for marker in ("future work", "limitation", "future research", "we conclude")):
            score *= 1.16

        if title == "Research Gap":
            strong = sum(1 for cue in GAP_STRONG_CUES if cue in lower)
            score *= 1.0 + min(0.5, strong * 0.15)

        if any(re.match(pattern, lower) for pattern in BOILERPLATE_PATTERNS):
            score *= 0.15

        scores.append(max(0.0, score))

    return np.array(scores, dtype=float), matrix


def _select_sentences(
    sentences: list[str],
    title: str,
    sentence_count: int,
    exclude: set[str] | None = None,
) -> list[int]:
    """Select sentences with Maximal Marginal Relevance over LexRank scores.

    MMR trades relevance against redundancy, so the result covers different
    aspects of a section instead of repeating the same claim.
    """
    exclude = exclude or set()
    candidates = [
        index
        for index, sentence in enumerate(sentences)
        if _informative(sentence, title) and _match_key(sentence) not in exclude
    ]
    if not candidates:
        candidates = [index for index in range(len(sentences)) if _match_key(sentences[index]) not in exclude]
    if not candidates:
        return []

    sentence_count = max(1, min(sentence_count, len(candidates)))
    if sentence_count >= len(candidates):
        return sorted(candidates)

    scores, matrix = _section_scores(sentences, title)
    for index in range(len(scores)):
        if index not in candidates:
            scores[index] = -1.0

    ranked = [int(index) for index in np.argsort(scores)[::-1] if scores[index] >= 0]
    if not ranked:
        return sorted(candidates)[:sentence_count]

    best_score = max(float(scores[index]) for index in ranked)
    normalised = {
        index: (float(scores[index]) / best_score if best_score > 0 else 1.0)
        for index in ranked
    }
    similarity = (
        cosine_similarity(matrix)
        if matrix is not None
        else np.zeros((len(sentences), len(sentences)), dtype=float)
    )

    if sentence_count == 1:
        best = ranked[0]
        # Lead sentences carry context in most academic sections. Keep one when
        # it is close to the strongest sentence rather than choosing an isolated
        # result from deep inside the section.
        if title in ("Introduction", "Conclusion") and 0 in candidates and scores[0] >= 0.86 * scores[best]:
            return [0]
        return [best]

    selected = [ranked[0]]
    while len(selected) < sentence_count:
        best_index, best_value = None, float("-inf")
        for index in ranked:
            if index in selected:
                continue
            redundancy = max(float(similarity[index, chosen]) for chosen in selected)
            value = 0.72 * normalised[index] - 0.28 * redundancy
            # Prefer sentences that continue the selected narrative.
            if any(abs(index - chosen) == 1 for chosen in selected):
                value += 0.02
            if value > best_value:
                best_index, best_value = index, value
        if best_index is None:
            break
        selected.append(best_index)

    return sorted(selected)


def _summarize_block(
    text: str,
    sentence_count: int,
    title: str,
    exclude: set[str] | None = None,
) -> dict:
    if title in ("Author", "Institution"):
        cleaned = clean_text(text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            return {"summary": cleaned, "selected_sentences": 1, "total_sentences": 1}
        return {"summary": "", "selected_sentences": 0, "total_sentences": 0}

    sentences = [_clean_sentence(sentence) for sentence in split_sentences(text)]
    sentences = [sentence for sentence in sentences if sentence]

    # Repeated headers and copied sentences otherwise win twice in a row.
    deduped: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        key = _match_key(sentence)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sentence)
    sentences = deduped

    exclude = exclude or set()
    if exclude:
        sentences = [sentence for sentence in sentences if _match_key(sentence) not in exclude]

    if not sentences:
        if exclude:
            return {"summary": "", "selected_sentences": 0, "total_sentences": 0}
        cleaned = clean_text(text)
        if cleaned:
            return {"summary": cleaned, "selected_sentences": 1, "total_sentences": 1}
        return {"summary": "", "selected_sentences": 0, "total_sentences": 0}

    selected_indices = _select_sentences(sentences, title, sentence_count, exclude=None)
    summary = " ".join(sentences[index] for index in selected_indices)
    return {
        "summary": summary,
        "selected_sentences": len(selected_indices),
        "total_sentences": len(sentences),
    }


CORE_SECTIONS = ("Introduction", "Methodology", "Results and Discussion", "Conclusion")


def _match_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _token_overlap(first: str, second: str) -> float:
    def words(text: str) -> set[str]:
        return {word for word in re.findall(r"[a-z]{3,}", text.lower())}

    left, right = words(first), words(second)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _find_gap_sentences(text: str, limit: int = 2) -> list[str]:
    """Find the sentences that state what prior work has not solved."""
    sentences = [_clean_sentence(sentence) for sentence in split_sentences(text)]
    sentences = [sentence for sentence in sentences if _informative(sentence, "Research Gap")]

    strong_scored: list[tuple[float, int, str]] = []
    medium_scored: list[tuple[float, int, str]] = []
    weak_scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        strong = sum(1 for cue in GAP_STRONG_CUES if cue in lower)
        medium = sum(1 for cue in GAP_MEDIUM_CUES if cue in lower)
        weak = sum(1 for cue in GAP_WEAK_CUES if cue in lower)
        if strong == 0 and medium == 0 and weak == 0:
            continue
        if len(sentence.split()) < 7:
            continue

        score = 4.0 * strong + 2.0 * medium + 0.7 * weak
        if "future work" in lower or "future research" in lower:
            score += 1.2
        if re.search(r"\bwe (address|fill|bridge|close|tackle|study|examine)\b", lower):
            score += 0.8
        if re.search(r"\b(this|our) (paper|study|work)\b", lower):
            score += 0.3
        item = (score, index, sentence)
        if strong > 0:
            strong_scored.append(item)
        elif medium > 0:
            medium_scored.append(item)
        else:
            weak_scored.append(item)

    # Prefer an explicit gap or scarcity statement over a generic "future
    # work" sentence. A novelty claim ("first model to...") is the next best
    # evidence; limitations are the last resort.
    scored = strong_scored + medium_scored

    if not scored:
        contrast: list[tuple[float, int, str]] = []
        cutoff = max(1, int(len(sentences) * 0.6))
        for index, sentence in enumerate(sentences[:cutoff]):
            lower = sentence.lower()
            contrast_hits = sum(1 for cue in GAP_CONTRAST_CUES if cue in lower)
            novelty_hits = sum(1 for cue in GAP_NOVELTY_CUES if cue in lower)
            if contrast_hits + novelty_hits == 0 or not (8 <= len(sentence.split()) <= 45):
                continue
            contrast.append((2.0 * novelty_hits + contrast_hits, index, sentence))
        if contrast:
            scored = [max(contrast, key=lambda item: item[0])]

    if not scored:
        scored = weak_scored

    if not scored:
        return []

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[str] = []
    for _, _, sentence in scored:
        if len(chosen) >= limit:
            break
        if any(_token_overlap(sentence, other) > 0.6 for other in chosen):
            continue
        chosen.append(sentence)
    return chosen


def _insert_gap_section(sections: list[dict], gap_sentences: list[str]) -> None:
    titles = [section["title"] for section in sections]
    if "Literature Review" in titles:
        insert_at = titles.index("Literature Review") + 1
    elif "Introduction" in titles:
        insert_at = titles.index("Introduction") + 1
    else:
        insert_at = 0
        for index, title in enumerate(titles):
            if title in ("Author", "Institution"):
                insert_at = index + 1
    text = " ".join(gap_sentences)
    sections.insert(
        insert_at,
        {
            "title": "Research Gap",
            "summary": text,
            "keywords": top_terms(text, 6),
            "selected_sentences": len(gap_sentences),
            "total_sentences": len(gap_sentences),
        },
    )


# Typical position bands for each part of a paper. Used only when headings are
# unavailable, so a paper whose headings were lost during PDF extraction still
# reads as a structured summary instead of one giant block.
_INFERENCE_SPECS_LONG = (
    ("Introduction", 0.20),
    ("Literature Review", 0.36),
    ("Methodology", 0.62),
    ("Results and Discussion", 0.86),
    ("Conclusion", 1.0),
)
_INFERENCE_SPECS_SHORT = (
    ("Introduction", 0.26),
    ("Methodology", 0.60),
    ("Results and Discussion", 0.85),
    ("Conclusion", 1.0),
)


def _cue_hits(sentence: str, title: str) -> int:
    lower = sentence.lower()
    return sum(1 for cue in SECTION_CUES.get(title, ()) if cue in lower)


def _paragraph_starts(text: str) -> set[int]:
    """Return sentence indices that begin a paragraph."""
    starts: set[int] = set()
    cursor = 0
    for paragraph in re.split(r"\n\s*\n", text):
        count = len(split_sentences(paragraph))
        if count:
            starts.add(cursor)
            cursor += count
    return starts


def _infer_sections(text: str) -> list[tuple[str, list[str]]] | None:
    """Infer Introduction/Methodology/Results/Conclusion when headings are absent."""
    sentences = split_sentences(text)
    if len(sentences) < 8:
        return None
    paragraph_starts = _paragraph_starts(text)

    specs = _INFERENCE_SPECS_LONG if len(sentences) >= 22 else _INFERENCE_SPECS_SHORT
    total = len(sentences)
    min_size = 2
    if total < len(specs) * min_size:
        return None

    boundaries = [0]
    for index in range(1, len(specs)):
        title = specs[index][0]
        remaining_sections = len(specs) - index
        latest = total - remaining_sections * min_size
        nominal = min(max(1, int(round(specs[index - 1][1] * total))), latest)
        window = max(2, int(round(total * 0.05)))
        low = max(boundaries[-1] + min_size, nominal - window)
        high = min(latest, nominal + window)
        if high < low:
            low = high = latest
        best_index, best_score = nominal, float("-inf")
        previous_title = specs[index - 1][0]
        for position in range(low, high + 1):
            # The boundary sentence opens the next section, so prefer one that
            # uses that section's vocabulary. Paragraph starts are stronger
            # evidence than a single ambiguous cue word ("approach" often
            # appears at the end of a literature review); break ties by
            # distance from the typical position.
            score = _cue_hits(sentences[position], title)
            score -= 0.35 * min(2, _cue_hits(sentences[position], previous_title))
            if position in paragraph_starts:
                score += 1.5
            score -= abs(position - nominal) / max(1, window) * 0.5
            if score > best_score:
                best_index, best_score = position, score
        boundaries.append(min(max(best_index, boundaries[-1] + min_size), latest))
    boundaries.append(total)

    inferred = []
    for (title, _), start, end in zip(specs, boundaries, boundaries[1:]):
        if end > start:
            inferred.append((title, [" ".join(sentences[start:end])]))
    return inferred


def _has_core_sections(segments: list[tuple[str, list[str]]]) -> bool:
    titles = {title for title, _ in segments}
    return all(title in titles for title in CORE_SECTIONS)


def summarize_text(text: str, sentence_count: int = 10) -> dict:
    """Create an extractive, section-aware summary of a paper."""
    sentence_count = max(1, min(sentence_count, 24))
    text = _repair_split_hyphenation(text)
    original_text = text
    author, institution, body = split_front_matter(text)
    if author or institution:
        prefix = []
        if author:
            prefix.append(f"Author\n{author}")
        if institution:
            prefix.append(f"Institution\n{institution}")
        text = "\n\n".join(prefix + [body]) if prefix else text

    segments = segment_document(text)
    structure_inferred = False
    if not _has_core_sections(segments):
        inferred = _infer_sections(body or original_text)
        if inferred:
            structure_inferred = True
            front_matter = []
            if author:
                front_matter.append(("Author", [author]))
            if institution:
                front_matter.append(("Institution", [institution]))
            segments = front_matter + inferred
    if not segments:
        raise ValueError("No readable sentences were found in the document.")

    # Add a Research Gap segment even when the paper has no such heading, by
    # locating the sentences that describe what prior work leaves unresolved.
    declared_gap = any(title == "Research Gap" for title, _ in segments)
    gap_sentences: list[str] = []
    if not declared_gap:
        gap_sentences = _find_gap_sentences(body or original_text, limit=2 if sentence_count >= 8 else 1)
        if gap_sentences:
            sentence_count = max(1, sentence_count - len(gap_sentences))
    gap_keys = {_match_key(sentence) for sentence in gap_sentences}

    segment_sentence_counts = [max(1, len(split_sentences("\n\n".join(paragraphs)))) for _, paragraphs in segments]
    if len(segments) == 1:
        allocations = [sentence_count]
    else:
        # Every detected section gets at least one sentence. Extra depth goes to
        # the sections readers rely on most, rather than only the longest block.
        allocations = [1 for _ in segments]
        remaining = sentence_count - len(segments)
        priority = [
            "Results and Discussion",
            "Methodology",
            "Introduction",
            "Conclusion",
            "Literature Review",
            "Author",
            "Institution",
        ]
        title_to_index = {title: index for index, (title, _) in enumerate(segments)}
        priority_indices = [title_to_index[title] for title in priority if title in title_to_index]
        priority_indices += [index for index in range(len(segments)) if index not in priority_indices]
        cursor = 0
        while remaining > 0 and priority_indices:
            progressed = False
            for _ in range(len(priority_indices)):
                index = priority_indices[cursor % len(priority_indices)]
                cursor += 1
                if allocations[index] < segment_sentence_counts[index]:
                    allocations[index] += 1
                    remaining -= 1
                    progressed = True
                    break
            if not progressed:
                break

    sections = []
    for (title, paragraphs), allocation in zip(segments, allocations):
        block_text = "\n\n".join(paragraphs)
        block_result = _summarize_block(block_text, allocation, title, exclude=gap_keys)
        if not block_result["summary"]:
            continue
        sections.append(
            {
                "title": title,
                "summary": block_result["summary"],
                "keywords": top_terms(block_text, 6),
                "selected_sentences": block_result["selected_sentences"],
                "total_sentences": block_result["total_sentences"],
            }
        )

    if gap_sentences and not any(section["title"] == "Research Gap" for section in sections):
        _insert_gap_section(sections, gap_sentences)

    if not sections:
        raise ValueError("No readable sentences were found in the document.")

    summary = "\n\n".join(f"{section['title']}: {section['summary']}" for section in sections)
    selected_total = sum(section["selected_sentences"] for section in sections)
    sentence_total = sum(section["total_sentences"] for section in sections)

    return {
        "summary": summary,
        "sections": sections,
        "keywords": top_terms(text),
        "selected_sentences": selected_total,
        "total_sentences": sentence_total,
        "compression_ratio": round(len(summary) / max(1, len(text)), 3),
        "structure_inferred": structure_inferred,
    }


QUERY_EXPANSIONS = {
    "method": ("methodology", "approach", "procedure"),
    "methodology": ("method", "approach"),
    "finding": ("results", "result"),
    "findings": ("results", "result"),
    "result": ("findings", "discussion"),
    "limitation": ("discussion", "future work"),
    "limitations": ("discussion", "future work"),
    "contribution": ("introduction", "objective", "goal"),
    "author": ("authors", "affiliation"),
    "institution": ("university", "department", "affiliation"),
}


def _expand_question(question: str) -> str:
    words = set(re.findall(r"[a-z]+", question.lower()))
    additions = [synonym for word, synonyms in QUERY_EXPANSIONS.items() if word in words for synonym in synonyms]
    return question if not additions else f"{question} {' '.join(additions)}"


def answer_question(text: str, question: str) -> dict:
    """Answer a question by retrieving the most relevant sentence."""
    sentences = split_sentences(text)
    question = question.strip()
    if not sentences:
        raise ValueError("No readable sentences were found in the document.")
    if not question:
        raise ValueError("A question is required.")

    expanded_question = _expand_question(question)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(sentences + [expanded_question])
    similarities = cosine_similarity(matrix[-1], matrix[:-1])[0]

    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])

    if best_score < 0.05:
        return {
            "answer": "I could not find a strong answer in the document.",
            "confidence": best_score,
            "source_sentence": None,
        }

    source = sentences[best_index]
    return {
        "answer": source,
        "confidence": round(best_score, 4),
        "source_sentence": source,
    }
