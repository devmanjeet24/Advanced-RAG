"""
Chunking Strategies
--------------------
1. recursive      - Standard recursive character splitting
2. parent_child   - Parent (large) + child (small) chunks  [in parent_child.py]
3. semantic       - Split on meaning boundaries using embeddings
4. section_aware  - Split on markdown/document headings
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────
# 1. Recursive (basic)
# ─────────────────────────────────────────────
def recursive(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    ).split_text(text)


# ─────────────────────────────────────────────
# 2. Semantic Chunking (similarity-based)
# ─────────────────────────────────────────────
def semantic_chunk(text: str, threshold: float = 0.75) -> list[str]:
    """
    Split text into chunks based on semantic similarity between sentences.
    Sentences that are semantically far apart trigger a new chunk.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 2:
            return sentences

        embeddings = model.encode(sentences, show_progress_bar=False)

        chunks = []
        current = [sentences[0]]

        for i in range(1, len(sentences)):
            # Cosine similarity between consecutive sentences
            a = embeddings[i - 1]
            b = embeddings[i]
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

            if sim < threshold:
                # Semantic boundary detected → new chunk
                chunks.append(" ".join(current))
                current = [sentences[i]]
            else:
                current.append(sentences[i])

        if current:
            chunks.append(" ".join(current))

        return chunks

    except Exception:
        # Fallback to recursive if sentence-transformers fails
        return recursive(text)


# ─────────────────────────────────────────────
# 3. Section-Aware Chunking
# ─────────────────────────────────────────────
def section_aware_chunk(text: str, chunk_size: int = 800, chunk_overlap: int = 80) -> list[dict]:
    """
    Split text by markdown headings / section titles.
    Returns list of {"text": ..., "section": ...} dicts.
    """
    # Pattern: lines starting with # or ALL CAPS titles
    heading_pattern = re.compile(
        r'^(#{1,6}\s+.+|[A-Z][A-Z\s]{4,}:?)$',
        re.MULTILINE
    )

    sections = []
    last_end = 0
    current_heading = "Introduction"

    for match in heading_pattern.finditer(text):
        section_text = text[last_end:match.start()].strip()
        if section_text:
            sections.append({"heading": current_heading, "text": section_text})
        current_heading = match.group().strip().lstrip("#").strip()
        last_end = match.end()

    # Last section
    remaining = text[last_end:].strip()
    if remaining:
        sections.append({"heading": current_heading, "text": remaining})

    # If no sections found, fallback
    if not sections:
        return [{"text": c, "section": "General"} for c in recursive(text, chunk_size, chunk_overlap)]

    # Further split large sections
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    result = []
    for sec in sections:
        sub_chunks = splitter.split_text(sec["text"])
        for chunk in sub_chunks:
            result.append({"text": chunk, "section": sec["heading"]})

    return result