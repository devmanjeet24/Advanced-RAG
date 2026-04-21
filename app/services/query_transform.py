"""
Query Transformation Strategies
--------------------------------
1. HyDE          - Hypothetical Document Embedding
2. Multi-Query   - Generate N paraphrased queries
3. Decompose     - Break complex query into sub-questions
4. Step-Back     - Generate a higher-level / abstract query
"""

from app.services.llm import generate_answer


# ─────────────────────────────────────────────
# 1. HyDE
# ─────────────────────────────────────────────
def hyde(query: str) -> str:
    prompt = (
        f"Write a detailed, factual passage that directly answers the following question. "
        f"Do not include the question itself, just write the answer as a document.\n\n"
        f"Question: {query}"
    )
    return generate_answer(prompt, max_tokens=256)


# ─────────────────────────────────────────────
# 2. Multi-Query
# ─────────────────────────────────────────────
def multi_query(query: str, n: int = 3) -> list[str]:
    prompt = (
        f"Generate {n} different phrasings/variations of the following question to help "
        f"retrieve relevant documents. Return ONLY the questions, one per line, no numbering.\n\n"
        f"Original question: {query}"
    )
    raw = generate_answer(prompt, max_tokens=200)
    queries = [q.strip() for q in raw.strip().split("\n") if q.strip()]
    # Always include the original
    if query not in queries:
        queries.insert(0, query)
    return queries[:n + 1]


# ─────────────────────────────────────────────
# 3. Decompose
# ─────────────────────────────────────────────
def decompose(query: str) -> list[str]:
    prompt = (
        f"Break the following complex question into 2-4 simpler sub-questions that, "
        f"when answered together, fully address the original question. "
        f"Return ONLY the sub-questions, one per line, no numbering or bullet points.\n\n"
        f"Question: {query}"
    )
    raw = generate_answer(prompt, max_tokens=200)
    subs = [q.strip() for q in raw.strip().split("\n") if q.strip()]
    return subs if subs else [query]


# ─────────────────────────────────────────────
# 4. Step-Back
# ─────────────────────────────────────────────
def step_back(query: str) -> str:
    prompt = (
        f"Given the specific question below, generate a more general, higher-level question "
        f"whose answer would help provide the background knowledge needed to answer the original.\n\n"
        f"Original question: {query}\n\n"
        f"Return ONLY the step-back question, nothing else."
    )
    return generate_answer(prompt, max_tokens=100)