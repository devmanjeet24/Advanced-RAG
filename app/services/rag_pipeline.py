"""
Advanced RAG Pipeline
----------------------
Supports:
- HyDE / Multi-Query / Decompose / Step-Back query strategies
- Hybrid retrieval (BM25 + Vector + RRF)
- Metadata filtering
- Cross-encoder reranking
- Groq LLM
"""

from app.services.vectorstore import get_vectorstore
from app.services.retrievers.bm25 import BM25Retriever    
from app.services.retrievers.hybrid import HybridRetriever 
from app.services.retrievers.rrf import rrf_multi   
from app.services.query_transform import hyde, multi_query, decompose, step_back
from app.services.reranker import rerank
from app.services.llm import generate_answer
from app.db.mongodb import db


# ─────────────────────────────────────────────
# Helper: fetch all docs from Mongo for BM25
# ─────────────────────────────────────────────
async def _get_bm25_docs(filters: dict = None) -> list[str]:
    query = {}
    if filters:
        if "user_id" in filters:
            query["metadata.user_id"] = filters["user_id"]
        if "file_id" in filters:
            query["metadata.file_id"] = filters["file_id"]

    cursor = db.chunks.find(query, {"text": 1, "_id": 0})
    chunks = await cursor.to_list(length=2000)
    return [c["text"] for c in chunks]


# ─────────────────────────────────────────────
# Helper: filter Chroma results by metadata
# ─────────────────────────────────────────────
def _chroma_where(filters: dict) -> dict | None:
    if not filters:
        return None
    clauses = []
    if "user_id" in filters:
        clauses.append({"user_id": {"$eq": filters["user_id"]}})
    if "file_id" in filters:
        clauses.append({"file_id": {"$eq": filters["file_id"]}})
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) > 1:
        return {"$and": clauses}
    return None


# ─────────────────────────────────────────────
# Core RAG pipeline
# ─────────────────────────────────────────────
async def rag_pipeline(
    query: str,
    filters: dict = None,
    strategy: str = "hybrid_multi",   # see strategies below
    top_k: int = 5,
) -> dict:
    """
    strategy options:
      - "basic"          : simple vector search, no query transform
      - "hyde"           : HyDE + vector search
      - "multi_query"    : multi-query + hybrid + RRF
      - "decompose"      : decompose → answer sub-questions → synthesize
      - "step_back"      : step-back + original + hybrid
      - "hybrid_multi"   : (default) multi-query + step-back + hybrid + RRF
    """
    try:
        vs = get_vectorstore()
        where = _chroma_where(filters or {})

        # ── Retrieve docs for BM25 ──
        bm25_docs = await _get_bm25_docs(filters)
        use_hybrid = len(bm25_docs) > 0

        if use_hybrid:
            hybrid = HybridRetriever(vs, bm25_docs)

        # ══════════════════════════════════════════
        # Strategy: basic
        # ══════════════════════════════════════════
        if strategy == "basic":
            kwargs = {"k": top_k}
            if where:
                kwargs["filter"] = where
            docs = vs.similarity_search(query, **kwargs)
            texts = [d.page_content for d in docs]

        # ══════════════════════════════════════════
        # Strategy: hyde
        # ══════════════════════════════════════════
        elif strategy == "hyde":
            hyp = hyde(query)
            kwargs = {"k": top_k}
            if where:
                kwargs["filter"] = where
            docs = vs.similarity_search(hyp, **kwargs)
            texts = [d.page_content for d in docs]

        # ══════════════════════════════════════════
        # Strategy: multi_query
        # ══════════════════════════════════════════
        elif strategy == "multi_query":
            queries = multi_query(query, n=3)
            if use_hybrid:
                texts = hybrid.search_multi_query(queries, k=top_k, top_n=top_k)
            else:
                all_ranked = []
                for q in queries:
                    kwargs = {"k": top_k}
                    if where:
                        kwargs["filter"] = where
                    docs = vs.similarity_search(q, **kwargs)
                    all_ranked.append([d.page_content for d in docs])
                texts = rrf_multi(all_ranked)[:top_k]

        # ══════════════════════════════════════════
        # Strategy: decompose
        # ══════════════════════════════════════════
        elif strategy == "decompose":
            sub_qs = decompose(query)
            sub_answers = []
            for sq in sub_qs:
                kwargs = {"k": 3}
                if where:
                    kwargs["filter"] = where
                docs = vs.similarity_search(sq, **kwargs)
                ctx = "\n".join([d.page_content for d in docs])
                ans = generate_answer(
                    f"Context:\n{ctx}\n\nQuestion: {sq}",
                    system="Answer briefly based only on the context."
                )
                sub_answers.append(f"Q: {sq}\nA: {ans}")

            synthesis_prompt = (
                f"Using the following sub-answers, provide a comprehensive answer to: {query}\n\n"
                + "\n\n".join(sub_answers)
            )
            final_answer = generate_answer(synthesis_prompt)
            return {
                "answer": final_answer,
                "strategy": "decompose",
                "sub_questions": sub_qs,
                "sources": []
            }

        # ══════════════════════════════════════════
        # Strategy: step_back
        # ══════════════════════════════════════════
        elif strategy == "step_back":
            sb_query = step_back(query)
            queries = [query, sb_query]
            if use_hybrid:
                texts = hybrid.search_multi_query(queries, k=top_k, top_n=top_k)
            else:
                all_ranked = []
                for q in queries:
                    kwargs = {"k": top_k}
                    if where:
                        kwargs["filter"] = where
                    docs = vs.similarity_search(q, **kwargs)
                    all_ranked.append([d.page_content for d in docs])
                texts = rrf_multi(all_ranked)[:top_k]

        # ══════════════════════════════════════════
        # Strategy: hybrid_multi (DEFAULT — best quality)
        # ══════════════════════════════════════════
        else:  # hybrid_multi
            queries = multi_query(query, n=3)
            sb_query = step_back(query)
            all_queries = queries + [sb_query]

            if use_hybrid:
                texts = hybrid.search_multi_query(all_queries, k=top_k, top_n=top_k + 3)
            else:
                all_ranked = []
                for q in all_queries:
                    kwargs = {"k": top_k}
                    if where:
                        kwargs["filter"] = where
                    docs = vs.similarity_search(q, **kwargs)
                    all_ranked.append([d.page_content for d in docs])
                texts = rrf_multi(all_ranked)[:top_k + 3]

        # ── Fallback ──
        if not texts:
            return {"answer": "No relevant documents found.", "sources": [], "strategy": strategy}

        # ── Rerank ──
        ranked = rerank(query, texts)
        top_context = ranked[:3]

        # ── Build prompt ──
        context = "\n\n---\n\n".join(top_context)
        final_prompt = (
            f"Answer the following question using ONLY the context provided below. "
            f"If the context doesn't contain the answer, say 'I don't have enough information.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )

        answer = generate_answer(final_prompt)

        return {
            "answer": answer,
            "strategy": strategy,
            "sources": top_context,
            "total_retrieved": len(texts),
        }

    except Exception as e:
        return {"error": str(e), "strategy": strategy}