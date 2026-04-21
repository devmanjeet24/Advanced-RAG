from app.services.retrievers.bm25 import BM25Retriever   # ✅
from app.services.retrievers.rrf import rrf_multi   


class HybridRetriever:
    """
    Combines semantic (vector) search with BM25 keyword search
    using Reciprocal Rank Fusion.
    """

    def __init__(self, vectorstore, docs: list[str]):
        self.vs = vectorstore
        self.bm25 = BM25Retriever(docs)

    def search(self, query: str, k: int = 10, top_n: int = 5) -> list[str]:
        # Semantic results
        sem_docs = self.vs.similarity_search(query, k=k)
        sem_texts = [d.page_content for d in sem_docs]

        # BM25 results
        kw_texts = self.bm25.search(query, n=k)

        # RRF fusion
        fused = rrf_multi([sem_texts, kw_texts])

        return fused[:top_n]

    def search_multi_query(self, queries: list[str], k: int = 5, top_n: int = 5) -> list[str]:
        """Run hybrid search for multiple query variants, then fuse with RRF."""
        all_ranked: list[list[str]] = []

        for q in queries:
            sem_docs = self.vs.similarity_search(q, k=k)
            sem_texts = [d.page_content for d in sem_docs]
            kw_texts = self.bm25.search(q, n=k)
            fused = rrf_multi([sem_texts, kw_texts])
            all_ranked.append(fused)

        final = rrf_multi(all_ranked)
        return final[:top_n]