from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, docs: list[str]):
        self.docs = docs
        self.bm = BM25Okapi([d.split() for d in docs])

    def search(self, query: str, n: int = 10) -> list[str]:
        return self.bm.get_top_n(query.split(), self.docs, n=n)

    def search_with_scores(self, query: str, n: int = 10) -> list[tuple[str, float]]:
        tokenized = query.split()
        scores = self.bm.get_scores(tokenized)
        doc_score_pairs = sorted(
            zip(self.docs, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return doc_score_pairs[:n]