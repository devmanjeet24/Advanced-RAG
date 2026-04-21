def rrf(a: list[str], b: list[str], k: int = 60) -> list[str]:
    """
    Reciprocal Rank Fusion of two ranked lists.
    Higher score = better rank.
    """
    scores: dict[str, float] = {}

    for i, doc in enumerate(a):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + i + 1)

    for i, doc in enumerate(b):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + i + 1)

    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


def rrf_multi(lists: list[list[str]], k: int = 60) -> list[str]:
    """RRF over any number of ranked lists."""
    scores: dict[str, float] = {}

    for ranked in lists:
        for i, doc in enumerate(ranked):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + i + 1)

    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)