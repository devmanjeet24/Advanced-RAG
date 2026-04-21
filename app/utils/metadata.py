def filter_docs(docs, filters):
    if not filters:
        return docs

    result = []
    for d in docs:
        ok = True
        for k, v in filters.items():
            if d.metadata.get(k) != v:
                ok = False
        if ok:
            result.append(d)

    return result