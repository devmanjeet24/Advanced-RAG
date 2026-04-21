from langchain_text_splitters import RecursiveCharacterTextSplitter

def parent_child(text, metadata):
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    parents = parent_splitter.split_text(text)

    chunks = []

    for i, p in enumerate(parents):
        children = child_splitter.split_text(p)
        for c in children:
            chunks.append({
                "text": c,
                "metadata": {
                    **metadata,
                    "parent_id": i
                }
            })

    return chunks