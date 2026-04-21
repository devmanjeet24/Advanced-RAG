from PyPDF2 import PdfReader
import docx

def load(path):
    if path.endswith(".pdf"):
        return " ".join([p.extract_text() for p in PdfReader(path).pages])

    if path.endswith(".docx"):
        return "\n".join([p.text for p in docx.Document(path).paragraphs])

    return open(path).read()