
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import DATA_DIR, FAISS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME

def load_documents(data_dir: str) -> List:
    docs = []
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dossier introuvable: {data_dir}")
    # PDF
    for pdf in data_path.rglob("*.pdf"):
        docs.extend(PyPDFLoader(str(pdf)).load())
    # TXT
    for txt in data_path.rglob("*.txt"):
        docs.extend(TextLoader(str(txt), encoding="utf-8").load())
    return docs

def chunk_documents(docs: List):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

def main():
    print("[1/3] Chargement des documents depuis:", DATA_DIR)
    docs = load_documents(DATA_DIR)
    print(f"  -> {len(docs)} documents bruts")

    print("[2/3] Découpage en chunks (size={}, overlap={})".format(CHUNK_SIZE, CHUNK_OVERLAP))
    chunks = chunk_documents(docs)
    print(f"  -> {len(chunks)} chunks")

    print("[3/3] Embeddings & FAISS ({})".format(EMBEDDING_MODEL_NAME))
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vs = FAISS.from_documents(chunks, embedder)

    os.makedirs(FAISS_DIR, exist_ok=True)
    vs.save_local(FAISS_DIR)
    print(f"Index FAISS sauvegardé dans: {FAISS_DIR}")

if __name__ == "__main__":
    main()
