
import os
import streamlit as st
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import APP_TITLE, FAISS_DIR, OPENAI_MODEL, OPENAI_API_KEY, TOP_K, EMBEDDING_MODEL_NAME
from prompts import QA_PROMPT, QA_SYSTEM_PROMPT, SUMMARY_PROMPT
from utils.rerank import hybrid_indices

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Sidebar
st.sidebar.header("Paramètres")
top_k = st.sidebar.slider("Top K (chunks)", min_value=2, max_value=10, value=TOP_K)
use_rerank = st.sidebar.checkbox("Rerank (BM25 / Cross-Encoder)", value=True)
use_crossencoder = st.sidebar.checkbox("Use Cross-Encoder when available", value=True)
temperature = st.sidebar.slider("Température LLM", 0.0, 1.0, 0.2, 0.1)
model_name = st.sidebar.text_input("OpenAI model", value=OPENAI_MODEL)
# HF Spaces: prefer secrets; fallback to env
api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY or "")
api_key_input = st.sidebar.text_input("OPENAI_API_KEY (optionnel, utilisé si présent)", value="", type="password")
if api_key_input.strip():
    api_key = api_key_input.strip()

mode = st.sidebar.radio("Mode", ["Q&A", "Résumé"], index=0)

# Load VectorStore
@st.cache_resource(show_spinner=True)
def load_vs():
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vs = FAISS.load_local(FAISS_DIR, embedder, allow_dangerous_deserialization=True)
    return vs

try:
    vs = load_vs()
except Exception as e:
    st.error("Impossible de charger l'index FAISS. Avez-vous lancé `python ingest.py` ?" )
    st.exception(e)
    st.stop()

# LLM
def get_llm():
    if not api_key:
        st.warning("Aucune clé OpenAI fournie. Renseignez OPENAI_API_KEY via Secrets (HF) ou la barre latérale.")
    os.environ["OPENAI_API_KEY"] = api_key or os.getenv("OPENAI_API_KEY", "")
    return ChatOpenAI(model=model_name, temperature=temperature)

llm = get_llm()
parser = StrOutputParser()

def format_context(docs: List[Document]) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        source = meta.get("source", "(source inconnue)")
        page = meta.get("page", None)
        tag = f"p.{page}" if page is not None else ""
        out.append(f"[Extrait {i}] {source} {tag}\n{d.page_content}".strip())
    return "\n\n".join(out)

def retrieve(query: str, k: int) -> List[Document]:
    docs = vs.similarity_search(query, k=k*3 if use_rerank else k)
    if use_rerank and docs:
        ranked = hybrid_indices(query, [d.page_content for d in docs], top_k=k, use_crossencoder=use_crossencoder)
        docs = [docs[i] for i in ranked]
    return docs[:k]

st.write("""
**Conseil** : commence par indexer tes documents (menu README) et pose ensuite ta question.
""")

with st.expander("ℹ️ Aide / README rapide", expanded=False):
    st.markdown("""
1. Place tes PDF/TXT dans `data/`.
2. Lance `python ingest.py` pour (ré)indexer.
3. Renseigne ta clé `OPENAI_API_KEY` via Secrets (HuggingFace) ou la barre latérale.
4. Pose ta question en Q&A, ou utilise le mode Résumé.
    """)

if mode == "Q&A":
    q = st.text_input("Ta question sur le Code de la route :", value="Quelle est la vitesse maximale autorisée en agglomération ?")
    if st.button("Répondre", use_container_width=True) and q.strip():
        with st.spinner("Récupération du contexte..."):
            ctx_docs = retrieve(q, top_k)
            ctx_text = format_context(ctx_docs)

        with st.spinner("Génération de la réponse..."):
            prompt = QA_PROMPT.format(system_prompt=QA_SYSTEM_PROMPT, question=q, context=ctx_text)
            chain = ChatPromptTemplate.from_template("{prompt}") | llm | parser
            answer = chain.invoke({"prompt": prompt})

        st.subheader("Réponse")
        st.markdown(answer)
        st.subheader("Contexte utilisé") 
        st.code(ctx_text)

else:  # Résumé
    if st.button("Générer un résumé des documents", use_container_width=True):
        with st.spinner("Récupération de larges extraits..."):
            queries = [
                "signalisation routière Sénégal",
                "priorités et dépassement Sénégal",
                "vitesse maximale Sénégal",
                "documents obligatoires contrôle Sénégal",
                "sanctions et infractions Sénégal",
            ]
            docs_all = []
            for qq in queries:
                docs_all.extend(retrieve(qq, max(2, top_k//2)))
            seen = set()
            unique_docs = []
            for d in docs_all:
                key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:80])
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(d)

            ctx_text = format_context(unique_docs[: min(12, len(unique_docs))])

        with st.spinner("Synthèse en cours..."):
            prompt = SUMMARY_PROMPT.format(context=ctx_text)
            chain = ChatPromptTemplate.from_template("{prompt}") | llm | parser
            summary = chain.invoke({"prompt": prompt})

        st.subheader("Résumé")
        st.markdown(summary)
        st.subheader("Contexte utilisé") 
        st.code(ctx_text)
