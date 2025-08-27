
"""
Script de bootstrap pour HuggingFace Spaces
- Charge les documents (PDF/TXT dans data/)
- Crée l'index FAISS s'il n'existe pas encore
- Peut être lancé manuellement ou via le build script du Space
"""

from pathlib import Path
from ingest import main as ingest_main
from config import FAISS_DIR

if __name__ == "__main__":
    if Path(FAISS_DIR).exists() and any(Path(FAISS_DIR).iterdir()):
        print(f"Index FAISS déjà présent dans: {FAISS_DIR} — aucune action nécessaire.")
    else:
        print("Aucun index trouvé. Lancement de l'ingestion initiale...")
        ingest_main()
        print("Index créé avec succès !")
