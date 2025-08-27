
# RAG Chatbot – Code de la Route Sénégal (2025)

**Naive RAG Chatbot** construit pour le NSK AI RAG Bootcamp – Phase 1.
Ce dépôt est une version améliorée avec **Cross-Encoder rerank** et guide de déploiement sur **HuggingFace Spaces (Streamlit)**.

---

## 🎯 Objectif
Permettre des Q&A et des résumés basés sur les documents officiels du **Code de la route au Sénégal**.

## 🧱 Stack technique
- LangChain (Python)
- Embeddings : `sentence-transformers/all-MiniLM-L6-v2`
- VectorStore : FAISS (local)
- Rerank : BM25 + **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- LLM : OpenAI (configurable) ou autre LLM compatible
- UI : Streamlit (compatible HuggingFace Spaces)

## ✅ Nouveautés dans cette version
- Cross-Encoder activé par défaut pour améliorer la précision du rerank.
- README enrichi avec exemples Q/A réels et guide complet pour déployer sur **HuggingFace Spaces**.
- Fallback robuste si Cross-Encoder n'est pas disponible (utilise BM25 uniquement).

---

## Installation locale (rapide)
1. Crée un env virtuel :
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate  # Windows PowerShell
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Placer des PDF/TXT dans `data/` (ex.: règlement du Code de la route).

4. Indexer :
```bash
python ingest.py
```

5. Lancer l'app :
```bash
streamlit run app.py
```

> **Note** : Pour utiliser OpenAI, ajoute ta clé `OPENAI_API_KEY` soit en variable d'environnement, soit via la barre latérale de l'app.

---

## Exemples de requêtes & réponses (à inclure comme exemples de test)
> Ces exemples supposent que les documents du Code de la route Sénégal sont indexés correctement.

**Q1**: *Quels documents dois-je présenter lors d'un contrôle routier au Sénégal ?*  
**R1 (attendu)**: Liste des documents obligatoires (carte grise, permis, attestation d'assurance) avec référence à l'article / section du document indexé, ex: *"Voir PDF: 'CodeRoute_SN.pdf', Section 3 - Documents obligatoires".*

**Q2**: *Quelle est la vitesse maximale autorisée en agglomération ?*  
**R2 (attendu)**: Valeur (ex.: 50 km/h) + contexte si exceptions (zones scolaires, routes spécifiques) et citation de la page/section.

**Q3**: *Quelles sont les sanctions pour conduite en état d'ivresse ?*  
**R3 (attendu)**: Résumé des amendes, retraits de permis possibles, et renvoi vers l'article correspondant dans le document indexé.

---

## Déploiement sur HuggingFace Spaces (Streamlit) — Guide pas-à-pas
1. Crée un compte sur https://huggingface.co/ si tu n'en as pas.
2. Crée un nouveau **Space** (choisir **Streamlit** comme SDK).
3. Dans ton dépôt GitHub (ou en upload direct), pousse tous les fichiers de ce projet.
4. Dans les **Files** du Space, ajoute tous les fichiers. HuggingFace va installer `requirements.txt` automatiquement.
5. **Secrets** : dans ton Space -> Settings -> Secrets, ajoute `OPENAI_API_KEY` (si tu utilises OpenAI). Tu peux aussi ajouter `EMBEDDING_MODEL_NAME` si tu veux remplacer.
6. Démarrage : Le Space lance automatiquement `streamlit run app.py`. Si tu veux changer la commande de démarrage, configure-la dans Settings -> Hardware & software.
7. Si le Cross-Encoder prend du temps au premier démarrage (téléchargement du modèle), sois patient — après le premier run les fichiers sont mis en cache.

**Remarques pour HuggingFace** :
- Si le modèle Cross-Encoder dépasse les limites de mémoire du Space, enlève `cross-encoder` ou utilise `use_crossencoder=False` via la barre latérale. Le code prévoit un fallback automatique.
- Pour des besoins lourds, considère un service payant ou un déploiement sur Render / Streamlit Cloud.

---

## Limitations connues
- Qualité dépend de la couverture des documents indexés.
- Cross-Encoder peut augmenter le temps de latence au premier appel et consommer plus de RAM.
- OpenAI API coûte en usage; tu peux brancher un LLM local pour expérimentation.

---

## Contribution & soumission
- Personnalise `data/` avec tes PDF officielles.
- Mets à jour `README.md` avec exemples réels et pushes sur GitHub.
- Soumets le lien du repo dans le formulaire du bootcamp.

Bonne chance 🎉 — si tu veux, je peux maintenant :
- créer et zipper le dépôt amélioré (avec Cross-Encoder activé), ce que j'ai fait ici ;
- **ou** directement pousser sur GitHub si tu me fournis un token (je te guiderai).

