
from langchain.prompts import PromptTemplate

QA_SYSTEM_PROMPT = '''Tu es un assistant pédagogique francophone.
Tu réponds de manière **claire, concise et sourcée** à partir du CONTEXTE.
- Si l'information n'est pas dans le contexte, dis que tu ne la trouves pas.
- Cite les numéros ou titres de sections si disponibles.
- Donne des listes lorsqu'elles aident la clarté.
- Réfère-toi au Sénégal lorsque les règles sont spécifiques.
'''

QA_PROMPT = PromptTemplate(
    input_variables=[
        "question",
        "context"
    ],
    template=(
        "{system_prompt}\n\n"
        "QUESTION:\n{question}\n\n"
        "CONTEXTE (extraits pertinents):\n{context}\n\n"
        "RÉPONSE (en français, sourcée et factuelle):"
    )
)

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context"],
    template=(
        "Tu es un assistant qui produit un **résumé synthétique et structuré** "
        "des règles du Code de la route au Sénégal à partir du CONTEXTE fourni.\n\n"
        "CONTEXTE:\n{context}\n\n"
        "Donne un résumé en 5 à 10 puces maximum, en français, avec les points clés."
    )
)
