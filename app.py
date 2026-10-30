import json
import random
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

st.set_page_config(page_title="Mon espace d'apprentissage", page_icon="🚀")

# =====================================================================
# CONFIGURATION (tout vient des "Secrets" de Streamlit)
# =====================================================================
S = st.secrets
MODEL = S.get("MODEL", "gemini-2.5-flash")
EMBED_MODEL = S.get("EMBED_MODEL", "text-embedding-004")
SHEET_URL = S["SHEET_URL"]
SHEET_TOKEN = S["SHEET_TOKEN"]
TEACHER_PASSWORD = S.get("TEACHER_PASSWORD", "")
RELEVANCE = float(S.get("RELEVANCE", 0.5))  # seuil de pertinence de la recherche
CONTACT_FORMATEUR = S.get("CONTACT_FORMATEUR", "(contact du formateur à renseigner)")
CONTACT_REFERENT = S.get("CONTACT_REFERENT", "(contact du référent du centre à renseigner)")
LIEN_FORUM = S.get("LIEN_FORUM", "(lien du forum ou du groupe d'étude à renseigner)")

# =====================================================================
# INITIALISATION DES CLIENTS (Gemini & Qdrant)
# =====================================================================
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=S["GEMINI_API_KEY"])

@st.cache_resource
def get_qdrant_client():
    # Retourne None si les clés Qdrant ne sont pas encore configurées
    if "QDRANT_URL" not in S or "QDRANT_API_KEY" not in S:
        return None
    return QdrantClient(
        url=S["QDRANT_URL"],
        api_key=S["QDRANT_API_KEY"]
    )

client = get_gemini_client()
qdrant = get_qdrant_client()

with open("prompt.txt", encoding="utf-8") as f:
    BASE_PROMPT = f.read()

RAG_RULES = """

# S'APPUYER SUR LE COURS
À chaque question de l'apprenant, l'application peut te fournir des « EXTRAITS DU COURS ».
- Appuie-toi sur ces extraits pour expliquer, et cite la source sous la forme (fichier, p. X).
- Si les extraits ne couvrent pas la question, dis-le honnêtement (« ce point ne semble pas traité dans le support »), n'invente rien, et invite l'apprenant à poser la question à son formateur.
- Garde ta posture de tuteur : guide par questions et indices, sans recopier les extraits.
"""

# =====================================================================
# STOCKAGE : Google Sheets via Apps Script
# =====================================================================
COLS = {
    "sessions": ["code", "session", "type", "energie", "stress", "minutes"],
    "messages": ["code", "session", "role", "texte"],
    "fiches": ["code", "fiche"],
    "quiz": ["code", "session", "notion", "question", "correct", "source"],
    "plans": ["code", "objectif", "etapes", "quand", "si_alors"],
    "pauses": ["code", "type"],
    "unanswered": ["code", "question"],
}

@st.cache_data(ttl=45, show_spinner=False)
def read(sheet, code=None):
    cols = ["date"] + COLS[sheet]
    try:
        params = {"token": SHEET_TOKEN, "sheet": sheet}
        if code:
            params["code"] = code
        rows = requests.get(SHEET_URL, params=params, timeout=25).json().get("rows", [])
    except Exception:
        rows = []
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.reindex(columns=range(len(cols)))
    df.columns = cols
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df

def log(sheet, row):
    try:
        requests.post(SHEET_URL, json={"token": SHEET_TOKEN, "sheet": sheet, "row": row}, timeout=25)
        read.clear()
    except Exception:
        pass  # on ne bloque jamais l'apprenant si l'enregistrement échoue

# =====================================================================
# GEMINI GENERATION
# =====================================================================
def to_contents(history):
    return [
        types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["text"])])
        for m in history
    ]

def gen(system, contents, json_mode=False, temperature=0.7):
    if isinstance(contents, str):
        contents = [types.Content(role="user", parts=[types.Part(text=contents)])]
    kwargs = dict(system_instruction=system, temperature=temperature)
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    for attempt in range(2):
        try:
            r = client.models.generate_content(
                model=MODEL, contents=contents, config=types.GenerateContentConfig(**kwargs)
            )
            return r.text or ""
        except Exception:
            if attempt == 1:
                raise
            time.sleep(3)

# =====================================================================
# RECHERCHE DANS LE COURS (via Qdrant Cloud)
# =====================================================================
def has_index():
    return qdrant is not None

def retrieve(query, k=4):
    if not has_index():
        return []
    try:
        # Embedding de la question utilisateur
        r = client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_vector = r.embeddings[0].values
        
        # Recherche sur Qdrant
        search_result = qdrant.search(
            collection_name="cours_tuteur",
            query_vector=query_vector,
            limit=k
        )
        
        # Formatage des résultats
        return [
            dict(
                src=hit.payload.get("source", "Support de cours"),
                page=hit.payload.get("page", "?"),
                text=hit.payload.get("text", ""),
                score=hit.score
            )
            for hit in search_result
        ]
    except Exception as e:
        print(f"Erreur de recherche vectorielle : {e}")
        return []

def random_passages(k=3):
    """Récupère des passages aléatoires dans Qdrant pour la génération de QCM."""
    if not has_index():
        return []
    try:
        # On fait une recherche vectorielle avec un vecteur aléatoire pour tomber sur un sujet au hasard
        random_vector = np.random.rand(768).tolist() 
        search_result = qdrant.search(
            collection_name="cours_tuteur",
            query_vector=random_vector,
            limit=k
        )
        return [
            dict(
                src=hit.payload.get("source", "Support de cours"),
                page=hit.payload.get("page", "?"),
                text=hit.payload.get("text", ""),
                score=1.0 # Score ignoré pour l'aléatoire
            )
            for hit in search_result
        ]
    except Exception:
        return []

def passages_block(ps):
    return "\n\n".join(f"[{n}] ({p['src']}, p. {p['page']})\n{p['text']}" for n, p in enumerate(ps, 1))

def src_label(p):
    return f"{p['src']}, p. {p['page']}"

# =====================================================================
# ÉTAT STREAMLIT
# =====================================================================
s = st.session_state
s.setdefault("tstep", "checkin")
s.setdefault("history", [])
s.setdefault("qid", 0)
s.setdefault("q", None)
s.setdefault("q_done", False)

NO_INDEX_MSG = (
    "Le support de cours n'est pas encore connecté à l'application : "
    "cette fonction n'est pas disponible (ou fonctionne sans le cours)."
)

# =====================================================================
# PAGE : TABLEAU DE BORD
# =====================================================================
def notion_stats(quiz):
    cols = ["notion", "n", "taux", "dernier", "rattrapee"]
    if quiz.empty:
        return pd.DataFrame(columns=cols)
    q = quiz.copy()
    q["correct"] = pd.to_numeric(q["correct"], errors="coerce").fillna(0)
    q = q.sort_values("date")
    rows = []
    for notion, g in q.groupby("notion"):
        c = g["correct"].tolist()
        rows.append(
            dict(
                notion=notion,
                n=len(c),
                taux=sum(c) / len(c),
                dernier=bool(c[-1]),
                rattrapee=bool(c[-1]) and (0 in c[:-1]),
            )
        )
    return pd.DataFrame(rows, columns=cols)

def page_dashboard():
    st.header("🏠 Mon tableau de bord")
    sess = read("sessions", s.code)
    quiz = read("quiz", s.code)
    fiches = read("fiches", s.code)

    for c in ("energie", "stress"):
        sess[c] = pd.to_numeric(sess[c], errors="coerce")
    debut = sess[sess["type"] == "debut"].sort_values("date")
    now = pd.Timestamp.
