import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
import time
import uuid
from datetime import datetime

st.set_page_config(page_title="Mon espace d'apprentissage", page_icon="🚀")

# =====================================================================
# CONFIGURATION
# =====================================================================
S = st.secrets
MODEL = S.get("MODEL", "gemini-1.5-flash")
SHEET_URL = S["SHEET_URL"]
SHEET_TOKEN = S["SHEET_TOKEN"]
TEACHER_PASSWORD = S.get("TEACHER_PASSWORD", "")
CONTACT_FORMATEUR = S.get("CONTACT_FORMATEUR", "(contact du formateur)")
CONTACT_REFERENT = S.get("CONTACT_REFERENT", "(contact du référent)")
LIEN_FORUM = S.get("LIEN_FORUM", "(lien du forum)")

@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=S["GEMINI_API_KEY"])

client = get_gemini_client()

COACH_RULES = """
Tu es un coach métacognitif bienveillant pour des adultes en formation à distance. 
Ton rôle N'EST PAS de faire cours ou de donner des réponses expertes. Ton rôle est de soutenir l'autorégulation de l'apprentissage.
- Aide l'apprenant à s'organiser, à gérer sa charge mentale et à planifier.
- S'il bloque sur un concept, ne lui donne pas la définition. Pose-lui des questions socratiques pour l'aider à structurer sa propre réflexion.
- Demande-lui d'expliquer ce qu'il a compris (technique de Feynman).
- Sois encourageant, concis et toujours orienté vers la méthode plutôt que vers le contenu.
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
        pass

# =====================================================================
# GEMINI GENERATION
# =====================================================================
def to_contents(history):
    return [
        types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["text"])])
        for m in history
    ]

def gen(system, contents, temperature=0.7):
    if isinstance(contents, str):
        contents = [types.Content(role="user", parts=[types.Part(text=contents)])]
    kwargs = dict(system_instruction=system, temperature=temperature)
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
# ÉTAT STREAMLIT
# =====================================================================
s = st.session_state
s.setdefault("tstep", "checkin")
s.setdefault("history", [])

# =====================================================================
# PAGE : TABLEAU DE BORD
# =====================================================================
def page_dashboard():
    st.header("🏠 Mon tableau de bord")
    sess = read("sessions", s.code)
    fiches = read("fiches", s.code)

    for c in ("energie", "stress"):
        sess[c] = pd.to_numeric(sess[c], errors="coerce")
    debut = sess[sess["type"] == "debut"].sort_values("date")
    now = pd.Timestamp.now(tz="UTC")

    c1, c2 = st.columns(2)
    c1.metric("Séances réalisées", len(debut))
    recent = debut[debut["date"] >= now - pd.Timedelta(days=28)]
    c2.metric("Semaines actives (sur 4)", int(recent["date"].dt.isocalendar().week.nunique()) if len(recent) else 0)

    if debut.empty:
        st.info("Ton tableau de bord se remplira dès ta première séance avec le coach. 🚀")
        return

    st.subheader("Mon énergie et mon stress")
    st.line_chart(debut.set_index("date")[["energie", "stress"]])
    st.caption("Apprendre à réguler son énergie est essentiel en formation à distance.")

    if len(fiches):
        with st.expander("Ma dernière fiche de séance"):
            st.markdown(fiches.iloc[-1]["fiche"])

# =====================================================================
# PAGE : TUTEUR / COACH
# =====================================================================
def start_session(energy, stress, minutes):
    fdf = read("fiches", s.code)
    fiche = fdf.iloc[-1]["fiche"] if len(fdf) else ""
    s.energy0, s.stress0 = energy, stress
    s.session_id = uuid.uuid4().hex[:8]
    s.system = (
        COACH_RULES
        + f"\n\nCONTEXTE : Énergie {energy}/5 ; Stress {stress}/5 ; Temps {minutes} min.\n"
        + f"Dernière fiche : {fiche if fiche else 'Aucune'}\n"
        + "Aide l'apprenant à fixer un petit objectif méthodologique pour aujourd'hui."
    )
    log("sessions", [s.code, s.session_id, "debut", energy, stress, minutes])
    s.history = [{"role": "user", "text": "Bonjour, je commence ma séance."}]
    s.history.append({"role": "model", "text": gen(s.system, to_contents(s.history))})
    s.tstep = "chat"

def page_tuteur():
    st.header("💬 Mon Coach d'organisation")

    if s.tstep == "checkin":
        st.write("Faisons un point avant de démarrer.")
        energy = st.slider("Ton énergie aujourd'hui (1 = très basse, 5 = très haute)", 1, 5, 3)
        stress = st.slider("Ton niveau de stress (1 = très bas, 5 = très élevé)", 1, 5, 3)
        minutes = st.number_input("Temps dont tu disposes (minutes)", 10, 240, 45, step=5)
        if st.button("Lancer la séance"):
            with st.spinner("Préparation de la séance..."):
                start_session(energy, stress, int(minutes))
            st.rerun()

    elif s.tstep == "chat":
        for m in s.history[1:]:
            with st.chat_message("user" if m["role"] == "user" else "assistant"):
                st.write(m["text"])

        if prompt := st.chat_input("Parle de ton organisation ou de tes blocages..."):
            s.history.append({"role": "user", "text": prompt})
            log("messages", [s.code, s.session_id, "apprenant", prompt])
            with st.spinner("Le coach réfléchit..."):
                try:
                    answer = gen(s.system, to_contents(s.history))
                except Exception:
                    answer = "Petit souci technique, peux-tu renvoyer ton message ?"
            s.history.append({"role": "model", "text": answer})
            log("messages", [s.code, s.session_id, "tuteur", answer])
            st.rerun()

        st.divider()
        if st.button("✅ Terminer la séance"):
            s.tstep = "bilan"
            st.rerun()

    elif s.tstep == "bilan":
        st.subheader("Bilan de séance")
        energy1 = st.slider("Ton énergie maintenant (1 à 5)", 1, 5, 3)
        stress1 = st.slider("Ton stress maintenant (1 à 5)", 1, 5, 3)
        learned = st.text_area("Qu'as-tu accompli aujourd'hui ?")
        nxt = st.text_input("Ta prochaine action concrète ?")
        if st.button("Générer ma fiche") and learned:
            demande = (
                f"Bilan : Accompli : {learned} | Prochaine action : {nxt}\n"
                "Génère une fiche de séance ultra courte et encourageante résumant son avancée."
            )
            with st.spinner("Préparation de la fiche..."):
                fiche = gen(COACH_RULES, demande)
            log("fiches", [s.code, f"[{datetime.now().strftime('%d/%m/%Y')}] {fiche}"])
            log("sessions", [s.code, s.session_id, "fin", energy1, stress1, ""])
            s.fiche_text = fiche
            s.tstep = "fin"
            st.rerun()

    elif s.tstep == "fin":
        st.success("Fiche enregistrée pour la prochaine fois !")
        st.markdown(s.fiche_text)
        if st.button("Nouvelle séance"):
            s.tstep, s.history = "checkin", []
            st.rerun()

# =====================================================================
# PAGE : AUTO-ÉVALUATION (Nouveau module sans PDF)
# =====================================================================
def page_test():
    st.header("🧠 Je m'auto-évalue")
    st.write("Saisis une notion que tu veux réviser. L'IA te posera une question ciblée pour vérifier ce que tu maîtrises et ce que tu dois approfondir.")
    
    topic = st.text_input("Quelle notion de ton cours veux-tu tester ?")
    
    if st.button("Générer ma question") and topic:
        with st.spinner("L'IA prépare ta question..."):
            sys = "Tu es un évaluateur. Pose UNE SEULE question ouverte et précise permettant de vérifier la compréhension de l'apprenant sur la notion demandée. Ne donne surtout pas la réponse."
            s.test_q = gen(sys, f"Notion à tester : {topic}")
            s.test_topic = topic
            s.test_ans = None
            s.test_eval = None
        st.rerun()

    if s.get("test_q") and not s.get("test_eval"):
        st.info(f"**Question sur la notion : {s.test_topic}**\n\n{s.test_q}")
        ans = st.text_area("Rédige ta réponse avec tes propres mots :")
        if st.button("Valider ma réponse") and ans:
            with st.spinner("Analyse de ta réponse en cours..."):
                sys_eval = (
                    "Tu es un tuteur bienveillant. Analyse la réponse de l'apprenant. "
                    "1/ Dis-lui clairement ce qui est juste et acquis. "
                    "2/ Explique-lui ce qui est faux, incomplet, ou ce qu'il n'a pas su dire. "
                    "3/ Donne-lui la réponse complète pour qu'il puisse réviser."
                )
                prompt_eval = f"Question posée : {s.test_q}\nRéponse de l'apprenant : {ans}"
                s.test_eval = gen(sys_eval, prompt_eval)
                log("quiz", [s.code, s.get("session_id", ""), s.test_topic, "Auto-évaluation ouverte", 1, "IA Générale"])
            st.rerun()
            
    if s.get("test_eval"):
        st.info(f"**Question posée :** {s.test_q}")
        st.success("🎯 Bilan de tes connaissances :")
        st.markdown(s.test_eval)
        if st.button("Tester une autre notion"):
            s.test_q = None
            s.test_eval = None
            st.rerun()

# =====================================================================
# PAGE : MON PLAN
# =====================================================================
def page_plan():
    st.header("📅 Mon plan")
    tab1, tab2 = st.tabs(["Mon plan de la semaine", "Je suis submergé(e)"])

    with tab1:
        prev = read("plans", s.code)
        if len(prev):
            p = prev.iloc[-1]
            st.info(f"**Dernier plan enregistré** : {p['objectif']}\n\nÉtapes : {p['etapes']}\n\nQuand : {p['quand']}")
        objectif = st.text_input("Mon objectif de la semaine (concret et réaliste)")
        e1 = st.text_input("Étape 1")
        e2 = st.text_input("Étape 2")
        quand = st.text_input("Quand ? (jours et horaires précis)")
        obstacle = st.text_input("Ce qui pourrait m'empêcher d'avancer")
        solution = st.text_input("Si cela arrive, alors je...")
        c1, c2 = st.columns(2)
        if c1.button("Enregistrer mon plan") and objectif:
            etapes = f"{e1} / {e2}"
            log("plans", [s.code, objectif, etapes, quand, f"Si {obstacle}, alors {solution}"])
            st.success("Plan enregistré !")
        if c2.button("💡 Améliorer mon plan") and objectif:
            demande = f"Objectif : {objectif}\nÉtapes : {e1} / {e2}\nQuand : {quand}\nObstacle : {obstacle} -> {solution}"
            with st.spinner("J'analyse ton plan..."):
                st.markdown(gen("Tu es un coach d'organisation. Analyse ce plan d'étude, dis ce qui est bien et propose 2 améliorations pour éviter la surcharge cognitive.", demande))

    with tab2:
        st.write("Quand tout s'accumule, on ne cherche pas à tout faire : on trie et on allège.")
        taches = st.text_area("Liste tout ce que tu as à faire")
        heures = st.number_input("Heures disponibles cette semaine", 1, 40, 5)
        if st.button("M'aider à trier") and taches.strip():
            with st.spinner("Je trie avec toi..."):
                rep = gen("L'apprenant est submergé. Propose un objectif ALLÉGÉ pour son temps disponible, et UNE petite action à faire dès maintenant. Ton rassurant.", f"Heures : {heures}\nTâches :\n{taches}")
            st.markdown(rep)
            log("pauses", [s.code, "submerge"])

# =====================================================================
# PAGE : MA PAUSE
# =====================================================================
def page_pause():
    st.header("🌿 Ma pause")
    st.write("Les pauses sont essentielles pour l'assimilation des connaissances.")
    st.markdown("- **20-20-20** : toutes les 20 min, regarde au loin pendant 20 secondes.\n- **Bouge** : étire-toi pendant 2 minutes.\n- **Hydrate-toi** : un verre d'eau, loin de l'écran.")
    if st.button("J'ai pris une vraie pause"):
        log("pauses", [s.code, "mini-pause"])
        st.success("Parfait, ton cerveau te remerciera pour l'autorégulation ! ✅")

# =====================================================================
# PAGE : BESOIN D'UN HUMAIN ?
# =====================================================================
def page_humain():
    st.header("🤝 Besoin d'un humain ?")
    st.markdown(f"- **Questions sur le cours** : {CONTACT_FORMATEUR}\n- **Situation compliquée** : {CONTACT_REFERENT}\n- **Échanger avec les autres** : {LIEN_FORUM}")

# =====================================================================
# PAGE : ESPACE FORMATEUR
# =====================================================================
def page_formateur():
    st.title("👩‍🏫 Espace formateur")
    if not s.get("teacher_ok"):
        pw = st.text_input("Mot de passe", type="password")
        if st.button("Entrer"):
            if pw == TEACHER_PASSWORD:
                s.teacher_ok = True
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
        return

    st.caption("Suivi de l'autorégulation des apprenants (anonymisé)")
    sess = read("sessions")
    for c in ("energie", "stress"):
        sess[c] = pd.to_numeric(sess[c], errors="coerce")
    debut = sess[sess["type"] == "debut"].sort_values("date")
    
    st.subheader("Suivi de l'engagement")
    rows = []
    for code, g in debut.groupby("code"):
        last = g["date"].max()
        recent_e = g.tail(2)["energie"].mean()
        rows.append({"Code": code, "Séances": len(g), "Énergie moyenne récente": round(recent_e, 1) if pd.notna(recent_e) else None})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.info("Aucune donnée.")

# =====================================================================
# ACCUEIL ET NAVIGATION
# =====================================================================
if st.query_params.get("formateur"):
    page_formateur()
    st.stop()

if "code" not in s:
    st.title("🚀 Mon espace d'apprentissage")
    st.info("Cet espace t'aide à t'organiser, à évaluer tes connaissances et à garder de l'énergie (autorégulation).")
    code = st.text_input("Ton code apprenant").strip()
    if st.button("Entrer") and code:
        s.code = code
        st.rerun()
    st.stop()

PAGES = {
    "🏠 Mon tableau de bord": page_dashboard,
    "💬 Mon Coach d'organisation": page_tuteur,
    "🧠 Je m'auto-évalue": page_test,
    "📅 Mon plan": page_plan,
    "🌿 Ma pause": page_pause,
    "🤝 Besoin d'un humain ?": page_humain,
}
with st.sidebar:
    st.write(f"Connecté : **{s.code}**")
    choice = st.radio("Menu", list(PAGES), label_visibility="collapsed")
PAGES[choice]()
