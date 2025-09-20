# streamlit_app_v6_7.py
# v6.7 — E0–E2, halvgarderingar, tipsrad + "Fredagsanalys" (GPT)
# - Ingen E3
# - OPENAI_API_KEY hämtas från Streamlit Secrets

import os
import json
from datetime import datetime
from urllib.request import urlretrieve
from collections import defaultdict, deque

import pandas as pd
import numpy as np
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier

# -------- OpenAI (GPT) ----------
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# =======================
# Grundinställningar
# =======================
st.set_page_config(page_title="Fotboll v6.7 — E0–E2 + Fredagsanalys", layout="wide")
st.title("⚽ Fotboll v6.7 — Tippa matcher (E0–E2) + halvgarderingar + Fredagsanalys")

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "model_v6.pkl")

BASE_URL = "https://www.football-data.co.uk/mmz4281"
LEAGUES = ["E0", "E1", "E2"]  # ⬅️ E3 borttagen

def season_code():
    y = datetime.now().year % 100
    prev = y - 1
    return f"{prev:02d}{y:02d}"

SEASON = season_code()
TEAMS_JSON = os.path.join(DATA_DIR, f"teams_{SEASON}.json")

# =======================
# Datahämtning
# =======================
def download_files(leagues=tuple(LEAGUES)):
    s = season_code()
    got = []
    for L in leagues:
        target = os.path.join(DATA_DIR, f"{L}_{s}.csv")
        url = f"{BASE_URL}/{s}/{L}.csv"
        try:
            urlretrieve(url, target)
            got.append(target)
        except Exception as e:
            st.warning(f"Kunde inte hämta {url}: {e}")
    return got

def load_all_data(files):
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="latin1")
        except Exception:
            continue
        league = os.path.basename(f).split("_")[0]
        df["League"] = league
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# =======================
# Features: form + ELO
# =======================
def calculate_5match_form(df):
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    home_pts, home_gd = defaultdict(lambda: deque([], maxlen=5)), defaultdict(lambda: deque([], maxlen=5))
    away_pts, away_gd = defaultdict(lambda: deque([], maxlen=5)), defaultdict(lambda: deque([], maxlen=5))

    df["HomeFormPts5"], df["HomeFormGD5"], df["AwayFormPts5"], df["AwayFormGD5"] = 0, 0, 0, 0

    for i, row in df.iterrows():
        home, away = row.get("HomeTeam",""), row.get("AwayTeam","")
        fthg, ftag, ftr = row.get("FTHG",0), row.get("FTAG",0), row.get("FTR","D")

        if len(home_pts[home]) > 0:
            df.at[i, "HomeFormPts5"] = float(np.mean(home_pts[home]))
            df.at[i, "HomeFormGD5"]  = float(np.mean(home_gd[home]))
        if len(away_pts[away]) > 0:
            df.at[i, "AwayFormPts5"] = float(np.mean(away_pts[away]))
            df.at[i, "AwayFormGD5"]  = float(np.mean(away_gd[away]))

        hp, ap = (3,0) if ftr=="H" else (1,1) if ftr=="D" else (0,3)
        gd_home, gd_away = fthg - ftag, ftag - fthg
        home_pts[home].append(hp); home_gd[home].append(gd_home)
        away_pts[away].append(ap); away_gd[away].append(gd_away)

    return df

def compute_elo(df, K=20):
    elo = defaultdict(lambda: 1500.0)
    df = df.copy()
    df["HomeElo"], df["AwayElo"] = 1500.0, 1500.0

    for i, row in df.iterrows():
        home, away = row.get("HomeTeam",""), row.get("AwayTeam","")
        ftr = row.get("FTR","D")
        Ra, Rb = elo[home], elo[away]
        Ea = 1/(1+10**((Rb-Ra)/400))
        Sa = 1 if ftr=="H" else 0.5 if ftr=="D" else 0
        Sb = 1 - Sa
        elo[home] = Ra + K*(Sa - Ea)
        elo[away] = Rb + K*(Sb - (1 - Ea))
        df.at[i, "HomeElo"], df.at[i, "AwayElo"] = elo[home], elo[away]
    return df

def prepare_features(df):
    expect = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","League"]
    for c in expect:
        if c not in df.columns:
            df[c] = np.nan
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["FTR"])
    df = calculate_5match_form(df)
    df = compute_elo(df)
    feature_cols = ["HomeFormPts5","HomeFormGD5","AwayFormPts5","AwayFormGD5","HomeElo","AwayElo"]
    return df, feature_cols

# =======================
# Modell
# =======================
def train_model(df, feature_cols):
    X = df[feature_cols].fillna(0)
    y = df["FTR"].map({"H":0,"D":1,"A":2}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_dist = {
        "n_estimators":[100,200],
        "max_depth":[3,5,7],
        "learning_rate":[0.05,0.1,0.2],
        "subsample":[0.8,1.0]
    }
    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    search = RandomizedSearchCV(model, param_dist, n_iter=5, cv=3, n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    joblib.dump(best, MODEL_FILE)
    return best

def load_or_train_model(df, feature_cols):
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return train_model(df, feature_cols)

def predict_probs(model, features, feature_cols):
    X = pd.DataFrame([features], columns=feature_cols)
    return model.predict_proba(X)[0]

# =======================
# Laglista (alfabetisk)
# =======================
def build_team_labels(df, leagues):
    pairs = set()
    for lg in leagues:
        sub = df[df["League"] == lg]
        teams = set(sub["HomeTeam"].dropna()) | set(sub["AwayTeam"].dropna())
        for t in teams:
            if t:
                pairs.add((str(t), lg))
    labels = [f"{t} ({lg})" for (t, lg) in pairs]
    labels = sorted(labels, key=lambda s: s.lower())
    return labels

def load_or_create_team_labels(df, leagues):
    if os.path.exists(TEAMS_JSON):
        try:
            with open(TEAMS_JSON, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if isinstance(labels, list) and labels:
                return labels
        except Exception:
            pass
    labels = build_team_labels(df, leagues)
    if labels:
        with open(TEAMS_JSON, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
    return labels

# =======================
# Halvgarderingar
# =======================
def pick_half_guards(match_probs, n_half):
    if n_half <= 0:
        return set()
    margins = []
    for i, p in enumerate(match_probs):
        s = np.sort(p)
        margin = s[-1] - s[-2]  # liten marginal = osäker → bra kandidat för halvgardering
        margins.append((i, margin))
    margins.sort(key=lambda x: x[1])
    return {i for i,_ in margins[:n_half]}

def halfguard_sign(probs):
    idxs = np.argsort(probs)[-2:]
    idxs = tuple(sorted(idxs))
    mapping = {(0,1): "1X", (0,2): "12", (1,2): "X2"}
    return mapping.get(idxs, "1X")

# =======================
# GPT "Fredagsanalys"
# =======================
def get_openai_client():
    if not _HAS_OPENAI:
        return None, "openai-biblioteket saknas (lägg till 'openai' i requirements.txt)."
    api_key = None
    # Streamlit Cloud: via secrets
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    # fallback: miljövariabel lokalt
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY saknas (lägg in i Streamlit Secrets)."
    try:
        client = OpenAI(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"Kunde inte initiera OpenAI-klient: {e}"

def gpt_match_brief(client, home, away, league, h_form_pts, h_form_gd, a_form_pts, a_form_gd, h_elo, a_elo, p1, px, p2):
    """
    Gör en kort briefing utan att hitta på nyheter:
    - Använd ENBART givna siffror (form, ELO, sannolikheter).
    - Inga påståenden om skador/rykten/etc.
    """
    prompt = f"""
Du är en sportanalytiker. Ge en kort briefing inför matchen {home} - {away} i {league}.
Använd endast siffrorna nedan (inga påhittade nyheter eller skador):
- Hemma form (5): poäng {h_form_pts:.2f}, målskillnad {h_form_gd:.2f}
- Borta form (5): poäng {a_form_pts:.2f}, målskillnad {a_form_gd:.2f}
- ELO: {home} {h_elo:.1f}, {away} {a_elo:.1f}
- Modellens sannolikheter: 1={p1:.1%}, X={px:.1%}, 2={p2:.1%}

Svara med 3 korta punkter:
1) Styrkebalans (ELO) och hemmaprofil.
2) Formkurvor (5 matcher) och vad det antyder.
3) Kort riskbedömning (t.ex. hög osäkerhet om 2 utfal ligger nära).
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"Du skriver kort, sakligt och utan spekulationer."},
                {"role":"user","content":prompt}
            ],
            temperature=0.2,
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Ingen GPT-analys: {e})"

# =======================
# Huvudflöde
# =======================
files = download_files()
df_raw = load_all_data(files)
if df_raw.empty:
    st.error("Ingen data.")
    st.stop()

df_prep, feat_cols = prepare_features(df_raw)
model = load_or_train_model(df_prep, feat_cols)

teams_all = load_or_create_team_labels(df_raw, LEAGUES)
if not teams_all:
    st.warning("Kunde inte skapa laglistan. Kontrollera att minst en match finns i varje liga.")
    st.stop()

# Inputs
n_matches = st.number_input("Antal matcher att tippa", 1, 13, value=13)
n_half = st.number_input("Antal halvgarderingar", 0, n_matches, value=7)

match_pairs = []
for i in range(n_matches):
    c1, c2 = st.columns(2)
    home = c1.selectbox(f"Hemmalag {i+1}", teams_all, key=f"h_{i}")
    away = c2.selectbox(f"Bortalag {i+1}", teams_all, key=f"a_{i}")
    if home and away and home != away:
        match_pairs.append((home, away))

# Tippa
if st.button("Tippa matcher"):
    rows, match_probs, tecken_list, match_meta = [], [], [], []

    # Beräkna sannolikheter + samla meta
    for (home, away) in match_pairs:
        home_team, home_lg = home.rsplit(" (",1)
        away_team, away_lg = away.rsplit(" (",1)
        home_team, home_lg = home_team.strip(), home_lg.strip(")")
        away_team, away_lg = away_team.strip(), away_lg.strip(")")

        h_row = df_prep[(df_prep["League"]==home_lg) & (df_prep["HomeTeam"]==home_team)].tail(1)
        a_row = df_prep[(df_prep["League"]==away_lg) & (df_prep["AwayTeam"]==away_team)].tail(1)
        if h_row.empty or a_row.empty:
            match_probs.append(np.array([0.0,0.0,0.0]))
            match_meta.append((home_team, away_team, f"{home_lg}/{away_lg}", 0,0,0,0,0,0))
            continue

        features = [
            float(h_row["HomeFormPts5"].values[0]),
            float(h_row["HomeFormGD5"].values[0]),
            float(a_row["AwayFormPts5"].values[0]),
            float(a_row["AwayFormGD5"].values[0]),
            float(h_row["HomeElo"].values[0]),
            float(a_row["AwayElo"].values[0])
        ]
        probs = predict_probs(model, features, feat_cols)
        match_probs.append(probs)

        match_meta.append((
            home_team, away_team, home_lg,                # visa ligan enligt hemmalagets liga
            features[0], features[1], features[2], features[3], features[4], features[5]
        ))

    # Välj halvgarderingar
    half_idxs = pick_half_guards(match_probs, n_half)

    # Tabell + tipsrad
    for idx, ((home_label, away_label), probs, meta) in enumerate(zip(match_pairs, match_probs, match_meta), start=1):
        if probs.sum() == 0:
            tecken, pct = "?", ""
        else:
            if (idx-1) in half_idxs:
                tecken = f"({halfguard_sign(probs)})"
                pct = "-"
            else:
                pred = int(np.argmax(probs))
                tecken = f"({['1','X','2'][pred]})"
                pct = f"{probs[pred]*100:.1f}%"

        rows.append([idx, "", f"{home_label} - {away_label}", tecken, "", "", pct])
        tecken_list.append(tecken)

    df_out = pd.DataFrame(rows, columns=["#","Status","Match","Tecken","Res.","%","Stats"])
    st.subheader("Resultat-tabell")
    st.dataframe(df_out, use_container_width=True)

    st.subheader("Tipsrad (kopiera)")
    st.code(" ".join(tecken_list), language=None)

    # Fredagsanalys (GPT) — valbart
    with st.expander("🔮 Fredagsanalys (GPT)"):
        client, err = get_openai_client()
        if err:
            st.warning(err)
        else:
            st.caption("Analysen bygger endast på form/ELO/sannolikheter (inga nyheter för att undvika påhitt).")
            for i, (home_team, away_team, lg, hfp, hfgd, afp, afgd, helo, aelo) in enumerate(match_meta, start=1):
                if match_probs[i-1].sum() == 0:
                    st.markdown(f"**{i}) {home_team} ({lg}) - {away_team}**\n*(Ingen data till analys)*")
                    continue
                p1, px, p2 = match_probs[i-1][0], match_probs[i-1][1], match_probs[i-1][2]
                summary = gpt_match_brief(
                    client, home_team, away_team, lg,
                    hfp, hfgd, afp, afgd, helo, aelo, p1, px, p2
                )
                st.markdown(f"**{i}) {home_team} ({lg}) - {away_team}**")
                st.write(summary)
