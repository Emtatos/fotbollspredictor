# streamlit_app_v6_7.py
# v6.8 — Stabil laglista + normalisering + auto-refresh
# - Laddar E0–E2 (Premier, Championship, League One)
# - Hanterar namnvarianter (t.ex. "Bradford", "Bradford C" -> "Bradford City")
# - Rensar och bygger om laglistan när CSV ändras (ingen fastnar i gammal JSON)
# - Timeout/retries vid nedladdning (app hänger inte)
# - "Fredagsanalys" via OpenAI (frivilligt)

import os
import json
import time
import hashlib
from datetime import datetime
from collections import defaultdict, deque

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import requests

from sklearn.model_selection import train_test_split
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
st.set_page_config(page_title="Fotboll v6.8 — E0–E2 + Fredagsanalys", layout="wide")
st.title("⚽ Fotboll v6.8 — Tippa matcher (E0–E2) + halvgarderingar + Fredagsanalys")

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "model_v6.pkl")
BASE_URL = "https://www.football-data.co.uk/mmz4281"
LEAGUES = ["E0", "E1", "E2"]  # Premier, Championship, League One

def season_code():
    y = datetime.now().year % 100
    prev = y - 1
    return f"{prev:02d}{y:02d}"

SEASON = season_code()

# =======================
# Hjälpare
# =======================
def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def _norm_space(s: str) -> str:
    return " ".join(str(s).strip().split())

# Namn-normalisering för Football-Data varianter → standardnamn
TEAM_ALIASES = {
    # Bradford-case + några vanliga exempel. Lägg gärna till fler efter behov.
    "Bradford": "Bradford City",
    "Bradford C": "Bradford City",
    "Bradford City": "Bradford City",
    # Cardiff (du nämnde matchen)
    "Cardiff": "Cardiff City",
    "Cardiff C": "Cardiff City",
    "Cardiff City": "Cardiff City",
    # Exempel (övriga vanliga)
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Sheffield Wed": "Sheffield Wednesday",
    "Sheff Wed": "Sheffield Wednesday",
    "Sheffield Utd": "Sheffield United",
    "Sheff Utd": "Sheffield United",
    "QPR": "Queens Park Rangers",
    "MK Dons": "Milton Keynes Dons",
}

def normalize_team_name(raw: str) -> str:
    s = _norm_space(raw)
    # Direkt alias
    if s in TEAM_ALIASES:
        return TEAM_ALIASES[s]
    # Football-Data använder ofta korta suffix med C, FC etc.
    # Enkel heuristik: " - " inte relevant här, men trimma "FC" vad gäller visning.
    if s.endswith(" FC"):
        s = s[:-3]
    # Om någon skriver t.ex. "Bradford C." (punkt)
    s = s.replace(" C.", " C")
    return TEAM_ALIASES.get(s, s)

# =======================
# HTTP med timeout/retries
# =======================
SESSION = requests.Session()
ADAPTER = requests.adapters.HTTPAdapter(max_retries=3)
SESSION.mount("https://", ADAPTER)
SESSION.mount("http://", ADAPTER)

def _http_get(url: str, timeout: float = 10.0) -> bytes | None:
    try:
        r = SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

@st.cache_data(ttl=6*3600, show_spinner=True)
def _download_one(league: str, s_code: str) -> str | None:
    target = os.path.join(DATA_DIR, f"{league}_{s_code}.csv")
    url = f"{BASE_URL}/{s_code}/{league}.csv"
    data = _http_get(url, timeout=10.0)
    if data is None:
        for delay in (1, 2, 4):
            time.sleep(delay)
            data = _http_get(url, timeout=10.0)
            if data is not None:
                break
    if data is None:
        return None
    try:
        with open(target, "wb") as f:
            f.write(data)
        return target
    except Exception:
        return None

@st.cache_data(ttl=6*3600, show_spinner=True)
def download_files(leagues=tuple(LEAGUES), s_code: str = SEASON):
    paths = []
    for L in leagues:
        p = _download_one(L, s_code)
        if p and os.path.exists(p):
            paths.append(p)
        else:
            st.warning(f"Kunde inte hämta {L} {s_code} (timeout/blockerat).")
    if not paths:
        st.error("Ingen liga kunde hämtas. Testa senare eller byt nät.")
    return tuple(paths)

@st.cache_data(ttl=6*3600, show_spinner=False)
def load_all_data(files: tuple[str, ...]) -> pd.DataFrame:
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="latin1")
            league = os.path.basename(f).split("_")[0]
            df["League"] = league
            for col in ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]:
                if col not in df.columns:
                    df[col] = np.nan
            # Normalisera lagnamn i rådata direkt
            df["HomeTeam"] = df["HomeTeam"].astype(str).apply(normalize_team_name)
            df["AwayTeam"] = df["AwayTeam"].astype(str).apply(normalize_team_name)
            dfs.append(df)
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# =======================
# Features: form + ELO
# =======================
def calculate_5match_form(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values(["Date"]).reset_index(drop=True)

    home_pts, home_gd = defaultdict(lambda: deque([], maxlen=5)), defaultdict(lambda: deque([], maxlen=5))
    away_pts, away_gd = defaultdict(lambda: deque([], maxlen=5)), defaultdict(lambda: deque([], maxlen=5))

    df["HomeFormPts5"], df["HomeFormGD5"], df["AwayFormPts5"], df["AwayFormGD5"] = 0.0, 0.0, 0.0, 0.0

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

@st.cache_data(ttl=6*3600, show_spinner=False)
def prepare_features(df: pd.DataFrame):
    if df.empty:
        return df, []
    df = df.dropna(subset=["FTR"])
    df = calculate_5match_form(df)
    df = compute_elo(df)
    feature_cols = ["HomeFormPts5","HomeFormGD5","AwayFormPts5","AwayFormGD5","HomeElo","AwayElo"]
    return df, feature_cols

# =======================
# Modell
# =======================
def _quick_train(df, feature_cols):
    X = df[feature_cols].fillna(0.0)
    y = df["FTR"].map({"H":0,"D":1,"A":2}).astype(int)

    if len(X) < 100:
        params = dict(n_estimators=120, max_depth=4, learning_rate=0.15, subsample=1.0, reg_lambda=1.0)
    else:
        params = dict(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.9, reg_lambda=1.0)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=min(0.2, max(0.1, 200/len(X))) if len(X) > 200 else 0.2,
        random_state=42, stratify=y
    )
    model = XGBClassifier(
        **params, objective="multi:softprob", num_class=3, n_jobs=1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

@st.cache_resource(show_spinner=True)
def load_or_train_model(df_signature: tuple[int, int] | None, df: pd.DataFrame, feature_cols: list[str]):
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass
    model = _quick_train(df, feature_cols)
    try:
        joblib.dump(model, MODEL_FILE)
    except Exception:
        pass
    return model

def predict_probs(model, features, feature_cols):
    X = pd.DataFrame([features], columns=feature_cols)
    return model.predict_proba(X)[0]

# =======================
# Laglista (ALLTID färsk + korrekta namn)
# =======================
def _league_signature(files: tuple[str, ...]) -> str:
    # Sig baserad på alla CSV-filers hash + storlek + mtime (om en fil ändras byggs listan om)
    parts = []
    for p in files:
        try:
            parts.append(f"{os.path.basename(p)}:{_hash_file(p)}:{os.path.getsize(p)}:{int(os.path.getmtime(p))}")
        except Exception:
            parts.append(os.path.basename(p))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

def build_team_labels(df_raw: pd.DataFrame, leagues: list[str]) -> list[str]:
    pairs = set()
    for lg in leagues:
        sub = df_raw[df_raw["League"] == lg]
        teams = set(sub["HomeTeam"].dropna()) | set(sub["AwayTeam"].dropna())
        for t in teams:
            t = normalize_team_name(t)
            if t:
                pairs.add((t, lg))
    labels = [f"{t} ({lg})" for (t, lg) in pairs]
    labels = sorted(labels, key=lambda s: s.lower())
    return labels

def load_or_create_team_labels(df_raw: pd.DataFrame, leagues: list[str], files_sig: str) -> list[str]:
    # JSON med inbakad signatur i filnamnet -> alltid färsk när CSV ändras
    teams_json = os.path.join(DATA_DIR, f"teams_{SEASON}_{files_sig}.json")

    # Rensa gamla team-filer för denna säsong (håller laglistan ren)
    try:
        for f in os.listdir(DATA_DIR):
            if f.startswith(f"teams_{SEASON}_") and f.endswith(".json") and f != os.path.basename(teams_json):
                os.remove(os.path.join(DATA_DIR, f))
    except Exception:
        pass

    if os.path.exists(teams_json):
        try:
            with open(teams_json, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if isinstance(labels, list) and labels:
                return labels
        except Exception:
            pass

    labels = build_team_labels(df_raw, leagues)
    if labels:
        try:
            with open(teams_json, "w", encoding="utf-8") as f:
                json.dump(labels, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return labels

# =======================
# GPT "Fredagsanalys"
# =======================
def get_openai_client():
    if not _HAS_OPENAI:
        return None, "openai-biblioteket saknas (lägg till 'openai' i requirements.txt)."
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY saknas (lägg in i Streamlit Secrets)."
    try:
        client = OpenAI(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"Kunde inte initiera OpenAI-klient: {e}"

def gpt_match_brief(client, home, away, league, h_form_pts, h_form_gd, a_form_pts, a_form_gd, h_elo, a_elo, p1, px, p2):
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
3) Kort riskbedömning (t.ex. hög osäkerhet om 2 utfall ligger nära).
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
with st.sidebar:
    st.header("Status")
    st.write("Säsongskod:", SEASON)

st.sidebar.write("Hämtar data…")
files = download_files(tuple(LEAGUES), SEASON)
st.sidebar.write("Filer klara:", len(files))

df_raw = load_all_data(files)
if df_raw.empty:
    st.error("Ingen data nedladdad. Kontrollera att football-data.co.uk är uppe eller testa senare.")
    st.stop()

df_prep, feat_cols = prepare_features(df_raw)
if not feat_cols:
    st.error("Kunde inte förbereda features (saknas FTR eller bas-kolumner).")
    st.stop()

# Modell-cache signatur
latest_ts = int(pd.to_datetime(df_prep["Date"], errors="coerce").max().timestamp()) if "Date" in df_prep.columns else 0
df_signature = (len(df_prep), latest_ts)

model = load_or_train_model(df_signature, df_prep, feat_cols)

# BYGG ALLTID FÄRSK LAGLISTA baserat på CSV-innehåll + normalisering
files_sig = _league_signature(files)
teams_all = load_or_create_team_labels(df_raw, LEAGUES, files_sig)

if not teams_all:
    st.warning("Kunde inte skapa laglistan. Kontrollera att minst en match finns i varje liga.")
    st.stop()

with st.sidebar:
    # Debug-info
    e2_home = set(df_raw.loc[df_raw["League"]=="E2","HomeTeam"].dropna().astype(str))
    e2_away = set(df_raw.loc[df_raw["League"]=="E2","AwayTeam"].dropna().astype(str))
    e2_teams = sorted(e2_home | e2_away)
    st.write("Lag:", len(teams_all))
    st.write("OPENAI:", "OK" if st.secrets.get("OPENAI_API_KEY") else "—")
    with st.expander("E2-lag (rådata, normaliserat)", expanded=False):
        st.write(", ".join(e2_teams))

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
if st.button("Tippa matcher", use_container_width=True):
    rows, match_probs, tecken_list, match_meta = [], [], [], []

    # Beräkna sannolikheter + samla meta
    for (home, away) in match_pairs:
        home_team, home_lg = home.rsplit(" (",1)
        away_team, away_lg = away.rsplit(" (",1)
        home_team, home_lg = home_team.strip(), home_lg.strip(")")
        away_team, away_lg = away_team.strip(), away_lg.strip(")")

        # Normalisera team vid uppslag (säkerställer att alias matchar features)
        home_team = normalize_team_name(home_team)
        away_team = normalize_team_name(away_team)

        h_row = df_prep[(df_prep["League"]==home_lg) & (df_prep["HomeTeam"]==home_team)].tail(1)
        a_row = df_prep[(df_prep["League"]==away_lg) & (df_prep["AwayTeam"]==away_team)].tail(1)

        if h_row.empty or a_row.empty:
            match_probs.append(np.array([0.0,0.0,0.0]))
            match_meta.append((home_team, away_team, f"{home_lg}", 0,0,0,0,0,0))
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
        match_meta.append((home_team, away_team, home_lg, *features))

    # Välj halvgarderingar
    def pick_half_guards(match_probs, n_half):
        if n_half <= 0:
            return set()
        margins = []
        for i, p in enumerate(match_probs):
            if p is None or len(p) != 3 or np.sum(p) == 0:
                margins.append((i, 1.0))
                continue
            s = np.sort(p)
            margin = s[-1] - s[-2]
            margins.append((i, margin))
        margins.sort(key=lambda x: x[1])
        return {i for i,_ in margins[:n_half]}

    def halfguard_sign(probs):
        idxs = np.argsort(probs)[-2:]
        idxs = tuple(sorted(map(int, idxs)))
        mapping = {(0,1): "1X", (0,2): "12", (1,2): "X2"}
        return mapping.get(idxs, "1X")

    half_idxs = pick_half_guards(match_probs, n_half)

    # Tabell + tipsrad
    for idx, ((home_label, away_label), probs, meta) in enumerate(zip(match_pairs, match_probs, match_meta), start=1):
        if probs is None or len(probs)!=3 or float(np.sum(probs)) == 0.0:
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
                if match_probs[i-1] is None or np.sum(match_probs[i-1]) == 0:
                    st.markdown(f"**{i}) {home_team} ({lg}) - {away_team}**\n*(Ingen data till analys)*")
                    continue
                p1, px, p2 = match_probs[i-1]
                try:
                    summary = gpt_match_brief(
                        client, home_team, away_team, lg,
                        hfp, hfgd, afp, afgd, helo, aelo, p1, px, p2
                    )
                except Exception as e:
                    summary = f"(Ingen GPT-analys: {e})"
                st.markdown(f"**{i}) {home_team} ({lg}) - {away_team}**")
                st.write(summary)
