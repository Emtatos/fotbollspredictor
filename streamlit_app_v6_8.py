# streamlit_app_v6_8.py
# v6.8 — Stabil laglista + normalisering + auto-refresh + manuell tipsrad
# - Laddar E0–E2 (Premier, Championship, League One) från football-data.co.uk
# - Namn-normalisering (Bradford/Bradford C → Bradford City, osv)
# - Robust laglista som byggs från rådata (rensar gamla cachefiler när CSV ändras)
# - Timeout/retries vid nedladdning
# - Manuell tipsrad (13 rader) med valfria ligataggar "(E0)/(E1)/(E2)" var som helst på raden
# - Fallback till GPT-kortanalys om en match saknar data i E0–E2
# - SÄKER hantering av hemligheter: inga direkta anrop till st.secrets

from __future__ import annotations

import os
import re
import json
import time
import glob
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------- OpenAI (valfritt) ----------
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# =======================
#   Grundinställningar
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

def _season_code() -> str:
    """Returnerar mmz4281-säsongskoden (ex 2425)."""
    y = datetime.now().year % 100
    prev = y - 1
    return f"{prev:02d}{y:02d}"

SEASON = _season_code()

# =======================
#   Hjälpfunktioner
# =======================
def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def _norm_space(s: str) -> str:
    return " ".join(str(s).strip().split())

def _unify_dash(s: str) -> str:
    # Tolka – (en-dash), — (em-dash) och − (minus) som bindestreck
    return re.sub(r"[–—−]", "-", s)

# Football-Data → standardnamn (inkl. vanliga kortformer)
TEAM_ALIASES = {
    "Bradford": "Bradford City",
    "Bradford C": "Bradford City",
    "Bradford City": "Bradford City",

    "Cardiff": "Cardiff City",
    "Cardiff C": "Cardiff City",
    "Cardiff City": "Cardiff City",

    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",

    "Sheffield Wed": "Sheffield Wednesday",
    "Sheff Wed": "Sheffield Wednesday",
    "Sheffield Utd": "Sheffield United",
    "Sheff Utd": "Sheffield United",

    "QPR": "Queens Park Rangers",
    "Queens Park Rangers": "Queens Park Rangers",

    "MK Dons": "Milton Keynes Dons",

    # Wolves / Forest-varianter
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Wolverhampton W": "Wolverhampton Wanderers",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",

    "Nott'm Forest": "Nottingham Forest",
    "Nottm Forest": "Nottingham Forest",
    "Nottingham F": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Nottingham Forest": "Nottingham Forest",
}

def normalize_team_name(raw: str) -> str:
    s = _norm_space(raw)
    if s in TEAM_ALIASES:
        return TEAM_ALIASES[s]
    if s.endswith(" FC"):
        s = s[:-3]
    s = s.replace(" C.", " C")
    return TEAM_ALIASES.get(s, s)

def _safe_secret(key: str) -> Optional[str]:
    """
    SÄKER hemlighetshämtning.
    - Försök env först (Render/Heroku m.fl.)
    - Försök st.secrets endast om det finns och är laddat, använd 'in' för att undvika parse-fel.
    """
    val = os.getenv(key)
    if val:
        return val
    try:
        if hasattr(st, "secrets") and (key in st.secrets):
            return st.secrets[key]
    except Exception:
        pass
    return None

def _has_openai_key() -> bool:
    return bool(_HAS_OPENAI and _safe_secret("OPENAI_API_KEY"))

# =======================
#   HTTP med retries
# =======================
SESSION = requests.Session()
ADAPTER = requests.adapters.HTTPAdapter(max_retries=3)
SESSION.mount("https://", ADAPTER)
SESSION.mount("http://", ADAPTER)

def _http_get(url: str, timeout: float = 10.0) -> Optional[bytes]:
    try:
        r = SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

# =======================
#   Nedladdning & laddning
# =======================
@st.cache_data(ttl=6*3600, show_spinner=True)
def _download_one(league: str, s_code: str) -> Optional[str]:
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
def download_files(leagues=tuple(LEAGUES), s_code: str = SEASON) -> Tuple[str, ...]:
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
def load_all_data(files: Tuple[str, ...]) -> pd.DataFrame:
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="latin1")
            league = os.path.basename(f).split("_")[0]
            df["League"] = league
            for col in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]:
                if col not in df.columns:
                    df[col] = np.nan
            df["HomeTeam"] = df["HomeTeam"].astype(str).apply(normalize_team_name)
            df["AwayTeam"] = df["AwayTeam"].astype(str).apply(normalize_team_name)
            dfs.append(df)
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# =======================
#   Features: form + ELO
# =======================
def calculate_5match_form(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values(["Date"]).reset_index(drop=True)

    home_pts = defaultdict(lambda: deque([], maxlen=5))
    home_gd  = defaultdict(lambda: deque([], maxlen=5))
    away_pts = defaultdict(lambda: deque([], maxlen=5))
    away_gd  = defaultdict(lambda: deque([], maxlen=5))

    df["HomeFormPts5"] = 0.0
    df["HomeFormGD5"]  = 0.0
    df["AwayFormPts5"] = 0.0
    df["AwayFormGD5"]  = 0.0

    for i, row in df.iterrows():
        home, away = row.get("HomeTeam", ""), row.get("AwayTeam", "")
        fthg, ftag, ftr = row.get("FTHG", 0), row.get("FTAG", 0), row.get("FTR", "D")

        if len(home_pts[home]) > 0:
            df.at[i, "HomeFormPts5"] = float(np.mean(home_pts[home]))
            df.at[i, "HomeFormGD5"]  = float(np.mean(home_gd[home]))
        if len(away_pts[away]) > 0:
            df.at[i, "AwayFormPts5"] = float(np.mean(away_pts[away]))
            df.at[i, "AwayFormGD5"]  = float(np.mean(away_gd[away]))

        hp, ap = (3, 0) if ftr == "H" else (1, 1) if ftr == "D" else (0, 3)
        gd_home, gd_away = fthg - ftag, ftag - fthg
        home_pts[home].append(hp); home_gd[home].append(gd_home)
        away_pts[away].append(ap);  away_gd[away].append(gd_away)

    return df

def compute_elo(df: pd.DataFrame, K: float = 20) -> pd.DataFrame:
    elo = defaultdict(lambda: 1500.0)
    df = df.copy()
    df["HomeElo"] = 1500.0
    df["AwayElo"] = 1500.0

    for i, row in df.iterrows():
        home, away = row.get("HomeTeam", ""), row.get("AwayTeam", "")
        ftr = row.get("FTR", "D")
        Ra, Rb = elo[home], elo[away]
        Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
        Sa = 1 if ftr == "H" else 0.5 if ftr == "D" else 0
        Sb = 1 - Sa
        elo[home] = Ra + K * (Sa - Ea)
        elo[away] = Rb + K * (Sb - (1 - Ea))
        df.at[i, "HomeElo"], df.at[i, "AwayElo"] = elo[home], elo[away]
    return df

@st.cache_data(ttl=6*3600, show_spinner=False)
def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if df.empty:
        return df, []
    df = df.dropna(subset=["FTR"])
    df = calculate_5match_form(df)
    df = compute_elo(df)
    feature_cols = ["HomeFormPts5", "HomeFormGD5", "AwayFormPts5", "AwayFormGD5", "HomeElo", "AwayElo"]
    return df, feature_cols

# =======================
#   Modell (XGBoost)
# =======================
def _quick_train(df: pd.DataFrame, feature_cols: List[str]) -> XGBClassifier:
    X = df[feature_cols].fillna(0.0)
    y = df["FTR"].map({"H": 0, "D": 1, "A": 2}).astype(int)

    if len(X) < 100:
        params = dict(n_estimators=120, max_depth=4, learning_rate=0.15, subsample=1.0, reg_lambda=1.0)
    else:
        params = dict(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.9, reg_lambda=1.0)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=min(0.2, max(0.1, 200/len(X))) if len(X) > 200 else 0.2,
        random_state=42, stratify=y
    )
    model = XGBClassifier(**params, objective="multi:softprob", num_class=3, n_jobs=1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

@st.cache_resource(show_spinner=True)
def load_or_train_model(df_signature: Tuple[int, int] | None,
                        df: pd.DataFrame,
                        feature_cols: List[str]) -> XGBClassifier:
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

def predict_probs(model: XGBClassifier, features: List[float], feature_cols: List[str]) -> np.ndarray:
    X = pd.DataFrame([features], columns=feature_cols)
    return model.predict_proba(X)[0]

# =======================
#   Laglista (alltid färsk)
# =======================
def _league_signature(files: Tuple[str, ...]) -> str:
    parts = []
    for p in files:
        try:
            parts.append(f"{os.path.basename(p)}:{_hash_file(p)}:{os.path.getsize(p)}:{int(os.path.getmtime(p))}")
        except Exception:
            parts.append(os.path.basename(p))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

def build_team_labels(df_raw: pd.DataFrame, leagues: List[str]) -> List[str]:
    pairs = set()
    for lg in leagues:
        sub = df_raw[df_raw["League"] == lg]
        teams = set(sub["HomeTeam"].dropna()) | set(sub["AwayTeam"].dropna())
        for t in teams:
            t = normalize_team_name(t)
            if t:
                pairs.add((t, lg))
    labels = [f"{t} ({lg})" for (t, lg) in pairs]
    return sorted(labels, key=lambda s: s.lower())

def load_or_create_team_labels(df_raw: pd.DataFrame, leagues: List[str], files_sig: str) -> List[str]:
    teams_json = os.path.join(DATA_DIR, f"teams_{SEASON}_{files_sig}.json")

    # Städa gamla filer för samma säsong
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
#   OpenAI (fredagsanalys)
# =======================
def get_openai_client():
    if not _HAS_OPENAI:
        return None, "openai-biblioteket saknas (lägg till 'openai' i requirements.txt)."
    api_key = _safe_secret("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY saknas (lägg i Render Environment Variables eller .streamlit/secrets.toml)."
    try:
        client = OpenAI(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"Kunde inte initiera OpenAI-klient: {e}"

def gpt_match_brief(client,
                    home, away, league,
                    h_form_pts, h_form_gd, a_form_pts, a_form_gd, h_elo, a_elo,
                    p1, px, p2) -> str:
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
                {"role": "system", "content": "Du skriver kort, sakligt och utan spekulationer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Ingen GPT-analys: {e})"

def gpt_predict_for_unknown(client, home: str, away: str) -> str:
    """Minimal fallback-analys när vi saknar data för matchen."""
    try:
        prompt = (
            f"Vi saknar E0–E2-data för {home} - {away}. "
            "Ge en kort försiktig bedömning i 2–3 meningar (hemmaplan liten fördel, oavgjort vanligt, osv). "
            "Inga djärva påståenden, inga skador eller nyheter."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Ingen GPT-fallback: {e})"

# =======================
#   Manuell tipsrad: tolkare
# =======================
LEAGUE_TAG_RE = re.compile(r"\((E0|E1|E2)\)", flags=re.IGNORECASE)

def _extract_league_tag(text: str) -> Tuple[str, Optional[str]]:
    """Plocka ut valfri ligatagg (E0/E1/E2) var som helst i texten."""
    m = LEAGUE_TAG_RE.search(text)
    league = None
    if m:
        league = m.group(1).upper()
        text = LEAGUE_TAG_RE.sub("", text)
    return _norm_space(text), league

def parse_manual_lines(s: str, expected_n: int) -> List[Tuple[str, str, Optional[str]]]:
    """
    Exempel:
      "Fulham - Brentford"
      "Man United (E0) - Chelsea"
      "Derby - Preston (E1)"
    Returnerar [(home, away, league_or_None), ...], endast upp till expected_n icke-tomma rader.
    """
    out = []
    for raw in s.splitlines():
        line = _unify_dash(raw.strip())  # <— viktig fix: tolka –/—/− som '-'
        if not line:
            continue
        line, tag = _extract_league_tag(line)
        if "-" not in line:
            continue
        h, a = line.split("-", 1)
        home, away = normalize_team_name(h), normalize_team_name(a)
        if home and away and home != away:
            out.append((home, away, tag))
        if len(out) >= expected_n:
            break
    return out

# =======================
#   UI – sidomeny
# =======================
with st.sidebar:
    st.header("Status")
    st.write("Säsongskod:", SEASON)

files = download_files(tuple(LEAGUES), SEASON)
df_raw = load_all_data(files)

# Siffror i sidomeny
with st.sidebar:
    st.write("Filer klara:", len(files))
    st.write("Rader i data:", len(df_raw))
    all_teams = (set(df_raw["HomeTeam"].dropna()) | set(df_raw["AwayTeam"].dropna())) if not df_raw.empty else set()
    st.write("Lag (alla E0–E2):", len(all_teams))
    st.write("OPENAI:", "OK" if _has_openai_key() else "—")

if df_raw.empty:
    st.error("Ingen data nedladdad. Kontrollera football-data.co.uk eller testa senare.")
    st.stop()

df_prep, feat_cols = prepare_features(df_raw)
if not feat_cols:
    st.error("Kunde inte förbereda features (saknas FTR eller bas-kolumner).")
    st.stop()

# Robust latest_ts (NaT-säker)
latest_dt = pd.to_datetime(df_prep["Date"], errors="coerce").max() if "Date" in df_prep.columns else None
latest_ts = int(latest_dt.timestamp()) if (latest_dt is not None and pd.notna(latest_dt)) else 0
df_signature = (len(df_prep), latest_ts)
model = load_or_train_model(df_signature, df_prep, feat_cols)

files_sig = _league_signature(files)
teams_all = load_or_create_team_labels(df_raw, LEAGUES, files_sig)
if not teams_all:
    st.warning("Kunde inte skapa laglistan. Kontrollera att minst en match finns i varje liga.")
    st.stop()

# Extra sidomeny-info (rålaglista E2 + filer)
with st.sidebar:
    with st.expander("E2-lag (rådata, normaliserat)", expanded=False):
        e2_home = set(df_raw.loc[df_raw["League"] == "E2", "HomeTeam"].dropna().astype(str))
        e2_away = set(df_raw.loc[df_raw["League"] == "E2", "AwayTeam"].dropna().astype(str))
        e2_teams = sorted({normalize_team_name(x) for x in (e2_home | e2_away)})
        st.write(", ".join(e2_teams))
    st.subheader("Filer i data/")
    try:
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        if csv_files:
            for f in csv_files:
                st.write(os.path.basename(f), f"({os.path.getsize(f)} bytes)")
        else:
            st.write("❌ Inga CSV-filer hittades i data/-mappen.")
    except Exception as e:
        st.write("Fel vid listning av data/-mappen:", e)

st.divider()
st.caption("Underhåll")
reset_model = st.checkbox(
    "Nollställ modell-cache också",
    value=False,
    help="Kryssa i om du vill tvinga omträning och rensa models/-filen."
)

if st.button("↻ Ladda om CSV-filer", use_container_width=True,
             help="Rensar data/-CSV-filer, tömmer cache och hämtar om E0–E2."):
    try:
        # 1) Ta bort gamla CSV:er
        removed = 0
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        for f in glob.glob(os.path.join(DATA_DIR, "*.csv")):
            try:
                os.remove(f)
                removed += 1
            except Exception:
                pass

        # 2) Rensa Streamlit-cachar
        st.cache_data.clear()
        if reset_model:
            st.cache_resource.clear()
            try:
                if os.path.exists(MODEL_FILE):
                    os.remove(MODEL_FILE)
            except Exception:
                pass

        # 3) Hämta om
        new_paths = download_files(tuple(LEAGUES), SEASON)

        # 4) Återrapport
        st.success(f"Klart! Tog bort {removed} filer och laddade ner {len(new_paths)} nya.")
        if new_paths:
            st.write("Nedladdade:")
            for p in new_paths:
                st.write("•", os.path.basename(p))

        st.rerun()
    except Exception as e:
        st.error(f"Något gick fel vid omladdning: {e}")

# =======================
#   Huvud – inputs
# =======================
n_matches = st.number_input("Antal matcher att tippa", min_value=1, max_value=13, value=13, step=1)
n_half = st.number_input("Antal halvgarderingar", min_value=0, max_value=int(n_matches), value=0, step=1)

st.markdown("### Förifyll tipsrad (manuellt)")
st.caption('Klistra in **13 rader** (valfri liga tillåten). Format-exempel:'
           '\n`Arsenal - Everton` eller `Arsenal (E0) - Everton (E0)` eller `AIK - Hammarby`.'
           '\nOm en match saknar data i E0–E2 används GPT-fallback (om nyckel finns).')

default_13 = (
    "Fulham - Brentford\n"
    "Man United - Chelsea\n"
    "Brighton - Tottenham\n"
    "West Ham - Crystal Palace\n"
    "Wolverhampton - Leeds\n"
    "Burnley - Nottingham\n"
    "Blackburn - Ipswich\n"
    "Derby - Preston\n"
    "Hull - Southampton\n"
    "Norwich - Wrexham\n"
    "Leicester - Middlesbrough\n"
    "Cardiff - Sunderland\n"
    "QPR - Stoke\n"
)

manual_text = st.text_area("Klistra in 13 rader (valfri liga tillåten).", value=default_13, height=180)
manual_pairs = parse_manual_lines(manual_text, expected_n=int(n_matches))
if len(manual_pairs) == int(n_matches):
    st.success(f"Upptäckte {len(manual_pairs)} manuella rader. Dessa används.")
else:
    st.info(f"Upptäckte {len(manual_pairs)} manuella rader (av {n_matches}). Tomma/felaktiga rader ignoreras.")

# =======================
#   Match → features
# =======================
def _latest_rows_for_match(df_feat: pd.DataFrame,
                           home: str, away: str,
                           league_hint: Optional[str]) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[str]]:
    """
    Hitta senaste rader (en rad vardera) för hemmasidan resp. bortasidan.
    Om league_hint (E0/E1/E2) ges filtrerar vi mot den ligan.
    Returnerar (h_row, a_row, league_used)
    """
    df = df_feat
    if league_hint:
        df = df[df["League"] == league_hint]

    h_row = df[(df["HomeTeam"] == home)].sort_values("Date").tail(1)
    a_row = df[(df["AwayTeam"] == away)].sort_values("Date").tail(1)

    if (h_row.empty or a_row.empty) and league_hint:
        df2 = df_feat
        h2 = df2[(df2["HomeTeam"] == home)].sort_values("Date").tail(1)
        a2 = df2[(df2["AwayTeam"] == away)].sort_values("Date").tail(1)
        if not h2.empty and not a2.empty:
            return h2.iloc[0], a2.iloc[0], league_hint
    if not h_row.empty and not a_row.empty:
        used = league_hint if league_hint else (h_row.iloc[0]["League"] if pd.notna(h_row.iloc[0]["League"]) else None)
        return h_row.iloc[0], a_row.iloc[0], used
    return None, None, league_hint

def _pick_half_guards(match_probs: List[Optional[np.ndarray]], n_half: int) -> set:
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
    return {i for i, _ in margins[:n_half]}

def _halfguard_sign(probs: np.ndarray) -> str:
    idxs = np.argsort(probs)[-2:]
    idxs = tuple(sorted(map(int, idxs)))
    mapping = {(0, 1): "1X", (0, 2): "12", (1, 2): "X2"}
    return mapping.get(idxs, "1X")

# =======================
#   Körning
# =======================
if st.button("Tippa matcher", use_container_width=True):
    rows = []
    match_probs: List[Optional[np.ndarray]] = []
    tecken_list: List[str] = []
    match_meta = []  # (home, away, league_used, hfp, hfgd, afp, afgd, helo, aelo)

    if len(manual_pairs) == int(n_matches):
        pairs_to_use = manual_pairs
    else:
        pairs_to_use = []  # användaren får korrigera texten

    for (home, away, lg_hint) in pairs_to_use:
        h_row, a_row, used_lg = _latest_rows_for_match(df_prep, home, away, lg_hint)

        if (h_row is None) or (a_row is None):
            match_probs.append(None)
            match_meta.append((home, away, used_lg or "—", 0, 0, 0, 0, 0, 0))
            continue

        features = [
            float(h_row["HomeFormPts5"]),
            float(h_row["HomeFormGD5"]),
            float(a_row["AwayFormPts5"]),
            float(a_row["AwayFormGD5"]),
            float(h_row["HomeElo"]),
            float(a_row["AwayElo"]),
        ]
        probs = predict_probs(model, features, feat_cols)
        match_probs.append(probs)
        match_meta.append(
            (home, away, used_lg or str(h_row.get("League", "—")),
             features[0], features[1], features[2], features[3], features[4], features[5])
        )

    # Halvgarderingar
    half_idxs = _pick_half_guards(match_probs, int(n_half))

    # Tabell
    for idx in range(1, len(pairs_to_use) + 1):
        home_label, away_label, _ = pairs_to_use[idx - 1]
        probs = match_probs[idx - 1]

        if (probs is None) or (len(probs) != 3) or float(np.sum(probs)) == 0.0:
            tecken, pct = "(X)", ""
        else:
            if (idx - 1) in half_idxs:
                tecken, pct = f"({_halfguard_sign(probs)})", "-"
            else:
                pred = int(np.argmax(probs))
                tecken, pct = f"({['1','X','2'][pred]})", f"{probs[pred]*100:.1f}%"

        rows.append([idx, "", f"{home_label} - {away_label}", tecken, "", "", pct])
        tecken_list.append(tecken)

    df_out = pd.DataFrame(rows, columns=["#", "Status", "Match", "Tecken", "Res.", "%", "Stats"])
    st.subheader("Resultat-tabell")
    st.dataframe(df_out, use_container_width=True)

    st.subheader("Tipsrad (kopiera)")
    st.code(" ".join(tecken_list), language=None)

    # Fredagsanalys
    with st.expander("🔮 Fredagsanalys (GPT)"):
        client, err = get_openai_client()
        if err:
            st.warning(err)
        else:
            st.caption("Analysen bygger endast på form/ELO/sannolikheter (inga nyheter för att undvika påhitt).")
            for i, (home_team, away_team, lg, hfp, hfgd, afp, afgd, helo, aelo) in enumerate(match_meta, start=1):
                if (i-1) < len(match_probs) and match_probs[i-1] is not None:
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
                else:
                    if _has_openai_key():
                        fallback = gpt_predict_for_unknown(client, home_team, away_team)
                    else:
                        fallback = "(Saknar data och ingen OPENAI_API_KEY – ingen analys.)"
                    st.markdown(f"**{i}) {home_team} - {away_team}** *(ingen E0–E2-data)*")
                    st.write(fallback)
