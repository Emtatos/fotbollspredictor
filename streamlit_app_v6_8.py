# streamlit_app_v6_8.py
# v6.8 — Stabil laglista + normalisering + auto-refresh + MANUELL TIPSRAD (utan ligataggar)
# - Laddar E0–E2 (Premier, Championship, League One)
# - Namn-normalisering (Bradford/Bradford C -> Bradford City, osv)
# - Låser upp-/nedflyttningar via rådatan (E0–E2)
# - Timeout/retries vid nedladdning
# - Valfri manuell tipsrad: "Fulham - Brentford" räcker; ligor gissas automatiskt
# - Matcher utanför E0–E2 markeras (EXT) och hanteras av GPT-kommentar (ingen modellprocent)
# - GPT "fredagsanalys" möjlig (om OPENAI_API_KEY finns)

from __future__ import annotations

import os
import json
import time
import re
import hashlib
from datetime import datetime
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import requests
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------- OpenAI (valfritt) ----------
try:
    from openai import OpenAI  # openai>=1.x
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


def season_code() -> str:
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
TEAM_ALIASES: dict[str, str] = {
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
    "MK Dons": "Milton Keynes Dons",
}


def normalize_team_name(raw: str) -> str:
    s = _norm_space(raw)
    if s in TEAM_ALIASES:
        return TEAM_ALIASES[s]
    if s.endswith(" FC"):
        s = s[:-3]
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


@st.cache_data(ttl=6 * 3600, show_spinner=True)
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


@st.cache_data(ttl=6 * 3600, show_spinner=True)
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


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_all_data(files: tuple[str, ...]) -> pd.DataFrame:
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
# Features: form + ELO
# =======================
def calculate_5match_form(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values(["Date"]).reset_index(drop=True)

    home_pts = defaultdict(lambda: deque([], maxlen=5))
    home_gd = defaultdict(lambda: deque([], maxlen=5))
    away_pts = defaultdict(lambda: deque([], maxlen=5))
    away_gd = defaultdict(lambda: deque([], maxlen=5))

    df["HomeFormPts5"] = 0.0
    df["HomeFormGD5"] = 0.0
    df["AwayFormPts5"] = 0.0
    df["AwayFormGD5"] = 0.0

    for i, row in df.iterrows():
        home, away = row.get("HomeTeam", ""), row.get("AwayTeam", "")
        fthg, ftag, ftr = row.get("FTHG", 0), row.get("FTAG", 0), row.get("FTR", "D")

        if len(home_pts[home]) > 0:
            df.at[i, "HomeFormPts5"] = float(np.mean(home_pts[home]))
            df.at[i, "HomeFormGD5"] = float(np.mean(home_gd[home]))
        if len(away_pts[away]) > 0:
            df.at[i, "AwayFormPts5"] = float(np.mean(away_pts[away]))
            df.at[i, "AwayFormGD5"] = float(np.mean(away_gd[away]))

        hp, ap = (3, 0) if ftr == "H" else (1, 1) if ftr == "D" else (0, 3)
        gd_home, gd_away = fthg - ftag, ftag - fthg
        home_pts[home].append(hp)
        home_gd[home].append(gd_home)
        away_pts[away].append(ap)
        away_gd[away].append(gd_away)

    return df


def compute_elo(df: pd.DataFrame, K: float = 20.0) -> pd.DataFrame:
    elo = defaultdict(lambda: 1500.0)
    df = df.copy()
    df["HomeElo"], df["AwayElo"] = 1500.0, 1500.0

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


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def prepare_features(df: pd.DataFrame):
    if df.empty:
        return df, []
    df = df.dropna(subset=["FTR"])
    df = calculate_5match_form(df)
    df = compute_elo(df)
    feature_cols = [
        "HomeFormPts5",
        "HomeFormGD5",
        "AwayFormPts5",
        "AwayFormGD5",
        "HomeElo",
        "AwayElo",
    ]
    return df, feature_cols


# =======================
# Modell
# =======================
def _quick_train(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].fillna(0.0)
    y = df["FTR"].map({"H": 0, "D": 1, "A": 2}).astype(int)

    if len(X) < 100:
        params = dict(n_estimators=120, max_depth=4, learning_rate=0.15, subsample=1.0, reg_lambda=1.0)
    else:
        params = dict(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.9, reg_lambda=1.0)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=min(0.2, max(0.1, 200 / len(X))) if len(X) > 200 else 0.2,
        random_state=42, stratify=y,
    )
    model = XGBClassifier(**params, objective="multi:softprob", num_class=3, n_jobs=1)
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
# Laglista och signatur
# =======================
def _league_signature(files: tuple[str, ...]) -> str:
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
    teams_json = os.path.join(DATA_DIR, f"teams_{SEASON}_{files_sig}.json")
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
    api_key = (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY saknas (lägg in i Render Environment eller Streamlit Secrets)."
    try:
        client = OpenAI(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"Kunde inte initiera OpenAI-klient: {e}"


def gpt_match_brief(client, home, away, league, h_form_pts, h_form_gd, a_form_pts, a_form_gd, h_elo, a_elo, p1, px, p2):
    prompt = (
        f"Du är en sportanalytiker. Ge en kort briefing inför matchen {home} - {away} i {league}.\n"
        "Använd endast siffrorna nedan (inga påhittade nyheter eller skador):\n"
        f"- Hemma form (5): poäng {h_form_pts:.2f}, målskillnad {h_form_gd:.2f}\n"
        f"- Borta form (5): poäng {a_form_pts:.2f}, målskillnad {a_form_gd:.2f}\n"
        f"- ELO: {home} {h_elo:.1f}, {away} {a_elo:.1f}\n"
        f"- Modellens sannolikheter: 1={p1:.1%}, X={px:.1%}, 2={p2:.1%}\n\n"
        "Svara med 3 korta punkter:\n"
        "1) Styrkebalans (ELO) och hemmaprofil.\n"
        "2) Formkurvor (5 matcher) och vad det antyder.\n"
        "3) Kort riskbedömning (t.ex. hög osäkerhet om 2 utfall ligger nära)."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Du skriver kort, sakligt och utan spekulationer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Ingen GPT-analys: {e})"


def gpt_external_brief(client, home, away):
    prompt = (
        "Du är en sportanalytiker. Gör en **väldigt kort** neutral kommentar inför matchen "
        f"{home} - {away} där vi saknar modell- och statistikdata. "
        "Håll dig försiktig: inga rykten, inga skador/övergångar. "
        "Nämn ev. klassisk styrkebild (storklubb vs mindre) och historiska trender om de är allmänt kända, "
        "annars bara att osäkerheten är hög. Max 2 meningar."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Du skriver kort, sakligt och utan spekulationer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Ingen GPT-kommentar: {e})"


# =======================
# Manuell tipsrad – parsing utan ligataggar
# =======================
LG_TAG_RE = re.compile(r"\((E0|E1|E2)\)\s*$", flags=re.IGNORECASE)


def _strip_lg_tag(s: str) -> tuple[str, str | None]:
    s = s.strip()
    m = LG_TAG_RE.search(s)
    if m:
        lg = m.group(1).upper()
        s = LG_TAG_RE.sub("", s).strip()
        return s, lg
    return s, None


def parse_manual_pairs(text: str, team_to_lg: dict[str, str]) -> list[tuple[str, str, bool]]:
    """
    Returnerar lista av (home_label, away_label, is_external)
    - home/away_label i format 'Team (E0/E1/E2)' när lag hittas eller taggas,
      annars 'Team (EXT)'.
    - is_external = True om någon sida blev EXT.
    """
    pairs = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = re.split(r"\s*[-–]\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        left, right = parts[0].strip(), parts[1].strip()

        l_name, l_tag = _strip_lg_tag(left)
        r_name, r_tag = _strip_lg_tag(right)

        l_norm = normalize_team_name(l_name)
        r_norm = normalize_team_name(r_name)

        lg_l = l_tag or team_to_lg.get(l_norm)
        lg_r = r_tag or team_to_lg.get(r_norm)

        left_label = f"{l_norm} ({lg_l})" if lg_l in ("E0", "E1", "E2") else f"{l_norm} (EXT)"
        right_label = f"{r_norm} ({lg_r})" if lg_r in ("E0", "E1", "E2") else f"{r_norm} (EXT)"

        is_external = ("(EXT)" in left_label) or ("(EXT)" in right_label)
        pairs.append((left_label, right_label, is_external))
    return pairs


# =======================
# UI + flöde
# =======================
with st.sidebar:
    st.header("Status")
    st.write("Säsongskod:", SEASON)

files = download_files(tuple(LEAGUES), SEASON)
df_raw = load_all_data(files)
if df_raw.empty:
    st.error("Ingen data nedladdad. Kontrollera att football-data.co.uk är uppe eller testa senare.")
    st.stop()

df_prep, feat_cols = prepare_features(df_raw)
if not feat_cols:
    st.error("Kunde inte förbereda features (saknas FTR eller bas-kolumner).")
    st.stop()

latest_ts = int(pd.to_datetime(df_prep["Date"], errors="coerce").max().timestamp()) if "Date" in df_prep.columns else 0
df_signature = (len(df_prep), latest_ts)
model = load_or_train_model(df_signature, df_prep, feat_cols)

files_sig = _league_signature(files)
teams_all = load_or_create_team_labels(df_raw, LEAGUES, files_sig)

# Bygg uppslag: normaliserat lagnamn -> liga (E0/E1/E2)
TEAM_TO_LG: dict[str, str] = {}
for lg in LEAGUES:
    sub = df_raw[df_raw["League"] == lg]
    for col in ("HomeTeam", "AwayTeam"):
        for t in sub[col].dropna().astype(str):
            name = normalize_team_name(t)
            TEAM_TO_LG[name] = lg

if not teams_all:
    st.warning("Kunde inte skapa laglistan. Kontrollera att minst en match finns i varje liga.")
    st.stop()

with st.sidebar:
    e2_home = set(df_raw.loc[df_raw["League"] == "E2", "HomeTeam"].dropna().astype(str))
    e2_away = set(df_raw.loc[df_raw["League"] == "E2", "AwayTeam"].dropna().astype(str))
    e2_teams = sorted(normalize_team_name(t) for t in (e2_home | e2_away))
    st.write("Filer klara:", len(files))
    st.write("Rader i data:", len(df_prep))
    st.write("Lag (alla E0–E2):", len(teams_all))
    st.write("OPENAI:", "OK" if ((_HAS_OPENAI) and ((hasattr(st, "secrets") and st.secrets.get("OPENAI_API_KEY")) or os.getenv("OPENAI_API_KEY"))) else "—")
    with st.expander("E2-lag (rådata, normaliserat)", expanded=False):
        st.write(", ".join(e2_teams))

# Antal matcher & halvgarderingar
n_matches = st.number_input("Antal matcher att tippa", min_value=1, max_value=13, value=13)
n_half = st.number_input("Antal halvgarderingar", min_value=0, max_value=n_matches, value=0)

# Manuell tipsrad (valfri)
st.subheader("Förifyll tipsrad (manuell)")
st.caption("Klistra in 13 rader (valfri liga tillåten). Format-exempel: Arsenal - Everton  •  Real Madrid (SP) - Barcelona (SP)  •  AIK - Hammarby")

manual_txt = st.text_area(
    "",
    height=180,
    placeholder="Ex:\nFulham - Brentford\nMan United - Chelsea\nBrighton - Tottenham\nWest Ham - Crystal Palace\nWolverhampton - Leeds\nBurnley - Nottingham\nBlackburn - Ipswich\nDerby - Preston\nHull - Southampton\nNorwich - Wrexham\n...",
)

manual_pairs: list[tuple[str, str]] = []
external_mask: list[bool] = []
if manual_txt.strip():
    parsed = parse_manual_pairs(manual_txt, TEAM_TO_LG)
    if parsed:
        manual_pairs = [(hl, al) for (hl, al, _) in parsed][: n_matches]
        external_mask = [is_ext for (_, _, is_ext) in parsed][: n_matches]
        st.success(f"Upptäckte {len(manual_pairs)} manuella rader. Dessa används nedan.")

# Fallback till selectboxar om ingen manuell lista
match_pairs: list[tuple[str, str]] = []
if manual_pairs:
    match_pairs = manual_pairs[: n_matches]
else:
    for i in range(int(n_matches)):
        c1, c2 = st.columns(2)
        home = c1.selectbox(f"Hemmalag {i+1}", teams_all, key=f"h_{i}")
        away = c2.selectbox(f"Bortalag {i+1}", teams_all, key=f"a_{i}")
        if home and away and home != away:
            match_pairs.append((home, away))
    external_mask = [False] * len(match_pairs)

# Kör tipsning
if st.button("Tippa matcher", use_container_width=True):
    rows, match_probs, tecken_list, match_meta = [], [], [], []

    # Hjälpare: välj halvgarderingar på minsta marginal
    def pick_half_guards(all_probs, n_half_):
        if n_half_ <= 0:
            return set()
        margins = []
        for i, p in enumerate(all_probs):
            if p is None or len(p) != 3 or np.sum(p) == 0:
                margins.append((i, 1.0))
                continue
            s = np.sort(p)
            margins.append((i, float(s[-1] - s[-2])))
        margins.sort(key=lambda x: x[1])
        return {i for i, _ in margins[: n_half_]}

    def halfguard_sign(probs):
        idxs = np.argsort(probs)[-2:]
        idxs = tuple(sorted(map(int, idxs)))
        mapping = {(0, 1): "1X", (0, 2): "12", (1, 2): "X2"}
        return mapping.get(idxs, "1X")

    # Beräkna sannolikheter / meta
    for k, (home, away) in enumerate(match_pairs):
        # EXT: utsides ligor -> inga modellprocent
        if home.endswith("(EXT)") or away.endswith("(EXT)") or (k < len(external_mask) and external_mask[k]):
            match_probs.append(np.array([0.0, 0.0, 0.0]))
            # meta
            h_clean = home.replace(" (EXT)", "")
            a_clean = away.replace(" (EXT)", "")
            match_meta.append((h_clean, a_clean, "EXT", 0, 0, 0, 0, 0, 0))
            continue

        # label -> team + liga
        home_team, home_lg = home.rsplit(" (", 1)
        away_team, away_lg = away.rsplit(" (", 1)
        home_team, home_lg = home_team.strip(), home_lg.strip(")")
        away_team, away_lg = away_team.strip(), away_lg.strip(")")

        # Hämta senaste rader för features
        h_row = df_prep[(df_prep["League"] == home_lg) & (df_prep["HomeTeam"] == home_team)].tail(1)
        a_row = df_prep[(df_prep["League"] == away_lg) & (df_prep["AwayTeam"] == away_team)].tail(1)

        if h_row.empty or a_row.empty:
            match_probs.append(np.array([0.0, 0.0, 0.0]))
            match_meta.append((home_team, away_team, home_lg, 0, 0, 0, 0, 0, 0))
            continue

        features = [
            float(h_row["HomeFormPts5"].values[0]),
            float(h_row["HomeFormGD5"].values[0]),
            float(a_row["AwayFormPts5"].values[0]),
            float(a_row["AwayFormGD5"].values[0]),
            float(h_row["HomeElo"].values[0]),
            float(a_row["AwayElo"].values[0]),
        ]
        probs = predict_probs(model, features, feat_cols)
        match_probs.append(probs)
        match_meta.append((home_team, away_team, home_lg, *features))

    half_idxs = pick_half_guards(match_probs, int(n_half))

    # Resultattabell
    for idx, ((home_label, away_label), probs, meta) in enumerate(zip(match_pairs, match_probs, match_meta), start=1):
        if probs is None or len(probs) != 3 or float(np.sum(probs)) == 0.0:
            tecken, pct = "?", ""
        else:
            if (idx - 1) in half_idxs:
                tecken = f"({halfguard_sign(probs)})"
                pct = "-"
            else:
                pred = int(np.argmax(probs))
                tecken = f"({['1', 'X', '2'][pred]})"
                pct = f"{probs[pred] * 100:.1f}%"

        rows.append([idx, "", f"{home_label} - {away_label}", tecken, "", "", pct])
        tecken_list.append(tecken)

    df_out = pd.DataFrame(rows, columns=["#", "Status", "Match", "Tecken", "Res.", "%", "Stats"])
    st.subheader("Resultat-tabell")
    st.dataframe(df_out, use_container_width=True, height=min(600, 46 + 35 * len(rows)))

    st.subheader("Tipsrad (kopiera)")
    st.code(" ".join(tecken_list), language=None)

    # GPT-kommentarer
    client, err = get_openai_client()
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("🔮 GPT-kommentarer för matcher **utan data** (EXT eller saknade features)")
        if err:
            st.warning(err)
        else:
            if st.button("GPT-kommentarer (EXT)"):
                for i, (home_team, away_team, lg, *_) in enumerate(match_meta, start=1):
                    if lg == "EXT":
                        txt = gpt_external_brief(client, home_team, away_team)
                        st.markdown(f"**{i}) {home_team} - {away_team}**  \n{txt}")

    with col_b:
        st.caption("🧪 Fredagsanalys (GPT) för E0–E2 med modellvärden")
        if err:
            st.warning(err)
        else:
            if st.button("GPT-fredagsanalys (E0–E2)"):
                st.caption("Analysen bygger endast på form/ELO/sannolikheter (inga nyheter för att undvika påhitt).")
                for i, (home_team, away_team, lg, hfp, hfgd, afp, afgd, helo, aelo) in enumerate(match_meta, start=1):
                    if lg == "EXT":
                        continue
                    p = match_probs[i - 1] if i - 1 < len(match_probs) else None
                    if p is None or np.sum(p) == 0:
                        st.markdown(f"**{i}) {home_team} ({lg}) - {away_team}**\n*(Ingen data till analys)*")
                        continue
                    p1, px, p2 = p
                    try:
                        summary = gpt_match_brief(client, home_team, away_team, lg, hfp, hfgd, afp, afgd, helo, aelo, p1, px, p2)
                    except Exception as e:
                        summary = f"(Ingen GPT-analys: {e})"
                    st.markdown(f"**{i}) {home_team} ({lg}) - {away_team}**")
                    st.write(summary)
