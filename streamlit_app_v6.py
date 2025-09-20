# streamlit_app_v6.py
# Version 6: XGBoost + 5-match form + ELO-rating
# För framtida matchprediktioner och bättre träffsäkerhet.

import os
import json
from datetime import datetime
from urllib.request import urlretrieve
from collections import defaultdict, deque

import pandas as pd
import numpy as np
import streamlit as st
import joblib
from filelock import FileLock

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# -----------------------
# Filstruktur
# -----------------------
DATA_DIR = "data"
MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "model_v6.pkl")
TIP_LOG = os.path.join(LOG_DIR, "tips.csv")
RESULT_LOG = os.path.join(LOG_DIR, "results.csv")

# -----------------------
# Datahämtning
# -----------------------
BASE_URL = "https://www.football-data.co.uk/mmz4281"

def season_code():
    y = datetime.now().year % 100
    prev = y - 1
    return f"{prev:02d}{y:02d}"

def download_files(leagues=("E0","E1"), force=False):
    s = season_code()
    got = []
    for L in leagues:
        target = os.path.join(DATA_DIR, f"{L}_{s}.csv")
        if os.path.exists(target) and not force:
            got.append(target)
            continue
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

# -----------------------
# Feature engineering
# -----------------------
def calculate_5match_form(df):
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    home_pts, home_gd = defaultdict(lambda: deque([], maxlen=5)), defaultdict(lambda: deque([], maxlen=5))
    away_pts, away_gd = defaultdict(lambda: deque([], maxlen=5)), defaultdict(lambda: deque([], maxlen=5))

    df["HomeFormPts5"], df["HomeFormGD5"], df["AwayFormPts5"], df["AwayFormGD5"] = 0,0,0,0

    for i, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        fthg, ftag, ftr = row["FTHG"], row["FTAG"], row["FTR"]

        # före match
        if len(home_pts[home]) > 0:
            df.at[i,"HomeFormPts5"] = np.mean(home_pts[home])
            df.at[i,"HomeFormGD5"] = np.mean(home_gd[home])
        if len(away_pts[away]) > 0:
            df.at[i,"AwayFormPts5"] = np.mean(away_pts[away])
            df.at[i,"AwayFormGD5"] = np.mean(away_gd[away])

        # efter match
        hp, ap = (3,0) if ftr=="H" else (1,1) if ftr=="D" else (0,3)
        gd_home, gd_away = fthg - ftag, ftag - fthg
        home_pts[home].append(hp); home_gd[home].append(gd_home)
        away_pts[away].append(ap); away_gd[away].append(gd_away)

    return df

def compute_elo(df, K=20):
    elo = defaultdict(lambda: 1500)
    df = df.copy()
    df["HomeElo"], df["AwayElo"] = 1500, 1500

    for i,row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        ftr = row["FTR"]
        Ra, Rb = elo[home], elo[away]
        Ea = 1/(1+10**((Rb-Ra)/400))
        Eb = 1-Ea
        Sa = 1 if ftr=="H" else 0.5 if ftr=="D" else 0
        Sb = 1-Sa
        elo[home] = Ra + K*(Sa-Ea)
        elo[away] = Rb + K*(Sb-Eb)
        df.at[i,"HomeElo"], df.at[i,"AwayElo"] = elo[home], elo[away]
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

    mapping = {"H":0, "D":1, "A":2}
    df["ResultLabel"] = df["FTR"].map(mapping)

    feature_cols = ["HomeFormPts5","HomeFormGD5","AwayFormPts5","AwayFormGD5","HomeElo","AwayElo"]
    return df, feature_cols

# -----------------------
# Modellträning
# -----------------------
def train_model(df, feature_cols):
    X = df[feature_cols].fillna(0)
    y = df["ResultLabel"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    param_dist = {
        "n_estimators":[100,200],
        "max_depth":[3,5,7],
        "learning_rate":[0.05,0.1,0.2],
        "subsample":[0.8,1.0]
    }
    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    search = RandomizedSearchCV(model, param_dist, n_iter=5, cv=3, n_jobs=-1, random_state=42)
    search.fit(X_train,y_train)
    best = search.best_estimator_
    acc = accuracy_score(y_test, best.predict(X_test))
    joblib.dump(best, MODEL_FILE)
    st.success(f"XGBoost tränad. Test-accuracy: {acc:.2%}")
    return best

def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

def predict_probs(model, features, feature_cols):
    X = pd.DataFrame([features], columns=feature_cols)
    probs = model.predict_proba(X)[0]
    return {"1": float(probs[0]), "X": float(probs[1]), "2": float(probs[2])}

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Fotboll v6", layout="wide")
st.title("⚽ Fotboll v6 — XGBoost + 5-match form + ELO")

col1,col2 = st.columns([1,2])
with col1:
    leagues = st.multiselect("Välj ligor",["E0","E1","E2"],default=["E0","E1"])
    if st.button("🔄 Ladda ner data"):
        files = download_files(tuple(leagues),force=True)
        st.success(f"Hämtade {len(files)} filer.")
    else:
        files = download_files(tuple(leagues))

with col2:
    df = load_all_data(files)
    if df.empty:
        st.warning("Ingen data.")
    else:
        df_prep, feat_cols = prepare_features(df)
        st.write(f"Matcher i datasetet: {len(df_prep)}")
        model = load_model()
        if model is None:
            if st.button("🔧 Träna XGBoost-modell"):
                model = train_model(df_prep, feat_cols)
        else:
            st.success("✅ Modell laddad")

# Framtida matcher
st.header("🔮 Framtida matcher")
if df is not None and not df.empty and 'model' in locals() and model is not None:
    teams = sorted(set(df["HomeTeam"].dropna().unique()).union(df["AwayTeam"].dropna().unique()))
    n = st.number_input("Antal matcher",1,13,3)
    matches = []
    for i in range(n):
        c1,c2 = st.columns(2)
        home = c1.selectbox(f"Hemmalag {i+1}",teams,key=f"h_{i}")
        away = c2.selectbox(f"Bortalag {i+1}",teams,key=f"a_{i}")
        if home!=away:
            matches.append((home,away))

    if st.button("Beräkna sannolikheter"):
        df_prep, feat_cols = prepare_features(df)
        latest = df_prep.sort_values("Date").groupby("HomeTeam").tail(1)
        for i,(home,away) in enumerate(matches,1):
            h_row = df_prep[df_prep["HomeTeam"]==home].tail(1)
            a_row = df_prep[df_prep["AwayTeam"]==away].tail(1)
            if h_row.empty or a_row.empty: 
                st.write(f"Match {i}: {home}-{away}: ingen data")
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
            st.write(f"Match {i}: {home} - {away}")
            st.json(probs)
