# streamlit_app_simple.py
# Enkel, komplett Streamlit-app för v5-uppgradering (endast-en-fil)
# Kör: streamlit run streamlit_app_simple.py

import os
import json
from datetime import datetime
from urllib.request import urlretrieve
from filelock import FileLock
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score

# -----------------------
# Filer & mappar
# -----------------------
DATA_DIR = "data"
MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TIP_LOG = os.path.join(LOG_DIR, "tips.csv")
RESULT_LOG = os.path.join(LOG_DIR, "results.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")

# -----------------------
# Hjälpfunktioner
# -----------------------
def season_code():
    y = datetime.now().year % 100
    prev = y - 1
    return f"{prev:02d}{y:02d}"

BASE_URL = "https://www.football-data.co.uk/mmz4281"

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

def read_csv_safe(path):
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise ValueError(f"Kan inte läsa {path}")

def load_all_data(files):
    dfs = []
    for f in files:
        df = read_csv_safe(f)
        league = os.path.basename(f).split("_")[0]
        df["League"] = league
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def safe_match_id(row):
    date = pd.to_datetime(row.get("Date", ""), dayfirst=True, errors="coerce")
    date_str = "unknown-date" if pd.isna(date) else date.strftime("%Y-%m-%d")
    home = str(row.get("HomeTeam","")).replace(" ", "_")
    away = str(row.get("AwayTeam","")).replace(" ", "_")
    league = str(row.get("League","UNK"))
    return f"{date_str}_{home}_{away}_{league}"

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# -----------------------
# Feature-beredning
# -----------------------
def prepare_features(df):
    expect = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","League"]
    for c in expect:
        if c not in df.columns:
            df[c] = np.nan

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["HomeGoals"] = pd.to_numeric(df["FTHG"], errors="coerce").fillna(0)
    df["AwayGoals"] = pd.to_numeric(df["FTAG"], errors="coerce").fillna(0)
    df["GoalDiff"] = df["HomeGoals"] - df["AwayGoals"]

    df = df.dropna(subset=["FTR"], how="all").copy()
    mapping = {"H":0, "D":1, "A":2}
    df["ResultLabel"] = df["FTR"].map(mapping)

    for col in ["HomeGoals", "AwayGoals", "GoalDiff"]:
        df[col + "_norm"] = df.groupby("League")[col].transform(
            lambda x: (x - x.mean()) / (x.std() if x.std()!=0 else 1)
        )

    feature_cols = ["HomeGoals_norm","AwayGoals_norm","GoalDiff_norm"]
    return df, feature_cols

# -----------------------
# Modell
# -----------------------
def train_model(df, feature_cols):
    X = df[feature_cols].fillna(0)
    y = df["ResultLabel"].astype(int)
    if len(X) < 50:
        st.warning("Få träningsrader (<50). Modellen kan bli dålig.")
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

    param_dist = {
        "n_estimators": [50,100,200],
        "max_depth": [5,10,None],
        "min_samples_split": [2,5],
        "min_samples_leaf": [1,2]
    }
    rf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(rf, param_dist, n_iter=6, cv=3, n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump(best, MODEL_FILE)
    st.success(f"Träning klar — test-accuracy: {acc:.2f}. Modell sparad.")
    return best

def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            st.warning("Kunde inte ladda model-filen, raderar och tränar ny vid behov.")
            os.remove(MODEL_FILE)
    return None

def predict_probs(model, feature_vector):
    probs = model.predict_proba([feature_vector])[0]
    classes = model.classes_
    label_map = {0:"1",1:"X",2:"2"}
    return {label_map[c]: float(p) for c,p in zip(classes,probs)}

# -----------------------
# Loggning
# -----------------------
def append_csv_atomic(path, df):
    lock = FileLock(path + ".lock")
    with lock:
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

def log_tip(match_id, home, away, tip, probs):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "match_id": match_id,
        "home": home,
        "away": away,
        "tip": tip,
        "probs": json.dumps(probs)
    }
    append_csv_atomic(TIP_LOG, pd.DataFrame([entry]))

def log_result(match_id, result):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "match_id": match_id,
        "result": result
    }
    append_csv_atomic(RESULT_LOG, pd.DataFrame([entry]))

def read_logs():
    tips = pd.read_csv(TIP_LOG) if os.path.exists(TIP_LOG) else pd.DataFrame()
    results = pd.read_csv(RESULT_LOG) if os.path.exists(RESULT_LOG) else pd.DataFrame()
    return tips, results

def reset_logs():
    for path in (TIP_LOG, RESULT_LOG):
        lock = FileLock(path + ".lock")
        with lock:
            if os.path.exists(path):
                os.remove(path)

# -----------------------
# Resultat-hantering
# -----------------------
def auto_update_results(df):
    tips, results = read_logs()
    if tips.empty:
        return 0
    updated = 0
    existing = set(results["match_id"].tolist()) if not results.empty else set()
    df = df.copy()
    df["Date_str"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
    df["match_id"] = df.apply(safe_match_id, axis=1)
    for _, t in tips.iterrows():
        mid = t["match_id"]
        if mid in existing:
            continue
        found = df[df["match_id"] == mid]
        if not found.empty and "FTR" in found.columns:
            ftr = found.iloc[0]["FTR"]
            if pd.notna(ftr) and ftr in ("H","D","A"):
                res = "1" if ftr=="H" else "X" if ftr=="D" else "2"
                log_result(mid, res)
                updated += 1
    return updated

# -----------------------
# Utvärdering
# -----------------------
def evaluate_performance():
    tips, results = read_logs()
    if tips.empty or results.empty:
        return None, None
    merged = tips.merge(results, on="match_id", how="inner")
    if merged.empty:
        return None, None
    merged["correct"] = merged["tip"] == merged["result"]
    total = len(merged)
    correct = merged["correct"].sum()
    accuracy = correct/total*100
    return merged, {"total": total, "correct": int(correct), "accuracy": float(accuracy)}

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Fotboll v5 - Enkel", layout="wide")
st.title("Fotboll v5 — Enkel uppgradering (endast-en-fil)")

# --- Verktyg / Admin ---
with st.expander("⚙️ Verktyg"):
    st.write("Snabbverktyg för nedladdning och hantering.")
    confirm_reset = st.checkbox("Jag förstår att loggarna raderas permanent (tips.csv & results.csv).")
    if st.button("🗑️ Nollställ loggar", disabled=not confirm_reset):
        reset_logs()
        st.success("Loggar nollställda.")

    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            st.download_button(
                "📥 Ladda ner modell.pkl",
                data=f,
                file_name="model.pkl",
                mime="application/octet-stream",
                key="dl_model_pkl"
            )
        if st.button("🗑️ Ta bort sparad modell.pkl"):
            os.remove(MODEL_FILE)
            st.success("Modell borttagen från disk.")

col1, col2 = st.columns([1,2])

with col1:
    st.header("Data")
    leagues = st.multiselect("Välj ligor (default E0,E1)", ["E0","E1","E2"], default=["E0","E1"])
    if st.button("🔄 Ladda ner / uppdatera data"):
        files = download_files(tuple(leagues), force=True)
        st.success(f"Hämtade {len(files)} filer.")
    else:
        files = download_files(tuple(leagues))
    st.write("Data-filer:", files)

with col2:
    st.header("Modell")
    df = load_all_data(files)
    if df.empty:
        st.warning("Ingen data tillgänglig. Hämta filer först.")
    else:
        df_prep, feat_cols = prepare_features(df)
        st.write(f"Data: {len(df_prep)} rader. Features: {feat_cols}")
        model = load_model()
        if model is None:
            if st.button("🔧 Träna modell (kan ta tid)"):
                model = train_model(df_prep, feat_cols)
        else:
            st.success("✅ Modell laddad från disk")

# Prediction
st.header("Prediktion & loggning")
if 'model' not in locals():
    model = load_model()

if df is None or df.empty:
    st.info("Ingen data för prediktion — ladda data först.")
else:
    sample_list = df_prep[["Date","HomeTeam","AwayTeam","League"]].dropna().drop_duplicates().tail(50).reset_index(drop=True)
    choice_idx = st.selectbox(
        "Välj rad att tippa (senaste 50):",
        sample_list.index,
        format_func=lambda i: f"{sample_list.loc[i,'Date'].date()} {sample_list.loc[i,'HomeTeam']} - {sample_list.loc[i,'AwayTeam']} ({sample_list.loc[i,'League']})"
    )
    chosen = sample_list.loc[choice_idx]
    chosen_row = df_prep[
        (df_prep["Date"]==chosen["Date"]) &
        (df_prep["HomeTeam"]==chosen["HomeTeam"]) &
        (df_prep["AwayTeam"]==chosen["AwayTeam"])
    ].iloc[0]
    st.write("Vald match:", chosen_row["HomeTeam"], "vs", chosen_row["AwayTeam"], "på", chosen_row["Date"])
    if model is None:
        st.info("Träna modellen innan du predikterar.")
    else:
        fv = [float(chosen_row[c]) for c in feat_cols]
        probs = predict_probs(model, fv)
        st.json(probs)
        tip = max(probs, key=probs.get)
        if st.button("📌 Logga tipset"):
            mid = safe_match_id(chosen_row)
            log_tip(mid, chosen_row["HomeTeam"].replace(" ","_"), chosen_row["AwayTeam"].replace(" ","_"), tip, probs)
            st.success("Tipset loggat!")

# Result & evaluation
st.header("Resultat & utvärdering")
if st.button("🔁 Kör automatisk uppdatering av resultat"):
    df_loaded = load_all_data(files)
    updated = auto_update_results(df_loaded)
    st.success(f"Uppdaterade {updated} tips med riktiga resultat (om några fanns).")

tips_df, results_df = read_logs()

st.subheader("Tips (senaste 10)")
if not tips_df.empty:
    st.dataframe(tips_df.tail(10))
    st.download_button(
        "⬇️ Ladda ner hela tips.csv",
        data=df_to_csv_bytes(tips_df),
        file_name="tips.csv",
        mime="text/csv",
        key="dl_tips_csv"
    )
else:
    st.write("Inga tips ännu.")

st.subheader("Resultat (senaste 10)")
if not results_df.empty:
    st.dataframe(results_df.tail(10))
    st.download_button(
        "⬇️ Ladda ner hela results.csv",
        data=df_to_csv_bytes(results_df),
        file_name="results.csv",
        mime="text/csv",
        key="dl_results_csv"
    )
else:
    st.write("Inga resultat ännu.")

st.header("Träffsäkerhet")
merged, stats = evaluate_performance()
if stats is None:
    st.info("Ingen träffstatistik att visa ännu (behöver både tips och resultat).")
else:
    met1, met2, met3 = st.columns(3)
    with met1:
        st.metric("Antal tippar med facit", stats["total"])
    with met2:
        st.metric("Rätt", stats["correct"])
    with met3:
        st.metric("Träffsäkerhet (%)", f"{stats['accuracy']:.1f}")

    st.subheader("Senaste tips med facit")
    st.dataframe(merged.tail(10))
    st.download_button(
        "⬇️ Ladda ner tabellen med facit (merged).csv",
        data=df_to_csv_bytes(merged),
        file_name="tips_med_facit.csv",
        mime="text/csv",
        key="dl_merged_csv"
    )