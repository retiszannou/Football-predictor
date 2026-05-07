"""
app.py
------
Application web Flask pour prédire le résultat d'un match
de Premier League.

Usage :
    uv run python app/app.py
    → Ouvre http://127.0.0.1:5000 dans ton navigateur
"""

import os
import sys

import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Ajouter le dossier racine au path pour importer les modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

app = Flask(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "matches.csv")

# ─── Entraînement du modèle au démarrage ──────────────────────────────
def train_model():
    """Charge les données et entraîne le modèle au lancement de l'app."""
    df = pd.read_csv(DATA_PATH).dropna(subset=["xg_home", "xg_away", "result"])

    le_home = LabelEncoder().fit(df["home"])
    le_away = LabelEncoder().fit(df["away"])

    df["home_enc"] = le_home.transform(df["home"])
    df["away_enc"] = le_away.transform(df["away"])
    df["xg_diff"]  = df["xg_home"] - df["xg_away"]

    X = df[["xg_home", "xg_away", "xg_diff", "home_enc", "away_enc"]]
    y = df["result"]

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6,
        random_state=42, class_weight="balanced"
    )
    model.fit(X, y)

    teams = sorted(df["home"].unique().tolist())
    return model, le_home, le_away, teams


print("[INFO] Chargement du modèle...")
model, le_home, le_away, TEAMS = train_model()
print(f"[OK] Modèle prêt. {len(TEAMS)} équipes disponibles.")


# ─── Routes ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", teams=TEAMS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    home    = data.get("home")
    away    = data.get("away")
    xg_home = float(data.get("xg_home", 1.5))
    xg_away = float(data.get("xg_away", 1.0))

    if home == away:
        return jsonify({"error": "Les deux équipes doivent être différentes."}), 400

    # Encoder les équipes
    home_enc = int(le_home.transform([home])[0]) if home in le_home.classes_ else 0
    away_enc = int(le_away.transform([away])[0]) if away in le_away.classes_ else 0
    xg_diff  = xg_home - xg_away

    X = pd.DataFrame(
        [[xg_home, xg_away, xg_diff, home_enc, away_enc]],
        columns=["xg_home", "xg_away", "xg_diff", "home_enc", "away_enc"],
    )

    pred   = model.predict(X)[0]
    probas = dict(zip(model.classes_, model.predict_proba(X)[0].tolist()))

    labels = {
        "H": f"Victoire {home}",
        "D": "Match nul",
        "A": f"Victoire {away}",
    }

    return jsonify({
        "prediction": pred,
        "label":      labels[pred],
        "probabilities": {
            "H": round(probas.get("H", 0) * 100, 1),
            "D": round(probas.get("D", 0) * 100, 1),
            "A": round(probas.get("A", 0) * 100, 1),
        },
        "home": home,
        "away": away,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
