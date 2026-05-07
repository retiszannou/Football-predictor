"""
predict.py
----------
Entraîne un modèle de Machine Learning pour prédire le résultat
d'un match de Premier League (H / D / A) à partir des stats.

Usage :
    uv run python ml/predict.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/matches.csv"


def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Charge le CSV et construit les features pour le modèle.
    Features utilisées :
      - xg_home, xg_away       : Expected Goals (qualité des occasions)
      - xg_diff                : Différence d'xG
      - home_encoded           : Équipe à domicile (encodée)
      - away_encoded           : Équipe à l'extérieur (encodée)
    """
    df = pd.read_csv(path).dropna(subset=["xg_home", "xg_away", "result"])

    le = LabelEncoder()
    df["home_encoded"] = le.fit_transform(df["home"])
    df["away_encoded"] = le.fit_transform(df["away"])
    df["xg_diff"] = df["xg_home"] - df["xg_away"]

    features = ["xg_home", "xg_away", "xg_diff", "home_encoded", "away_encoded"]
    X = df[features]
    y = df["result"]

    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Entraîne et évalue un Random Forest."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=" * 50)
    print(f"  Précision globale : {acc:.2%}")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Away", "Draw", "Home"]))

    # --- Matrice de confusion ---
    cm = confusion_matrix(y_test, y_pred, labels=["H", "D", "A"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Home", "Draw", "Away"],
        yticklabels=["Home", "Draw", "Away"],
    )
    plt.title("Matrice de confusion — Premier League Predictor")
    plt.ylabel("Réel")
    plt.xlabel("Prédit")
    plt.tight_layout()
    plt.savefig("data/confusion_matrix.png", dpi=150)
    print("[OK] Matrice de confusion sauvegardée dans 'data/confusion_matrix.png'.")

    # --- Importance des features ---
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=True
    )
    plt.figure(figsize=(6, 4))
    importances.plot(kind="barh", color="steelblue")
    plt.title("Importance des variables")
    plt.tight_layout()
    plt.savefig("data/feature_importance.png", dpi=150)
    print("[OK] Graphique d'importance sauvegardé dans 'data/feature_importance.png'.")

    return model


def predict_match(model: RandomForestClassifier, xg_home: float, xg_away: float,
                  home_enc: int = 0, away_enc: int = 1) -> str:
    """Prédit le résultat d'un match à partir des xG estimés."""
    xg_diff = xg_home - xg_away
    X = pd.DataFrame(
        [[xg_home, xg_away, xg_diff, home_enc, away_enc]],
        columns=["xg_home", "xg_away", "xg_diff", "home_encoded", "away_encoded"],
    )
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = model.classes_

    print("\n--- Prédiction ---")
    for cls, p in zip(classes, proba):
        label = {"H": "Victoire domicile", "D": "Match nul", "A": "Victoire extérieur"}[cls]
        print(f"  {label:25s} : {p:.1%}")
    print(f"  → Résultat prédit : {pred}")
    return pred


if __name__ == "__main__":
    X, y = load_and_prepare(DATA_PATH)
    model = train_model(X, y)

    # Exemple de prédiction : Manchester City (xG 2.1) vs Arsenal (xG 1.4)
    print("\nExemple : Man City (xG=2.1) vs Arsenal (xG=1.4) à domicile")
    predict_match(model, xg_home=2.1, xg_away=1.4)
