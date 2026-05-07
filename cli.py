"""
cli.py
------
Interface en ligne de commande pour prédire le résultat d'un match
de Premier League à partir des Expected Goals estimés.

Usage :
    uv run python cli.py

    # Ou directement avec les arguments :
    uv run python cli.py --home "Manchester City" --away "Arsenal" --xg-home 2.1 --xg-away 1.3
"""

import argparse
import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Équipes Premier League 2023-2024
PREMIER_LEAGUE_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford",
    "Brighton", "Burnley", "Chelsea", "Crystal Palace",
    "Everton", "Fulham", "Liverpool", "Luton Town",
    "Manchester City", "Manchester United", "Newcastle United",
    "Nottingham Forest", "Sheffield United", "Tottenham",
    "West Ham", "Wolverhampton",
]

DATA_PATH = "data/matches.csv"


# ─── Couleurs terminal ───────────────────────────────────────────────
class Colors:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GREY   = "\033[90m"


def banner():
    print(f"""
{Colors.BOLD}{Colors.BLUE}
  ╔══════════════════════════════════════════════╗
  ║   🏴󠁧󠁢󠁥󠁮󠁧󠁿  Premier League Match Predictor       ║
  ║      Webscraping + Machine Learning          ║
  ╚══════════════════════════════════════════════╝
{Colors.RESET}""")


def color_result(result: str) -> str:
    colors = {"H": Colors.BLUE, "D": Colors.YELLOW, "A": Colors.RED}
    labels = {
        "H": "🏠 Victoire Domicile",
        "D": "🤝 Match Nul",
        "A": "✈️  Victoire Extérieur",
    }
    return f"{colors.get(result, '')}{Colors.BOLD}{labels[result]}{Colors.RESET}"


def bar(proba: float, width: int = 30) -> str:
    filled = int(proba * width)
    empty  = width - filled
    return f"[{'█' * filled}{'░' * empty}] {proba:.1%}"


# ─── Chargement et entraînement ──────────────────────────────────────
def load_and_train(path: str):
    """Charge les données et entraîne le modèle. Retourne (model, le_home, le_away)."""
    if not os.path.exists(path):
        print(f"{Colors.RED}[ERREUR]{Colors.RESET} Fichier '{path}' introuvable.")
        print(f"  → Lance d'abord : {Colors.CYAN}uv run python scraper/scrape_fbref.py{Colors.RESET}")
        sys.exit(1)

    df = pd.read_csv(path).dropna(subset=["xg_home", "xg_away", "result"])

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
    return model, le_home, le_away


# ─── Prédiction ──────────────────────────────────────────────────────
def predict(model, le_home, le_away, home: str, away: str, xg_home: float, xg_away: float):
    """Effectue et affiche la prédiction."""

    # Encoder les équipes (fallback sur 0 si inconnue)
    home_enc = le_home.transform([home])[0] if home in le_home.classes_ else 0
    away_enc = le_away.transform([away])[0] if away in le_away.classes_ else 0
    xg_diff  = xg_home - xg_away

    X = pd.DataFrame(
        [[xg_home, xg_away, xg_diff, home_enc, away_enc]],
        columns=["xg_home", "xg_away", "xg_diff", "home_enc", "away_enc"],
    )

    pred   = model.predict(X)[0]
    probas = dict(zip(model.classes_, model.predict_proba(X)[0]))

    # ── Affichage ──
    print(f"\n  {Colors.BOLD}{'─' * 44}{Colors.RESET}")
    print(f"  🏟️  {Colors.BOLD}{home:20s}{Colors.RESET}  vs  {Colors.BOLD}{away}{Colors.RESET}")
    print(f"  xG domicile : {Colors.CYAN}{xg_home:.2f}{Colors.RESET}   |   xG extérieur : {Colors.CYAN}{xg_away:.2f}{Colors.RESET}")
    print(f"  {'─' * 44}")

    order = [("H", "🏠 Victoire Domicile "), ("D", "🤝 Match Nul         "), ("A", "✈️  Victoire Extérieur")]
    for key, label in order:
        p = probas.get(key, 0)
        highlight = Colors.BOLD if key == pred else Colors.GREY
        print(f"  {highlight}{label} {bar(p)}{Colors.RESET}")

    print(f"\n  → Résultat prédit : {color_result(pred)}")
    print(f"  {'─' * 44}\n")


# ─── Mode interactif ─────────────────────────────────────────────────
def interactive_mode(model, le_home, le_away):
    """Pose les questions à l'utilisateur en mode interactif."""
    print(f"{Colors.CYAN}Mode interactif — tape 'q' pour quitter.{Colors.RESET}\n")

    while True:
        print(f"{Colors.BOLD}Équipes disponibles :{Colors.RESET}")
        for i, team in enumerate(PREMIER_LEAGUE_TEAMS, 1):
            print(f"  {i:2}. {team}")

        print()
        home = input(f"  Équipe {Colors.BLUE}domicile{Colors.RESET} (nom exact) : ").strip()
        if home.lower() == "q":
            break

        away = input(f"  Équipe {Colors.RED}extérieur{Colors.RESET} (nom exact) : ").strip()
        if away.lower() == "q":
            break

        try:
            xg_home = float(input(f"  xG estimé pour {Colors.BLUE}{home}{Colors.RESET} : "))
            xg_away = float(input(f"  xG estimé pour {Colors.RED}{away}{Colors.RESET} : "))
        except ValueError:
            print(f"{Colors.RED}[ERREUR]{Colors.RESET} Valeur numérique attendue.\n")
            continue

        predict(model, le_home, le_away, home, away, xg_home, xg_away)

        again = input("  Faire une autre prédiction ? (o/n) : ").strip().lower()
        if again != "o":
            break

    print(f"\n{Colors.GREY}À bientôt !{Colors.RESET}\n")


# ─── Point d'entrée ──────────────────────────────────────────────────
def main():
    banner()

    parser = argparse.ArgumentParser(
        description="Prédit le résultat d'un match de Premier League.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--home",     type=str,   help="Équipe à domicile")
    parser.add_argument("--away",     type=str,   help="Équipe à l'extérieur")
    parser.add_argument("--xg-home",  type=float, help="Expected Goals domicile")
    parser.add_argument("--xg-away",  type=float, help="Expected Goals extérieur")
    args = parser.parse_args()

    print(f"{Colors.GREY}[INFO] Chargement des données et entraînement du modèle...{Colors.RESET}")
    model, le_home, le_away = load_and_train(DATA_PATH)
    print(f"{Colors.GREEN}[OK]{Colors.RESET} Modèle prêt.\n")

    # Mode direct (arguments fournis) ou mode interactif
    if all([args.home, args.away, args.xg_home is not None, args.xg_away is not None]):
        predict(model, le_home, le_away, args.home, args.away, args.xg_home, args.xg_away)
    else:
        interactive_mode(model, le_home, le_away)


if __name__ == "__main__":
    main()
