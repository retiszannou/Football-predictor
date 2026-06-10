"""
cli.py
------
Interface en ligne de commande pour prédire le résultat d'un match
de Premier League. L'utilisateur saisit uniquement les deux noms d'équipes.

Les xG estimés et tous les indicateurs sont calculés automatiquement
depuis l'historique (data/epl_all_seasons.csv) :
  - Historique des confrontations directes (H2H)
  - xG moyen home/away sur toute l'historique
  - Points et performance home/away de la dernière saison
    (0 si l'équipe était en D2 la saison dernière)

La liste des équipes affichées est celle de la saison en cours,
récupérée dynamiquement depuis Wikipedia (scraper/scrape_wikipedia.py).

Usage :
    uv run python cli.py
    uv run python cli.py --home "Manchester City" --away "Arsenal"
"""

import argparse
import datetime
import os
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import GradientBoostingClassifier


DATA_PATH = "data/epl_all_seasons.csv"


# ─── Scraper Wikipedia — équipes de la saison en cours ──────────────────────
def get_current_season_years() -> str:
    now = datetime.datetime.now()
    year, month = now.year, now.month
    if month >= 6:
        return f"{year}–{str(year + 1)[2:]}"
    return f"{year - 1}–{str(year)[2:]}"


def scrape_premier_league_teams() -> list[str]:
    season = get_current_season_years()
    url = f"https://en.wikipedia.org/wiki/{season}_Premier_League"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
    except Exception:
        return []

    soup  = BeautifulSoup(response.text, "html.parser")
    teams = []
    for table in soup.find_all("table", {"class": "wikitable"}):
        headers_text = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if "team" not in headers_text and "club" not in headers_text:
            continue
        col_idx = headers_text.index("team") if "team" in headers_text else headers_text.index("club")
        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) <= col_idx:
                continue
            link = cells[col_idx].find("a")
            if not link:
                continue
            raw = link.get_text(strip=True)
            if raw.startswith("[") or len(raw) <= 2:
                raw = link.get("title", "")
            name = raw.replace("F.C.", "").replace("A.F.C.", "").strip()
            # Harmonisation avec les noms Understat
            if "Brighton"    in name: name = "Brighton"
            elif "Tottenham" in name: name = "Tottenham"
            elif "Bournemouth" in name: name = "Bournemouth"
            elif "Ipswich"   in name: name = "Ipswich Town"
            if name and name not in teams:
                teams.append(name)
        break
    return sorted(set(teams))[:20]


def load_current_season_teams() -> list[str]:
    """
    Récupère les 20 équipes de la saison en cours depuis Wikipedia.
    Si Wikipedia est inaccessible, retourne une liste de secours.
    """
    print(f"{C.GREY}[INFO] Récupération des équipes de la saison en cours (Wikipedia)...{C.RESET}")
    teams = scrape_premier_league_teams()
    if teams and len(teams) >= 18:
        season = get_current_season_years()
        print(f"{C.GREEN}[OK]{C.RESET} {len(teams)} équipes synchronisées — saison {season}.")
        return teams
    # Fallback
    print(f"{C.YELLOW}[WARN]{C.RESET} Wikipedia inaccessible — liste de secours utilisée.")
    return [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool",
        "Manchester City", "Manchester United", "Newcastle United",
        "Nottingham Forest", "Tottenham", "West Ham", "Wolverhampton",
    ]


# ─── Couleurs terminal ───────────────────────────────────────────────────────
class C:
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
{C.BOLD}{C.BLUE}
  ╔══════════════════════════════════════════════╗
  ║   🏴󠁧󠁢󠁥󠁮󠁧󠁿  Premier League Match Predictor       ║
  ║      Historique + Machine Learning           ║
  ╚══════════════════════════════════════════════╝
{C.RESET}""")


def bar(proba: float, width: int = 28) -> str:
    filled = int(proba * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {proba:.1%}"


def color_result(result: str) -> str:
    labels = {
        "H": f"{C.BLUE}{C.BOLD}🏠 Victoire Domicile{C.RESET}",
        "D": f"{C.YELLOW}{C.BOLD}🤝 Match Nul{C.RESET}",
        "A": f"{C.RED}{C.BOLD}✈️  Victoire Extérieur{C.RESET}",
    }
    return labels.get(result, result)


# ─── Calcul des features depuis le CSV ──────────────────────────────────────
class FeatureEngine:
    """
    Calcule toutes les features nécessaires à partir de l'historique des matchs.
    Appelé une seule fois au démarrage pour construire les tables de stats.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.last_season = df["season"].max()

        # Stats globales par équipe (sur tout l'historique)
        self._global_home = self._build_global_home()
        self._global_away = self._build_global_away()

        # Stats de la dernière saison
        self._last_season_stats = self._build_last_season_stats()

        # H2H (confrontations directes)
        self._h2h = self._build_h2h()

        # Toutes les équipes connues
        self.all_teams = sorted(
            set(df["home"].unique()) | set(df["away"].unique())
        )

    # ── Builders ────────────────────────────────────────────────────────────

    def _build_global_home(self) -> dict:
        """xG moyen quand l'équipe joue à domicile (tout l'historique)."""
        return (
            self.df.groupby("home")["xg_home"]
            .mean()
            .to_dict()
        )

    def _build_global_away(self) -> dict:
        """xG moyen quand l'équipe joue à l'extérieur (tout l'historique)."""
        return (
            self.df.groupby("away")["xg_away"]
            .mean()
            .to_dict()
        )

    def _build_last_season_stats(self) -> dict:
        """
        Pour chaque équipe : points totaux, xG home moyen, xG away moyen
        lors de la dernière saison disponible.
        Si l'équipe est absente -> elle était en D2 -> tout à 0.
        """
        ls = self.df[self.df["season"] == self.last_season].copy()

        # Points home (3 si H, 1 si D, 0 si A)
        ls_home = ls[["home", "result", "xg_home"]].copy()
        ls_home["pts"] = ls_home["result"].map({"H": 3, "D": 1, "A": 0})
        home_stats = ls_home.groupby("home").agg(
            pts_home=("pts", "sum"),
            xg_home_last=("xg_home", "mean"),
        )

        # Points away (3 si A, 1 si D, 0 si H)
        ls_away = ls[["away", "result", "xg_away"]].copy()
        ls_away["pts"] = ls_away["result"].map({"A": 3, "D": 1, "H": 0})
        away_stats = ls_away.groupby("away").agg(
            pts_away=("pts", "sum"),
            xg_away_last=("xg_away", "mean"),
        )

        combined = home_stats.join(away_stats, how="outer").fillna(0)
        combined["pts_total"] = combined["pts_home"] + combined["pts_away"]
        return combined.to_dict(orient="index")

    def _build_h2h(self) -> dict:
        """
        H2H : pour chaque paire (home, away), xG moyen historique
        de chaque équipe dans ce duel exact.
        Clé : (home_team, away_team)
        """
        h2h = (
            self.df.groupby(["home", "away"])
            .agg(
                h2h_xg_home=("xg_home", "mean"),
                h2h_xg_away=("xg_away", "mean"),
                h2h_count=("xg_home", "count"),
            )
            .to_dict(orient="index")
        )
        return h2h

    # ── Getters ─────────────────────────────────────────────────────────────

    def get_last_season_stats(self, team: str) -> dict:
        """Retourne les stats de la dernière saison (0 si équipe absente = D2)."""
        return self._last_season_stats.get(
            team,
            {"pts_home": 0, "pts_away": 0, "pts_total": 0,
             "xg_home_last": 0.0, "xg_away_last": 0.0},
        )

    def estimate_xg(self, home: str, away: str) -> tuple[float, float]:
        """
        Estime les xG probables pour ce match via une moyenne pondérée de :
          - H2H historique (poids fort si beaucoup de matchs joués ensemble)
          - Performance globale home/away de chaque équipe
          - Performance home/away de la dernière saison
        """
        h2h = self._h2h.get((home, away), {})
        n_h2h = h2h.get("h2h_count", 0)

        # Poids H2H : monte jusqu'à 0.5 à partir de 5 matchs
        w_h2h = min(n_h2h / 10, 0.5)
        w_rest = 1 - w_h2h

        # xG home
        xg_h_h2h    = h2h.get("h2h_xg_home", self._global_home.get(home, 1.4))
        xg_h_global = self._global_home.get(home, 1.4)
        xg_h_last   = self.get_last_season_stats(home).get("xg_home_last", 1.4)
        xg_home = w_h2h * xg_h_h2h + w_rest * (0.6 * xg_h_global + 0.4 * xg_h_last)

        # xG away
        xg_a_h2h    = h2h.get("h2h_xg_away", self._global_away.get(away, 1.1))
        xg_a_global = self._global_away.get(away, 1.1)
        xg_a_last   = self.get_last_season_stats(away).get("xg_away_last", 1.1)
        xg_away = w_h2h * xg_a_h2h + w_rest * (0.6 * xg_a_global + 0.4 * xg_a_last)

        return round(xg_home, 3), round(xg_away, 3)

    def build_feature_row(self, home: str, away: str) -> pd.DataFrame:
        """Construit la ligne de features pour le modèle."""
        xg_home, xg_away = self.estimate_xg(home, away)
        h2h = self._h2h.get((home, away), {})
        stats_home = self.get_last_season_stats(home)
        stats_away = self.get_last_season_stats(away)

        return pd.DataFrame([{
            # xG estimés
            "xg_home":        xg_home,
            "xg_away":        xg_away,
            "xg_diff":        round(xg_home - xg_away, 3),

            # H2H
            "h2h_xg_home":    h2h.get("h2h_xg_home", xg_home),
            "h2h_xg_away":    h2h.get("h2h_xg_away", xg_away),
            "h2h_count":      h2h.get("h2h_count", 0),

            # Dernière saison — performance home/away
            "xg_home_last":   stats_home["xg_home_last"],
            "xg_away_last":   stats_away["xg_away_last"],

            # Points dernière saison (force actuelle)
            "pts_home_team":  stats_home["pts_total"],
            "pts_away_team":  stats_away["pts_total"],
            "pts_diff":       stats_home["pts_total"] - stats_away["pts_total"],
        }])


# ─── Chargement + entraînement ───────────────────────────────────────────────
def load_and_train(path: str) -> tuple:
    """Charge le CSV, construit les features, entraîne le modèle."""
    if not os.path.exists(path):
        print(f"{C.RED}[ERREUR]{C.RESET} Fichier '{path}' introuvable.")
        print(f"  → Lance d'abord : {C.CYAN}uv run python scrape_understat_all.py{C.RESET}")
        sys.exit(1)

    df = pd.read_csv(path).dropna(subset=["xg_home", "xg_away", "result"])
    engine = FeatureEngine(df)

    # Construire les features pour chaque match historique
    rows = []
    for _, row in df.iterrows():
        h2h    = engine._h2h.get((row["home"], row["away"]), {})
        sh     = engine.get_last_season_stats(row["home"])
        sa     = engine.get_last_season_stats(row["away"])
        rows.append({
            "xg_home":       row["xg_home"],
            "xg_away":       row["xg_away"],
            "xg_diff":       row["xg_home"] - row["xg_away"],
            "h2h_xg_home":   h2h.get("h2h_xg_home", row["xg_home"]),
            "h2h_xg_away":   h2h.get("h2h_xg_away", row["xg_away"]),
            "h2h_count":     h2h.get("h2h_count", 0),
            "xg_home_last":  sh["xg_home_last"],
            "xg_away_last":  sa["xg_away_last"],
            "pts_home_team": sh["pts_total"],
            "pts_away_team": sa["pts_total"],
            "pts_diff":      sh["pts_total"] - sa["pts_total"],
            "result":        row["result"],
        })

    train_df = pd.DataFrame(rows)
    X = train_df.drop(columns=["result"])
    y = train_df["result"]

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)
    return model, engine


# ─── Affichage de la prédiction ──────────────────────────────────────────────
def predict_and_display(model, engine: FeatureEngine, home: str, away: str):
    xg_home, xg_away = engine.estimate_xg(home, away)
    h2h_data  = engine._h2h.get((home, away), {})
    stats_h   = engine.get_last_season_stats(home)
    stats_a   = engine.get_last_season_stats(away)
    n_h2h     = h2h_data.get("h2h_count", 0)
    last_s    = engine.last_season

    X      = engine.build_feature_row(home, away)
    pred   = model.predict(X)[0]
    probas = dict(zip(model.classes_, model.predict_proba(X)[0]))

    # ── En-tête du match ────────────────────────────────────────────
    sep = "─" * 50
    print(f"\n  {C.BOLD}{sep}{C.RESET}")
    print(f"  🏟️  {C.BOLD}{home:22s}{C.RESET} vs  {C.BOLD}{away}{C.RESET}")
    print(f"  {sep}")

    # ── Contexte calculé ────────────────────────────────────────────
    print(f"\n  {C.CYAN}{C.BOLD}Données calculées automatiquement{C.RESET}")

    # H2H
    if n_h2h > 0:
        print(f"  {'H2H (confrontations directes)':35s} {n_h2h} matchs")
        print(f"  {'  → xG moyen ' + home:35s} {h2h_data['h2h_xg_home']:.2f}")
        print(f"  {'  → xG moyen ' + away:35s} {h2h_data['h2h_xg_away']:.2f}")
    else:
        print(f"  {C.GREY}H2H : aucun historique direct trouvé{C.RESET}")

    # xG estimés
    print(f"  {'xG estimé ' + home + ' (domicile)':35s} {C.CYAN}{xg_home:.2f}{C.RESET}")
    print(f"  {'xG estimé ' + away + ' (extérieur)':35s} {C.CYAN}{xg_away:.2f}{C.RESET}")

    # Dernière saison
    h_in_last = stats_h["pts_total"] > 0
    a_in_last = stats_a["pts_total"] > 0
    print(f"\n  {C.BOLD}Dernière saison ({last_s}){C.RESET}")
    print(f"  {home:35s} "
          f"{C.GREEN if h_in_last else C.GREY}"
          f"{int(stats_h['pts_total'])} pts"
          f"{'' if h_in_last else '  (absent → D2, poids=0)'}"
          f"{C.RESET}")
    print(f"  {away:35s} "
          f"{C.GREEN if a_in_last else C.GREY}"
          f"{int(stats_a['pts_total'])} pts"
          f"{'' if a_in_last else '  (absent → D2, poids=0)'}"
          f"{C.RESET}")

    # ── Résultats du modèle ─────────────────────────────────────────
    print(f"\n  {C.BOLD}Probabilités prédites{C.RESET}")
    order = [
        ("H", f"🏠 {home[:18]:20s}"),
        ("D", f"🤝 {'Match Nul':20s}"),
        ("A", f"✈️  {away[:18]:20s}"),
    ]
    for key, label in order:
        p         = probas.get(key, 0)
        highlight = C.BOLD if key == pred else C.GREY
        color     = {
            "H": C.BLUE, "D": C.YELLOW, "A": C.RED
        }.get(key, "")
        print(f"  {highlight}{color}{label} {bar(p)}{C.RESET}")

    print(f"\n  → Résultat prédit : {color_result(pred)}")
    print(f"  {C.BOLD}{sep}{C.RESET}\n")


# ─── Sélection interactive de l'équipe ──────────────────────────────────────
def pick_team(prompt: str, teams: list[str]) -> str:
    """
    Affiche la liste et laisse l'utilisateur saisir le nom
    (correspondance insensible à la casse, partielle acceptée).
    """
    while True:
        print(f"\n  {C.BOLD}Équipes disponibles :{C.RESET}")
        cols = 2
        for i in range(0, len(teams), cols):
            row_teams = teams[i:i + cols]
            line = "".join(f"  {i+j+1:3}. {t:<28}" for j, t in enumerate(row_teams))
            print(line)
        print()

        val = input(f"  {prompt} (nom ou numéro, 'q' pour quitter) : ").strip()

        if val.lower() == "q":
            sys.exit(0)

        # Sélection par numéro
        if val.isdigit():
            idx = int(val) - 1
            if 0 <= idx < len(teams):
                return teams[idx]
            print(f"  {C.RED}Numéro invalide.{C.RESET}")
            continue

        # Correspondance exacte insensible à la casse
        matches = [t for t in teams if t.lower() == val.lower()]
        if matches:
            return matches[0]

        # Correspondance partielle
        matches = [t for t in teams if val.lower() in t.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"  {C.YELLOW}Plusieurs équipes correspondent : {', '.join(matches)}{C.RESET}")
            continue

        print(f"  {C.RED}Équipe introuvable. Réessaie.{C.RESET}")


# ─── Mode interactif ─────────────────────────────────────────────────────────
def interactive_mode(model, engine: FeatureEngine, current_teams: list[str]):
    print(f"{C.CYAN}Mode interactif — tape 'q' pour quitter.{C.RESET}\n")

    while True:
        home = pick_team(f"Équipe {C.BLUE}domicile{C.RESET}", current_teams)
        away = pick_team(f"Équipe {C.RED}extérieur{C.RESET}", current_teams)

        if home == away:
            print(f"  {C.RED}Les deux équipes doivent être différentes.{C.RESET}")
            continue

        predict_and_display(model, engine, home, away)

        again = input("  Faire une autre prédiction ? (o/n) : ").strip().lower()
        if again != "o":
            break

    print(f"\n{C.GREY}À bientôt !{C.RESET}\n")


# ─── Point d'entrée ──────────────────────────────────────────────────────────
def main():
    banner()

    # Récupération des équipes de la saison en cours (avant le banner de chargement)
    current_teams = load_current_season_teams()

    parser = argparse.ArgumentParser(
        description="Prédit le résultat d'un match EPL (noms d'équipes uniquement).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--home", type=str, help="Équipe à domicile")
    parser.add_argument("--away", type=str, help="Équipe à l'extérieur")
    args = parser.parse_args()

    print(f"{C.GREY}[INFO] Chargement des données et entraînement du modèle...{C.RESET}")
    model, engine = load_and_train(DATA_PATH)
    print(f"{C.GREEN}[OK]{C.RESET} Modèle prêt — dernière saison connue : {engine.last_season}\n")

    if args.home and args.away:
        # Résolution des noms sur la liste saison en cours (partielle acceptée)
        home_match = [t for t in current_teams if args.home.lower() in t.lower()]
        away_match = [t for t in current_teams if args.away.lower() in t.lower()]

        if not home_match:
            print(f"{C.RED}[ERREUR]{C.RESET} Équipe domicile '{args.home}' introuvable dans la saison en cours.")
            sys.exit(1)
        if not away_match:
            print(f"{C.RED}[ERREUR]{C.RESET} Équipe extérieur '{args.away}' introuvable dans la saison en cours.")
            sys.exit(1)

        predict_and_display(model, engine, home_match[0], away_match[0])
    else:
        interactive_mode(model, engine, current_teams)


if __name__ == "__main__":
    main()