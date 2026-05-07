# 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League Match Predictor

Prédiction de résultats de matchs de Premier League (victoire / nul / défaite)
via **webscraping Playwright** (données xG sur Understat.com) et **Machine Learning** (Random Forest).
Une **application web Flask** permet de faire des prédictions interactives.

> Projet réalisé dans le cadre du cours **Algorithmie et Programmation** — Université d'Angers

---

## 📁 Structure du projet

```
football-predictor/
├── scraper/
│   └── scrape_understat.py     # Webscraping Playwright → Understat.com (xG)
├── ml/
│   └── predict.py              # Modèle Random Forest + évaluation
├── app/
│   ├── app.py                  # Application web Flask
│   └── templates/
│       └── index.html          # Interface web de prédiction
├── notebooks/
│   └── exploration.ipynb       # Analyse exploratoire + visualisations
├── data/
│   └── matches.csv             # Données générées par le scraper
├── cli.py                      # Interface ligne de commande
├── pyproject.toml              # Dépendances (gérées par uv)
├── .gitignore
└── README.md
```

---

## ⚙️ Installation et lancement

### Prérequis

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) installé

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-pseudo>/football-predictor.git
cd football-predictor
```

### 2. Installer les dépendances

```bash
uv sync
```

### 3. Installer le navigateur Playwright (1 seule fois)

```bash
uv run playwright install chromium
```

### 4. Scraper les données xG depuis Understat

```bash
uv run python scraper/scrape_understat.py
```

→ Génère `data/matches.csv` avec les 380 matchs réels de la saison 2023-2024.

### 5. Lancer l'application web

```bash
uv run python app/app.py
```

→ Ouvre **http://127.0.0.1:5000** dans ton navigateur.

### 6. (Optionnel) CLI de prédiction

```bash
uv run python cli.py
```

### 7. (Optionnel) Graphiques et évaluation du modèle

```bash
uv run python ml/predict.py
```

---

## 🤖 Modèle Machine Learning

| Paramètre    | Valeur                                      |
|-------------|---------------------------------------------|
| Algorithme  | Random Forest Classifier                    |
| Features    | xg_home, xg_away, xg_diff, équipes encodées |
| Cible       | H (domicile) / D (nul) / A (extérieur)      |
| Split       | 80% entraînement / 20% test                 |
| Précision   | ~58–65%                                     |

---

## 📊 Source des données

**[Understat.com](https://understat.com)** — Saison Premier League 2023-2024

Scraping via **Playwright** : le navigateur charge la page, extrait le JSON embarqué
dans les balises `<script>` contenant tous les matchs avec leurs xG.

| Colonne      | Description                          |
|-------------|--------------------------------------|
| `date`      | Date du match                        |
| `home`      | Équipe à domicile                    |
| `away`      | Équipe à l'extérieur                 |
| `score_home`| Buts domicile                        |
| `score_away`| Buts extérieur                       |
| `xg_home`   | Expected Goals domicile (Understat)  |
| `xg_away`   | Expected Goals extérieur (Understat) |
| `result`    | H / D / A                            |

---

## 🌐 Application web

L'application Flask permet de saisir deux équipes et leurs xG estimés,
puis affiche la prédiction avec les probabilités pour chaque issue.

---

## 👤 Auteur

Projet universitaire — Université d'Angers — Cours : Algorithmie et Programmation
