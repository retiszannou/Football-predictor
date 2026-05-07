"""
scrape_understat.py
-------------------
Scrape les données xG des matchs de Premier League 2023-2024
depuis Understat.com en interceptant les requêtes réseau avec Playwright.

Usage :
    uv run playwright install chromium   (1 seule fois)
    uv run python scraper/scrape_understat.py
"""

import asyncio
import json
import os
import re

import pandas as pd
from playwright.async_api import async_playwright

OUTPUT_PATH = "data/matches.csv"


async def scrape_understat_xg() -> pd.DataFrame:
    print("[INFO] Lancement du navigateur Playwright...")

    intercepted = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        # Intercepter toutes les réponses réseau
        async def handle_response(response):
            url = response.url
            try:
                if "understat.com" in url and response.status == 200:
                    ct = response.headers.get("content-type", "")
                    if "json" in ct or "javascript" in ct or "text" in ct:
                        body = await response.text()
                        if any(k in body for k in ["isResult", "xG", "goals", "datetime"]):
                            intercepted[url] = body
                            print(f"[INFO] Données interceptées depuis : {url}")
            except Exception:
                pass

        page.on("response", handle_response)

        print("[INFO] Chargement de la page Understat...")
        await page.goto(
            "https://understat.com/league/EPL/2023",
            wait_until="networkidle",
            timeout=60000,
        )

        # Attendre un peu pour que toutes les requêtes finissent
        await asyncio.sleep(3)

        # Essayer aussi d'exécuter JS pour récupérer les données déjà en mémoire
        js_data = None
        for var_name in ["datesData", "matchesData", "fixturesData", "resultsData"]:
            try:
                result = await page.evaluate(f"() => typeof {var_name} !== 'undefined' ? JSON.stringify({var_name}) : null")
                if result:
                    js_data = json.loads(result)
                    print(f"[INFO] Variable JS '{var_name}' récupérée depuis le contexte de la page.")
                    break
            except Exception:
                pass

        # Récupérer le HTML final si besoin
        content = await page.content()
        await browser.close()

    # ── Méthode 1 : données récupérées via JS context ──────────────
    if js_data:
        return parse_matches(js_data)

    # ── Méthode 2 : données interceptées via réseau ─────────────────
    for url, body in intercepted.items():
        try:
            data = json.loads(body)
            if isinstance(data, list) and len(data) > 10:
                if "goals" in str(data[0]) or "xG" in str(data[0]):
                    print(f"[INFO] Parsing des données depuis {url}")
                    return parse_matches(data)
            elif isinstance(data, dict) and "data" in data:
                return parse_matches(data["data"])
        except Exception as e:
            print(f"[WARN] Échec parsing {url}: {e}")

    # ── Méthode 3 : chercher dans le HTML avec regex étendue ────────
    print("[INFO] Tentative extraction depuis le HTML...")
    patterns = [
        r"var\s+\w*[Dd]ates?\w*\s*=\s*JSON\.parse\('(.+?)'\)",
        r"var\s+\w*[Mm]atch\w*\s*=\s*JSON\.parse\('(.+?)'\)",
        r"JSON\.parse\('(\\x7B\\x22datetime.+?)'\)",
        r'JSON\.parse\("(\\x7B\\x22datetime.+?)"\)',
    ]
    for pattern in patterns:
        m = re.search(pattern, content, re.DOTALL)
        if m:
            raw = m.group(1)
            try:
                decoded = bytes(raw, "utf-8").decode("unicode_escape")
                data = json.loads(decoded)
                if isinstance(data, list) and len(data) > 0:
                    print(f"[INFO] Données trouvées via pattern HTML.")
                    return parse_matches(data)
            except Exception as e:
                print(f"[WARN] Décodage échoué : {e}")

    # Sauvegarder pour debug
    os.makedirs("data", exist_ok=True)
    with open("data/debug_understat.html", "w", encoding="utf-8") as f:
        f.write(content)
    with open("data/debug_intercepted.json", "w", encoding="utf-8") as f:
        json.dump(list(intercepted.keys()), f, indent=2)

    raise ValueError(
        "Impossible de récupérer les données.\n"
        f"URLs interceptées : {list(intercepted.keys())}\n"
        "Fichiers de debug sauvegardés dans data/"
    )


def parse_matches(matches_data: list) -> pd.DataFrame:
    """Transforme la liste de matchs en DataFrame propre."""
    rows = []
    for m in matches_data:
        if not m.get("isResult"):
            continue
        try:
            sh = int(m["goals"]["h"])
            sa = int(m["goals"]["a"])
            result = "H" if sh > sa else ("A" if sa > sh else "D")
            rows.append({
                "date":       m["datetime"][:10],
                "home":       m["h"]["title"],
                "away":       m["a"]["title"],
                "score_home": sh,
                "score_away": sa,
                "xg_home":    round(float(m["xG"]["h"]), 2),
                "xg_away":    round(float(m["xG"]["a"]), 2),
                "result":     result,
            })
        except (KeyError, ValueError):
            continue

    if not rows:
        raise ValueError("Aucun match valide trouvé dans les données.")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print(f"[INFO] {len(df)} matchs parsés avec succès.")
    return df


def print_stats(df: pd.DataFrame):
    print("\n" + "=" * 52)
    print("  Premier League 2023-2024 (Understat)")
    print("=" * 52)
    print(f"  Matchs total   : {len(df)}")
    print(f"  Période        : {df['date'].min()}  ->  {df['date'].max()}")
    print(f"  xG moyen dom.  : {df['xg_home'].mean():.2f}")
    print(f"  xG moyen ext.  : {df['xg_away'].mean():.2f}")
    print(f"\n  Distribution des résultats :")
    labels = {"H": "Victoire domicile ", "D": "Match nul         ", "A": "Victoire extérieur"}
    for key, label in labels.items():
        count = (df["result"] == key).sum()
        pct   = count / len(df) * 100
        print(f"    {label} {count:3d}  ({pct:.1f}%)")
    print("=" * 52)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = asyncio.run(scrape_understat_xg())
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[OK] Données sauvegardées dans '{OUTPUT_PATH}'.")
    print_stats(df)
    print(f"\nAperçu :")
    print(df.head(5).to_string(index=False))
