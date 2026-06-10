"""
scrape_understat_all.py
-----------------------
Scrape TOUS les scores de matchs EPL disponibles sur Understat.com,
de 2014 à aujourd'hui, et se met à jour automatiquement chaque année.

Logique "infinie" :
  - Détecte automatiquement la saison en cours selon la date du jour
  - Chaque 01/07, une nouvelle saison devient disponible
  - Re-scrape toutes les saisons pour avoir les données à jour
  - Peut être lancé via un cron job ou un scheduler

Usage :
    uv run playwright install chromium   (1 seule fois)
    uv run python scrape_understat_all.py

    # Ou en mode scheduler (tourne en continu, se met à jour chaque 01/07) :
    uv run python scrape_understat_all.py --scheduler
"""

import argparse
import asyncio
import json
import os
import re
import time
from datetime import date, datetime

import pandas as pd
from playwright.async_api import async_playwright

# ── Constantes ──────────────────────────────────────────────────────────────
OUTPUT_PATH   = "data/epl_all_seasons.csv"
FIRST_SEASON  = 2014          # Première saison disponible sur Understat
SCRAPE_DELAY  = 4             # Secondes entre chaque saison (politesse)
REFRESH_MONTH = 7             # Juillet : nouvelle saison disponible
REFRESH_DAY   = 1


# ── Détection dynamique de la dernière saison disponible ────────────────────
def get_last_available_season() -> int:
    """
    Retourne la dernière année de début de saison disponible.

    Logique :
      - La saison 2024-25 est identifiée par l'année 2024 sur Understat.
      - Elle devient disponible à partir du 01/08/2024 (début de saison).
      - Les résultats complets sont exploitables dès le 01/07/2025.
      - Donc au 01/07/YYYY, on peut scraper jusqu'à la saison YYYY-1.

    Exemples :
      - Aujourd'hui = 15/06/2026  -> dernière saison = 2025 (2025-26 en cours, pas finie)
        En fait avant le 01/07/2026 -> on prend 2024
      - Aujourd'hui = 01/07/2026  -> dernière saison = 2025 (2025-26 terminée)
      - Aujourd'hui = 01/07/2027  -> dernière saison = 2026
    """
    today = date.today()
    year  = today.year

    # Avant le 01/07 de l'année en cours : la saison (year-1) n'est pas encore
    # entièrement terminée et archivée -> on s'arrête à (year-2)
    if today < date(year, REFRESH_MONTH, REFRESH_DAY):
        return year - 2
    else:
        return year - 1


def get_all_seasons() -> list[int]:
    """Retourne la liste de toutes les années de début de saison à scraper."""
    last = get_last_available_season()
    return list(range(FIRST_SEASON, last + 1))


# ── Scraping d'une saison ────────────────────────────────────────────────────
async def scrape_season(season: int) -> pd.DataFrame | None:
    """
    Scrape les matchs d'une saison EPL donnée depuis Understat.
    Retourne un DataFrame ou None si échec.
    """
    url = f"https://understat.com/league/EPL/{season}"
    print(f"\n[INFO] Scraping saison {season}-{season+1} -> {url}")

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

        async def handle_response(response):
            try:
                if "understat.com" in response.url and response.status == 200:
                    ct = response.headers.get("content-type", "")
                    if "json" in ct or "javascript" in ct or "text" in ct:
                        body = await response.text()
                        if any(k in body for k in ["isResult", "xG", "goals", "datetime"]):
                            intercepted[response.url] = body
            except Exception:
                pass

        page.on("response", handle_response)

        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            await asyncio.sleep(3)
        except Exception as e:
            print(f"[WARN] Timeout/erreur pour la saison {season}: {e}")
            await browser.close()
            return None

        # Méthode 1 : variables JS en mémoire
        js_data = None
        for var_name in ["datesData", "matchesData", "fixturesData", "resultsData"]:
            try:
                result = await page.evaluate(
                    f"() => typeof {var_name} !== 'undefined' ? JSON.stringify({var_name}) : null"
                )
                if result:
                    js_data = json.loads(result)
                    print(f"[INFO] Variable JS '{var_name}' récupérée (saison {season}).")
                    break
            except Exception:
                pass

        content = await page.content()
        await browser.close()

    # Méthode 1
    if js_data:
        return parse_matches(js_data, season)

    # Méthode 2 : réseau intercepté
    for resp_url, body in intercepted.items():
        try:
            data = json.loads(body)
            if isinstance(data, list) and len(data) > 10:
                if "goals" in str(data[0]) or "xG" in str(data[0]):
                    return parse_matches(data, season)
            elif isinstance(data, dict) and "data" in data:
                return parse_matches(data["data"], season)
        except Exception:
            pass

    # Méthode 3 : regex HTML
    patterns = [
        r"var\s+\w*[Dd]ates?\w*\s*=\s*JSON\.parse\('(.+?)'\)",
        r"var\s+\w*[Mm]atch\w*\s*=\s*JSON\.parse\('(.+?)'\)",
        r"JSON\.parse\('(\\x7B\\x22datetime.+?)'\)",
        r'JSON\.parse\("(\\x7B\\x22datetime.+?)"\)',
    ]
    for pattern in patterns:
        m = re.search(pattern, content, re.DOTALL)
        if m:
            try:
                raw     = m.group(1)
                decoded = bytes(raw, "utf-8").decode("unicode_escape")
                data    = json.loads(decoded)
                if isinstance(data, list) and len(data) > 0:
                    return parse_matches(data, season)
            except Exception:
                pass

    print(f"[WARN] Aucune donnée trouvée pour la saison {season}.")
    return None


# ── Parsing ──────────────────────────────────────────────────────────────────
def parse_matches(matches_data: list, season: int) -> pd.DataFrame:
    """Transforme la liste brute en DataFrame propre."""
    rows = []
    for m in matches_data:
        if not m.get("isResult"):
            continue
        try:
            sh     = int(m["goals"]["h"])
            sa     = int(m["goals"]["a"])
            result = "H" if sh > sa else ("A" if sa > sh else "D")
            rows.append({
                "season":     f"{season}-{season+1}",
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
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print(f"[INFO] {len(df)} matchs parsés pour {season}-{season+1}.")
    return df


# ── Scraping complet de toutes les saisons ───────────────────────────────────
async def scrape_all_seasons() -> pd.DataFrame:
    seasons = get_all_seasons()
    last    = get_last_available_season()
    print(f"\n{'='*55}")
    print(f"  Saisons à scraper : {seasons[0]}-{seasons[0]+1} -> {last}-{last+1}")
    print(f"  Nombre de saisons : {len(seasons)}")
    print(f"  Date du jour      : {date.today()}")
    print(f"{'='*55}")

    all_dfs = []
    for i, season in enumerate(seasons):
        df = await scrape_season(season)
        if df is not None and not df.empty:
            all_dfs.append(df)

        # Délai entre les saisons (sauf la dernière)
        if i < len(seasons) - 1:
            print(f"[INFO] Pause {SCRAPE_DELAY}s avant la prochaine saison...")
            await asyncio.sleep(SCRAPE_DELAY)

    if not all_dfs:
        raise ValueError("Aucune donnée récupérée pour aucune saison.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[OK] Total : {len(df_all)} matchs sur {len(all_dfs)} saisons.")
    return df_all


# ── Stats globales ───────────────────────────────────────────────────────────
def print_stats(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  EPL — Toutes saisons (Understat)")
    print("=" * 55)
    print(f"  Matchs total   : {len(df)}")
    print(f"  Saisons        : {df['season'].nunique()}")
    print(f"  Période        : {df['date'].min()}  ->  {df['date'].max()}")
    print(f"  xG moyen dom.  : {df['xg_home'].mean():.2f}")
    print(f"  xG moyen ext.  : {df['xg_away'].mean():.2f}")
    print(f"\n  Distribution des résultats :")
    labels = {
        "H": "Victoire domicile  ",
        "D": "Match nul          ",
        "A": "Victoire extérieur ",
    }
    for key, label in labels.items():
        count = (df["result"] == key).sum()
        pct   = count / len(df) * 100
        print(f"    {label} {count:4d}  ({pct:.1f}%)")
    print("=" * 55)


# ── Scheduler annuel ─────────────────────────────────────────────────────────
def seconds_until_next_refresh() -> float:
    """
    Calcule le nombre de secondes avant le prochain 01/07.
    C'est à cette date que la saison écoulée est considérée complète.
    """
    today = datetime.now()
    next_refresh = datetime(today.year, REFRESH_MONTH, REFRESH_DAY)
    if today >= next_refresh:
        next_refresh = datetime(today.year + 1, REFRESH_MONTH, REFRESH_DAY)
    delta = next_refresh - today
    return delta.total_seconds()


async def run_scheduler():
    """
    Boucle infinie :
      1. Scrape toutes les saisons disponibles
      2. Sauvegarde le CSV
      3. Attend jusqu'au prochain 01/07
      4. Recommence
    """
    while True:
        print(f"\n[SCHEDULER] Lancement du scraping — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            df = await scrape_all_seasons()
            os.makedirs("data", exist_ok=True)
            df.to_csv(OUTPUT_PATH, index=False)
            print(f"[SCHEDULER] Données sauvegardées : {OUTPUT_PATH}")
            print_stats(df)
        except Exception as e:
            print(f"[SCHEDULER] Erreur lors du scraping : {e}")

        wait_sec = seconds_until_next_refresh()
        wait_h   = wait_sec / 3600
        next_dt  = datetime.now().replace(microsecond=0).fromtimestamp(
            time.time() + wait_sec
        )
        print(f"\n[SCHEDULER] Prochain scraping le {next_dt.strftime('%Y-%m-%d')} "
              f"(dans {wait_h:.0f}h / {wait_sec/86400:.1f} jours)")
        print(f"[SCHEDULER] En attente... (Ctrl+C pour arrêter)")

        await asyncio.sleep(wait_sec)


# ── Point d'entrée ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scraper Understat EPL — toutes saisons")
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Mode scheduler : tourne en continu et se met à jour chaque 01/07",
    )
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    if args.scheduler:
        print("[MODE] Scheduler annuel activé.")
        asyncio.run(run_scheduler())
    else:
        # Mode one-shot : scrape une fois et quitte
        print("[MODE] Scraping unique.")
        df = asyncio.run(scrape_all_seasons())
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n[OK] Données sauvegardées dans '{OUTPUT_PATH}'.")
        print_stats(df)
        print(f"\nAperçu (5 premiers matchs) :")
        print(df.head(5).to_string(index=False))
        print(f"\nAperçu (5 derniers matchs) :")
        print(df.tail(5).to_string(index=False))