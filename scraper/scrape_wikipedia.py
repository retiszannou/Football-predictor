import datetime
import requests
from bs4 import BeautifulSoup

def get_current_season_years() -> str:
    now = datetime.datetime.now()
    current_year = now.year
    current_month = now.month

    # Juin ou après -> Nouvelle saison
    if current_month >= 6:
        start_year = current_year
        end_year = str(current_year + 1)[2:]
    else:
        start_year = current_year - 1
        end_year = str(current_year)[2:]

    return f"{start_year}–{end_year}"

def scrape_premier_league_teams() -> list:
    season = get_current_season_years()
    url = f"https://en.wikipedia.org/wiki/{season}_Premier_League"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
    except Exception:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    teams = []
    
    # Trouver le tableau des équipes
    for table in soup.find_all("table", {"class": "wikitable"}):
        headers_text = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        
        if "team" in headers_text or "club" in headers_text:
            # On cherche l'index exact de la colonne "Team" (généralement 0 ou 1)
            team_col_index = headers_text.index("team") if "team" in headers_text else headers_text.index("club")
            
            for row in table.find_all("tr")[1:]:  # On saute la ligne d'en-tête
                cells = row.find_all(["td", "th"])
                if len(cells) > team_col_index:
                    # SÉCURITÉ CRITIQUE : On prend UNIQUEMENT la cellule de la colonne Team
                    target_cell = cells[team_col_index]
                    link = target_cell.find("a")
                    
                    if link:
                        raw_name = link.get_text(strip=True)
                        
                        # Si le texte du lien est un numéro (note de bas de page comme [c]), on prend le title
                        if raw_name.startswith("[") or len(raw_name) <= 2:
                            raw_name = link.get("title", "").replace("F.C.", "").replace("A.F.C.", "").strip()
                        
                        # Nettoyage et harmonisation pour que ça colle avec le modèle prédictif
                        clean_name = raw_name.replace("F.C.", "").replace("A.F.C.", "").strip()
                        if "Brighton" in clean_name:
                            clean_name = "Brighton"
                        elif "Tottenham" in clean_name:
                            clean_name = "Tottenham"
                        elif "Bournemouth" in clean_name:
                            clean_name = "Bournemouth"
                        elif "Ipswich" in clean_name:
                            clean_name = "Ipswich Town"
                            
                        if clean_name and clean_name not in teams:
                            teams.append(clean_name)
            break

    return sorted(list(set(teams)))[:20]

if __name__ == "__main__":
    print(scrape_premier_league_teams())