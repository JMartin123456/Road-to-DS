from rivals_api import RivalsClient
import inspect


def analyze_match(client, match, my_uid):
    detail = client.matches.get(match.match_uid)

    # Nájdeme tím, v ktorom hráš ty
    my_team = None

    for team in detail.teams:
        if any(p.player_uid == my_uid for p in team.players):
            my_team = team
            break

    if my_team is None:
        print(f"Nepodarilo sa nájsť môj tím: {match.match_uid}")
        return None

    # Druhý tím sú súperi
    enemy_teams = [
        team for team in detail.teams
        if team.camp != my_team.camp
    ]

    enemies = []

    for team in enemy_teams:
        for p in team.players:
            enemies.append({
                "name": p.name,
                "hero": p.hero,
                "hero_id": p.top_hero_id,
                "win": my_team.is_win
            })

    return enemies


with RivalsClient() as client:
    player = client.get_player(1302818279)

    history = player.matches.fetch(
        limit=10
    )

    match = history.matches[0]

    print("Match ID:", match.match_uid)
    print("Win:", match.is_win)
    print("Season:", match.season)
    print("Hero:", match.hero_id)

    detail = client.matches.get(match.match_uid)

    team = detail.teams[0]

    
    for i, team in enumerate(detail.teams):
        print(f"\n--- TÍM {i + 1} | Camp: {team.camp} | Výhra: {team.is_win} ---")

        for p in team.players:
            print(
                "Meno:", p.name,
                "| UID:", p.player_uid,
                "| Hero ID:", p.top_hero_id,
                "| Hero:", p.hero,
                "| Počet kills:", p.kills,
                "| Deaths:", p.deaths
            )

    enemies = analyze_match(client, match, 1302818279)

    for enemy in enemies or []:
        print(
            enemy["hero"],
            "| Výsledok:", "Výhra" if enemy["win"] else "Prehra"
        )





##1302818279