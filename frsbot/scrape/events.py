"""Scrape completed REBUILT events into per-event match records (scores, breakdowns, alliances).

No replay files here: the public API only exposes scores and breakdowns per match.
Usage: python -m frsbot.scrape.events
"""
import json
import urllib.request
from pathlib import Path

API = "https://reefscape-ws.frsim.com"
DATA = Path(__file__).resolve().parents[2] / "data" / "events"


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as r:
        return json.loads(r.read())


def scrape_event(ev: dict) -> dict:
    full = get(f"/events/{ev['id']}")["event"]
    series = (full.get("bracket") or {}).get("series") or []
    matches = []
    for s in series:
        if not s.get("games"):
            continue
        m = get(f"/events/{ev['id']}/match/{s['id']}")["match"]
        matches.append({
            "series": s["id"],
            "label": m.get("label"),
            "a": m.get("a"),
            "b": m.get("b"),
            "result": m.get("result"),
            "games": [
                {"n": g["n"], "score": g["score"], "winner": g["winner"], "forfeit": g.get("forfeit"),
                 "dq": g.get("dq"), "breakdown": g.get("breakdown"), "at": g.get("at")}
                for g in m.get("games", [])
            ],
        })
    return {
        "id": full["id"], "name": ev.get("name"), "game": ev.get("game"), "style": ev.get("style"),
        "size": ev.get("size"), "official": full.get("official"), "champion": full.get("champion"),
        "started": full.get("started"), "ended": full.get("ended"),
        "teams": [{"name": t.get("name"), "bot": t.get("bot", False),
                   "members": [{"name": m["name"], "bot": m.get("bot", False), "elo": m.get("elo")} for m in t.get("members", [])]}
                  for t in full.get("teams", [])],
        "matches": matches,
    }


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    events = [e for e in get("/events")["events"] if e["game"] == "rebuilt" and e["status"] == "done"]
    print(f"done rebuilt events: {len(events)}")
    for ev in events:
        out = DATA / f"{ev['id']}.json"
        if out.exists():
            continue
        try:
            rec = scrape_event(ev)
            out.write_text(json.dumps(rec, separators=(",", ":")))
            print(f"{ev['id']} {ev['name']!r}: {len(rec['matches'])} matches")
        except Exception as ex:
            print(f"ERR {ev['id']}: {ex}")


if __name__ == "__main__":
    main()
