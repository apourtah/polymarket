"""Fetch all closed Polymarket weather markets (tag 'weather', id 84) with an
end date on/after 2026-01-01 from the Gamma API, using keyset pagination.

Pages are appended to data/events.jsonl (one event per line) with the cursor
kept in data/cursor.txt so the script resumes cleanly. Final output is
data/markets.json: one row per market with the fields needed downstream.
"""
import json, os, sys, time, requests

GAMMA = "https://gamma-api.polymarket.com"
TAG_WEATHER = 84
END_DATE_MIN = "2026-01-01T00:00:00Z"
EVENTS = "data/events.jsonl"
CURSOR = "data/cursor.txt"
OUT = "data/markets.json"

def fetch_events():
    cursor = open(CURSOR).read().strip() if os.path.exists(CURSOR) else None
    n = sum(1 for _ in open(EVENTS)) if os.path.exists(EVENTS) else 0
    if cursor == "DONE":
        print("  already complete", file=sys.stderr); return
    print(f"  resuming: {n} events on disk", file=sys.stderr)
    with open(EVENTS, "a") as out:
        while True:
            params = {"tag_id": TAG_WEATHER, "closed": "true", "limit": 100,
                      "end_date_min": END_DATE_MIN}
            if cursor:
                params["after_cursor"] = cursor
            for attempt in range(5):
                r = requests.get(f"{GAMMA}/events/keyset", params=params, timeout=60)
                if r.status_code == 200:
                    break
                time.sleep(2 * (attempt + 1))
            r.raise_for_status()
            d = r.json()
            page = d.get("events", [])
            for e in page:
                out.write(json.dumps(e) + "\n")
            out.flush()
            n += len(page)
            print(f"  {n} events...", file=sys.stderr, flush=True)
            cursor = d.get("next_cursor")
            if not page or not cursor:
                cursor = "DONE"
            open(CURSOR, "w").write(cursor)
            if cursor == "DONE":
                break

def flatten(events):
    rows = []
    for e in events:
        for m in e.get("markets", []):
            try:
                outcomes = json.loads(m.get("outcomes") or "[]")
                prices = json.loads(m.get("outcomePrices") or "[]")
                tokens = json.loads(m.get("clobTokenIds") or "[]")
            except json.JSONDecodeError:
                continue
            rows.append({
                "event_id": e["id"], "event_slug": e["slug"], "event_title": e["title"],
                "event_end": e.get("endDate"),
                "market_id": m["id"], "condition_id": m.get("conditionId"),
                "question": m.get("question"), "slug": m.get("slug"),
                "outcomes": outcomes, "outcome_prices": prices, "tokens": tokens,
                "closed": m.get("closed"), "uma_status": m.get("umaResolutionStatus"),
                "created_at": m.get("createdAt"), "start_date": m.get("startDate"),
                "end_date": m.get("endDate"), "closed_time": m.get("closedTime"),
                "volume": m.get("volumeNum"), "liquidity": m.get("liquidityNum"),
                "tags": [t["slug"] for t in e.get("tags", [])],
            })
    return rows

if __name__ == "__main__":
    fetch_events()
    seen = set(); ev = []
    for line in open(EVENTS):
        e = json.loads(line)
        if e["id"] not in seen:
            seen.add(e["id"]); ev.append(e)
    rows = flatten(ev)
    json.dump(rows, open(OUT, "w"))
    print(f"{len(ev)} events, {len(rows)} markets -> {OUT}")
