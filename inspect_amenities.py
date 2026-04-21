import requests

r = requests.get(
    "https://api.qarba.com/api/v1/properties/",
    headers={"User-Agent": "TinahBot/2.0"},
    timeout=20
)

all_amenities = {}
page = 1
url  = "https://api.qarba.com/api/v1/properties/"

while url:
    r    = requests.get(url, headers={"User-Agent": "TinahBot/2.0"}, timeout=20)
    data = r.json()
    props = data.get("data", {}).get("results", [])
    for p in props:
        for a in (p.get("amenities") or []):
            aid  = a.get("id")
            name = a.get("name", "").strip()
            if aid and name:
                all_amenities[aid] = name
    url = data.get("data", {}).get("next")
    print(f"Page {page} — {len(props)} props — {len(all_amenities)} unique amenities so far")
    page += 1
    if page > 50:
        break

print()
print("=== ALL UNIQUE AMENITIES ===")
for aid, name in sorted(all_amenities.items()):
    print(f"  id={aid}: '{name}'")