import requests, json

r = requests.get(
    "https://api.qarba.com/api/v1/properties/",
    headers={"User-Agent": "TinahBot/2.0"},
    timeout=20
)
props = r.json().get("data", {}).get("results", [])

if props:
    p = props[0]
    print("=== FIELD NAMES ===")
    print(list(p.keys()))
    print()
    print("=== KEY FIELD VALUES (first property) ===")
    print("property_name        :", repr(p.get("property_name")))
    print("location             :", repr(p.get("location")))
    print("city                 :", repr(p.get("city")))
    print("state                :", repr(p.get("state")))
    print("property_type_display:", repr(p.get("property_type_display")))
    print("listing_type_display :", repr(p.get("listing_type_display")))
    print("rent_price           :", repr(p.get("rent_price")))
    print("sale_price           :", repr(p.get("sale_price")))
    print("rent_frequency       :", repr(p.get("rent_frequency")))
    print("bedrooms             :", repr(p.get("bedrooms")))
    print("bathrooms            :", repr(p.get("bathrooms")))
    print("amenities            :", repr(p.get("amenities")))
    print()
    print("=== LOCATION FIELDS — first 10 properties ===")
    for i, prop in enumerate(props[:10]):
        loc   = repr(prop.get("location"))
        city  = repr(prop.get("city"))
        state = repr(prop.get("state"))
        name  = repr(prop.get("property_name"))
        print(f"{i+1}. name={name}")
        print(f"   location={loc} | city={city} | state={state}")
        print()
else:
    print("No results found")