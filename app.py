"""
Tinah — Qarba Real Estate AI Assistant (app.py)
─────────────────────────────────────────────────────────────────────────────
Agentic architecture:  gpt-4o-mini via OpenRouter  +  5 tools:
  1. search_properties  — live Qarba property listings
  2. search_blogs       — Qarba blog articles
  3. answer_faq         — local FAQ knowledge base
  4. browse_website     — scrapes any Qarba public page (Next.js SSR aware)
  5. search_agents      — INACTIVE until QARBA_AGENT_API is added to .env

Cache: TTL-based (60 min) — auto-heals stale/error data, thread-safe.
─────────────────────────────────────────────────────────────────────────────
"""

from flask import Flask, request, jsonify, render_template, session
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import numpy as np
import os
import re
import json
import time
import requests
import threading

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════
load_dotenv()

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL    = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
QARBA_PROPERTY_API = os.getenv("QARBA_PROPERTY_API", "https://api.qarba.com/api/v1/properties/")
QARBA_BLOG_API     = os.getenv("QARBA_CLIENT_API",   "https://api.qarba.com/api/v1/blogs/")
QARBA_AGENT_API    = os.getenv("QARBA_AGENT_API")     # optional — activate later

#print("✅ OPENAI_API_KEY    :", bool(OPENAI_API_KEY))
print("✅ OPENAI_BASE_URL   :", OPENAI_BASE_URL)
print("✅ PROPERTY_API      :", QARBA_PROPERTY_API)
print("✅ BLOG_API          :", QARBA_BLOG_API)
print("✅ AGENT_API         :", QARBA_AGENT_API or "NOT SET (agent tool inactive)")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecret_change_me")

# ─── Rate Limiting ────────────────────────────────────────────────────────────
# Prevents a single user from spamming the /chat endpoint and exhausting
# OpenRouter credits. Limit: 15 requests per minute per IP address.
# Falls back gracefully if Flask-Limiter is not installed.
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=[],
        storage_uri="memory://",
    )
    _limiter_available = True
    print("✅ Rate limiter active (15 req/min per IP)")
except ImportError:
    limiter          = None
    _limiter_available = False
    print("⚠️  flask-limiter not installed — rate limiting disabled.")
    print("   Install with: pip install flask-limiter")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EMBEDDING MODEL
# ═══════════════════════════════════════════════════════════════════════════════
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TEXT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def extract_readable_text(html: str, max_chars: int = 4000) -> str:
    """
    Pull clean readable text from raw HTML.
    Works on Next.js SSR pages because the server-rendered HTML contains
    full text content inside <main>, <article>, <section>, <p> tags.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "meta", "link", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.find(id="__next") or soup
    text = main.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TTL CACHE
# ═══════════════════════════════════════════════════════════════════════════════
class TTLCache:
    """
    Thread-safe in-process cache with 60-minute TTL.
    Errors are NOT cached so failed fetches self-heal on the next request.
    """
    TTL_SECONDS = 3600  # 60 minutes

    def __init__(self):
        self._store = {}
        self._lock  = threading.Lock()

    def get(self, key):
        with self._lock:
            entry = self._store.get(key)
            if entry and (time.time() - entry["ts"]) < self.TTL_SECONDS:
                return entry["data"]
            return None

    def set(self, key, data):
        with self._lock:
            self._store[key] = {"data": data, "ts": time.time()}

    def clear(self):
        with self._lock:
            self._store.clear()

    def info(self) -> dict:
        with self._lock:
            now = time.time()
            return {
                k: {
                    "age_seconds": int(now - v["ts"]),
                    "expires_in":  max(0, int(self.TTL_SECONDS - (now - v["ts"]))),
                    "records":     len(v["data"]) if isinstance(v["data"], list) else "text",
                }
                for k, v in self._store.items()
            }

_cache = TTLCache()


# ─── Cache pre-warmer ─────────────────────────────────────────────────────────
# Loads all 490+ properties and blogs into cache on startup in a background
# thread so the first real user request is served instantly, not after a
# 30-60 second wait while 42 pages of properties are fetched sequentially.
def _prewarm_cache():
    print("🔥 Pre-warming cache in background...")
    fetch_properties()
    fetch_blogs()
    print("✅ Cache pre-warm complete.")

threading.Thread(target=_prewarm_cache, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. API FETCHERS
# ═══════════════════════════════════════════════════════════════════════════════
_HEADERS = {"User-Agent": "TinahBot/2.0"}


def _unwrap(raw) -> list:
    """
    Unwrap any Qarba API response shape into a plain list.

    Handles all known shapes:
      - bare list:                              [...]
      - {"data": [...]}                         flat list inside data
      - {"data": {"results": [...], ...}}       paginated (live Qarba shape)
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        data = raw.get("data", raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            results = data.get("results", [])
            if isinstance(results, list):
                return results
    return []


def _fetch_json(url: str, cache_key: str, label: str) -> list:
    """
    Fetch a JSON list from a Qarba API endpoint with TTL caching.
    Uses _unwrap() to handle all response shapes.
    """
    cached = _cache.get(cache_key)
    if cached is not None:
        print(f"📦 {label}: served from cache.")
        return cached

    print(f"🌐 Fetching {label}...")
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        if resp.status_code != 200:
            print(f"⚠️  {label} API {resp.status_code}: {resp.text[:200]}")
            return []
        data = _unwrap(resp.json())
        if not data:
            print(f"⚠️  {label}: no records found in response.")
            return []
        print(f"✅ {label}: {len(data)} records fetched.")
        _cache.set(cache_key, data)
        return data
    except Exception as e:
        print(f"❌ {label} fetch error:", e)
        return []


def fetch_properties() -> list:
    """
    Fetch ALL properties across all pages.
    Qarba property API is paginated:
      {"data": {"count": 490, "next": "...?page=2", "results": [...]}}
    We follow "next" until null, then cache the full list.
    """
    cached = _cache.get("properties")
    if cached is not None:
        print(f"📦 Properties: served from cache ({len(cached)} records).")
        return cached

    all_props = []
    url       = QARBA_PROPERTY_API
    page      = 1

    while url:
        print(f"🌐 Fetching Properties page {page}...")
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20)
            if resp.status_code != 200:
                print(f"⚠️  Properties page {page} error: {resp.status_code}")
                break
            raw     = resp.json()
            results = _unwrap(raw)
            if not results:
                print(f"⚠️  Properties page {page}: no results found.")
                break
            all_props.extend(results)

            # Follow the "next" pagination link
            data_block = raw.get("data", {}) if isinstance(raw, dict) else {}
            url        = data_block.get("next") if isinstance(data_block, dict) else None
            page      += 1

        except Exception as e:
            print(f"❌ Properties page {page} fetch error:", e)
            break

    if all_props:
        print(f"✅ Properties: {len(all_props)} total across {page - 1} page(s).")
        _cache.set("properties", all_props)
        # Build the embedding index in a background thread so it doesn't
        # block the pre-warm or the first user request
        threading.Thread(
            target=_build_property_index,
            args=(all_props,),
            daemon=True
        ).start()
    else:
        print("⚠️  Properties: no data retrieved.")

    return all_props


def fetch_blogs() -> list:
    return _fetch_json(QARBA_BLOG_API, "blogs", "Blogs")

def fetch_agents() -> list:
    if not QARBA_AGENT_API:
        return []
    return _fetch_json(QARBA_AGENT_API, "agents", "Agents")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. FAQ LOADER
# ═══════════════════════════════════════════════════════════════════════════════
def load_faqs() -> list:
    try:
        with open("faqs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data["faqs"] if (isinstance(data, dict) and "faqs" in data) else data
    except FileNotFoundError:
        print("⚠️  faqs.json not found — FAQ tool disabled.")
        return []
    except Exception as e:
        print("⚠️  faqs.json load error:", e)
        return []

    valid = []
    for faq in raw:
        q, a = faq.get("question"), faq.get("answer")
        if q and a:
            faq["embedding"] = embedding_model.encode(q)
            valid.append(faq)
    print(f"✅ FAQs: {len(valid)} loaded.")
    return valid

faqs = load_faqs()


def find_best_faq(query: str):
    if not faqs:
        return None
    user_emb = embedding_model.encode(query)
    scores   = [cosine_similarity([user_emb], [f["embedding"]])[0][0] for f in faqs]
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    # Primary: semantic similarity
    if best_score > 0.50:
        return faqs[best_idx]
    # Fallback: fuzzy keyword match for short/simple queries
    q_lower = query.lower()
    fuzz_scores = [fuzz.partial_ratio(q_lower, f["question"].lower()) for f in faqs]
    fuzz_best = int(np.argmax(fuzz_scores))
    if fuzz_scores[fuzz_best] >= 70:
        return faqs[fuzz_best]
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PROPERTY SEARCH ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
#
# ARCHITECTURE — three-stage pipeline:
#
# Stage 1 — HARD FILTERS (run before any scoring)
#   Extract structured criteria from the query: location (city/state), bedrooms,
#   listing type, property type, price ceiling.  Apply them as hard yes/no
#   filters directly against the API field values.  Only properties that pass
#   ALL stated criteria enter Stage 2.
#
# Stage 2 — SOFT RANKING (score the survivors)
#   Rank the filtered set by a weighted combination of:
#     • Fuzzy string match on the full property text  (fast, no GPU needed)
#     • Semantic cosine similarity via SentenceTransformer (uses pre-built index)
#
# Stage 3 — INTERNATIONAL GUARD
#   If the query mentions a country or city outside Nigeria, return a clear
#   "Qarba only covers Nigeria" message before touching any property data.
#
# Why this approach beats pure semantic search for this use-case:
#   Semantic models compress meaning, so "Abuja" and "Ebonyi" both relate to
#   "Nigeria → government → property" and score similarly.  Hard filtering on
#   the actual state/city field values sidesteps this entirely.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Pre-built embedding index (speeds up soft ranking) ───────────────────────
_property_index = {
    "texts":      [],
    "embeddings": None,
    "properties": [],
}
_index_lock = threading.Lock()


def _build_property_index(properties: list):
    """Encode all property texts once; reused for every search."""
    with _index_lock:
        texts = [_property_text(p) for p in properties]
        print(f"🧠 Building embedding index for {len(properties)} properties...")
        embeddings = embedding_model.encode(texts, batch_size=64, show_progress_bar=False)
        _property_index["texts"]      = texts
        _property_index["embeddings"] = embeddings
        _property_index["properties"] = properties
        print("✅ Embedding index ready.")


def _property_text(p: dict) -> str:
    """Canonical text representation of a property for embedding/fuzzy matching."""
    amenity_names = " ".join(a.get("name", "") for a in (p.get("amenities") or []))
    bedrooms = p.get("bedrooms") or 0
    bed_str  = f"{bedrooms} bedroom" if bedrooms else ""
    return clean_text(" ".join([
        str(p.get("property_name",         "")),
        str(p.get("location",              "")),
        str(p.get("city",                  "")),
        str(p.get("state",                 "")),
        str(p.get("property_type_display", "")),
        str(p.get("listing_type_display",  "")),
        bed_str,
        amenity_names,
    ]))


# ── International location guard ─────────────────────────────────────────────
INTERNATIONAL_LOCATIONS = {
    "canada", "usa", "united states", "america", "uk", "united kingdom",
    "england", "ghana", "kenya", "south africa", "egypt", "ethiopia",
    "cameroon", "senegal", "tanzania", "uganda", "rwanda", "zimbabwe",
    "zambia", "angola", "mozambique", "botswana", "namibia", "malawi",
    "australia", "new zealand", "germany", "france", "italy", "spain",
    "portugal", "netherlands", "belgium", "sweden", "norway", "denmark",
    "china", "japan", "india", "brazil", "mexico", "argentina",
    "dubai", "uae", "qatar", "saudi arabia", "kuwait", "bahrain",
    "london", "new york", "toronto", "accra", "nairobi", "johannesburg",
    "cape town", "cairo", "paris", "berlin", "sydney",
    "los angeles", "chicago", "houston", "miami", "atlanta",
}


def _check_international(query: str):
    """Return detected international location name, or None."""
    q = query.lower()
    for loc in sorted(INTERNATIONAL_LOCATIONS, key=len, reverse=True):
        if loc in q:
            return loc
    return None


# ── Query parser — extract hard filter criteria ───────────────────────────────
import re as _re

# Nigerian states — both short form and "X State" variant
_NIGERIAN_STATES = [
    "abia", "adamawa", "akwa ibom", "anambra", "bauchi", "bayelsa",
    "benue", "borno", "cross river", "delta", "ebonyi", "edo", "ekiti",
    "enugu", "gombe", "imo", "jigawa", "kaduna", "kano", "katsina",
    "kebbi", "kogi", "kwara", "lagos", "nasarawa", "niger", "ogun",
    "ondo", "osun", "oyo", "plateau", "rivers", "sokoto", "taraba",
    "yobe", "zamfara", "abuja", "fct",
]

# Nigerian cities / areas known to appear in Qarba listings
_NIGERIAN_CITIES = [
    # Ebonyi / Abakaliki and surroundings
    "abakaliki", "ugwuachara", "nkaliki", "igweokpu", "waterworks",
    "presco", "presco campus", "mile 50", "afikpo road", "ishieke",
    "kpirikpiri", "achara layout", "achara", "azugwu", "onueke",
    "isieke", "ogoja road", "waterworks road", "afikpo", "edda",
    "ezza south", "ezza north", "ikwo", "ohaukwu",
    # Port Harcourt
    "port harcourt", "rumuola", "rumuokoro", "trans amadi",
    "old gra", "new gra", "gra", "peter odili", "eliozu", "rumuigbo",
    "elechi", "diobu", "mile 1", "mile 2", "mile 3", "mile 4",
    "woji", "rukpokwu", "ozuoba", "eneka", "obia akpor", "obio",
    # Lagos
    "lagos island", "lekki", "victoria island", "ikoyi", "yaba",
    "surulere", "ajah", "ikeja", "festac", "gbagada", "magodo",
    "maryland", "ojodu", "berger", "ogba", "ifako", "agege",
    "mushin", "oshodi", "isolo", "egbeda", "iyana ipaja", "ipaja",
    "badagry", "epe", "ikorodu", "sangotedo", "chevron", "osapa",
    "awoyaya", "ibeju lekki", "ojota",
    # Abuja
    "wuse", "garki", "maitama", "asokoro", "gwarinpa", "kubwa",
    "lokogoma", "apo", "jabi", "utako", "gudu", "lugbe",
    "kado", "dawaki", "gaduwa", "life camp", "lifecamp", "nbora",
    "kaura", "katampe", "wuye", "mpape", "pyakasa", "bwari",
    # Enugu
    "enugu", "nsukka", "agbani", "emene", "independence layout",
    "gra enugu", "new haven", "achara layout enugu",
    # Other cities
    "aba", "umuahia", "owerri", "onitsha", "awka", "nnewi",
    "benin city", "warri", "asaba", "sapele", "calabar", "uyo",
    "akure", "ado ekiti", "ibadan", "abeokuta", "ilorin",
    "minna", "lokoja", "jos", "kaduna city", "kano city",
    "zaria", "sokoto", "maiduguri", "yola", "makurdi",
]

# Property type synonyms → canonical display value substrings.
# Sorted longest-first at match time so more specific terms win
# (e.g. "single room" wins over "room", "room and parlour" wins over "room").
_PROPERTY_TYPE_MAP = {
    # Specific room types (check before generic "room" or "flat")
    "room and parlour":   "parlour",
    "room and parlor":    "parlour",
    "room & parlour":     "parlour",
    "room & parlor":      "parlour",
    "room self contain":  "single",
    "single room":        "single",
    "mini flat":          "mini",
    "miniflat":           "mini",
    "boys quarters":      "boys quarter",
    "boys quarter":       "boys quarter",
    "bedsitter":          "bedsit",
    "bed sitter":         "bedsit",
    "self contain":       "self",
    "selfcon":            "self",
    "self-con":           "self",
    # Standard types
    "penthouse":          "penthouse",
    "apartment":          "apartment",
    "bungalow":           "bungalow",
    "townhouse":          "townhouse",
    "terrace":            "terrace",
    "mansion":            "mansion",
    "duplex":             "duplex",
    "studio":             "studio",
    "villa":              "villa",
    "flat":               "flat",
    "house":              "house",
    # Commercial
    "warehouse":          "warehouse",
    "commercial":         "commercial",
    "office":             "office",
    "shop":               "shop",
    "land":               "land",
}

# Listing type synonyms
_LISTING_TYPE_MAP = {
    "rent": "rent", "rental": "rent", "for rent": "rent",
    "lease": "rent", "let": "rent",
    "buy": "sale", "purchase": "sale", "for sale": "sale",
    "sale": "sale", "sell": "sale",
    "shortlet": "shortlet", "short let": "shortlet",
    "short-let": "shortlet", "shortterm": "shortlet",
}

# Price extraction pattern: optional ₦, digits, optional k/m suffix
_PRICE_RE = _re.compile(
    r"(?:under|below|less than|max|maximum|not more than|within)?\s*"
    r"[₦#]?\s*(\d[\d,]*)\s*(k|thousand|m|million)?",
    _re.IGNORECASE
)

# Bedroom extraction
_BED_RE = _re.compile(
    r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*(?:bed(?:room)?s?|br\b)",
    _re.IGNORECASE
)

# Price range: between 500k and 1m
_PRICE_RANGE_RE = _re.compile(
    r"(?:between|from)\s*[₦#]?\s*(\d[\d,]*)\s*(k|thousand|m|million)?\s*"
    r"(?:to|and)\s*[₦#]?\s*(\d[\d,]*)\s*(k|thousand|m|million)?",
    _re.IGNORECASE
)

_NON_LOCATION_WORDS = {
    "the", "a", "an", "this", "that", "these", "those", "some", "any",
    "my", "your", "our", "their", "its", "me", "you", "us",
    "bedroom", "bedrooms", "bed", "beds", "bathroom", "bathrooms",
    "flat", "apartment", "house", "property", "properties",
    "room", "rooms", "studio", "duplex", "bungalow",
    "rent", "rental", "sale", "shortlet", "lease", "buy", "purchase",
    "for", "with", "and", "or", "but", "of", "to", "from",
    "affordable", "cheap", "nice", "good", "modern", "new", "old",
    "available", "furnished", "unfurnished", "serviced",
    "nigeria", "nigerian", "one", "two", "three", "four", "five",
}

_FREE_LOC_RE = _re.compile(
    r"(?<![a-z0-9])(?:at|on|along|around|near|off|beside|opposite|in)\s+"
    r"([a-z][a-z0-9\s]{1,45}?)(?=\s+(?:in|for|with|and|,|\.)| *$)",
    _re.IGNORECASE
)


def _extract_free_location(q: str, known_locs: list, known_states: list) -> list:
    """
    Extract street/road/landmark names from the query not already captured
    in the hardcoded city or state lists.
    """
    cleaned = q
    for loc in sorted(known_locs + known_states, key=len, reverse=True):
        cleaned = cleaned.replace(loc, " ")
    cleaned = _re.sub(r"\s+", " ", cleaned).strip()
    result = []
    for m in _FREE_LOC_RE.finditer(cleaned):
        phrase = m.group(1).strip()
        phrase = _re.sub(r"\s+(?:in|for|with|that|which|and|or)$", "", phrase).strip()
        words = phrase.split()
        if not phrase or len(phrase) < 3:
            continue
        if len(words) == 1 and words[0] in _NON_LOCATION_WORDS:
            continue
        if phrase in _PROPERTY_TYPE_MAP or phrase in _LISTING_TYPE_MAP:
            continue
        if phrase not in result:
            result.append(phrase)
    return result

# Amenity synonym map → exact Qarba amenity name (from API inspection)
# Keys are user query terms; values are the exact name stored in the API.
# All 13 Qarba amenities are covered. Add new ones as Qarba expands.
_AMENITY_MAP = {
    # Visitors toilet
    "visitors toilet":   "Visitors toilet",
    "visitor toilet":    "Visitors toilet",
    "guest toilet":      "Visitors toilet",
    "visitor wc":        "Visitors toilet",
    # Ensuite bedrooms
    "ensuite":           "Ensuite bedrooms",
    "en suite":          "Ensuite bedrooms",
    "en-suite":          "Ensuite bedrooms",
    "ensuite bedroom":   "Ensuite bedrooms",
    # Parking space
    "parking":           "Parking space",
    "parking space":     "Parking space",
    "car park":          "Parking space",
    "garage":            "Parking space",
    "car space":         "Parking space",
    # Modern kitchen
    "modern kitchen":    "Modern kitchen",
    "fitted kitchen":    "Modern kitchen",
    "kitchen":           "Modern kitchen",
    # Dining
    "dining":            "Dining",
    "dining room":       "Dining",
    "dining area":       "Dining",
    # Wardrobes
    "wardrobe":          "Wardrobes",
    "wardrobes":         "Wardrobes",
    "built in wardrobe": "Wardrobes",
    "inbuilt wardrobe":  "Wardrobes",
    # CCTV Camera
    "cctv":              "CCTV Camera",
    "cctv camera":       "CCTV Camera",
    "security camera":   "CCTV Camera",
    "surveillance":      "CCTV Camera",
    # Internet / Wi-Fi
    "wifi":              "Internet/Wi-Fi",
    "wi-fi":             "Internet/Wi-Fi",
    "internet":          "Internet/Wi-Fi",
    "wi fi":             "Internet/Wi-Fi",
    "broadband":         "Internet/Wi-Fi",
    # Swimming pool
    "swimming pool":     "Swimming pool",
    "pool":              "Swimming pool",
    "swim":              "Swimming pool",
    # Security Guard
    "security guard":    "Security Guard",
    "gateman":           "Security Guard",
    "gate man":          "Security Guard",
    "security":          "Security Guard",
    "guard":             "Security Guard",
    # POP Ceiling
    "pop ceiling":       "POP Ceiling",
    "pop":               "POP Ceiling",
    "false ceiling":     "POP Ceiling",
    "ceiling":           "POP Ceiling",
    # Electricity
    "electricity":       "Electricity",
    "prepaid meter":     "Electricity",
    "light":             "Electricity",
    "power":             "Electricity",
    "meter":             "Electricity",
    # Water supply
    "water supply":      "Water supply",
    "running water":     "Water supply",
    "borehole":          "Water supply",
    "tap water":         "Water supply",
    "water":             "Water supply",
}



def _parse_query(query: str) -> dict:
    """
    Extract all structured criteria from a natural language query.
    Returns: locations, states, free_location, property_type, listing_type,
             min_beds, max_beds, min_price, max_price, amenities.
    """
    q = query.lower()
    criteria = {
        "locations":     [],
        "states":        [],
        "free_location": [],
        "property_type": None,
        "listing_type":  None,
        "min_beds":      None,
        "max_beds":      None,
        "min_price":     None,
        "max_price":     None,
        "amenities":     [],
    }

    for state in sorted(_NIGERIAN_STATES, key=len, reverse=True):
        if _re.search(r"\b" + _re.escape(state) + r"\b", q):
            criteria["states"].append(state)

    for city in sorted(_NIGERIAN_CITIES, key=len, reverse=True):
        if _re.search(r"(?<![a-z0-9])" + _re.escape(city) + r"(?![a-z0-9])", q):
            criteria["locations"].append(city)

    criteria["free_location"] = _extract_free_location(
        q, criteria["locations"], criteria["states"]
    )

    for keyword, canonical in sorted(_PROPERTY_TYPE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if keyword in q:
            criteria["property_type"] = canonical
            break

    for keyword, canonical in sorted(_LISTING_TYPE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if keyword in q:
            criteria["listing_type"] = canonical
            break

    _WORD_NUMS = {"one":1,"two":2,"three":3,"four":4,"five":5,
                  "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    bed_matches = _BED_RE.findall(q)
    if bed_matches:
        counts = [_WORD_NUMS[m.lower()] if m.lower() in _WORD_NUMS else int(m)
                  for m in bed_matches]
        criteria["min_beds"] = min(counts)
        criteria["max_beds"] = max(counts)

    range_m = _PRICE_RANGE_RE.search(q)
    if range_m:
        def _to_naira(s, suf):
            a = float(s.replace(",", ""))
            s2 = (suf or "").lower()
            if s2 in ("k", "thousand"): a *= 1_000
            elif s2 in ("m", "million"): a *= 1_000_000
            return a
        criteria["min_price"] = _to_naira(range_m.group(1), range_m.group(2))
        criteria["max_price"] = _to_naira(range_m.group(3), range_m.group(4))
    else:
        for amount_str, suffix in _PRICE_RE.findall(q):
            amount = float(amount_str.replace(",", ""))
            suffix = suffix.lower() if suffix else ""
            if suffix in ("k", "thousand"): amount *= 1_000
            elif suffix in ("m", "million"): amount *= 1_000_000
            if amount > 1000:
                criteria["max_price"] = amount
                break

    for keyword, canonical in sorted(_AMENITY_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if _re.search(r"\b" + _re.escape(keyword) + r"\b", q):
            if canonical not in criteria["amenities"]:
                criteria["amenities"].append(canonical)

    return criteria


def _normalise(s) -> str:
    """Lowercase + strip for consistent comparison."""
    return str(s or "").lower().strip()


def _hard_filter(properties: list, criteria: dict) -> list:
    """
    Apply hard yes/no filters. Returns properties passing ALL stated
    constraints. Unstated constraints (None/[]) are skipped entirely.
    """
    result = []
    for p in properties:
        prop_state    = _normalise(p.get("state",    ""))
        prop_city     = _normalise(p.get("city",     ""))
        prop_location = _normalise(p.get("location", ""))
        prop_type     = _normalise(p.get("property_type_display", ""))
        prop_listing  = _normalise(p.get("listing_type_display",  ""))
        prop_name     = _normalise(p.get("property_name", ""))
        prop_beds     = p.get("bedrooms") or 0
        prop_rent     = p.get("rent_price")  or 0
        prop_sale     = p.get("sale_price")  or 0
        prop_price    = prop_rent or prop_sale

        # State
        if criteria["states"]:
            state_match = False
            for s in criteria["states"]:
                clean_prop_state = prop_state.replace(" state", "").strip()
                if s == clean_prop_state or s in prop_state:
                    state_match = True
                    break
            if not state_match:
                continue

        # Known city
        if criteria["locations"]:
            loc_match = False
            for loc in criteria["locations"]:
                if loc in prop_city or loc in prop_location:
                    loc_match = True
                    break
            if not loc_match:
                continue

        # Free-text location (street / road / landmark)
        if criteria.get("free_location"):
            floc_match = False
            for floc in criteria["free_location"]:
                if (floc in prop_location or
                        floc in prop_city or
                        floc in prop_name or
                        fuzz.partial_ratio(floc, prop_location) >= 80):
                    floc_match = True
                    break
            if not floc_match:
                continue

        # Property type (also check property name for non-standard labels)
        if criteria["property_type"]:
            if (criteria["property_type"] not in prop_type and
                    criteria["property_type"] not in prop_name):
                continue

        # Listing type
        if criteria["listing_type"]:
            if criteria["listing_type"] not in prop_listing:
                continue

        # Bedrooms
        if criteria["min_beds"] is not None:
            if prop_beds < criteria["min_beds"]:
                continue
        if criteria["max_beds"] is not None:
            if prop_beds > criteria["max_beds"]:
                continue

        # Price ceiling
        if criteria["max_price"] is not None and prop_price > 0:
            if prop_price > criteria["max_price"]:
                continue

        # Price floor
        if criteria.get("min_price") is not None and prop_price > 0:
            if prop_price < criteria["min_price"]:
                continue

        # Amenities: ALL requested must be present
        if criteria.get("amenities"):
            prop_amenity_names = [
                _normalise(a.get("name", ""))
                for a in (p.get("amenities") or [])
            ]
            if not all(
                _normalise(req) in prop_amenity_names
                for req in criteria["amenities"]
            ):
                continue

        result.append(p)
    return result


def _describe_search(criteria: dict) -> str:
    """Build a human-readable description of search criteria."""
    naira = "₦"
    parts = []
    if criteria.get("free_location"):
        parts.append(", ".join(l.title() for l in criteria["free_location"]))
    if criteria["locations"]:
        parts.append(", ".join(l.title() for l in criteria["locations"]))
    if criteria["states"]:
        parts.append(", ".join(s.title() + " State" for s in criteria["states"]))
    if criteria["property_type"]:
        parts.append(criteria["property_type"])
    if criteria["listing_type"]:
        parts.append(f"for {criteria['listing_type']}")
    if criteria.get("min_beds"):
        parts.append(f"{criteria['min_beds']} bedroom(s)")
    if criteria.get("max_price"):
        parts.append(f"under {naira}{int(criteria['max_price']):,}")
    if criteria.get("min_price"):
        parts.append(f"above {naira}{int(criteria['min_price']):,}")
    if criteria.get("amenities"):
        parts.append("with " + ", ".join(criteria["amenities"]))
    return " | ".join(parts) if parts else "your criteria"


def _sort_by_price_proximity(properties: list, min_price, max_price) -> list:
    """Sort properties by closeness to the requested price (ascending distance)."""
    def dist(p):
        price = p.get("rent_price") or p.get("sale_price") or 0
        if not price:
            return float("inf")
        if min_price and max_price:
            mid = (min_price + max_price) / 2
            return abs(price - mid)
        elif max_price:
            return abs(price - max_price)
        elif min_price:
            return abs(price - min_price)
        return 0
    return sorted(properties, key=dist)


# Property types that are close enough to suggest as alternatives
_TYPE_GROUPS = {
    "flat":      {"apartment", "mini", "studio"},
    "apartment": {"flat", "mini", "studio"},
    "mini":      {"flat", "apartment", "studio"},
    "studio":    {"flat", "apartment", "mini"},
    "duplex":    {"bungalow", "terrace", "detached"},
    "bungalow":  {"duplex", "terrace", "semi"},
    "terrace":   {"bungalow", "semi", "duplex"},
    "semi":      {"terrace", "bungalow", "detached"},
    "detached":  {"duplex", "semi", "bungalow"},
    "self":      {"single", "bedsit", "mini"},
    "single":    {"self", "bedsit"},
    "bedsit":    {"single", "self"},
    "parlour":   {"single", "self", "bedsit"},
}


def _search_with_fallback(properties: list, criteria: dict):
    """
    Try exact search first, then progressively relax constraints until results
    are found. Returns (results, exact_match: bool, notice: str or None).
    The notice is an honest message explaining what was not found and why.
    """
    naira = "₦"

    exact = _hard_filter(properties, criteria)
    if exact:
        return exact, True, None

    # Step 1: relax street/road/landmark, keep known city + state
    if criteria.get("free_location"):
        r = {**criteria, "free_location": []}
        res = _hard_filter(properties, r)
        if res:
            locs = ", ".join(l.title() for l in criteria["free_location"])
            nearby = (
                ", ".join(l.title() for l in criteria["locations"]) or
                ", ".join(s.title() + " State" for s in criteria["states"]) or
                "the area"
            )
            notice = (
                f"I currently don't have any properties listed on/near {locs}. "
                f"Here are the closest available options in {nearby}:"
            )
            return res, False, notice

    # Step 2: relax specific city, keep state + type + beds + price
    if criteria["locations"]:
        r = {**criteria, "locations": [], "free_location": []}
        res = _hard_filter(properties, r)
        if res:
            locs = ", ".join(l.title() for l in criteria["locations"])
            state_ctx = (
                ", ".join(s.title() + " State" for s in criteria["states"])
                if criteria["states"] else "the surrounding area"
            )
            notice = (
                f"I don't have any properties listed in {locs} at the moment. "
                f"Here are similar options available in {state_ctx}:"
            )
            return res, False, notice

    # Step 3: relax price, keep location + type + beds
    if criteria.get("max_price") or criteria.get("min_price"):
        r = {**criteria, "min_price": None, "max_price": None}
        res = _hard_filter(properties, r)
        if res:
            # Sort by closeness to the requested price so nearest options show first
            res = _sort_by_price_proximity(
                res, criteria.get("min_price"), criteria.get("max_price")
            )
            if criteria.get("min_price") and criteria.get("max_price"):
                p_desc = f"between {naira}{int(criteria['min_price']):,} and {naira}{int(criteria['max_price']):,}"
            elif criteria.get("max_price"):
                p_desc = f"under {naira}{int(criteria['max_price']):,}"
            else:
                p_desc = f"above {naira}{int(criteria['min_price']):,}"
            notice = (
                f"I don't currently have any properties under {naira}{int(criteria.get('max_price') or criteria.get('min_price')):,}. "
                f"Here are the closest options to your budget:"
            ) if criteria.get("max_price") and not criteria.get("min_price") else (
                f"No properties found {p_desc}. "
                f"Here are the closest options to your budget:"
            )
            return res, False, notice

    # Step 3b: relax type to sibling types, keep location + price
    if criteria.get("property_type") and criteria["property_type"] in _TYPE_GROUPS:
        siblings = _TYPE_GROUPS[criteria["property_type"]]
        for sib_type in siblings:
            r = {**criteria, "property_type": sib_type, "free_location": []}
            res = _hard_filter(properties, r)
            if res:
                notice = (
                    f"I don't have exact {criteria['property_type']} listings matching your criteria. "
                    f"Here are similar properties you might like:"
                )
                return res, False, notice

    # Step 4: relax amenities, keep location + type + beds
    if criteria.get("amenities"):
        r = {**criteria, "amenities": [], "free_location": []}
        res = _hard_filter(properties, r)
        if res:
            amen = ", ".join(criteria["amenities"])
            notice = (
                f"No properties found with all requested amenities ({amen}). "
                f"Here are similar options:"
            )
            return res, False, notice

    # Step 5: relax bedrooms, keep location + type
    if criteria.get("min_beds") or criteria.get("max_beds"):
        r = {**criteria, "min_beds": None, "max_beds": None, "free_location": []}
        res = _hard_filter(properties, r)
        if res:
            bed_n = criteria.get("min_beds", "?")
            notice = (
                f"No {bed_n}-bedroom properties found matching your criteria. "
                f"Here are similar options with different bedroom counts:"
            )
            return res, False, notice

    # Step 6: type + listing type only (all of Nigeria)
    if criteria["property_type"] or criteria["listing_type"]:
        r = {
            "locations": [], "states": [], "free_location": [],
            "property_type": criteria["property_type"],
            "listing_type":  criteria["listing_type"],
            "min_beds": None, "max_beds": None,
            "min_price": None, "max_price": None,
            "amenities": [],
        }
        res = _hard_filter(properties, r)
        if res:
            notice = (
                "I couldn't find properties matching all your criteria. "
                "Here are some similar options currently available on Qarba:"
            )
            return res, False, notice

    return [], False, None


def _soft_rank(query: str, properties: list) -> list:
    """
    Rank a list of properties by relevance to the query.
    Uses pre-built embedding index when available for speed,
    falls back to fuzzy-only for small lists or when index is not ready.
    Returns properties sorted best-first (no threshold cut — caller decides).
    """
    if not properties:
        return []

    q_lower = query.lower()

    with _index_lock:
        index_ready = (
            _property_index["embeddings"] is not None
            and len(_property_index["properties"]) > 0
            and _property_index["properties"][0] is properties[0]
        )

    if index_ready and len(properties) > 5:
        # Fast path: look up pre-computed embeddings by position
        with _index_lock:
            all_props  = _property_index["properties"]
            all_texts  = _property_index["texts"]
            embeddings = _property_index["embeddings"]

        # Build index map for this subset
        prop_positions = {id(p): i for i, p in enumerate(all_props)}
        query_emb = embedding_model.encode(query)

        scored = []
        for p in properties:
            pos = prop_positions.get(id(p))
            if pos is not None:
                text  = all_texts[pos]
                sem   = float(cosine_similarity([query_emb], [embeddings[pos]])[0][0])
            else:
                text  = _property_text(p)
                sem   = 0.0
            fuzzy = fuzz.partial_ratio(q_lower, text.lower()) / 100.0
            score = 0.55 * sem + 0.45 * fuzzy
            scored.append((score, p))
    else:
        # Slow path: encode on the fly (only for small subsets)
        query_emb = embedding_model.encode(query)
        scored    = []
        for p in properties:
            text  = _property_text(p)
            sem   = float(cosine_similarity([query_emb], [embedding_model.encode(text)])[0][0])
            fuzzy = fuzz.partial_ratio(q_lower, text.lower()) / 100.0
            score = 0.55 * sem + 0.45 * fuzzy
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


def hybrid_match(query: str, items: list, text_fn) -> list:
    """
    Legacy entry point kept for compatibility with blog/FAQ callers.
    For properties, tool_search_properties calls _hard_filter + _soft_rank
    directly instead of this function.
    For non-property items (blogs etc), falls back to fuzzy-only ranking.
    """
    if not items:
        return []
    q_lower = query.lower()
    scored  = []
    for item in items:
        text  = clean_text(text_fn(item))
        fuzzy = fuzz.partial_ratio(q_lower, text.lower()) / 100.0
        scored.append((fuzzy, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored if score > 0.30]


# 9. QARBA PUBLIC PAGES — STATIC KNOWLEDGE BASE + PLAYWRIGHT FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY STATIC CONTENT:
# Qarba's frontend is built with Next.js in client-side rendering mode.
# requests.get() receives the HTML shell but the actual page content is
# injected by JavaScript after load — so scraping returns empty text.
#
# SOLUTION: Curated static content for key pages (reliable, instant, no cost).
# FALLBACK: Playwright headless browser for any page not in the static store.
#
# TO UPDATE: Edit the text values below whenever Qarba updates a page.
# TO ADD A NEW PAGE: Add a new key to both QARBA_PAGES and QARBA_STATIC_CONTENT.
# ═══════════════════════════════════════════════════════════════════════════════

QARBA_PAGES = {
    "about":                "https://qarba.com/about-us",
    "about us":             "https://qarba.com/about-us",
    "contact":              "https://qarba.com/contact",
    "contact us":           "https://qarba.com/contact",
    "privacy":              "https://qarba.com/privacy-policy",
    "privacy policy":       "https://qarba.com/privacy-policy",
    "terms":                "https://qarba.com/terms-conditions",
    "terms and conditions": "https://qarba.com/terms-conditions",
    "manage":               "https://qarba.com/manage",
    "careers":              "https://qarba.com/careers",
    "jobs":                 "https://qarba.com/careers",
    "register":             "https://qarba.com/create-account",
    "create account":       "https://qarba.com/create-account",
    "sign up":              "https://qarba.com/create-account",
}

# ── Static knowledge base ─────────────────────────────────────────────────────
# Fill in the real content from each page. The more detail you add here,
# the better Tinah can answer questions about Qarba.
# Instructions: Visit each URL, copy the key content, paste it below.

QARBA_STATIC_CONTENT = {
    "https://qarba.com/about-us": """
Your Trusted Partner in Nigerian Real Estate
At QARBA, we're revolutionizing property transactions across Nigeria through innovation, transparency,
 and exceptional service. Our platform connects property seekers with their perfect spaces and helps property 
 owners reach the right audience.

 Our Core Values are:
Trust & Security
We prioritize the security of your transactions and personal information above all else.
Innovation
We continuously evolve our platform to provide cutting-edge solutions for property transactions.
Customer First
Your satisfaction drives every decision we make and every feature we develop.

Our Mission
To simplify real estate transactions by creating a transparent, secure, and user-friendly digital 
environment for property buyers, renters, and sellers in Nigeria.

Our Vision
To become Nigeria's most trusted digital real estate platform, revolutionizing how people buy, sell, 
and rent properties through innovative technology and unwavering commitment to transparency and 
customer satisfaction.

Meet the Team
Behind QARBA is a diverse and passionate team of experts

Dr. Francis Nwebonyi
Dr. Francis Nwebonyi
Founder & CEO

Augustine Anwuchie
Augustine Anwuchie
Project Manager

Barr Obinna Nwali
Barr Obinna Nwali
Legal Advisor

Ngozi Paschaline
Ngozi Paschaline
Director of Communications

Steve Tylor
Steve Tylor
Strategic Advisor

Dr Elias Eze
Dr Elias Eze
Consultant

Michael Nwogha
Michael Nwogha
Backend Engineer

Godwin Chisom .H.
Godwin Chisom .H.
Frontend Engineer

Onuorah Victor .M.
Onuorah Victor .M.
Product Designer

Nwebonyi Harrison A
Ai Engineer

Qarba provides a wide range of property listings including residential apartments,
houses, commercial properties, and land across major Nigerian cities including
Lagos, Abuja, Port Harcourt, and more.

Our platform enables users to search for properties by location, price, type,
and amenities. We connect users directly with verified real estate agents to
ensure a smooth and trustworthy property transaction experience.

Vision: To be Africa's leading technology-driven real estate platform.
Mission: To simplify property discovery and transactions for every Nigerian.

Website: https://qarba.com
    """,

    "https://qarba.com/contact": """
Contact Qarba:

Contact Us
Get in touch with us for any inquiries about properties, listings, or general questions. 
We're here to help you with your property journey.

Contact Information
Phone
+234 815 590 1163

WhatsApp
+234 815 590 1163

Email
contact@qarba.com

Office Address
10 Brackenbury Street, Ebonyi state, Nigeria

Business Hours
Monday - Saturday: 9:00 AM - 6:00 PM

Ready to Get Started?
Join thousands of satisfied customers who trust us with their property needs.
Call Us Now
WhatsApp Us

Website contact form: https://qarba.com/contact
Email: contact@qarba.com
    """,

    "https://qarba.com/privacy-policy": """
QARBA Privacy Policy
Last updated: October 17, 2025

QARBA Properties ("QARBA," "we," "our," or "us") operate the QARBA website and mobile application 
(collectively, the "Platform") as a real estate technology service that connects buyers, sellers, 
renters, and agents.

This Privacy Policy describes how we collect, use, disclose, and protect your information when you use our Platform.
 By accessing or using QARBA, you agree to the terms of this Privacy Policy. If you do not agree, please do not
   proceed with using the Platform.

1. Information We Collect
We collect different types of information to provide and improve our services:

a. Personal Information
When you create an account or use certain features, we may collect:

Full name, email address, and phone number
Profile information (e.g., agency name, professional license, or company details)
Property details, descriptions, and uploaded media (images/videos/documents)
Location data, and/or payment or financial related information, where applicable

b. Usage and Log Data
When you use QARBA, we may automatically collect:
IP address and device identifiers
Browser type and operating system
Pages viewed, actions performed, and time spent on the Platform
Access times, dates, location data, and referring URLs

c. Cookies and Tracking Technologies
QARBA and our partners may use cookies and similar technologies to:
Recognize returning users
Improve performance and user experience
Deliver personalized content and advertisements
You can manage or disable cookies through your browser settings, but doing so may affect certain features of the Platform.

2. How We Use Your Information
We use collected data to:
Provide, maintain, and improve our Platform and services
Facilitate property listings, communications, and transactions
Verify user identity and prevent fraudulent activity
Customize user experience and recommendations
Communicate updates, promotions, or service-related information
Comply with legal obligations and enforce our Terms and Conditions

3. Sharing and Disclosure
We do not sell or rent your personal data. However, we may share your information under these limited circumstances:
With Service Providers: Trusted third-party vendors that assist in hosting, analytics, payment processing, or marketing.
With Other Users: When you interact on the Platform (e.g., viewing listings or contacting an agent).
For Legal Reasons: If required by law, regulation, or court order.
Business Transfers: In case of a merger, acquisition, or sale of company assets.
All third parties are obligated to use your information solely for the services they perform for QARBA.

4. Data Retention
We retain personal information as long as your account is active or as necessary to provide services and comply with 
legal obligations. You may request deletion of your data by contacting contact@qarba.com, subject to applicable 
retention policies or laws.

5. Security
We implement appropriate administrative, technical, and physical measures to protect your data. However, 
please note that no online system is a 100% secure, and we cannot guarantee absolute protection against 
unauthorized access.

6. Third-Party Services
QARBA may use or link to third-party tools and services such as:

Google Analytics
Facebook Business Tools
These third parties may collect data as described in their respective privacy policies. We encourage you to review 
their policies for more details.

7. Links to External Websites
Our Platform may contain links to third-party websites. QARBA is not responsible for the privacy practices or 
content of these external sites. Please review their privacy policies before providing any personal information.

8. Children's Privacy
QARBA does not knowingly collect or process data from individuals under the age of 18. If we discover that we 
have inadvertently collected personal data from a minor, we will promptly delete it. Parents or guardians may 
contact contact@qarba.com to request removal.

9. Your Rights
Depending on your jurisdiction, you may have the right to:

Access, correct, or delete your personal information
Withdraw consent for marketing communications
Request data portability or restriction of processing
To exercise these rights, please contact us at contact@qarba.com

10. Changes to This Privacy Policy
We may update this Privacy Policy periodically, and without (prior) notice. When we do, the updated version will be 
posted on this page with a revised date of update. Significant changes may be communicated via email or in-app 
notification.

11. Contact Us
If you have questions, concerns, or complaints regarding this Privacy Policy or our data practices, please contact us 
at contact@qarba.com or via our website: www.qarba.com.

For the full privacy policy, visit: https://qarba.com/privacy-policy
    """,

    "https://qarba.com/terms-conditions": """
QARBA – Terms and Conditions
Last updated: October 22, 2025

Introduction
Welcome to QARBA, a real estate technology platform that connects buyers, sellers, renters, and agents. 
By accessing or using our website, mobile app, or related services ("Platform"), you agree to comply with 
and be bound by these Terms and Conditions ("Terms"). If you do not agree to these Terms, please do not use
 our Platform.

About QARBA
QARBA provides real estate listings, property management tools, and digital services that facilitate 
transactions between Agents, Landlords and Clients (renter/buyers). We may not act as a real estate agent
 or broker and may not participate in direct property transactions between users. If at any point we are 
 acting directly as an Agent, it will be explicitly made clear to both parties.

User Eligibility
You must be at least 18 years old to use our services.
You must provide accurate, complete, and current information when creating an account.
Real estate agents or agencies must possess valid licenses as required by law. It is your responsibility to ensure
 you possess due licences before registering on our platform.
You are responsible for maintaining the confidentiality of your login credentials and for all activities
 under your account.
You must promptly report any unauthorized access or suspicious activity on your account.
Property Listings & Transactions
I. Listing Requirements
All property information, images, and descriptions must be accurate, truthful, and up to date.
Listed properties must be legally available for sale, lease, or rent, as indicated.
Images and videos must represent the actual property and comply with copyright laws.
Prices must include all mandatory fees and charges.
Listing of fake property may lead to prosecution.
II. Transaction Rules
All transactions conducted via the Platform must comply with applicable real estate laws.
QARBA is not responsible for any transactions or agreements made between users.
Users are encouraged to independently verify all property information, ownership, and legal documents.
Payment terms must be clearly stated within listings or user agreements, and be disclosed to Qarba.
5. User Responsibilities
i. Account Usage
Keep your profile information accurate and updated.
Maintain control of your account and do not share login details.
Report unauthorized use immediately.
Comply with all applicable laws and regulations while using QARBA.
ii. Prohibited Activities
You agree not to:

Post fraudulent, misleading, fake, or duplicate listings.
Harass, discriminate, or defraud other users.
Use automated systems to scrape, harvest, or manipulate listings.
Upload or share content that infringes copyright or intellectual property.
Use the Platform for illegal or prohibited activities.
Platform Rules & Safety
i. Safety Guidelines
It is your responsibility to verify properties and (other) users before physical meetings.
Use secure payment channels and avoid cash transactions (where possible).
Report suspicious users or listings via our support system.
Follow due safety protocols during property viewings, as much as possible, have someone accompanying you,
 and do not go to dodgy locations.
Do not meet at suspicious venues, and apply safety protocol.
ii. Content Guidelines
No discriminatory, offensive, or defamatory content.
No false or deceptive advertising.
Respect intellectual property and privacy rights.
Maintain a professional and respectful tone in all communications.
Fees and Payments
QARBA may charge fees for certain services (e.g., subscription, premium listings, featured placements, 
or professional tools).
All fees are non-refundable unless otherwise stated in writing.
Payment terms, billing cycles, and applicable taxes are disclosed at the time of purchase. 
This may be subject to changes.
Subscription fees may be billed according to your selected plan.
Refunds and cancellations follow the policy provided in the applicable service agreement. 
Generally, a no refund policy apply.
Intellectual Property Rights
All content, design elements, software, and trademarks on QARBA are the exclusive property of QARBA Properties.
You may not copy, reproduce, or distribute any content from the Platform without prior written permission.
You grant QARBA a non-exclusive, royalty-free license to display and promote your listings on the Platform and
 partner channels.
Disclaimer of Warranties
QARBA provides its services on an "as is" and "as available" basis. We do not guarantee that the Platform will 
be error-free, uninterrupted, or free of viruses. QARBA makes no warranties, express or implied, regarding the accuracy, 
reliability, or completeness of listings or other user-generated content. Nothing on this Platform constitutes legal,
 financial, or professional advice.

Limitation of Liability
To the maximum extent permitted by law, QARBA shall not be liable for any loss, damage, or claim arising from:
Errors or omissions in property information;
Transactions or disputes between users;
Interruption of service or technical issues;
Unauthorized access to your account or data.
Fraudulent or fake listings or transactions.
Other similar actions.
In no event shall QARBA's total liability exceed the amount you paid directly to QARBA for any paid services.

Indemnity
You agree to indemnify and hold harmless QARBA, its affiliates, employees, and agents from any claims, damages, 
liabilities, or expenses arising out of your:

Violation of these Terms;
Misuse of the Platform; or
Violation of third-party rights, including intellectual property or privacy.
Carelessness or inability to duly verify listings, users, or the likes.
Termination of Account
QARBA reserves the right to suspend or terminate your account for violations of these Terms or inappropriate behaviours.
Users may terminate their accounts at any time by written notice or through the platform's account settings.
Upon termination, your listings and associated data may be removed or retained per QARBA's data retention policy.
Termination does not affect any outstanding obligations or payments owed to QARBA or other users that you may be 
directly transacting with.
Modification of Terms
QARBA may update or modify these Terms at any time and without notice. Updates will be effective once published on 
our website. Continued use of the Platform after changes constitutes your acceptance of the revised Terms.

Governing Law & Jurisdiction
These Terms shall be governed by and construed in accordance with the laws of the Federal Republic of Nigeria. 
Any dispute arising under or in connection with these Terms shall be subject to the exclusive jurisdiction of 
Nigerian courts.

Severability
If any provision of these Terms is found invalid or unenforceable by a court, the remaining provisions shall 
continue in full force and effect.

Contact Information
If you have any questions, complaints, or feedback regarding these Terms or our services, 
please contact us at: contact@qarba.com, or via our website: www.qarba.com.
For the full terms and conditions, visit: https://qarba.com/terms-conditions
    """,

    "https://qarba.com/careers": """
Careers at Qarba:

Qarba is always looking for talented, passionate individuals to join our growing team.
We are building the future of real estate technology in Nigeria and Africa.

Open roles includes positions in software engineering, product management,
sales, marketing, customer support, and real estate operations.
Roles can be internship-based, full-time or hybrid.
To view current openings or submit an application, visit: https://qarba.com/careers

Why work at Qarba:
- Be part of a fast-growing Nigerian tech company
- Work on products used by thousands of Nigerians daily
- Collaborative and innovative work environment
- Competitive compensation and growth opportunities
    """,

    "https://qarba.com/manage": """
Qarba Manage — Property Management Portal:

Let Us Manage Your Properties
Focus on your investments while we handle the day-to-day management of your properties. Our team is here to help!

Property Management
Let us handle your property portfolio with professional care and attention to detail.

Listing Services
We'll create, manage, and optimize your property listings for maximum visibility.

Scheduling & Maintenance
We'll coordinate viewings, handle maintenance requests, and manage tenant relations.

Performance Tracking
Get regular updates on your property performance, occupancy rates, and revenue.

To access the management portal, visit: https://qarba.com/manage
You will need a registered Qarba agent or seller account to use this portal.
    """,

    "https://qarba.com/create-account": """
Create a Qarba Account:

To get started on Qarba, create a free account at: https://qarba.com/create-account

Registration requires:
- Full name
- Email address
- Phone number
- Password

Once registered, you can:
- Save favourite property listings
- Contact agents directly
- Set up property search alerts
- List properties for sale or rent (agent/seller accounts)

Account creation is free. Visit https://qarba.com/create-account to sign up.
    """,
}


def scrape_qarba_page(page_hint: str) -> str:
    """
    Return content for a Qarba public page.

    Strategy (in order):
    1. Resolve URL from QARBA_PAGES via exact or fuzzy key match.
    2. Return static curated content from QARBA_STATIC_CONTENT if available
       (most reliable — bypasses Next.js CSR rendering issue entirely).
    3. Attempt Playwright headless browser scrape as fallback (renders JS).
    4. Attempt plain requests.get() + BeautifulSoup as last resort.
    """
    hint = page_hint.lower().strip()
    url  = QARBA_PAGES.get(hint)

    if not url:
        best_score, best_url = 0, None
        for key, page_url in QARBA_PAGES.items():
            score = fuzz.partial_ratio(hint, key)
            if score > best_score:
                best_score, best_url = score, page_url
        if best_score >= 60:
            url = best_url

    if not url and hint.startswith("http"):
        url = hint

    if not url:
        known = ", ".join(sorted(set(QARBA_PAGES.keys())))
        return (
            f"I don't have '{page_hint}' in my page registry. "
            f"Known pages: {known}. Visit qarba.com directly for more."
        )

    # ── 1. Static content (fastest, most reliable) ────────────────────────
    static = QARBA_STATIC_CONTENT.get(url)
    if static and len(static.strip()) > 100:
        print(f"✅ Static content served for: {url}")
        return static.strip()

    # ── 2. Playwright headless browser (renders JavaScript) ───────────────
    cache_key = f"page:{url}"
    cached    = _cache.get(cache_key)
    if cached:
        print(f"📦 Page cache hit: {url}")
        return cached

    print(f"🌐 Attempting Playwright scrape: {url}")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page    = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            content = page.inner_text("body")
            browser.close()

        if content and len(content.strip()) > 100:
            text = re.sub(r"\s+", " ", content).strip()[:4000]
            print(f"✅ Playwright extracted: {len(text)} chars")
            _cache.set(cache_key, text)
            return text

    except ImportError:
        print("⚠️  Playwright not installed — skipping headless scrape.")
    except Exception as e:
        print(f"⚠️  Playwright scrape failed: {e}")

    # ── 3. Plain requests + BeautifulSoup (last resort) ───────────────────
    print(f"🌐 Attempting plain HTTP scrape: {url}")
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        if resp.status_code == 200:
            text = extract_readable_text(resp.text)
            if len(text) > 100:
                print(f"✅ HTML extracted: {len(text)} chars")
                _cache.set(cache_key, text)
                return text
    except Exception as e:
        print(f"❌ Plain scrape error ({url}):", e)

    return (
        f"I was unable to retrieve the content for that page automatically. "
        f"Please visit {url} directly for the most accurate information."
    )



# ═══════════════════════════════════════════════════════════════════════════════
# 10. TOOL EXECUTORS
# ═══════════════════════════════════════════════════════════════════════════════

def tool_search_properties(query: str) -> str:
    """
    Search Qarba property listings.
    Pipeline: international guard -> parse criteria ->
    progressive fallback search -> soft rank results.
    """
    # Stage 1: International guard
    intl = _check_international(query)
    if intl:
        return (
            f"Qarba is a Nigerian real estate platform and currently only covers "
            f"properties within Nigeria. I don't have listings for {intl.title()}. "
            f"If you're looking for properties in Nigeria, I'd be happy to help — "
            f"just let me know your preferred location within Nigeria."
        )

    properties = fetch_properties()
    if not properties:
        return (
            "Property listings are temporarily unavailable. "
            "Please visit qarba.com to browse directly."
        )

    # Stage 2: Parse all criteria from the query
    criteria = _parse_query(query)
    print(f"🔍 Criteria: {criteria}")

    # Stage 3: Search with progressive fallback
    filtered, exact_match, notice = _search_with_fallback(properties, criteria)

    if not filtered:
        desc = _describe_search(criteria)
        return (
            f"I wasn't able to find any properties matching {desc} on Qarba right now. "
            f"Qarba may not have listings in that area or matching those specifications yet. "
            f"You can browse all available listings directly at qarba.com."
        )

    # Stage 4: Soft rank by relevance
    ranked = _soft_rank(query, filtered)
    count  = len(ranked)

    if exact_match:
        header = f"Found {count} matching propert{'y' if count == 1 else 'ies'} on Qarba:\n"
    else:
        header = (notice + "\n") if notice else \
                 f"Here are {count} available propert{'y' if count == 1 else 'ies'} on Qarba:\n"

    lines = [header]

    for p in ranked[:10]:
        name       = p.get("property_name", "Unnamed Property")
        location   = p.get("location", "")
        city       = p.get("city",     "")
        state      = p.get("state",    "")
        price      = p.get("rent_price") or p.get("sale_price") or 0
        freq       = p.get("rent_frequency", "")
        ptype      = p.get("property_type_display", "")
        ltype      = p.get("listing_type_display",  "")
        bedrooms   = p.get("bedrooms") or 0
        amenities  = ", ".join([a.get("name","") for a in p.get("amenities",[])]) or "None listed"
        agent_obj  = p.get("listed_by", {})
        agent_name = (
            f"{agent_obj.get('first_name','')} {agent_obj.get('last_name','')}".strip()
            or "Unknown Agent"
        )
        thumbnail  = p.get("thumbnail", "")
        slug       = p.get("slug", "")
        link       = f"https://qarba.com/properties/{slug}" if slug else "https://qarba.com"
        price_str  = f"₦{int(price):,} {freq}".strip() if price else "Price on request"
        bed_str    = f"{bedrooms} bed" if bedrooms else "N/A"

        entry = (
            f"Property: {name}\n"
            f"Location: {location}, {city}, {state}\n"
            f"Type: {ptype}  |  Listing: {ltype}  |  Bedrooms: {bed_str}\n"
            f"Price: {price_str}\n"
            f"Amenities: {amenities}\n"
            f"Agent: {agent_name}\n"
            f"Link: {link}"
        )
        if thumbnail:
            entry += f"\n[IMAGE]{thumbnail}[/IMAGE]"
        lines.append(entry)

    if count > 10:
        lines.append(f"...and {count - 10} more. Visit qarba.com to see all listings.")

    return "\n\n".join(lines)


def tool_search_blogs(query: str = "") -> str:
    blogs = fetch_blogs()
    if not blogs:
        return "Blog articles are temporarily unavailable. Please visit qarba.com."

    if query:
        q        = query.lower()
        filtered = [
            b for b in blogs
            if q in (b.get("title","") + b.get("summary","") + b.get("writers_name","")).lower()
        ]
        if filtered:
            blogs = filtered
        else:
            return (
                f"No blog articles found matching '{query}' on Qarba. "
                f"Visit qarba.com for the latest articles."
            )

    lines = [f"Here are Qarba's latest blog articles ({len(blogs)} found):\n"]
    for b in blogs[:6]:
        title   = b.get("title",        "Untitled")
        author  = b.get("writers_name", "Qarba Team")
        summary = clean_text(b.get("summary", ""))[:250]
        date    = b.get("created_at", "")
        cover   = b.get("cover_image_url", "")
        slug    = b.get("slug", "")
        link    = f"https://qarba.com/blog/{slug}" if slug else "https://qarba.com"

        entry = (
            f"Title: {title}\n"
            f"Author: {author}  |  Date: {date}\n"
            f"Summary: {summary}...\n"
            f"Read more: {link}"
        )
        if cover:
            entry += f"\n[IMAGE]{cover}[/IMAGE]"
        lines.append(entry)

    return "\n\n".join(lines)


def tool_answer_faq(question: str) -> str:
    match = find_best_faq(question)
    if match:
        return f"FAQ Answer:\n{clean_text(match['answer'])}"
    return (
        "No FAQ entry found for that specific question. "
        "Try asking about Qarba's services, how to list a property, or payment options."
    )


def tool_browse_website(page: str) -> str:
    return scrape_qarba_page(page)


def tool_search_agents(query: str = "") -> str:
    if not QARBA_AGENT_API:
        return (
            "The agent directory is not yet available through the assistant. "
            "Please visit qarba.com to find a verified Qarba agent."
        )
    agents = fetch_agents()
    if not agents:
        return "Agent data is temporarily unavailable."
    if query:
        agents = [a for a in agents if query.lower() in json.dumps(a).lower()]
    if not agents:
        return f"No agents found matching '{query}'. Visit qarba.com for the full directory."

    lines = [f"Qarba has {len(agents)} verified agent(s):\n"]
    for a in agents[:8]:
        name  = f"{a.get('first_name','')} {a.get('last_name','')}".strip()
        phone = a.get("phone_number") or a.get("phone") or "N/A"
        email = a.get("email", "N/A")
        lines.append(f"Agent: {name}  |  Phone: {phone}  |  Email: {email}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. TOOL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_properties",
            "description": (
                "Search Qarba's live property listings. "
                "Use for ANY question about buying, renting, property prices, "
                "locations, bedrooms, amenities, or available listings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language description of what the user wants. "
                            "E.g. '3-bedroom apartment in Lekki for rent under 2 million'."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_blogs",
            "description": (
                "Search Qarba's blog articles for real estate news, market updates, "
                "property tips, and investment guides. Use when the user asks about "
                "articles, news, blog posts, or real estate advice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic or keywords. Leave empty to get latest articles."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "answer_faq",
            "description": (
                "Answer general questions about Qarba using the FAQ knowledge base. "
                "Use for questions about how Qarba works, listing a property, "
                "fees, payment, support, or platform policies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question exactly as asked."
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_website",
            "description": (
                "Read a Qarba public webpage to answer questions about company information. "
                "Available pages: about-us, contact, privacy-policy, terms-conditions, "
                "manage, careers, create-account. "
                "Use for questions about Qarba's story, mission, team, contact details, "
                "legal policies, job openings, or how to create an account."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "string",
                        "description": (
                            "Page name or topic. E.g. 'about', 'contact', "
                            "'privacy', 'terms', 'careers', 'manage', 'create account'."
                        )
                    }
                },
                "required": ["page"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_agents",
            "description": (
                "Retrieve Qarba verified agents and their contact details. "
                "Use when the user asks about agents, realtors, or who to contact."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Agent name or location filter. Leave empty for all agents."
                    }
                },
                "required": []
            }
        }
    },
]

TOOL_MAP = {
    "search_properties": lambda args: tool_search_properties(args.get("query",    "")),
    "search_blogs":      lambda args: tool_search_blogs(args.get("query",         "")),
    "answer_faq":        lambda args: tool_answer_faq(args.get("question",        "")),
    "browse_website":    lambda args: tool_browse_website(args.get("page",        "")),
    "search_agents":     lambda args: tool_search_agents(args.get("query",        "")),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 12. LLM CALL HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def call_llm(messages: list, tools: list = None) -> dict:
    import time
    payload = {
        "model":       "openai/gpt-4o-mini",
        "messages":    messages,
        "max_tokens":  1024,
        "temperature": 0.4,
    }
    if tools:
        payload["tools"]       = tools
        payload["tool_choice"] = "auto"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type":  "application/json",
    }

    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            result = resp.json()
            if "error" in result:
                raise RuntimeError(f"LLM API error: {result['error']}")
            return result
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_err = e
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise last_err


# ═══════════════════════════════════════════════════════════════════════════════
# 13. SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are Tinah — the official AI assistant for Qarba.com, Nigeria's intelligent real estate platform.

You have four active tools:
- search_properties  → search live Qarba property listings
- search_blogs       → retrieve Qarba blog articles and real estate news
- answer_faq         → answer questions using Qarba's FAQ knowledge base
- browse_website     → read any Qarba public page (About Us, Contact, Privacy Policy, Terms, Careers, Manage, Create Account)

You also have search_agents but it is currently unavailable. If a user asks about agents, direct them to qarba.com.

Your rules:
1. ALWAYS call the relevant tool before answering. Never answer property, blog, FAQ, or company questions from memory.
2. For ANY question about how to use Qarba, listing a property, renting, buying, account management, payments, passwords, mobile app, agents, or platform features — call answer_faq FIRST. This includes questions like "how do I list a property", "how can I contact a landlord", "is listing free", "how do I reset my password", etc.
3. If a user asks about Qarba's story, team, contact info, phone number, email, legal policies, or careers — call browse_website with the relevant page name.
4. Base your answers ONLY on data returned by your tools. If a tool returns no results, say so honestly and direct the user to qarba.com.
5. Write in clear, professional English. No markdown symbols (*, _, ##). Use plain text only.
6. For property results, always include the property name, location, price, and link.
7. Images in tool results appear as [IMAGE]URL[/IMAGE] — pass them through unchanged. Never describe or modify image tags.
8. Keep answers concise. For long property lists, show the top results and mention more are available at qarba.com.
9. Never make up property details, prices, or contact information. Only use what your tools return.
10. You represent Qarba professionally. Be helpful, warm, and accurate at all times.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 14. CHAT ROUTE  (agentic loop)
# ═══════════════════════════════════════════════════════════════════════════════
# Rate limit applied conditionally — decorator only active when flask-limiter installed
_chat_route = app.route("/chat", methods=["POST"])

def _apply_limiter(f):
    if _limiter_available:
        return limiter.limit("15 per minute")(f)
    return f

@_chat_route
@_apply_limiter
def chat():
    try:
        body       = request.get_json(force=True) or {}
        user_query = body.get("message", "").strip()
        if not user_query:
            return jsonify({"response": "Please enter a message."})

        # Sanitise input — prevent huge payloads and prompt injection attempts
        if len(user_query) > 500:
            return jsonify({"response": "Please keep your message under 500 characters."})
        # Strip any attempt to inject system-level instructions
        user_query = user_query.replace("<|system|>", "").replace("<|user|>", "").replace("<|assistant|>", "")

        history  = session.get("chat_history", [])
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for h in history[-4:]:
            messages.append({"role": "user",     "content": h["user"]})
            messages.append({"role": "assistant", "content": h["bot"]})
        messages.append({"role": "user", "content": user_query})

        # ── Agentic loop — model decides which tools to call ──────────────────
        MAX_ROUNDS = 4
        for round_num in range(MAX_ROUNDS):
            result = call_llm(messages, tools=TOOLS)
            choice = result["choices"][0]
            msg    = choice["message"]

            messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                break   # model produced its final answer

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"].get("arguments", "{}"))
                except json.JSONDecodeError:
                    fn_args = {}

                print(f"🔧 [{round_num + 1}] {fn_name}({fn_args})")

                tool_result = (
                    TOOL_MAP[fn_name](fn_args)
                    if fn_name in TOOL_MAP
                    else f"Unknown tool '{fn_name}'."
                )

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      tool_result,
                })

        # ── Extract final reply ───────────────────────────────────────────────
        last     = messages[-1]
        ai_reply = (last.get("content") or "").strip()

        if last.get("role") == "tool" or not ai_reply:
            ai_reply = "I wasn't able to generate a response. Please try again."

        # Strip [IMAGE]...[/IMAGE] tags from the saved bot reply before
        # storing in the session cookie — images make the cookie too large
        # (Flask session limit is ~4093 bytes). The full response with images
        # is still returned to the frontend below; only the session copy is trimmed.
        bot_for_history = re.sub(r'\[IMAGE\].*?\[/IMAGE\]', '[image]', ai_reply, flags=re.DOTALL)
        # Also trim very long responses in history to a summary
        if len(bot_for_history) > 400:
            bot_for_history = bot_for_history[:400] + "..."

        history.append({"user": user_query, "bot": bot_for_history})
        session["chat_history"] = history[-4:]  # keep 4 turns max to stay under cookie limit

        return jsonify({"response": ai_reply})

    except RuntimeError as e:
        print("❌ LLM error:", e)
        return jsonify({"response": "I'm having trouble reaching the AI right now. Please try again shortly."})
    except Exception as e:
        print("❌ Unexpected /chat error:")
        import traceback; traceback.print_exc()
        return jsonify({"response": "An internal error occurred. Please try again."})


# ═══════════════════════════════════════════════════════════════════════════════
# 15. ADMIN & UTILITY ROUTES
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/admin/cache/clear", methods=["POST"])
def clear_cache():
    """
    Force-clears the data cache.
    Use after a Qarba API update so Tinah picks up new data immediately
    rather than waiting for the 60-minute TTL.

    Usage:
        curl -X POST https://your-server/admin/cache/clear \\
             -H "X-Admin-Secret: your_admin_secret"
    """
    secret = request.headers.get("X-Admin-Secret", "")
    if not secret or secret != os.getenv("ADMIN_SECRET", ""):
        return jsonify({"error": "Unauthorised"}), 403
    _cache.clear()
    print("🗑️  Cache cleared by admin.")
    return jsonify({"status": "Cache cleared. Next requests will fetch fresh data."})


@app.route("/health")
def health():
    """
    Live status check. Visit /health after deploying to verify everything works.
    Shows record counts, cache state, active tools, and registered pages.
    """
    return jsonify({
        "status":          "ok",
        "model":           "openai/gpt-4o-mini",
        "cache_ttl":       f"{TTLCache.TTL_SECONDS // 60} minutes",
        "cache_state":     _cache.info(),
        "data": {
            "properties":  len(fetch_properties()),
            "blogs":       len(fetch_blogs()),
            "agents":      len(fetch_agents()),
            "faqs":        len(faqs),
        },
        "tools_active":    [t["function"]["name"] for t in TOOLS],
        "pages_available": sorted(set(QARBA_PAGES.values())),
    })


@app.route("/")
def home():
    return render_template("index.html")


# ═══════════════════════════════════════════════════════════════════════════════
# 16. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # IMPORTANT: debug=False for any deployment (staging or production).
    # debug=True exposes an interactive debugger to anyone who triggers an error.
    # Use the DEBUG env var to enable debug mode only in local development.
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(
        debug=debug_mode,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
    )