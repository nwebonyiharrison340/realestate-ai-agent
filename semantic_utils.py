# semantic_utils.py
import re
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util

# Load lightweight semantic model for hybrid similarity
model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text: str) -> str:
    """Remove HTML, special chars, and normalize whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII chars
    return text.strip()

def hybrid_match(user_query, items, text_fn, fuzzy_threshold=65, semantic_threshold=0.45):
    """Hybrid fuzzy + semantic matching of query to list of items."""
    query = clean_text(user_query)
    matched = []

    # Encode query once
    query_emb = model.encode(query, convert_to_tensor=True)

    for item in items:
        text = clean_text(text_fn(item))
        if not text:
            continue

        # Fuzzy match
        fuzzy_score = fuzz.partial_ratio(query.lower(), text.lower())

        # Semantic similarity
        item_emb = model.encode(text, convert_to_tensor=True)
        semantic_score = float(util.pytorch_cos_sim(query_emb, item_emb))

        # Hybrid condition
        if fuzzy_score > fuzzy_threshold or semantic_score > semantic_threshold:
            matched.append(item)

    return matched
