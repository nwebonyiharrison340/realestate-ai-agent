from flask import Flask, request, jsonify, render_template, session
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
import numpy as np
import os
import json
import requests
from openai import OpenAI

# Load environment variables
load_dotenv()
print(" Loaded OpenAI Key:", os.getenv("OPENAI_API_KEY"))
print("✅ Loaded Qarba Agent API:", os.getenv("QARBA_AGENT_API"))
print("✅ Loaded Qarba Property API:", os.getenv("QARBA_PROPERTY_API"))
print("✅ Loaded Qarba Client API:", os.getenv("QARBA_CLIENT_API"))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecret")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
QARBA_AGENT_API = os.getenv("QARBA_AGENT_API")
QARBA_PROPERTY_API = os.getenv("QARBA_PROPERTY_API")
QARBA_CLIENT_API = os.getenv("QARBA_CLIENT_API")



client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# Load sentence transformer model for semantic similarity
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAQs
def load_faqs():
    with open("faqs.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both possible JSON structures
    if isinstance(data, dict) and "faqs" in data:
        faqs = data["faqs"]
    else:
        faqs = data

    # Debug: show first FAQ entry
    if len(faqs) > 0:
        print(" First FAQ loaded:", faqs[0])
    else:
        print(" No FAQs found in JSON!")

    valid_faqs = []
    for faq in faqs:
        question = faq.get("question")
        answer = faq.get("answer")

        if question and answer:
            faq["embedding"] = embedding_model.encode(question)
            valid_faqs.append(faq)
        else:
            print(f" Skipping invalid FAQ entry: {faq}")

    print(f" Loaded {len(valid_faqs)} valid FAQs.")
    return valid_faqs


faqs = load_faqs()


# Semantic FAQ Finder
def find_best_faq(user_query):
    user_embedding = embedding_model.encode(user_query)
    similarities = [cosine_similarity([user_embedding], [faq["embedding"]])[0][0] for faq in faqs]
    best_index = int(np.argmax(similarities))
    best_score = similarities[best_index]
    return faqs[best_index] if best_score > 0.65 else None

from flask import session  # add this import at the top if not already there

def fetch_properties():
    try:
        response = requests.get(QARBA_PROPERTY_API)
        print("🌐 Fetching properties from:", QARBA_PROPERTY_API)
        print("📡 Status code:", response.status_code)

        if response.status_code == 200:
            data = response.json()
            print("✅ Property API returned keys:", list(data.keys()))

            # Access the actual list inside "data"
            if "data" in data and isinstance(data["data"], list):
                properties = data["data"]
                if len(properties) > 0:
                    print("🏠 First property item:", properties[0])
                else:
                    print("⚠️ No properties found in API response")
                return properties
            else:
                print("⚠️ 'data' key not found or not a list")
                return []
        else:
            print("⚠️ Property API error:", response.status_code, "->", response.text[:200])
            return []
    except Exception as e:
        print("❌ Error fetching properties:", e)
        return []


def fetch_agents():
    """Fetch live agent data from Qarba API."""
    try:
        response = requests.get(QARBA_AGENT_API, headers={"User-Agent": "QarbaBot/1.0"})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"⚠️ Agent API error: {response.status_code} -> {response.text[:200]}")
            return []
    except Exception as e:
        print("⚠️ Error fetching agents:", e)
        return []




@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_query = data.get("message", "").strip()
        if not user_query:
            return jsonify({"response": "Please enter a question."})

        # Retrieve chat history
        history = session.get("chat_history", [])
        best_match = find_best_faq(user_query)

        # 🧠 Build context from FAQs and Qarba.com data
        context_parts = []
        if best_match:
            context_parts.append(f"FAQ info: {best_match['answer']}")

        try:
            properties = fetch_properties()
            agents = fetch_agents()

            property_data = []
            if isinstance(properties, list):
                property_data = properties
            elif isinstance(properties, dict) and "data" in properties:
                property_data = properties["data"]

            matched_properties = []
            user_terms = user_query.lower().split()

            # 🧩 Intelligent property matching
            for p in property_data:
                searchable_text = " ".join([
                    str(p.get("property_name", "")),
                    str(p.get("location", "")),
                    str(p.get("city", "")),
                    str(p.get("state", "")),
                    str(p.get("property_type_display", "")),
                    str(p.get("listing_type_display", "")),
                    str(p.get("rent_price", "")),
                    str(p.get("sale_price", "")),
                    " ".join([a.get("name", "") for a in p.get("amenities", [])])
                ]).lower()

                if any(term in searchable_text for term in user_terms):
                    matched_properties.append(p)

            if matched_properties:
                context_parts.append(f"Found {len(matched_properties)} matching Qarba property results:")
                for p in matched_properties[:10]:  # limit to 10 to avoid token overflow
                    name = p.get("property_name", "Unnamed property")
                    location = p.get("location", "Unknown location")
                    price = p.get("rent_price") or p.get("sale_price") or "N/A"
                    freq = p.get("rent_frequency", "")
                    type_ = p.get("property_type_display", "")
                    listing_type = p.get("listing_type_display", "")
                    amenities = ", ".join([a.get("name", "") for a in p.get("amenities", [])])
                    agent_name = f"{p.get('listed_by', {}).get('first_name', '')} {p.get('listed_by', {}).get('last_name', '')}".strip()
                    context_parts.append(
                        f"🏠 {name} — {type_} for {listing_type.lower()} at {location}. "
                        f"Price: ₦{price:,} {freq if freq else ''}. "
                        f"Amenities: {amenities}. "
                        f"Agent: {agent_name or 'N/A'}."
                    )
            else:
                # If no match found, still summarize
                if property_data:
                    context_parts.append(f"There are {len(property_data)} properties listed on Qarba.com.")
                    sample_list = []
                    for p in property_data[:5]:
                        name = p.get("property_name", "Unnamed property")
                        location = p.get("location", "Unknown location")
                        price = p.get("rent_price") or p.get("sale_price") or "N/A"
                        type_ = p.get("property_type_display", "")
                        listing_type = p.get("listing_type_display", "")
                        sample_list.append(f"{name} — {type_} for {listing_type.lower()} at {location} (₦{price:,})")
                    context_parts.append("Some available listings:\n" + "\n".join(sample_list))
                else:
                    context_parts.append("No property data found on Qarba.com.")

            # Agent info
            if isinstance(agents, dict) and agents.get("data"):
                context_parts.append(f"There are {len(agents['data'])} registered agents available on Qarba.com.")

        except Exception as e:
            print("⚠️ Error fetching Qarba data:", e)

        # Build final context for AI
        context = "\n".join(context_parts) if context_parts else "No extra data found."

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        # 🧩 Build the conversation for AI
        conversation = [
            {
                "role": "system",
                "content": (
                    "You are QARBA — a smart real estate AI assistant for Qarba.com.\n"
                    "You use live Qarba data (properties, agents, clients) to answer user questions.\n"
                    "When a user searches, show accurate details such as property name, location, type, price, frequency, "
                    "agent name, and amenities in a clean, human-readable format — no markdown or asterisks.\n"
                    "If multiple results exist, summarize neatly.\n"
                    "Never invent or guess details."
                )
            }
        ]

        # Add previous history
        for h in history:
            conversation.append({"role": "user", "content": h["user"]})
            conversation.append({"role": "assistant", "content": h["bot"]})

        # Add new user query and context
        conversation.append({
            "role": "user",
            "content": f"""
User asked: {user_query}

<qarba_data>
{context}
</qarba_data>
"""
        })

        # Shorten if too long (avoid token overflow)
        conversation = conversation[-10:]

        payload = {
            "model": "google/gemma-2-9b-it",
            "messages": conversation
        }

        response = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
        result = response.json()

        if "error" in result:
            print("❌ API Error:", result["error"])
            return jsonify({"response": "Error communicating with AI model. Please try again later."})

        ai_message = result["choices"][0]["message"]["content"].strip()

        # Save recent chat history (last 5)
        history.append({"user": user_query, "bot": ai_message})
        session["chat_history"] = history[-5:]

        return jsonify({"response": ai_message})

    except Exception as e:
        print("❌ Error in chat:", e)
        return jsonify({"response": "An error occurred while processing your request."})





#User asked: {user_query}

#<faq_context>
#{best_match['answer'] if best_match else 'No FAQ match'}
#</faq_context>

#<qarba_data>
#{context}
#</qarba_data>

#})


        payload = {
            "model": "google/gemma-2-9b-it",
            "messages": conversation
        }

        response = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
        result = response.json()

        if "error" in result:
            print("❌ API Error:", result["error"])
            return jsonify({"response": "Error communicating with AI model. Please try again later."})

        ai_message = result["choices"][0]["message"]["content"].strip()

        # Save conversation context
        history.append({"user": user_query, "bot": ai_message})
        session["chat_history"] = history[-5:]  # keep last 5 messages

        return jsonify({"response": ai_message})

    except Exception as e:
        print(" Error in /chat route:", str(e))
        return jsonify({"response": "An internal error occurred. Please try again later."})

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
