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

def fetch_blogs():
    """Fetch live blog data from Qarba API."""
    try:
        response = requests.get(QARBA_CLIENT_API)
        print("🌐 Fetching blogs from:", QARBA_CLIENT_API)
        print("📡 Status code:", response.status_code)

        if response.status_code == 200:
            data = response.json()
            # If it's a list, return directly
            if isinstance(data, list):
                print(f"✅ Blog API returned {len(data)} articles.")
                return data
            # If it's wrapped in "data", unwrap it
            elif isinstance(data, dict) and "data" in data:
                print(f"✅ Blog API (dict) returned {len(data['data'])} articles.")
                return data["data"]
            else:
                print("⚠️ Unexpected blog API structure:", type(data))
                return []
        else:
            print(f"⚠️ Blog API error: {response.status_code} -> {response.text[:200]}")
            return []
    except Exception as e:
        print("❌ Error fetching blogs:", e)
        return []



@app.route("/chat", methods=["POST"])
def chat():
    try:
        #from fuzzywuzzy import fuzz
        data = request.get_json()
        user_query = data.get("message", "").strip()
        if not user_query:
            return jsonify({"response": "Please enter a question."})

        # Retrieve session history and best FAQ match
        history = session.get("chat_history", [])
        best_match = find_best_faq(user_query)

        context_parts = []
        if best_match:
            context_parts.append(f"FAQ info: {best_match['answer']}")

        try:
            print("🌐 Fetching Qarba data...")
            properties = fetch_properties()
            agents = fetch_agents()
            blogs = fetch_blogs()

            # 🏠 PROPERTY CONTEXT
            matched_properties = []
            if isinstance(properties, list) and len(properties) > 0:
                query_terms = user_query.lower().split()
                for p in properties:
                    searchable_text = " ".join([
                        str(p.get("property_name", "")),
                        str(p.get("location", "")),
                        str(p.get("state", "")),
                        str(p.get("city", "")),
                        str(p.get("property_type_display", "")),
                        str(p.get("listing_type_display", "")),
                        str(p.get("rent_price", "")),
                        str(p.get("sale_price", "")),
                        " ".join([a.get("name", "") for a in p.get("amenities", [])])
                    ]).lower()

                    if any(fuzz.partial_ratio(term, searchable_text) > 70 for term in query_terms):
                        matched_properties.append(p)

                # fallback if user mentions property-related terms
                if not matched_properties and any(
                    t in user_query.lower() for t in ["property", "house", "apartment", "flat", "selfcon", "rent", "buy", "location", "price"]
                ):
                    matched_properties = properties

                if matched_properties:
                    context_parts.append(f"🏠 Found {len(matched_properties)} matching properties on Qarba.com.")
                    show_images = any(term in user_query.lower() for term in ["photo", "picture", "image", "show", "display"])

                    for p in matched_properties:
                        name = p.get("property_name", "Unnamed Property")
                        location = p.get("location", "Unknown location")
                        city = p.get("city", "")
                        state = p.get("state", "")
                        price = p.get("rent_price") or p.get("sale_price") or 0
                        freq = p.get("rent_frequency", "")
                        type_ = p.get("property_type_display", "")
                        listing_type = p.get("listing_type_display", "")
                        amenities = ", ".join([a.get("name", "") for a in p.get("amenities", [])]) or "No listed amenities"
                        agent = p.get("listed_by", {}).get("first_name", "Unknown Agent")
                        thumbnail = p.get("thumbnail", "")

                        details = (
                            f"🏘 {name}\n"
                            f"📍 Location: {location}, {city}, {state}\n"
                            f"🏠 Type: {type_} for {listing_type.lower()}\n"
                            f"💰 Price: ₦{int(price):,} {freq if freq else ''}\n"
                            f"✨ Amenities: {amenities}\n"
                            f"👤 Agent: {agent}"
                        )
                        context_parts.append(details)

                        if show_images and thumbnail:
                            context_parts.append(f"[IMAGE]{thumbnail}[/IMAGE]")
                else:
                    context_parts.append("⚠️ No matching properties found on Qarba.com.")

            # 👥 AGENT CONTEXT
            if isinstance(agents, dict) and agents.get("data"):
                context_parts.append(f"👥 There are {len(agents['data'])} verified agents available on Qarba.com.")

            # 📰 --- BLOG CONTEXT (fixed) ---
            if isinstance(blogs, list) and len(blogs) > 0:
             blog_terms = ["blog", "news", "article", "post"]
             if any(term in user_query.lower() for term in blog_terms):
                   context_parts.append(f"📰 Qarba currently has {len(blogs)} blog article(s). Here are some:")
        
             for b in blogs[:5]:  # show only first 5
                     title = b.get("title", "Untitled Blog")
                     author = b.get("writers_name", "Unknown Author")
                     summary = b.get("summary", "No summary available.")
                     date = b.get("created_at", "Unknown Date")
                     cover = b.get("cover_image_url", "")

            blog_text = (
                f"📝 Title: {title}\n"
                f"👤 Author: {author}\n"
                f"📅 Published: {date}\n"
                f"🗒 Summary: {summary.strip()}"
            )
            context_parts.append(blog_text)

            # Include cover image if user asks for it
            if any(t in user_query.lower() for t in ["photo", "image", "cover"]) and cover:
                context_parts.append(f"[IMAGE]{cover}[/IMAGE]")



        except Exception as e:
            print("⚠️ Error fetching Qarba data:", e)

        # Combine all context
        context = "\n\n".join(context_parts) if context_parts else "No Qarba data found."

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        # 🧠 Build AI conversation
        conversation = [{
            "role": "system",
            "content": (
                "You are QARBA — a professional AI real estate assistant for Qarba.com.\n"
                "Answer user queries using Qarba's live property, agent, and blog data.\n"
                "Always reply clearly and cleanly — avoid markdown asterisks or special symbols.\n"
                "If a property has an [IMAGE] tag, it represents the property's picture.\n"
                "If the user asks for property details, summarize all details clearly."
            )
        }]

        # Append chat history
        for h in history:
            conversation.append({"role": "user", "content": h["user"]})
            conversation.append({"role": "assistant", "content": h["bot"]})

        # Append new query
        conversation.append({
            "role": "user",
            "content": f"User asked: {user_query}\n\n<QarbaContext>\n{context}\n</QarbaContext>"
        })

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

        # Save chat history (keep only last 5)
        history.append({"user": user_query, "bot": ai_message})
        session["chat_history"] = history[-5:]

        return jsonify({"response": ai_message})

    except Exception as e:
        print("❌ Error in chat:", e)
        return jsonify({"response": "An internal error occurred. Please try again later."})





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
