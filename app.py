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
QARBA_AGENT_API_URL = os.getenv("AGENT_API_URL")
QARBA_PROPERTIES_API_URL = os.getenv("PROPERTIES_API_URL")
QARBA_CLIENT_API_URL = os.getenv("CLIENT_API_URL")


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
    """Fetch live property data from Qarba API."""
    try:
        response = requests.get(os.getenv("QARBA_PROPERTIES_API"))
        if response.status_code == 200:
            return response.json()
        else:
            print("⚠️ Property API error:", response.status_code)
            return []
    except Exception as e:
        print("⚠️ Error fetching properties:", e)
        return []

def fetch_agents():
    """Fetch live agent data from Qarba API."""
    try:
        response = requests.get(os.getenv("QARBA_AGENTS_API"))
        if response.status_code == 200:
            return response.json()
        else:
            print("⚠️ Agent API error:", response.status_code)
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

        # Retrieve chat history for contextual understanding
        history = session.get("chat_history", [])
        best_match = find_best_faq(user_query)

        # 🧠 Build context from FAQs and Qarba.com data
        context_parts = []
        if best_match:
            context_parts.append(f"FAQ info: {best_match['answer']}")

        try:
            properties = fetch_properties()
            agents = fetch_agents()
            if properties:
                context_parts.append(f"There are currently {len(properties)} properties listed on Qarba.com.")
            if agents:
                context_parts.append(f"There are {len(agents)} real estate agents available.")
        except Exception as e:
            print("⚠️ Error fetching Qarba data:", e)

        context = "\n".join(context_parts) if context_parts else "No extra data found."

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        # Combine conversation and context
        conversation = [{
    "role": "system",
    "content": (
        "You are QARBA — a real estate AI assistant for Qarba.com.\n"
        "Your job is to answer questions using live Qarba data when possible.\n"
        "If the question relates to properties, agents, or clients, use Qarba APIs’ info "
        "(fetched data shown in the context) to form your response.\n"
        "If the question is general or not covered by Qarba data, use your general real estate knowledge "
        "and the FAQ answers provided.\n"
        "Always respond clearly, professionally, and naturally.\n"
        "Avoid making up listings, names, or prices that aren’t in Qarba data."
    )
}]

        for h in history:
            conversation.append({"role": "user", "content": h["user"]})
            conversation.append({"role": "assistant", "content": h["bot"]})
        #conversation.append({"role": "user", "content": f"User asked: {user_query}\nRelevant info:\n{context}"})
        conversation.append({
    "role": "user",
    "content": f"""
User asked: {user_query}

<faq_context>
{best_match['answer'] if best_match else 'No FAQ match'}
</faq_context>

<qarba_data>
{context}
</qarba_data>
"""
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
