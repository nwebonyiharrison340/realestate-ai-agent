from flask import Flask, request, jsonify, render_template
from fuzzywuzzy import fuzz
import os
import json
from dotenv import load_dotenv
import requests
from openai import OpenAI

# Load environment variables
load_dotenv()
print("✅ Loaded OpenAI Key:", os.getenv("OPENAI_API_KEY"))


app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Load FAQs
def load_faqs():
    with open("faqs.json", "r", encoding="utf-8") as f:
        return json.load(f)

faqs = load_faqs()

# Find best FAQ
def find_best_faq(user_query):
    user_query_lower = user_query.lower()
    best_match = None
    best_score = 0

    for faq in faqs:
        question = faq.get("question", "").lower()
        if not question:
            continue
        similarity = fuzz.ratio(user_query_lower, question)
        if similarity > best_score:
            best_score = similarity
            best_match = faq

    return best_match if best_score > 60 else None


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_query = data.get("message", "").strip()

        if not user_query:
            return jsonify({"response": "Please enter a question."})

        best_match = find_best_faq(user_query)
        context = best_match["answer"] if best_match else "No matching FAQ found."

        # Prepare request to OpenRouter API
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "google/gemma-2-9b-it",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant for a real estate platform. Use FAQ info if available."},
                {"role": "user", "content": f"User asked: {user_query}\nFAQ answer: {context}"}
            ]
        }

        response = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
        result = response.json()

        if "error" in result:
            print("Error from API:", result["error"])
            return jsonify({"response": "Error communicating with AI model. Please try again later."})

        ai_message = result["choices"][0]["message"]["content"].strip()
        return jsonify({"response": ai_message})

    except Exception as e:
        print("Error in /chat route:", str(e))
        return jsonify({"response": "An internal error occurred. Please try again later."})


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
