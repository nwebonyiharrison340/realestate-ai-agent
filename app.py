from openai import OpenAI
import os
import json
import difflib
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from fuzzywuzzy import fuzz

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize OpenAI client (via OpenRouter)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Load FAQs from JSON file
def load_faqs():
    with open("faqs.json", "r", encoding="utf-8") as f:
        return json.load(f)

faqs = load_faqs()

# Function to find the most relevant FAQ
def find_best_faq(user_query):
    user_query_lower = user_query.lower()
    best_match = None
    best_score = 0

    for faq in faqs:
        if not isinstance(faq, dict):
            continue  # skip bad entries

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

        # Find best FAQ match
        best_match = find_best_faq(user_query)
        if best_match:
            context = best_match["answer"]
        else:
            context = "Sorry, I couldn’t find an exact answer to that question."

        # Generate AI response using OpenRouter model
        response = client.chat.completions.create(
            model="google/gemma-2-9b-it",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant for a real estate platform. "
                        "Always rely on the FAQ answer if available."
                    )
                },
                {
                    "role": "user",
                    "content": f"User asked: {user_query}\nFAQ answer: {context}"
                }
            ]
        )

        ai_message = response.choices[0].message.content.strip()
        return jsonify({"response": ai_message})

    except Exception as e:
        # Print the error in your terminal and return an error message to the frontend
        print("Error in /chat route:", str(e))
        return jsonify({"response": "An internal error occurred. Please try again later."})


@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)