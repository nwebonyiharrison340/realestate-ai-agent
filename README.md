# 🏡 Tinah — Qarba Real Estate AI Assistant

> An intelligent conversational AI assistant built for [Qarba.com](https://qarba.com) — Nigeria's forward-thinking real estate platform. Tinah allows users to interact naturally with live property listings, agent data, and real estate FAQs through a seamless chat interface.

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Aim and Objectives](#aim-and-objectives)
- [Significance of the Project](#significance-of-the-project)
- [Scope](#scope)
- [Project Outline](#project-outline)
- [Methodology and System Design](#methodology-and-system-design)
- [Tools and Libraries](#tools-and-libraries)
- [System Architecture](#system-architecture)
- [Frontend Design](#frontend-design)
- [Backend (Flask Application)](#backend-flask-application)
- [Conclusion](#conclusion)
- [References](#references)

---

## Introduction

The real estate sector in Nigeria is evolving rapidly, driven by urbanization, demographic expansion, and a growing digital economy. **Qarba**, a pioneering Nigerian real estate platform, has positioned itself as a forward-thinking leader committed to transforming how Nigerians discover, evaluate, and acquire property.

**Tinah** serves as an intelligent conversational interface that allows users to interact directly with Qarba's live property listings, agent profiles, and knowledge base — all through natural language. Rather than navigating complex filters or static search bars, users can simply ask:

> *"Show me 3-bedroom apartments in Lekki under ₦2 million"*

...and receive instant, contextually rich responses.

The integration of Tinah into the Qarba ecosystem marks a strategic transition from static property browsing to dynamic, AI-powered real estate discovery. Through this project, Qarba demonstrates a commitment to leveraging artificial intelligence not only as a technological enhancement, but as a core driver of user value.

---

## Problem Statement

Despite increased digitalization in Nigeria's real estate sector, property discovery and acquisition processes remain fragmented, time-consuming, and inaccessible to a significant portion of the population. Users face:

- Overwhelming property listings with poor filtering mechanisms
- Lack of instant, personalised responses to property inquiries
- Inability to interact with platforms using natural, conversational language
- Limited 24/7 availability of real estate support and guidance

These challenges underscore the absence of an intelligent system capable of understanding natural language and retrieving relevant, real-time property information. To address this gap, **Tinah** was developed — leveraging artificial intelligence, semantic search, and real-time API integration to deliver a conversational real estate experience.

---

## Aim and Objectives

### Aim

The primary aim of this project is to design and implement an AI-powered conversational assistant, code-named **Tinah**, that enhances the property discovery experience on Qarba.com through natural language interaction.

### Objectives

The objectives of this project are:

- ✅ To implement a conversational AI interface for Qarba.com
- ✅ To integrate real estate data through Qarba's APIs
- ✅ To enhance information retrieval using **Semantic Search** and **Fuzzy Matching**
- ✅ To leverage generative AI for natural and adaptive responses
- ✅ To ensure a secure, modular, and scalable backend architecture
- ✅ To deliver an intelligent and visually appealing user experience

---

## Significance of the Project

Tinah's significance lies in its ability to transform static browsing into dynamic engagement. By leveraging Natural Language Processing (NLP), generative AI, and live data integration, Tinah bridges the gap between complex property databases and everyday users who simply want answers.

At its core, Tinah embodies the convergence of **business innovation**, **technological intelligence**, and **human-centric design** — making it a landmark step toward redefining digital real estate interaction in Nigeria and across Africa.

---

## Scope

The scope of this project encompasses the design, development, and deployment of an intelligent conversational system, Tinah, integrated into the Qarba real estate platform.

It covers both **frontend** and **backend** implementation:

- **Frontend:** A responsive and interactive chat widget, accessible across all Qarba web pages
- **Backend:** A Flask-based server handling AI reasoning, API calls, semantic search, and response formatting
- **AI Layer:** SentenceTransformer embeddings + Gemma 2-9B generative model
- **Data Layer:** Live property listings, agent data, and blog content via Qarba's APIs, supplemented by a locally cached FAQ dataset

---

## Project Outline

The development of Tinah was carried out in **six major stages**:

### Stage 1 — Requirement Analysis and System Conceptualization

Identified key functional and non-functional requirements through analysis of Qarba's existing platform. Established the project's architectural direction integrating AI-based Natural Language Understanding (NLU) and REST API communication.

**Deliverables:**
- Clear project roadmap and system objectives
- Identification of Qarba's live data sources (property, agent, and blog APIs)
- Determination of the AI model pipeline (SentenceTransformer + LLM integration)
- Definition of the chatbot's functional boundaries (property retrieval, FAQs, blogs)

---

### Stage 2 — Data Preparation and Knowledge Base Development

Curated foundational data sources for Tinah's reasoning and contextual understanding. Collected FAQs, property descriptions, agent profiles, and blog content. Formatted content into structured FAQ datasets (`faqs.json`) and semantically encoded using SentenceTransformer embeddings.

**Deliverables:**
- Integration-ready FAQ database with semantic embeddings
- Automated web scraping script (`scraper.py`) for dynamic content retrieval
- Preprocessing utilities for noise reduction, tokenization, and text embedding

---

### Stage 3 — Backend Development and AI Integration

Designed and implemented the Flask-based backend architecture as the central communication hub.

The backend employed a modular structure combining:
- Natural language understanding using the SentenceTransformer model
- Hybrid similarity computation through cosine similarity and fuzzy matching
- Dynamic API requests to Qarba's property, agent, and client endpoints
- Response generation using OpenAI's `chat.completions` endpoint powered by **Gemma 2-9B**

**Deliverables:**
- Fully functional Flask server
- AI pipeline integration for hybrid semantic reasoning
- Modular error handling and logging mechanisms

---

### Stage 4 — Frontend Interface Design and User Experience Development

Built a responsive, visually engaging chat interface using HTML, CSS, and JavaScript.

**Key interface features:**
- A floating chat widget accessible across all Qarba pages
- Real-time message rendering and smooth animation transitions
- A typing indicator to simulate human-like interaction
- Support for media embedding (e.g., property images)

The frontend communicates asynchronously with the Flask backend via AJAX (Fetch API), ensuring non-blocking, real-time responses.

**Deliverables:**
- `index.html` — UI structure
- `style.css` — Design consistency and responsiveness
- `script.js` — Interaction logic and asynchronous communication

---

### Stage 5 — Integration, Testing, and Optimization

Unified all components — frontend, backend, APIs, and AI — into a cohesive system.

**Testing covered:**
- Functional testing to verify correct data retrieval and chatbot logic
- Performance testing to assess API response time and caching efficiency
- User Acceptance Testing (UAT) with simulated queries and sample property searches

Caching mechanisms were introduced using Python's `functools.lru_cache` to reduce redundant API calls and enhance speed.

**Deliverables:**
- Fully functional integrated system
- Performance-optimized backend

---

### Stage 6 — Deployment and Evaluation

Deployed Tinah to a live environment for pilot testing using cloud-based infrastructure.

**Evaluation metrics:**
- Response accuracy (alignment between user intent and retrieved data)
- System stability under concurrent user loads
- User satisfaction metrics from pilot usage

**Deliverables:**
- Deployed version of Tinah integrated into Qarba.com
- Post-deployment performance analysis
- Feedback documentation for future model iterations

---

## Methodology and System Design

### 7.1 Technical Workflow

Tinah was developed using a **data-driven, modular, and iterative methodology**, ensuring that each layer (frontend, backend, AI) was independently testable yet cohesive in production.

#### 7.1.1 User Query Collection

The chat interface captures the user's text input. When the user hits **"Send"**, the message is passed to the Flask `/chat` endpoint via an AJAX call:

```javascript
// Frontend sends message to Flask backend
fetch('/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: userInput })
});
```

#### 7.1.2 Query Processing and Context Building

Once Flask receives the query, `app.py` executes these steps:

1. **Text cleaning and normalization** — Removes unnecessary characters, converts to lowercase
2. **FAQ Semantic Matching** — Uses `SentenceTransformer` (`all-MiniLM-L6-v2`) to generate embeddings for both the user query and stored FAQs, then computes cosine similarity

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def find_best_faq(user_query, faqs):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    faq_embeddings = model.encode([faq['question'] for faq in faqs], convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, faq_embeddings)
    best_idx = scores.argmax().item()
    if scores[0][best_idx] >= 0.65:
        return faqs[best_idx]['answer']
    return None
```

If the similarity score is **above 0.65**, Tinah uses the FAQ answer directly.

#### 7.1.3 Response Generation (AI Model)

After gathering context (FAQs, property data, blogs), Tinah compiles it into a structured system prompt sent to the AI model. The model (`google/gemma-2-9b-it`) is queried using OpenAI's `chat.completions` endpoint.

This ensures that responses:
- Use Qarba-specific context
- Avoid hallucination
- Remain natural and professional

#### 7.1.4 Display and Formatting

Once the AI's response is received:
- It is formatted for web display via `formatBotMessage()`
- Images are embedded dynamically for properties (using regex)
- The final text is appended to the chat window in real time

---

## Tools and Libraries

| Library | Purpose |
|---|---|
| **Flask** | Web server and API handling |
| **SentenceTransformer** | Semantic similarity model for FAQs |
| **OpenAI / Google API** | AI-driven text generation |
| **Requests** | HTTP client for API integration |
| **Playwright** | Dynamic web scraping (for FAQs) |
| **Dotenv** | Environment variable management |
| **FuzzyWuzzy** | Fuzzy string matching for flexible searches |
| **BeautifulSoup** | HTML parsing for structured data extraction |

---

## System Architecture

Tinah follows a **modular, multi-layered architecture** that ensures scalability, maintainability, and reliability:

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│          Chat Interface — HTML + CSS + JavaScript            │
└───────────────────────┬─────────────────────────────────────┘
                        │  JSON / HTTP POST (/chat)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                          │
│          Flask Backend (app.py) — Request Handling,          │
│          Session Management, API Orchestration               │
└──────────┬─────────────────────────────┬────────────────────┘
           │                             │
           ▼                             ▼
┌──────────────────────┐    ┌────────────────────────────────┐
│  INTELLIGENCE LAYER  │    │         DATA LAYER             │
│  SentenceTransformer │    │  Qarba Property / Agent APIs   │
│  Semantic Search     │    │  Blog Content API              │
│  Gemma 2-9B LLM      │    │  Local FAQ Dataset (faqs.json) │
│  Fuzzy Matching      │    │  LRU Cache                     │
└──────────────────────┘    └────────────────────────────────┘
```

This architecture ensures data flows **bidirectionally** — from user queries to AI responses — with Flask orchestrating all communication.

---

## Frontend Design

### `index.html`
The root HTML file housing Tinah's chat widget. Provides the visual skeleton; all interactivity is managed by `script.js`.

- Chat widget is hidden by default, activated via a floating **"Chat with Tinah"** button
- `#chat-toggle-btn` toggles visibility of the chat widget
- `.chat-header` contains branding and chatbot identity
- `.chat-body` dynamically displays messages using JavaScript
- `.chat-input` accepts user text and triggers `sendMessage()` events
- Uses Flask's `url_for()` to reference static assets safely

### `style.css`
Ensures a professional, modern chat interface consistent with Qarba's branding.

| Component | Style |
|---|---|
| Floating chat button | Fixed bottom-right, hover effects |
| Chat window | Rounded card design, drop shadow, smooth animation |
| Message bubbles | Blue for user, light blue for Tinah |
| Typing indicator | Three pulsing dots simulating live conversation |
| Layout | Mobile-first responsive design |

### `script.js`
Handles all interactivity and communication with Flask via the Fetch API.

- Captures and sends user input to the `/chat` endpoint
- Receives AI responses and renders them instantly
- Handles "Tinah is typing…" animation for natural conversational flow
- Auto-scrolls to new messages for seamless UX

---

## Backend (Flask Application)

The backend (`app.py`) serves as the **brain and coordinator** of the system, connecting the frontend to Qarba's APIs, the AI model, and the semantic search engine.

### Environment Configuration
```python
from dotenv import load_dotenv
load_dotenv()  # Loads sensitive credentials from .env
```

### Semantic Matching for FAQs
```python
# Converts user queries and FAQ questions into numerical embeddings.
# Measures semantic closeness using cosine similarity.
# Returns the best-matching FAQ if similarity exceeds 0.65.
similarity = util.cos_sim(query_embedding, faq_embeddings)
```

### Qarba API Data Integration
```python
@lru_cache(maxsize=None)
def fetch_properties():
    # Fetches live property data from Qarba's API endpoint.
    # Includes timeout and error handling.
    # Data is cached using @lru_cache to prevent redundant requests.
    response = requests.get(QARBA_API_URL, timeout=10)
    return response.json()
```

### AI Communication
```python
# Sends structured user–context conversation to the AI model.
# The model generates coherent, natural responses grounded in Qarba's data.
# Tinah filters and formats this output before sending to the frontend.
response = openai_client.chat.completions.create(
    model="google/gemma-2-9b-it",
    messages=conversation_history
)
```

### Response Delivery
Sends structured JSON back to the frontend, ready for immediate display in the UI.

```python
return jsonify({"response": formatted_reply})
```

---

## Conclusion

The development and implementation of **Tinah — the Qarba Real Estate AI Assistant** represents a significant advancement in the intersection of artificial intelligence and Nigeria's property technology (PropTech) sector.

This project successfully demonstrates how conversational AI can:
- Enhance user engagement on real estate platforms
- Streamline property search through natural language interaction
- Facilitate efficient, 24/7 access to real estate information

Through a carefully designed system architecture — integrating frontend interactivity, Flask-based backend logic, API connectivity, and AI-driven reasoning — Tinah delivers a seamless, context-aware user experience.

The project's success lies not merely in building a chatbot interface, but in constructing a **knowledge-integrated assistant** capable of reasoning over dynamic, real-world property data. It validates the commercial viability of AI-powered assistants in the African real estate market and establishes a strong foundation for future enhancements including voice interaction, multilingual support, and predictive property analytics.

---

## References

- Abubakar, M., & Yusuf, K. (2023). *Adoption of Artificial Intelligence in Nigeria's Real Estate Sector: Opportunities and Challenges.* Journal of African Technology Studies.
- Alshahrani, A., & Alhajj, R. (2021). *Chatbot Systems in Real Estate: Enhancing Customer Experience through Natural Language Processing.* International Journal of Advanced Computer Science.
- Amaral, J. P., Rodrigues, J. J. P. C., & Alberti, A. M. (2020). *A Smart Real Estate Management System Based on Cloud and IoT Technologies.* IEEE Access, 8, 112134–112150.
- Brown, T. B., Mann, B., et al. (2020). *Language Models are Few-Shot Learners.* Advances in Neural Information Processing Systems (NeurIPS).
- Feng, Y., & Lin, Z. (2022). *AI-Powered Customer Service: The Case of Real Estate Chatbots in Online Marketplaces.* Journal of Information Systems, 36(3).
- Google DeepMind. (2024). *Gemma Model Card: Google's Lightweight Family of Open-Weight Language Models.* Retrieved from https://ai.google.dev/gemma
- Hugging Face. (2023). *SentenceTransformers Documentation: Semantic Search and Similarity Models.* Retrieved from https://www.sbert.net
- Nguyen, P., Do, H., & Hoang, M. (2021). *Combining Semantic Search and Fuzzy Matching for Intelligent Information Retrieval.* International Conference on Information Technology.
- OpenAI. (2024). *OpenAI API Documentation.* Retrieved from https://platform.openai.com/docs
- Oyelami, T., & Nwogu, A. (2022). *Artificial Intelligence Adoption in African Property Technology (PropTech).* Journal of African Digital Transformation.
- Python Software Foundation. (2023). *Flask Documentation: Lightweight Web Framework for Python.* Retrieved from https://flask.palletsprojects.com/
- Qarba Technologies. (2025). *Qarba Real Estate API Reference.* Internal API Documentation, Qarba.com.
- Rahman, M., & Sharma, P. (2022). *Data-Driven Chatbots for Business Automation: A Framework for Real Estate Applications.* International Journal of Emerging Technologies.
- Rosenfeld, A., & Kraus, S. (2019). *Explainable AI in Human-Agent Systems.* ACM Transactions on Interactive Intelligent Systems, 9(4), 1–26.
- Scikit-learn Developers. (2023). *Scikit-learn: Machine Learning in Python.* Retrieved from https://scikit-learn.org/stable/
- Wang, X., & Li, J. (2020). *Design and Implementation of Intelligent Conversational Systems Using RESTful APIs.* International Journal of Artificial Intelligence Research.

---

<div align="center">
  <p>Built with ❤️ for <strong>Qarba.com</strong> — Nigeria's Intelligent Real Estate Platform</p>
</div>
