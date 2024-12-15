# cred_check.py
import requests
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy

# Constants

GOOGLE_API_KEY = ""
CUSTOM_SEARCH_ENGINE_ID = ""

TRUSTED_SOURCES = [
    "bbc.com", "reuters.com", "apnews.com", "snopes.com", "theguardian.com", "nytimes.com", "washingtonpost.com",
    "bbc.co.uk", "cnn.com", "forbes.com", "npr.org", "wsj.com", "time.com", "usatoday.com", "bloomberg.com",
    "thehill.com", "guardian.co.uk", "huffpost.com", "independent.co.uk", "scientificamerican.com", "wired.com",
    "nationalgeographic.com", "marketwatch.com", "businessinsider.com", "abcnews.go.com", "news.yahoo.com",
    "theverge.com", "techcrunch.com", "theatlantic.com", "axios.com", "cnbc.com", "newsweek.com", "bbc.co.uk",
    "latimes.com", "thetimes.co.uk", "sky.com", "reuters.uk", "thehindu.com", "straitstimes.com", "foreignpolicy.com",
    "dw.com", "indianexpress.com", "dailymail.co.uk", "smh.com.au", "mint.com", "livemint.com"
]

# Initialize the tokenizer and model for BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Load the spaCy NER model
nlp = spacy.load("en_core_web_sm")

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    return embeddings.detach().numpy()

def google_search(query, num_results=5):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={CUSTOM_SEARCH_ENGINE_ID}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get("items", [])
        return [
            {"title": item["title"], "snippet": item["snippet"], "link": item["link"]}
            for item in results[:num_results]
        ]
    else:
        return {"error": f"Error {response.status_code}: {response.text}"}

def check_trusted_source(link):
    for source in TRUSTED_SOURCES:
        if source in link:
            return True
    return False

def calculate_similarity(headline, search_results):
    headline_emb = get_embeddings(headline)
    similarities = []

    for result in search_results:
        result_text = result["title"] + " " + result["snippet"]
        result_emb = get_embeddings(result_text)
        similarity = cosine_similarity(headline_emb, result_emb)[0][0]
        similarities.append(similarity)

    return similarities

def extract_named_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def contextual_analysis(headline, search_results):
    headline_entities = extract_named_entities(headline)
    context_score = 0

    for result in search_results:
        result_entities = extract_named_entities(result["title"] + " " + result["snippet"])
        if any(entity in headline_entities for entity in result_entities):
            context_score += 1

    return context_score / len(search_results) if search_results else 0

def enhance_credibility_score(link, headline):
    credibility_score = 0

    if check_trusted_source(link):
        credibility_score += 0.5

    if 'bbc' in link or 'reuters' in link:
        credibility_score += 0.3
    if 'factcheck' in headline.lower():
        credibility_score += 0.2

    return credibility_score

def fake_news_detector(headline):
    search_results = google_search(headline.strip())
    if isinstance(search_results, dict) and "error" in search_results:
        return search_results

    similarities = calculate_similarity(headline, search_results)
    credibility_scores = [
        enhance_credibility_score(result["link"], result["title"]) for result in search_results
    ]
    context_score = contextual_analysis(headline, search_results)

    average_similarity = float(np.mean(similarities)) if similarities else 0
    average_credibility = float(np.mean(credibility_scores)) if credibility_scores else 0
    is_fake = (average_similarity < 0.75) and (average_credibility <= 0.01)

    return {
        "headline": headline,
        "average_similarity": average_similarity,
        "average_credibility": average_credibility,
        "context_score": context_score,
        "is_fake": bool(is_fake),
    }