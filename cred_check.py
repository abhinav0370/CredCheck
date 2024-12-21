import requests
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize the tokenizer (shared between both models)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Initialize the base BERT model for embeddings
try:
    base_model = AutoModel.from_pretrained("bert-base-uncased")
    base_model.eval()  # Set to evaluation mode
    logger.info("Base BERT model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load the base BERT model: {e}")
    st.stop()

# Initialize your custom classification model for confidence scores
try:
    custom_model_path = Path("./fake_news_detector_model")  # Adjust the path as needed
    confidence_model = AutoModelForSequenceClassification.from_pretrained(custom_model_path)
    confidence_model.eval()  # Set to evaluation mode
    logger.info(f"Custom confidence model loaded successfully from {custom_model_path.resolve()}")
except Exception as e:
    logger.error(f"Failed to load the custom confidence model: {e}")
    st.stop()

def get_embeddings(text):
    """
    Generates embeddings for the given text using the base BERT model.
    """
    try:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            # Obtain the hidden states from the base BERT model
            outputs = base_model(**inputs)
        
        # Compute the mean of the token embeddings to get a single vector
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        return embeddings.numpy()
    except Exception as e:
        logger.error(f"Error generating embeddings for text '{text}': {e}")
        return np.array([])  # Return an empty array in case of failure

def predict_confidence(headline):
    """
    Predicts the confidence score of the headline being true using the custom classification model.
    """
    try:
        # Tokenize the headline
        inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            # Obtain logits from the custom classification model
            outputs = confidence_model(**inputs)
            logits = outputs.logits
        
        # Apply softmax to convert logits to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Extract the probability of the 'true' class (assuming it's at index 1)
        confidence_score = probs[0][1].item()
        logger.debug(f"Confidence score for headline '{headline}': {confidence_score}")
        return confidence_score
    except Exception as e:
        logger.error(f"Error predicting confidence for headline '{headline}': {e}")
        return 0.0  # Return a default low confidence score in case of failure

def google_search(query, num_results=5):
    """
    Performs a Google search using the Custom Search API.
    """
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={CUSTOM_SEARCH_ENGINE_ID}"
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json().get("items", [])
            logger.info(f"Received {len(results)} search results for query '{query}'.")
            return [
                {"title": item.get("title", ""), "snippet": item.get("snippet", ""), "link": item.get("link", "")}
                for item in results[:num_results]
            ]
        else:
            error_message = f"Error {response.status_code}: {response.text}"
            logger.error(error_message)
            return {"error": error_message}
    except Exception as e:
        logger.error(f"Exception during Google search: {e}")
        return {"error": str(e)}

def check_trusted_source(link):
    """
    Checks if the link is from a trusted source.
    """
    for source in TRUSTED_SOURCES:
        if source in link:
            logger.debug(f"Link '{link}' is from a trusted source: {source}")
            return True
    logger.debug(f"Link '{link}' is not from a trusted source.")
    return False

def calculate_similarity(headline, search_results):
    """
    Calculates the cosine similarity between the headline and each search result.
    """
    try:
        # Generate embeddings for the headline
        headline_emb = get_embeddings(headline)
        if headline_emb.size == 0:
            logger.warning("Headline embeddings are empty. Skipping similarity calculation.")
            return []
        
        similarities = []

        for result in search_results:
            # Combine title and snippet to form the result text
            result_text = f"{result['title']} {result['snippet']}"
            
            # Generate embeddings for the result text
            result_emb = get_embeddings(result_text)
            if result_emb.size == 0:
                similarity = 0.0  # Assign zero similarity if embeddings are missing
            else:
                # Calculate cosine similarity between headline and result embeddings
                similarity = cosine_similarity(headline_emb, result_emb)[0][0]
            
            similarities.append(similarity)
            logger.debug(f"Similarity between headline and result '{result['title']}': {similarity}")

        return similarities
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return []

def enhance_credibility_score(link, headline):
    """
    Enhances the credibility score based on the source and headline content.
    """
    credibility_score = 0

    if check_trusted_source(link):
        credibility_score += 0.5

    if 'bbc' in link or 'reuters' in link:
        credibility_score += 0.3
    if 'factcheck' in headline.lower():
        credibility_score += 0.2

    logger.debug(f"Credibility score for link '{link}': {credibility_score}")
    return credibility_score

def fake_news_detector(headline):
    """
    Detects whether the given headline is fake news based on similarity, credibility, and confidence score.
    """
    try:
        logger.info(f"Analyzing headline: '{headline}'")
        
        # Perform a Google search for the headline
        search_results = google_search(headline.strip())
        if isinstance(search_results, dict) and "error" in search_results:
            logger.warning(f"Google Search Error: {search_results['error']}")
            return search_results

        # Calculate cosine similarities between the headline and search results
        similarities = calculate_similarity(headline, search_results)
        
        # Enhance credibility scores based on sources and headline content
        credibility_scores = [
            enhance_credibility_score(result["link"], result["title"]) for result in search_results
        ]
        
        # Predict confidence score using the custom classification model
        confidence_score = predict_confidence(headline)

        # Compute average similarity and credibility
        average_similarity = float(np.mean(similarities)) if similarities else 0
        average_credibility = float(np.mean(credibility_scores)) if credibility_scores else 0

        logger.info(f"Average Similarity: {average_similarity}")
        logger.info(f"Average Credibility: {average_credibility}")
        logger.info(f"Confidence Score: {confidence_score}")

        # Determine if the headline is fake based on thresholds
        is_fake = (average_similarity < 0.75) and (average_credibility <= 0.01) and (confidence_score <= 0.7)

        logger.info(f"Is Fake: {is_fake}")

        return {
            "headline": headline,
            "average_similarity": average_similarity,
            "average_credibility": average_credibility,
            "confidence_score": confidence_score,
            "is_fake": bool(is_fake),
        }
    except Exception as e:
        logger.error(f"Error in fake_news_detector: {e}")
        return {"error": str(e)}

# Example usage:
if __name__ == "__main__":
    test_headline = "Breaking News: Scientists Discover Cure for Common Cold"
    result = fake_news_detector(test_headline)
    print(result)
