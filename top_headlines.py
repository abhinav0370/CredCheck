# top_headlines.py
import requests
import streamlit as st

# NewsAPI endpoint and API key
API_KEY = ""
BASE_URL = "https://newsapi.org/v2/top-headlines"

# Parameters for the API request
params = {
    "country": "us",  # Change 'us' to other country codes as needed
    "apiKey": API_KEY,
    "pageSize": 10,    # Fetch only 10 news articles
}

def fetch_headlines():
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        news_data = response.json()
        if news_data.get("status") == "ok":
            articles = news_data.get("articles", [])
            headlines = [article['title'] for article in articles]
            return headlines
        else:
            return {"error": "Unable to fetch news. Check API key or parameters."}
    else:
        return {"error": f"HTTP Error: {response.status_code}"}