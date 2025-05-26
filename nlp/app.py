# email_analyzer_api.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import re
from transformers import pipeline
import os


nltk_resources = {
    'punkt': 'tokenizers/punkt',
    'punkt_tab': 'tokenizers/punkt_tab',
    'stopwords': 'corpora/stopwords',
    'vader_lexicon': 'sentiment/vader_lexicon'
}




for resource_name, resource_path in nltk_resources.items():
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource_name)



#  models
sentiment_analyzer = SentimentIntensityAnalyzer()
summarizer_pipeline = None


SUMMARY_MIN_LENGTH = 20
SUMMARY_MAX_LENGTH = 100


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


class EmailInput(BaseModel):
    subject: str
    body: str



# Helper Functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(filtered_tokens)



def summarize_email(text):
    global summarizer_pipeline
    if summarizer_pipeline is None:
        try:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        except Exception:
            summarizer_pipeline = "fallback"
            return "Summarization service unavailable."

    if summarizer_pipeline == "fallback":
        sentences = nltk.sent_tokenize(text)
        return " ".join(sentences[:2]) + ("..." if len(sentences) > 2 else "")

    try:
        if len(text) > 10000:
            text = text[:10000] + "..."

        summary = summarizer_pipeline(
            text,
            max_length=SUMMARY_MAX_LENGTH,
            min_length=SUMMARY_MIN_LENGTH,
            do_sample=False
        )[0]['summary_text']
        return summary
    except Exception:
        sentences = nltk.sent_tokenize(text)
        return " ".join(sentences[:1]) + " (Summarization error, basic fallback)"



def analyze_sentiment(text):
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {"label": label, "scores": scores}



def classify_priority(text, compound_score):
    text = text.lower()
    priority_keywords = {
        "high": ["urgent", "asap", "immediate", "deadline", "critical", "important", "action required", "escalate"],
        "medium": ["follow up", "review", "request", "please", "kindly", "update"]
    }
    for keyword in priority_keywords["high"]:
        if keyword in text:
            if compound_score < -0.2:
                return "High (Critical)"
            return "High (Urgent)"
    for keyword in priority_keywords["medium"]:
        if keyword in text:
            return "Medium"
    return "Low"



#Endpoint html

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()


# accepts

@app.post("/analyze")
async def analyze_email(email: EmailInput):
    full_text = email.subject + ". " + email.body
    preprocessed = preprocess_text(full_text)
    summary = summarize_email(full_text)
    sentiment = analyze_sentiment(full_text)
    priority = classify_priority(full_text, sentiment['scores']['compound'])
    return JSONResponse(content={
        "summary": summary,
        "sentiment": sentiment,
        "priority": priority,
        "preprocessed": preprocessed
    })
