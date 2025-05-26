import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import re
from transformers import pipeline
import textwrap
import os

# --- NLTK Downloads (ENSURE THESE RUN BEFORE ANY NLTK FUNCTION IS CALLED) ---
# It's crucial that these downloads happen before SentimentIntensityAnalyzer() or any tokenize function is called.
print("Checking/Downloading NLTK resources... (This runs once)")

# List of NLTK resources required
nltk_resources = {
    'punkt': 'tokenizers/punkt',
    'punkt_tab': 'tokenizers/punkt_tab', # The problematic one!
    'stopwords': 'corpora/stopwords',
    'vader_lexicon': 'sentiment/vader_lexicon'
}

for resource_name, resource_path in nltk_resources.items():
    try:
        nltk.data.find(resource_path)
        print(f"  -> '{resource_name}' is already downloaded.")
    except LookupError: # Catch the specific LookupError raised by nltk.data.find
        print(f"  -> '{resource_name}' not found. Downloading...")
        try:
            nltk.download(resource_name)
            print(f"  -> Downloaded '{resource_name}'.")
        except Exception as e:
            print(f"  -> Error downloading '{resource_name}': {e}")
            print(f"     Please try running 'python -m nltk.downloader {resource_name}' manually.")
print("NLTK resources check complete.\n")


# --- Global Models/Analyzers (Initialized once for efficiency) ---
sentiment_analyzer = SentimentIntensityAnalyzer()
summarizer_pipeline = None # Will be initialized lazily

# --- Configuration ---
SUMMARY_MIN_LENGTH = 20
SUMMARY_MAX_LENGTH = 100
TEXT_WRAP_WIDTH = 80

# --- 1. Introduction ---
def display_introduction():
    """Displays a brief introduction to the Smart Email Analyzer."""
    print("=" * TEXT_WRAP_WIDTH)
    print("🚀 Smart Email Analyzer 🚀".center(TEXT_WRAP_WIDTH))
    print("=" * TEXT_WRAP_WIDTH)
    print("\nIn today's fast-paced digital world, email overload is a real challenge.")
    print("Natural Language Processing (NLP) offers powerful solutions to manage this.")
    print("This 'Smart Email Analyzer' demonstrates how NLP can help:")
    print("  - Generate quick summaries (TL;DR)")
    print("  - Determine email sentiment (positive, neutral, negative)")
    print("  - Classify email priority (high, medium, low)")
    print("\nLet's dive in and make your inbox smarter!\n")
    print("-" * TEXT_WRAP_WIDTH + "\n")

# --- 2. Sample Emails ---
def load_sample_emails():
    """
    Loads a predefined set of sample emails.
    In a real application, these would be loaded from an actual inbox (e.g., via IMAP).
    """
    sample_emails = [
        {
            "id": 1,
            "subject": "Urgent: Project Deadline Approaching",
            "body": "Hi Team,\n\nThis is an urgent reminder that the Project Alpha deadline is this Friday, EOD. We need to finalize all pending tasks and ensure the presentation is ready. Please provide an immediate update on your progress. Any delays will impact the entire project timeline. Your prompt attention is critical.\n\nBest regards,\nSarah"
        },
        {
            "id": 2,
            "subject": "Follow-up: Meeting Notes from Yesterday",
            "body": "Hi All,\n\nHope you're well. Attached are the meeting notes from yesterday's discussion. Please review them at your convenience and let me know if there are any corrections or additions. We'll revisit these points next week. Thanks!\n\nCheers,\nJohn"
        },
        {
            "id": 3,
            "subject": "Great Job on the Q3 Report!",
            "body": "Dear Alex,\n\nI just wanted to express my sincere appreciation for the excellent work on the Q3 financial report. The insights were incredibly valuable, and the presentation was flawless. This will significantly help our strategic planning. Keep up the fantastic work!\n\nBest,\nCEO"
        },
        {
            "id": 4,
            "subject": "Regarding your recent inquiry",
            "body": "Dear Customer,\n\nThank you for reaching out. We have received your inquiry regarding order #12345. Our support team is currently reviewing your case, and we aim to respond within 24-48 business hours with a detailed resolution. We appreciate your patience.\n\nSincerely,\nSupport Team"
        },
        {
            "id": 5,
            "subject": "System Maintenance Notification",
            "body": "Dear User,\n\nPlease be advised that our systems will undergo scheduled maintenance on Saturday, Oct 28th, from 10 PM to 2 AM UTC. During this period, some services may experience temporary interruptions. We apologize for any inconvenience this may cause. No action is required from your side.\n\nThank you for your understanding,\nIT Department"
        },
        {
            "id": 6,
            "subject": "Complaint: Service Disruption",
            "body": "I am extremely disappointed with the recent service disruption. My internet has been down for over 12 hours, and I haven't received a clear explanation or an estimated fix time. This is unacceptable and impacting my work. I demand an immediate resolution and compensation.\n\nFrustrated User"
        }
    ]
    print(f"Loaded {len(sample_emails)} sample emails.\n")
    return sample_emails

# --- 3. Email Preprocessing ---
def preprocess_text(text):
    """
    Performs NLTK-based cleaning:
    - Lowercasing
    - Removing punctuation
    - Tokenization
    - Stopword removal
    - Removing extra spaces
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # Remove punctuation
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    cleaned_text = " ".join(filtered_tokens)
    return cleaned_text

# --- 4. Summarization (LLM/BERT - using HuggingFace BART) ---
def summarize_email(text):
    """
    Generates a TL;DR summary using a pre-trained BART model.
    Initializes the model on first call.
    """
    global summarizer_pipeline
    if summarizer_pipeline is None:
        print("\n⏳ Initializing summarization model (facebook/bart-large-cnn)... This may take a moment.")
        try:
            # Suppress excessive logging from transformers
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            # Using device=-1 for CPU or specific GPU index (e.g., 0)
            # In a local environment, -1 for CPU is generally safer unless you have CUDA setup.
            summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
            print("✅ Summarization model initialized.")
        except Exception as e:
            print(f"❌ Error initializing summarization model: {e}")
            print("Please ensure you have enough memory/disk space and internet connection.")
            print("Falling back to a simpler summary method for this session.")
            summarizer_pipeline = "fallback" # Indicate fallback mode
            return "Summarization service unavailable."

    if summarizer_pipeline == "fallback":
        # Simple fallback: first few sentences
        sentences = nltk.sent_tokenize(text)
        return " ".join(sentences[:2]) + ("..." if len(sentences) > 2 else "")


    try:
        # Limit input length for the model to avoid truncation issues for very long emails
        # BART models typically handle up to 1024 tokens.
        # We can trim the text if it's excessively long.
        if len(text) > 10000: # Arbitrary character limit for very long texts
            text = text[:10000] + "..."

        summary = summarizer_pipeline(
            text,
            max_length=SUMMARY_MAX_LENGTH,
            min_length=SUMMARY_MIN_LENGTH,
            do_sample=False # For deterministic summaries
        )[0]['summary_text']
        return summary
    except Exception as e:
        print(f"⚠️ Error during summarization: {e}")
        # Fallback to simple first sentence if model fails on a specific text
        sentences = nltk.sent_tokenize(text)
        return " ".join(sentences[:1]) + " (Summarization error, basic fallback)"

# --- 5. Sentiment Analysis (NLTK VADER) ---
def analyze_sentiment(text):
    """
    Analyzes the sentiment of the email text using NLTK VADER.
    Returns polarity scores and a categorized label.
    """
    scores = sentiment_analyzer.polarity_scores(text)
    compound_score = scores['compound']

    if compound_score >= 0.05:
        label = "Positive 😊"
    elif compound_score <= -0.05:
        label = "Negative 😠"
    else:
        label = "Neutral 😐"

    return {"label": label, "scores": scores}

# --- 6. Priority Classification ---
def classify_priority(text, sentiment_score):
    """
    Classifies email priority based on keywords and sentiment.
    This is a rule-based approach. An LLM could also be prompted for this.
    """
    text_lower = text.lower()
    priority_keywords = {
        "high": ["urgent", "asap", "immediate", "deadline", "critical", "important", "action required", "escalate"],
        "medium": ["follow up", "review", "request", "please", "kindly", "update"],
        # Low is default if no high/medium keywords are found
    }

    # Check for high priority keywords
    for keyword in priority_keywords["high"]:
        if keyword in text_lower:
            # If negative sentiment with high priority keyword, it's very high
            if sentiment_score < -0.2:
                return "High (Critical! 🚨)"
            return "High (Urgent! ⬆️)"

    # Check for medium priority keywords
    for keyword in priority_keywords["medium"]:
        if keyword in text_lower:
            return "Medium (Standard ➡️)"

    # Default to low if no specific keywords are found
    return "Low (Informational ⬇️)"

# --- 7. Results Display ---
def display_results(processed_emails_data):
    """Displays the analysis results for each email."""
    print("\n" + "=" * TEXT_WRAP_WIDTH)
    print("✨ Analysis Results ✨".center(TEXT_WRAP_WIDTH))
    print("=" * TEXT_WRAP_WIDTH + "\n")

    for data in processed_emails_data:
        original_email = data['original_email']
        summary = data['summary']
        sentiment = data['sentiment']
        priority = data['priority']
        preprocessed_text = data['preprocessed_text']

        print(f"\n--- Email ID: {original_email['id']} ---")
        print(f"Subject: {original_email['subject']}")
        print(f"Original Body:\n{textwrap.fill(original_email['body'], width=TEXT_WRAP_WIDTH)}")

        print("\n--- Preprocessing Example (for keywords/traditional NLP) ---")
        print(f"Cleaned Text:\n{textwrap.fill(preprocessed_text, width=TEXT_WRAP_WIDTH)}")

        print("\n--- Smart Analysis ---")
        print(f"Summary (TL;DR):")
        print(f"  {textwrap.fill(summary, width=TEXT_WRAP_WIDTH - 2)}") # Indent summary

        print(f"\nSentiment: {sentiment['label']}")
        print(f"  (Compound Score: {sentiment['scores']['compound']:.2f}, Pos: {sentiment['scores']['pos']:.2f}, Neu: {sentiment['scores']['neu']:.2f}, Neg: {sentiment['scores']['neg']:.2f})")

        print(f"\nPriority: {priority}")
        print("-" * (TEXT_WRAP_WIDTH // 2) + "\n")

# --- 8. Conclusion ---
def display_conclusion():
    """Provides a brief conclusion and potential improvements."""
    print("=" * TEXT_WRAP_WIDTH)
    print("🎉 Analysis Complete! 🎉".center(TEXT_WRAP_WIDTH))
    print("=" * TEXT_WRAP_WIDTH)
    print("\nBy combining preprocessing, advanced summarization, sentiment analysis,")
    print("and priority classification, we've transformed raw emails into actionable insights.")
    print("This allows users to quickly grasp the essence and urgency of their communications.")

    print("\n💡 Potential Improvements & Next Steps:")
    print("  1.  **Fine-tuning Models:** Train summarization/priority models on domain-specific email datasets for higher accuracy.")
    print("  2.  **Entity Extraction:** Identify names, dates, locations, company names (e.g., with SpaCy).")
    print("  3.  **Topic Modeling:** Discover the main topics discussed across a large set of emails (e.g., using LDA).")
    print("  4.  **Action Item Extraction:** Automatically pull out tasks or requests.")
    print("  5.  **Integration:** Connect to a real email client (Gmail API, Outlook API) for live analysis.")
    print("  6.  **User Interface:** Build a web or desktop app for a more interactive experience.")
    print("  7.  **More Sophisticated Priority:** Use a classification model (e.g., Logistic Regression, BERT) trained on labeled data.")
    print("  8.  **Thread Analysis:** Analyze entire email conversations for context.")
    print("\nThank you for exploring the Smart Email Analyzer!")
    print("-" * TEXT_WRAP_WIDTH + "\n")

# --- Main Execution Flow ---
def main():
    display_introduction()
    emails = load_sample_emails()
    processed_emails_data = []

    print("\n--- Starting Email Analysis ---\n")
    for i, email in enumerate(emails):
        # Combine subject and body for analysis, as subject often contains key info
        full_text = email['subject'] + ". " + email['body']

        # 3. Email Preprocessing Example (for reference/traditional keyword tasks)
        preprocessed_text_example = preprocess_text(full_text)

        # 4. Summarization (using original full text for context)
        summary = summarize_email(full_text)

        # 5. Sentiment Analysis (using original full text for nuance)
        sentiment_analysis_result = analyze_sentiment(full_text)

        # 6. Priority Classification (using original full text and sentiment)
        priority = classify_priority(full_text, sentiment_analysis_result['scores']['compound'])

        processed_emails_data.append({
            'original_email': email,
            'preprocessed_text': preprocessed_text_example, # Stored for display example
            'summary': summary,
            'sentiment': sentiment_analysis_result,
            'priority': priority
        })

    display_results(processed_emails_data)
    display_conclusion()

if __name__ == "__main__":
    main()