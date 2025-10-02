from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

def analyze_vendor_report(text):
    """Analyzes text for sentiment and key risk keywords."""
    if not text:
        return {"sentiment": 0.0, "risks_found": []}
    
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)['compound']
    
    risk_keywords = ["delay", "shortage", "issue", "problem", "escalate", "pending", "halt", "concern"]
    risks_found = [word for word in risk_keywords if word in text.lower()]
    
    return {"sentiment": sentiment, "risks_found": risks_found}