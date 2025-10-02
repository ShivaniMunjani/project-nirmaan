from nltk.sentiment.vader import SentimentIntensityAnalyzer
# We no longer need to download the data here. We will do it in app.py

def analyze_vendor_report(text):
    """Analyzes text for sentiment and key risk keywords."""
    if not text:
        return {"sentiment": 0.0, "risks_found": []}
    
    # We can now safely create the analyzer, as the data will be pre-downloaded
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)['compound']
    
    risk_keywords = ["delay", "shortage", "issue", "problem", "escalate", "pending", "halt", "concern"]
    risks_found = [word for word in risk_keywords if word in text.lower()]
    
    return {"sentiment": sentiment, "risks_found": risks_found}
