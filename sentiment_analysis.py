import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load your comments CSV
df = pd.read_csv("Sentimentanalysis-ready.csv")  # Ensure there's a column named 'Comment'

# Load CardiffNLP model and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Sentiment labels
labels = ['Negative', 'Neutral', 'Positive']

# Preprocessing
def preprocess(text):
    new_text = []
    for t in str(text).split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Sentiment scoring function
def analyze_sentiment(text):
    try:
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        probs = softmax(scores)

        # Get label
        sentiment_label = labels[probs.argmax()]

        # Calculate polarity score: -1 * negative_prob + 0 * neutral + 1 * positive_prob
        polarity_score = -1 * probs[0] + 0 * probs[1] + 1 * probs[2]

        return sentiment_label, probs[0], probs[1], probs[2], polarity_score
    except:
        return "Neutral", 0.0, 1.0, 0.0, 0.0

# Apply function
df[['Sentiment_Label', 'Negative_Prob', 'Neutral_Prob', 'Positive_Prob', 'Polarity_Score']] = df['Comment'].apply(
    lambda x: pd.Series(analyze_sentiment(x))
)

# Save to CSV
df.to_csv("comments_with_cardiff_polarity.csv", index=False)
print("✅ Sentiment scores and polarity computed.")