from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd

# === Step 1: Load cleaned comments ===
df = pd.read_csv("final_dataset.csv")  # use your latest cleaned file
docs = df["cleaned_comment"].dropna().astype(str).tolist()

# === Step 2: UMAP - tighten clustering but keep global structure ===
umap_model = UMAP(
    n_neighbors=20,
    n_components=5,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)

# === Step 3: HDBSCAN - do NOT allow clusters < 10 ===
hdbscan_model = HDBSCAN(
    min_cluster_size=10,      # strict minimum
    min_samples=2,            # still allow flexibility
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# === Step 4: Vectorizer with unigrams and bigrams ===
vectorizer_model = CountVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

# === Step 5: Initialize BERTopic ===
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=10,
    nr_topics=None,
    calculate_probabilities=True,
    verbose=True
)

# === Step 6: Fit the model ===
topics, probs = topic_model.fit_transform(docs)
df["BERTopic_Topic_10plus"] = topics

# === Step 7: Save final dataset ===
df.to_csv("Berttopics.csv", index=False)

# === Step 8: View topic distribution ===
topic_info = topic_model.get_topic_info()
print(type(topic_info))
print(topic_info['Representation'])
for topic in topic_info['Representation']:
    for keyword in topic:
        print(keyword)
    print("------------------------------------------------------------------")
