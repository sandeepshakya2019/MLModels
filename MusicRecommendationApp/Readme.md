# Music Recommendation App 🎵

This project demonstrates a content-based music recommendation system using song lyrics. The system recommends similar songs based on the textual similarity of their lyrics using TF-IDF vectorization and cosine similarity.

---

## 📊 Workflow Overview

1. **Import Dependencies**
2. **Load and Preprocess Data**
3. **Exploratory Data Analysis**
4. **Text Cleaning & Feature Engineering**
5. **TF-IDF Vectorization**
6. **Cosine Similarity Calculation**
7. **Recommendation Function**
8. **Visualization**

---

## 1. Import Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
```

---

## 2. Load and Preprocess Data

```python
df = pd.read_csv("../Datasets/spotify_millsongdata.csv")
df = df[:20000]  # Limit to first 20,000 rows for performance
df.drop(columns=["link"], inplace=True)
```

---

## 3. Exploratory Data Analysis

```python
df.head()
df.shape
df.describe()
df.isnull().sum()
df['artist'].value_counts().head(10)
```

---

## 4. Text Cleaning & Feature Engineering

```python
# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['cleaned_text'] = df['text'].apply(preprocess_text)
```

---

## 5. TF-IDF Vectorization

```python
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])
```

---

## 6. Cosine Similarity Calculation

```python
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

---

## 7. Recommendation Function

```python
def recommend_songs(song_name, cosine_sim=cosine_sim, df=df, top_n=5):
    idx = df[df['song'].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        return "Song not found in the dataset!"
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    song_indices = [i[0] for i in sim_scores]
    return df[['artist', 'song']].iloc[song_indices]
```

**Example Usage:**
```python
print("Recommendations for the song 'For The First Time':")
recommendations = recommend_songs("For The First Time")
print(recommendations)
```

---

## 8. Visualization

**WordCloud for Song Lyrics:**
```python
all_lyrics = " ".join(df['text'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_lyrics)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words in Lyrics")
plt.show()
```
<!-- ![Workflow](../images/workflow.png) -->

---

## 📁 Project Structure

```
MusicRecommendationApp/
│
├── notebooks/
│   └── index.ipynb
├── images/
│   └── workflow.png
├── Datasets/
│   └── spotify_millsongdata.csv
└── README.md
```

---

## 📝 Insights

- **Content-Based Filtering:** Recommends songs based on lyric similarity.
- **Text Feature Engineering:** Uses TF-IDF vectorization for lyrics.
- **Cosine Similarity:** Measures similarity between songs for recommendation.
- **Visualization:** WordCloud shows most frequent words in the dataset.
- **Scalable:** Can be extended to include more features or hybrid approaches.

---