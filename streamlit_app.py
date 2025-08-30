import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\prati\OneDrive\Desktop\guvi\project 5\imdb_movies_2024.csv")  
    return df

df = load_data()

# Preprocessing Function

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

df["Cleaned_Storyline"] = df["Storyline"].apply(clean_text)

# Vectorization

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Storyline"])

# Recommendation Function

def recommend_movies(storyline, top_n=5):
    query_vec = vectorizer.transform([clean_text(storyline)])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    recommendations = df.iloc[top_indices][["Movie Name", "Storyline"]].copy()
    recommendations["Similarity Score"] = sim_scores[top_indices]
    return recommendations


# Streamlit Interface

st.set_page_config(page_title="IMDb 2024 Movie Recommendation System", layout="wide")
st.title("🎬 IMDb 2024 Movie Recommendation System")

st.markdown("Enter a **movie storyline/plot** below, and get the **Top 5 recommended movies** from IMDb 2024 dataset based on storyline similarity.")

# Sidebar

st.sidebar.header("⚙️ Settings")
top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)
show_scores = st.sidebar.checkbox("Show Similarity Scores", value=True)

# User input

user_input = st.text_area("✍️ Enter a storyline or movie plot:", height=150)

if st.button("🔍 Recommend Movies"):
    if user_input.strip():
        results = recommend_movies(user_input, top_n=top_n)

        st.subheader(f"🎯 Top {top_n} Recommended Movies")
        
        for idx, row in results.iterrows():
            with st.expander(f"🎥 {row['Movie Name']}", expanded=True):
                st.write(row["Storyline"])
                if show_scores:
                    st.progress(float(row["Similarity Score"]))  # visual score bar
                    st.caption(f"Similarity Score: {row['Similarity Score']:.2f}")
            st.markdown("---")
        
    else:
        st.warning("🎯 Please enter a storyline to get recommendations.")

# Footer
st.markdown("---")
st.markdown("✅ Built with Streamlit | 🔗 IMDb 2024 Data | 🎯 NLP + TF-IDF + Cosine Similarity")
