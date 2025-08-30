# IMDB-Movie-Recommendation-System-Using-Storylines
This project focuses on extracting movie data from IMDb for 2024, specifically focusing on the movie name and storyline. Using Selenium, the program will scrape IMDb to collect movie names and their associated storylines. The storylines will then be pre-processed and analyzed using Natural Language Processing (NLP) techniques, such as TF-IDF (Term Frequency-Inverse Document Frequency) or Count Vectorizer. Using these methods, the project will calculate Cosine Similarity or other Machine Learning algorithms to recommend similar movies based on a given storyline. The project will provide an interactive user interface built with Streamlit where users can input a movie storyline and receive the top 5 recommended movies.


---

# 🎬 IMDb 2024 Movie Recommendation System

This project builds a **storyline-based movie recommendation system** using IMDb 2024 movies. It scrapes data with Selenium, processes storylines with NLP, applies **TF-IDF + Cosine Similarity**, and delivers recommendations through an interactive **Streamlit app**.

---

## 📌 Features

* ✅ **IMDb 2024 Data Scraping** using Selenium (Movie Name + Storyline)
* ✅ **Data Preprocessing** with NLP (stopwords removal, tokenization, text cleaning)
* ✅ **Vectorization** with TF-IDF
* ✅ **Recommendation Engine** using Cosine Similarity
* ✅ **Streamlit App** for interactive movie recommendations
* ✅ **User Input**: Enter a custom storyline to get Top 5 movie suggestions
* ✅ **Interactive UI**: Expandable cards, similarity score bars, sidebar controls

---

## 🏗️ Tech Stack

* **Languages:** Python
* **Libraries/Tools:** Selenium, Pandas, scikit-learn, NLTK, SpaCy, Streamlit
* **NLP Techniques:** TF-IDF, Cosine Similarity
* **Visualization:** Streamlit Components

---

## 📂 Project Structure

```
📁 imdb-2024-recommender
│── 📄 scraping.ipynb          # Jupyter Notebook for IMDb scraping  
│── 📄 imdb_movies_2024.csv    # Scraped dataset (Movie Name + Storyline)  
│── 📄 nlp_recommendation.ipynb # NLP preprocessing & recommendation  
│── 📄 streamlit_app.py        # Streamlit application  
│── 📄 README.md               # Project documentation  
```

---

## 🚀 How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/imdb-2024-recommender.git
cd imdb-2024-recommender
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 🎯 Usage

* Enter a **movie storyline** in the input box.
* Get **Top 5 most similar movies** from IMDb 2024 dataset.
* Explore recommendations with expandable cards and similarity scores.

---

## 📊 Example

Input:
*"A young hero fights to save the world from destruction."*

Output (Top 5 Recommendations):

1. 🎥 Movie A – Storyline …
2. 🎥 Movie B – Storyline …
3. 🎥 Movie C – Storyline …

---

## 📌 Business Use Cases

* 🎥 **Movie Recommendation:** Suggests movies based on plot similarity.
* 🍿 **Entertainment Suggestions:** Helps users discover new movies aligned with their preferences.

---

## 🙌 Acknowledgements

* IMDb for movie data
* Python & Streamlit community

