import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("movies.csv")

# Fill missing values
df['genres'] = df['genres'].fillna('')
df['keywords'] = df['keywords'].fillna('')
df['cast'] = df['cast'].fillna('')
df['director'] = df['director'].fillna('')
df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')

# Combine important features into a single column
df['combined_features'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['cast'] + ' ' + df['director']

# Create a CountVectorizer object
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix)

# Functions to map movie titles and indices
def get_index_from_title(title):
    return df[df['title'] == title].index[0]

def get_title_from_index(index):
    return df.iloc[index]['title']

# Function to recommend movies
def recommend_movies(movie_title, num_recommendations=10):
    movie_index = get_index_from_title(movie_title)
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_movie_indices = [x[0] for x in sorted_scores[1:num_recommendations+1]]
    return [get_title_from_index(idx) for idx in recommended_movie_indices]

# Streamlit app
st.title("Movie Recommendation System")

# Input for movie name
movie_name = st.text_input("Enter a movie name:")
if movie_name:
    try:
        recommendations = recommend_movies(movie_name)
        st.write(f"Movies similar to '{movie_name}':")
        for movie in recommendations:
            st.write(movie)
    except Exception as e:
        st.write("Movie not found in dataset. Try another title.")

# Dark mode toggle
dark_mode = st.checkbox("Enable Dark Mode")

# Set chart styles based on theme
sns.set_theme(
    style="darkgrid" if dark_mode else "whitegrid",
    rc={
        "axes.facecolor": "#212121" if dark_mode else "#FFFFFF",
        "axes.labelcolor": "#FFFFFF" if dark_mode else "#000000",
        "xtick.color": "#FFFFFF" if dark_mode else "#000000",
        "ytick.color": "#FFFFFF" if dark_mode else "#000000",
        "text.color": "#FFFFFF" if dark_mode else "#000000",
        "grid.color": "#383838" if dark_mode else "#DDDDDD",
    },
)

# Genre Analysis Visualization
st.write("### Top Movie Genres")
top_genres = df['genres'].str.split('|').explode().value_counts().head(10)

fig, ax = plt.subplots()
sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax, alpha=0.8, palette="dark" if dark_mode else "deep")
ax.set_title('Top Genres', color="#FFFFFF" if dark_mode else "#000000")
ax.set_xlabel('Count', color="#FFFFFF" if dark_mode else "#000000")
ax.set_ylabel('Genre', color="#FFFFFF" if dark_mode else "#000000")
st.pyplot(fig)
