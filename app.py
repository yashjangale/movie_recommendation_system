import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset and handle missing values
df = pd.read_csv("movies.csv")
df['genres'] = df['genres'].fillna('')
df['keywords'] = df['keywords'].fillna('')
df['cast'] = df['cast'].fillna('')
df['director'] = df['director'].fillna('')
df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')

# Combine important features into a single column
df['combined_features'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['cast'] + ' ' + df['director']

# Create a CountVectorizer object and compute the cosine similarity matrix
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(count_matrix)

# Function to get the index of a movie from its title
def get_index_from_title(title):
    return df[df['title'] == title].index[0]

# Function to get the title of a movie from its index
def get_title_from_index(index):
    return df.iloc[index]['title']

# Streamlit App Title
st.title("Movie Recommendation System")
st.markdown("""
    <style>
        .main {
            background-color: #f1f1f1;
        }
    </style>
""", unsafe_allow_html=True)

# Movie Recommendation System Function
def recommend_movies(movie_title, num_recommendations=10):
    movie_index = get_index_from_title(movie_title)
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    # Get top recommendations
    recommended_movie_indices = [x[0] for x in sorted_scores[1:num_recommendations+1]]
    return [get_title_from_index(idx) for idx in recommended_movie_indices]

# Input for movie name
movie_name = st.text_input("Enter a movie name:", key="movie_input")
if movie_name:
    try:
        recommendations = recommend_movies(movie_name)
        st.write(f"Movies similar to '{movie_name}':")
        for movie in recommendations:
            st.write(movie)
    except Exception as e:
        st.write("Movie not found in dataset. Try another title.")

# Genre Analysis Visualization with Improved Styling
st.write("### Top Movie Genres")

# Process genres data
top_genres = df['genres'].str.split('|').explode().value_counts().head(10)

# Create a more sophisticated bar plot with transparency and enhanced styling
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax, palette='viridis', alpha=0.5)

# Adding some extra customizations to the plot
ax.set_title('Top 10 Movie Genres', fontsize=16, fontweight='bold', color='darkblue')
ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('Genre', fontsize=12)
ax.tick_params(axis='both', labelsize=10)

# Customize background color
ax.set_facecolor('#f5f5f5')

# Render plot in Streamlit
st.pyplot(fig)

# Add another engaging visualization: A histogram of movie ratings (if available in your dataset)
if 'vote_average' in df.columns:
    st.write("### Movie Rating Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['vote_average'], bins=20, kde=True, color='teal', alpha=0.3, ax=ax)
    ax.set_title('Movie Rating Distribution', fontsize=16, fontweight='bold', color='darkblue')
    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    st.pyplot(fig)

# Add some additional recommendations or features
st.sidebar.write("### About the Movie Recommender System")
st.sidebar.write("This system recommends movies based on their genres, keywords, cast, and director. Simply enter a movie title, and get a list of similar movies!")
