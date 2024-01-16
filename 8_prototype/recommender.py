import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


curr_path = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(os.path.dirname(curr_path), 'data', 'anime-dataset-2023.csv')
#print(data_file_path)

df_anime = pd.read_csv(data_file_path)
print(df_anime['Synopsis'].head())

# Create the TF-IDF matrix for text comparison
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_anime['Synopsis'])

# Compute cosine similarity between all anime synopses
similarity = cosine_similarity(tfidf_matrix)
similarity_df = pd.DataFrame(similarity, 
                             index=df_anime['Name'], 
                             columns=df_anime['Name'])

anime_list = similarity_df.columns.values


#streamlit app
st.title("Anime Recommender")

#sidebar with user input
input_anime = st.sidebar.selectbox("Select an anime:", df_anime['Name'].values)

# Number of recommendations to show
top_n = st.sidebar.slider("Number of recommendations", 1, 30, 10)

#get recommendations
def content_anime_recommender(input_anime, similarity_database=similarity_df, anime_database_list=anime_list, top_n=10):
    anime_sim = similarity_database[similarity_database.index == input_anime].values[0]
    sorted_anime_ids = np.argsort(anime_sim)[::-1]
    recommended_anime = anime_database_list[sorted_anime_ids[1:top_n+1]]
    return recommended_anime

#get image url
def get_image_url(anime_name):
    return df_anime.loc[df_anime['Name'] == anime_name, 'Image URL'].values[0]

#display recommendations
recommended_anime = content_anime_recommender(input_anime, top_n=top_n)

st.write(f"\n\nTop Recommended Anime for {input_anime} are:\n")

for anime in recommended_anime:
    image_url = get_image_url(anime)
    st.image(image_url, caption=anime, use_column_width=False, width=150)