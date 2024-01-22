import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


curr_path = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(os.path.dirname(curr_path), 'data', 'anime-dataset-2023.csv')
#print(data_file_path)

df_anime = pd.read_csv(data_file_path)
print(df_anime['Synopsis'].head())


#basic filtering for duplicates
duplicates_all = df_anime[df_anime.duplicated()]
print("All Duplicates:")
print(len(duplicates_all))

duplicates = df_anime[df_anime.duplicated(['Name'])].sort_values(by='Name')
print("Duplicates based on Name:")
print(len(duplicates))
duplicates = duplicates[['anime_id', 'Name']]
print(duplicates)

df_anime_new = df_anime.drop_duplicates(['Name'])
print("Cleaned anime shape: {} \n".format(df_anime_new.shape))
print("Old anime shape: {}".format(df_anime.shape))



#filter out certain genre
to_exclude = df_anime[df_anime['Genres'].str.contains('Hentai', case=False, na=False)]
filtered_df = df_anime[~df_anime.index.isin(to_exclude.index)]


# Convert Name column to lowercase and remove spaces
filtered_df['Processed_Name'] = filtered_df['Name'].str.lower().replace(' ', '')

# Filter out rows with titles in lowercase and without spaces
duplicate_rows = filtered_df[filtered_df.duplicated(subset='Processed_Name', keep=False) | 
                             ~filtered_df.duplicated(subset='Processed_Name', keep=False) & 
                             ~filtered_df['Processed_Name'].str.contains(' ')]

# Filter out rows that are upper case and have no spacing, e.g. between Death Note and DEATHNOTE, keep Death Note
filtered_df = filtered_df[~((filtered_df['Processed_Name'].isin(duplicate_rows['Processed_Name'])) 
                            & (filtered_df.duplicated(subset='Processed_Name', keep=False)))]

# Drop the intermediate 'Processed_Name' column
filtered_df = filtered_df.drop(columns='Processed_Name')

#drop rows with unknown genres
unknown_rows = filtered_df[filtered_df['Genres'].str.lower() == 'unknown']
filtered_df = filtered_df.drop(unknown_rows.index)


# Create the TF-IDF matrix for text comparison
tfidf = TfidfVectorizer(stop_words='english')
synopsis_vectors = tfidf.fit_transform(filtered_df['Synopsis'])

#use one-hot encoder to include genre in the recommendation
encoder = OneHotEncoder(sparse_output=True)

genre_encoded_sparse = encoder.fit_transform(filtered_df[['Genres']].explode('Genres'))

# Step 4: Combine the sparse matrices horizontally (hstack)
combined_sparse_matrix = hstack([genre_encoded_sparse, synopsis_vectors])




# Compute cosine similarity between all anime synopses
similarity = cosine_similarity(combined_sparse_matrix)
similarity_df = pd.DataFrame(similarity, 
                             index=filtered_df['Name'], 
                             columns=filtered_df['Name'])

anime_list = similarity_df.columns.values


#streamlit app
st.title("Anime Recommender")

#sidebar with user input
input_anime = st.sidebar.selectbox("Select an anime:", filtered_df['Name'].values)

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
    return filtered_df.loc[filtered_df['Name'] == anime_name, 'Image URL'].values[0]

#display recommendations
recommended_anime = content_anime_recommender(input_anime, top_n=top_n)

st.write(f"\n\nTop Recommended Anime for {input_anime} are:\n")

for anime in recommended_anime:
    image_url = get_image_url(anime)
    st.image(image_url, caption=anime, use_column_width=False, width=150)