import streamlit as st
import pandas as pd
import numpy as np
import os, logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@st.cache_data
def load_data(curr_path):
    logger.debug(curr_path)
    data_file_path = os.path.join(curr_path, 'data', 'anime-dataset-2023.csv')
    logger.debug(data_file_path)

    df_anime = pd.read_csv(data_file_path)
    return df_anime

@st.cache_data
def filter_data(df_anime):

    #filtering for duplicates
    duplicates = df_anime[df_anime.duplicated(['Name'])].sort_values(by='Name')
    logger.debug("Duplicates based on Name:")
    logger.debug(len(duplicates))
    duplicates = duplicates[['anime_id', 'Name']]
    logger.debug(duplicates)

    df_anime_new = df_anime.drop_duplicates(['Name'])
    logger.debug("Without duplicates, anime shape: {} \n".format(df_anime_new.shape))
    logger.debug("Old anime shape: {}".format(df_anime.shape))

    #filter out certain genre
    to_exclude = df_anime[df_anime['Genres'].str.contains('Hentai', case=False, na=False)]
    filtered_df = df_anime[~df_anime.index.isin(to_exclude.index)]


    # convert Name column to lowercase and remove spaces
    filtered_df['Processed_Name'] = filtered_df['Name'].str.lower().replace(' ', '')

    # filter out rows with titles in lowercase and without spaces
    duplicate_rows = filtered_df[filtered_df.duplicated(subset='Processed_Name', keep=False) | 
                                ~filtered_df.duplicated(subset='Processed_Name', keep=False) & 
                                ~filtered_df['Processed_Name'].str.contains(' ')]

    # filter out rows that are upper case and have no spacing, e.g. between Death Note and DEATHNOTE, keep Death Note
    filtered_df = filtered_df[~((filtered_df['Processed_Name'].isin(duplicate_rows['Processed_Name'])) 
                                & (filtered_df.duplicated(subset='Processed_Name', keep=False)))]

    # drop the intermediate 'Processed_Name' column
    filtered_df = filtered_df.drop(columns='Processed_Name')

    # drop rows with unknown genres
    unknown_rows = filtered_df[filtered_df['Genres'].str.lower() == 'unknown']
    filtered_df = filtered_df.drop(unknown_rows.index)
    logger.debug("final filtered_df shape: {} \n".format(filtered_df.shape))

    return filtered_df

@st.cache_resource
def find_similarity(filtered_df):
    # create the TF-IDF matrix for text comparison
    # max_features is the max # of unique words to consider
    # ignore terms that appear in < min_df documents or > max_df% documents
    # apply sublinear scaling which benefits high variation of text length
    # consider unigrams and bigrams (groups of terms)
    tfidf = TfidfVectorizer(stop_words='english',
                            max_features=10000,
                            max_df=0.9,
                            min_df=2)
    synopsis_vectors = tfidf.fit_transform(filtered_df['Synopsis'])

    # use one-hot encoder to include genre in the recommendation
    encoder = OneHotEncoder(sparse_output=True)

    genre_encoded_sparse = encoder.fit_transform(filtered_df[['Genres']].explode('Genres'))

    #include Studios in the recommendation
    exclude_studios = ['Animation', 'Studio', 'UNKNOWN']
    studios = [studio if studio not in exclude_studios else '' for studio in filtered_df['Studios']]

    studios_encoder = OneHotEncoder(sparse_output=True)

    studios_encoded_sparse = studios_encoder.fit_transform(filtered_df[['Studios']].explode('Studios'))

    #include Rating (PG, etc) in the recommendation
    rating_encoder = OneHotEncoder(sparse_output=True)
    rating_encoded_sparse = rating_encoder.fit_transform(filtered_df[['Rating']].explode('Rating'))


    # apply weights of importance to feature
    weight_synopsis = 3.0
    weight_genres = 2.0
    weight_studios = 1.0
    weight_rating = 1.5

    weighted_synopsis = weight_synopsis * synopsis_vectors
    weighted_genres = weight_genres * genre_encoded_sparse
    weighted_studios = weight_studios * studios_encoded_sparse
    weighted_rating = weight_rating * rating_encoded_sparse


    # combine the sparse matrices horizontally (hstack)
    combined_sparse_matrix = hstack([weighted_synopsis, weighted_genres, weighted_studios, weighted_rating])

    # compute cosine similarity between all anime synopses
    similarity = cosine_similarity(combined_sparse_matrix)

    similarity_df = pd.DataFrame(similarity, 
                                index=filtered_df['Display Name'], 
                                columns=filtered_df['Display Name'])
    logger.info(similarity)
    logger.info(filtered_df.head())
    anime_list = similarity_df.columns.values
    return anime_list, similarity_df


# get recommendations
def content_anime_recommender(input_anime, top_n=10):
    anime_database_list, similarity_database = find_similarity(st.session_state.filtered_df)
    anime_sim = similarity_database[similarity_database.index == input_anime].values[0]
    sorted_anime_ids = np.argsort(anime_sim)[::-1]
    recommended_anime = anime_database_list[sorted_anime_ids[1:top_n+1]]
    return recommended_anime

# get image url
@st.cache_data
def get_image_url(anime_name):
    return st.session_state.filtered_df.loc[st.session_state.filtered_df['Display Name'] == anime_name, 'Image URL'].values[0]

# get synopsis
def get_synopsis(anime_name):
    return st.session_state.filtered_df.loc[st.session_state.filtered_df['Display Name'] == anime_name, 'Synopsis'].values[0]

# get genres
def get_genres(anime_name):
    return st.session_state.filtered_df.loc[st.session_state.filtered_df['Display Name'] == anime_name, 'Genres'].values[0]

# condition for creating display name for UI display
def create_display_name(row):
    return row['English name'] if row['English name'].lower() != "unknown" else row['Name']



# streamlit app
st.set_page_config(layout="wide")

curr_path = os.path.dirname(os.path.abspath(__file__))
# df_anime = load_data(curr_path)
# filtered_df = filter_data(df_anime)
# # create new col Display Name for showing to end user
# filtered_df['Display Name'] = filtered_df.apply(create_display_name, axis=1)


styles_file_path = os.path.join(curr_path, 'assets/css', 'styles.css')
# logger.info(styles_file_path)

# with open (styles_file_path, 'r') as f:
#     custom_css = f.read()

# # apply styles, trust the custom style
# st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)


# st.title("Anime Recommender")

# # sidebar with user input

# st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"][aria-expanded="true"]{
#         min-width: 400px;
#         max-width: 400px;
#     }
#     """,
#     unsafe_allow_html=True,
# )   

# with st.sidebar:
   
#     input_anime = st.sidebar.selectbox("Select an anime:", filtered_df['Display Name'])
    
#     # num of recommendations to show
#     top_n = st.sidebar.slider("Number of recommendations", 1, 30, 10)

# # display recommendations
# recommended_anime = content_anime_recommender(input_anime, top_n=top_n)

# recommended_heading = '<h4>Recommended Anime for ' + input_anime + ':</h4>'
# st.markdown(recommended_heading, unsafe_allow_html=True)

# count = 1
# for anime in recommended_anime:

#     anime_title = '<h5>' + str(count) + '. ' + anime + '</h5>'
#     st.markdown(anime_title, unsafe_allow_html=True)

#     col1, col2 = st.columns([1, 3])

#     image_url = get_image_url(anime)
#     col1.image(image_url, use_column_width=False, width=200)

#     genres_title = '<h6>' + get_genres(anime) + '</h6>'
#     col1.markdown(genres_title, unsafe_allow_html=True)

#     synopsis_title = '<h7>' + get_synopsis(anime).replace('\n', '<br>') + '</h7>'
#     col2.markdown(synopsis_title, unsafe_allow_html=True)

#     st.markdown('<hr>', unsafe_allow_html=True)
#     count+=1

# check if data in session_state
if 'loaded_df' not in st.session_state:
    st.session_state.loaded_df = None

if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None

if 'selected_anime' not in st.session_state:
    st.session_state.selected_anime = None

if 'num_recommendations' not in st.session_state:
    st.session_state.num_recommendations = 10

# load data if not loaded
if st.session_state.loaded_df is None:
    curr_path = os.path.dirname(os.path.abspath(__file__))
    st.session_state.loaded_df = load_data(curr_path)

# filter data if not filtered
if st.session_state.filtered_df is None:
    st.session_state.filtered_df = filter_data(st.session_state.loaded_df)
    filtered_df = st.session_state.filtered_df

# create new col Display Name for showing to end user
st.session_state.filtered_df['Display Name'] = st.session_state.filtered_df.apply(create_display_name, axis=1)

with open (styles_file_path, 'r') as f:
    custom_css = f.read()

# Load custom styles from styles.css
styles_file_path = os.path.join(curr_path, 'assets/css', 'styles.css')
st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)

# Sidebar with user input
with st.sidebar:
    st.sidebar.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            min-width: 400px;
            max-width: 400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    input_anime = st.sidebar.selectbox("Select an anime:", st.session_state.filtered_df['Display Name'])
    st.session_state.selected_anime = input_anime

    # Num of recommendations to show
    top_n = st.sidebar.slider("Number of recommendations", 1, 30, st.session_state.num_recommendations)

# Display recommendations
recommended_anime = content_anime_recommender(input_anime, top_n=top_n)

recommended_heading = '<h4>Recommended Anime for ' + st.session_state.selected_anime + ':</h4>'
st.markdown(recommended_heading, unsafe_allow_html=True)

count = 1
for anime in recommended_anime:
    anime_title = '<h5>' + str(count) + '. ' + anime + '</h5>'
    st.markdown(anime_title, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    image_url = get_image_url(anime)
    col1.image(image_url, use_column_width=False, width=200)

    genres_title = '<h6>' + get_genres(anime) + '</h6>'
    col1.markdown(genres_title, unsafe_allow_html=True)

    synopsis_title = '<h7>' + get_synopsis(anime).replace('\n', '<br>') + '</h7>'
    col2.markdown(synopsis_title, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    count += 1