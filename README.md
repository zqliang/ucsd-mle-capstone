# Anime Recommender

A Streamlit app that generates a list of anime recommendations given an anime title.

## Models

Content-based and collaborative filtering models were experimented with.

In the content-based model finds TF-IDF and cosine similarities and uses the 
Synopsis, Genres, Rating, and Studios features for text comparison.

The collaborative approach uses a deep-learning recommender which predicts ratings based on 
ratings of other users.

The content-based recommender is used in the deployed app.


### Running locally
git clone https://github.com/zqliang/ucsd-mle-capstone
cd ucsd-mle-capstone
streamlit run app.py