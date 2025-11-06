import pandas as pd
import numpy as np
import ast

credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')

credits.head()

movies.head(1)

movies = movies.merge(credits, left_on='title', right_on='title')

movies.head(1)

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.head(1)

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)

movies['genres']

movies['keywords'] = movies['keywords'].apply(convert)

movies['keywords']

movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])

movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])

movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies['tags']

movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

movies['tags']

movies = movies[['movie_id', 'title', 'overview', 'tags']]

movies['tags'] = movies['tags'].apply(lambda x: x.lower())

movies.head()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

print(get_recommendations('Aliens'))

import pickle
with open('movie_data.pkl', 'wb') as file:
    pickle.dump((movies, cosine_sim), file)  explain in simple words