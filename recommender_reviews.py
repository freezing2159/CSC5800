# Import necessary libraries
import pandas as pd
import numpy as np
import re
import os
import string
import warnings
import folium
from folium import plugins
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel
from langdetect import detect
from geopy.geocoders import Nominatim

# Ignore warning messages for clean output
warnings.simplefilter(action = 'ignore', category = FutureWarning)

# Define stopwords
custom_stopwords = list(ENGLISH_STOP_WORDS.union({'no', 'nothing'}))

# Load preprocessed hotel review data
new_df = pd.read_pickle('data/review.pkl')

# Select relevant columns for analysis
hotel_review = new_df.loc[:, ['hotel_name', 'positive_review', 'negative_review', 'city', 'lat_x', 'lng_x']].copy()

# Combine positive and negative reviews into one text field
hotel_review.loc[:, 'review_text'] = hotel_review['positive_review'].astype(str) + hotel_review['negative_review'].astype(str)

# Detect language of reviews
hotel_review.loc[:, 'lang'] = hotel_review['review_text'].apply(lambda x: detect(x))

# Function: Get top N frequent n-grams without removing stop words
def get_top_n_words(corpus, n = None, y = (1,1)):
    vec = CountVectorizer(ngram_range = y).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key = lambda x: x[1], reverse = True)[:n]

# Function: Get top N frequent n-grams with stop words removed
def get_top_n_words_with_stop_words(corpus, n = None, y = (1,1)):
    vec = CountVectorizer(ngram_range = y, stop_words = custom_stopwords)
    bag_of_words = vec.fit_transform(corpus.fillna('').astype(str))
    sum_words = bag_of_words.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key = lambda x: x[1], reverse = True)[:n]

# Analyze word counts in each review
hotel_review['word_count'] = hotel_review['review_text'].apply(lambda x: len(str(x).split()))
desc_lengths = hotel_review['word_count']

# Plot word count distribution
fig = px.histogram(hotel_review, x = 'word_count', nbins=100,
                   title = 'Word Count Distribution in Reviews',
                   labels = {'word_count': 'Word Count', 'count': 'Frequency'})
fig.write_html("image/fig.html")

# 1-gram before stop words
fig1 = px.bar(pd.DataFrame(get_top_n_words(hotel_review['review_text'], 20, (1,1)), columns = ['Word', 'Count']),
              x = 'Count', y = 'Word', orientation = 'h', title = 'Top 20 1-grams (Before Stop Words)')
fig1.write_html("image/fig1.html")

# 1-gram after stop words
fig2 = px.bar(pd.DataFrame(get_top_n_words_with_stop_words(hotel_review['review_text'], 20, (1,1)), columns = ['Word', 'Count']),
              x = 'Count', y = 'Word', orientation = 'h', title = 'Top 20 1-grams (After Stop Words)')
fig2.write_html("image/fig2.html")

# 2-gram before and after stop words
fig3 = px.bar(pd.DataFrame(get_top_n_words(hotel_review['review_text'], 20, (2,2)), columns = ['Bigram', 'Count']),
              x = 'Bigram', y = 'Count', title = 'Top 20 Bigrams (Before Stop Words)')
fig3.write_html("image/fig3.html")

fig4 = px.bar(pd.DataFrame(get_top_n_words_with_stop_words(hotel_review['review_text'], 20, (2,2)), columns = ['Bigram', 'Count']),
              x = 'Bigram', y = 'Count', title = 'Top 20 Bigrams (After Stop Words)')
fig4.write_html("image/fig4.html")

# 3-gram before and after stop words
fig5 = px.bar(pd.DataFrame(get_top_n_words(hotel_review['review_text'], 20, (3,3)), columns = ['Trigram', 'Count']),
              x = 'Trigram', y = 'Count', title = 'Top 20 Trigrams (Before Stop Words)')
fig5.write_html("image/fig5.html")

fig6 = px.bar(pd.DataFrame(get_top_n_words_with_stop_words(hotel_review['review_text'], 20, (3,3)), columns = ['Trigram', 'Count']),
              x = 'Trigram', y = 'Count', title = 'Top 20 Trigrams (After Stop Words)')
fig6.write_html("image/fig6.html")

# Text cleaning for modeling
replace_space = re.compile(r'[/(){}\[\]\|@,;]')
symbol = re.compile('[^0-9a-z #+_]')
stopwordset = ENGLISH_STOP_WORDS

def clean_text(text):
    text = text.lower()
    text = replace_space.sub(' ', text)
    text = symbol.sub('', text)
    return ' '.join(word for word in text.split() if word not in stopwordset)

# Apply text cleaning
hotel_review['review_text_clean'] = hotel_review['review_text'].apply(clean_text)

# Create TF-IDF matrix using bigrams and trigrams
tfidf = TfidfVectorizer(analyzer = 'word', ngram_range = (2, 3), min_df = 1, stop_words = custom_stopwords)
tfidf_matrix = tfidf.fit_transform(hotel_review['review_text_clean'])

# Compute cosine similarity between all hotels
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function: get latitude and longitude of a city
def get_lat_lng(city_name):
    geolocator = Nominatim(user_agent = "hotel_recommender_app")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    else:
        raise ValueError(f"City not found: {city_name}")

# Function: recommend similar hotels within the same city
def new_recommendations(name, city, cosine_similarities):
    recommended_hotels = []
    city_index = list(hotel_review[hotel_review.city == city].index)
    idx = hotel_review[hotel_review.hotel_name == name].index[0]
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending = False)
    top_10_indexes = list(score_series.index)

    for i in top_10_indexes:
        if i in city_index and hotel_review.loc[i, 'hotel_name'] != name:
            recommended_hotels.append(hotel_review.loc[i, 'hotel_name'])
        if len(recommended_hotels) == 10:
            break

    location_dict = hotel_review.set_index('hotel_name')[['lat_x', 'lng_x']].T.to_dict('list')
    return {hotel: location_dict[hotel] for hotel in recommended_hotels if hotel in location_dict}

# Function: generate a folium map to show recommended hotel locations
def get_hotel_fn_pin(flat_dict, city):
    lat, lng = get_lat_lng(city)
    map_ = folium.Map(location = [lat, lng], zoom_start = 13)
    folium.TileLayer('OpenStreetMap').add_to(map_)

    for i, (hotel, coords) in enumerate(flat_dict.items(), 1):
        folium.Marker(
            location = coords,
            tooltip = hotel,
            popup = hotel,
            icon = plugins.BeautifyIcon(
                number = i,
                icon = 'hotel',
                border_color = 'blue',
                border_width = 0.5,
                text_color = 'red',
                inner_icon_style = 'margin-top:0px;')
        ).add_to(map_)

    return map_

# Save recommendation map as HTML
get_hotel_fn_pin(new_recommendations('The Belgrave Hotel', 'Paris', cosine_similarities), 'Paris').save('image/reviews.html')
get_hotel_fn_pin(new_recommendations('The Belgrave Hotel', 'Vienna', cosine_similarities), 'Vienna').save('image/reviews2.html')