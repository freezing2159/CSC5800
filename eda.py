import pandas as pd
import numpy as np
import re
from langdetect import detect
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import folium
from folium import plugins
import reverse_geocode
import os
import warnings

warnings.simplefilter(action = 'ignore', category = FutureWarning)

# Load and clean raw hotel data
df = pd.read_csv('Hotel_Reviews.csv')
df.columns = [col.lower() for col in df.columns]

# Fix malformed hotel names
df['hotel_name'] = df['hotel_name'].apply(lambda x: x.replace('H tel', 'Hotel'))

# Aggregate lat/lng by hotel
geocode_df = df.groupby('hotel_name').agg({'lat': 'first', 'lng': 'first'}).reset_index()
geocode_df = geocode_df.dropna(subset = ['lat', 'lng']).reset_index(drop = True)

# Get reverse geocode location info (city, country)
geocode_df['location'] = ''

def search_location(df):
    for i in range(df.shape[0]):
        coor = ((df.at[i, 'lat'], df.at[i, 'lng']),)
        df.at[i, 'location'] = reverse_geocode.search(coor)

search_location(geocode_df)

geocode_df['country'] = geocode_df['location'].map(lambda x: x[0]['country'])
geocode_df['city'] = geocode_df['location'].map(lambda x: x[0]['city'])

# Save map showing hotel distribution per city
city_df = geocode_df.drop_duplicates(subset = 'city')
city_counts = geocode_df['city'].value_counts().to_dict()

map2 = folium.Map(location = [48.8566, 2.3522], zoom_start = 4)  # centered on France

for _, row in city_df.iterrows():
    city = row['city']
    count = city_counts.get(city, 0)
    folium.Marker(
        location = [row['lat'], row['lng']],
        tooltip = f"{city}: {count} hotels",
        icon = folium.Icon(color = 'blue', icon = 'info-sign')
    ).add_to(map2)

map2.save('image/Europe_overview.html')
geocode_df.to_pickle('data/geocode.pkl')


# Process hotel tags
tag_df = df[['hotel_name', 'tags', 'lat', 'lng']]
tag_df = tag_df.groupby('hotel_name').agg({
    'tags': ', '.join,
    'lat': 'first',
    'lng': 'first'
}).reset_index()

tag_df.dropna(subset = ['lat', 'lng'], inplace = True)
tag_df.to_pickle("data/hoteltag.pkl")

# Clean placeholder values
df['negative_review'] = df['negative_review'].apply(lambda x: x.replace("No Negative", "No"))
df['positive_review'] = df['positive_review'].apply(lambda x: x.replace("No Positive", "No"))

# Replace duplicate reviews (positive = negative) with 'nothing'
duplicates = (df['positive_review'] == df['negative_review'])
df.loc[duplicates, 'positive_review'] = 'nothing'

# Combine reviews and aggregate by hotel
review_df = df.groupby('hotel_name').agg({
    'positive_review': ', '.join,
    'negative_review': ', '.join,
    'lat': 'first',
    'lng': 'first',
    'hotel_address': 'first',
    'tags': ', '.join
}).reset_index()

review_df.dropna(subset = ['lat', 'lng'], inplace = True)

# Merge city info
geocode = pd.read_pickle('data/geocode.pkl')
final_df = pd.merge(review_df, geocode, on = 'hotel_name', how = 'outer')
final_df.to_pickle('data/review.pkl')
