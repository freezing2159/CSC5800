import pandas as pd
import numpy as np
import string
import folium
from folium import plugins
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from geopy.geocoders import Nominatim
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load data from pickle files
hoteltags = pd.read_pickle('data/hoteltag.pkl')
geocode = pd.read_pickle('data/geocode.pkl')

# Merge tag and geolocation data
hoteltags_geo = pd.merge(hoteltags, geocode, on = 'hotel_name', how = 'outer')
hoteltags_geo = hoteltags_geo[['hotel_name', 'tags', 'lat_x', 'lng_x', 'city']]

# Clean raw tags: remove punctuation and lowercase them
exclude = set(string.punctuation)
def clean(x):
    return set([''.join(ch for ch in i.lower() if ch not in exclude).strip() for i in x[2:-2].split(',')])

hoteltags_geo['new_tags'] = hoteltags_geo['tags'].map(clean)

# Flatten all tags to build a tag set
tag_sum_list = []
def get_tag_sum_elems(tag_sum_string):
    global tag_sum_list
    tag_sum_list.extend(tag_sum_string)
    return True

for i in hoteltags_geo['new_tags']:
    get_tag_sum_elems(i)
tag_sum_set = set(tag_sum_list)

# Remove unwanted tag
hoteltags_geo['new_tags'] = hoteltags_geo['new_tags'].apply(lambda x: x - {'submitted from a mobile device'})

# Get hotels by keyword (used for refactoring)
def get_special(s):
    spike_cols = [col for col in tag_sum_set if s in col]
    hotel_list = set()
    for i in range(len(hoteltags_geo.tags)):
        for j in range(len(spike_cols)):
            if spike_cols[j] in hoteltags_geo.new_tags[i]:
                hotel_list.add(hoteltags_geo[hoteltags_geo.index == i]['hotel_name'][i])
    return hoteltags_geo[hoteltags_geo.hotel_name.isin(list(hotel_list))][['hotel_name', 'lat_x', 'lng_x', 'city']]

# Map plot function
def get_map(df, imagepath):
    if df.empty:
        print("DataFrame is empty, cannot generate map.")
        return None
    zoom = 12 if df.shape[0] < 15 else 4
    map2 = folium.Map(location = [df.iloc[0].lat_x, df.iloc[0].lng_x], zoom_start = zoom)
    for i, row in enumerate(df.itertuples(), 1):
        folium.Marker(
            location = [row.lat_x, row.lng_x],
            tooltip = f"{i}. {row.hotel_name}",
            popup = row.hotel_name,
            icon = folium.Icon(color = 'blue', icon = 'info-sign')
        ).add_to(map2)
    map2.save(imagepath)
    return map2

# Group columns for tag replacement
grp_col = ['single room','river view','private pool','breakfast','spa bath','twin room','double room',
           'superior room','king room','executive room','city view','sea view','stayed ','eiffel twin',
           'eiffel tower view','suite','triple room','penthouse','standard room','wheelchair accessible',
           'family room ','deluxe room','apartment','terrace']

# Find all similar tags for replacement
newarray = []
for i in grp_col:
    newarray.extend([col for col in tag_sum_set if i in col])

# Apply new label for refactored tags
def new_fn(a, b):
    efv = list(get_special(a).index)
    for i in efv:
        hoteltags_geo.at[i, 'new_tags'].add(b)

# Apply refactor labels
new_fn("single room", "single_room")
new_fn("river view", "river_view")
new_fn("private pool", "private_pool")
new_fn("breakfast", "break_fast")
new_fn("spa bath", "spa_bath")
new_fn("twin room", "twin_room")
new_fn("double room", "double_room")
new_fn("superior room", "superior_room")
new_fn("king room", "king_room")
new_fn("executive room", "executive_room")
new_fn("city view", "city_view")
new_fn("sea view", "sea_view")
new_fn("eiffel tower view", "eiffel_tower_view")
new_fn("suite", "_suite_")
new_fn("triple room", "triple_room")
new_fn("penthouse", "_penthouse_")
new_fn("standard room", "standard_room")
new_fn("wheelchair accessible", "wheelchair_accessible")
new_fn("family room ", "family_room")
new_fn("deluxe room", "deluxe_room")
new_fn("guest room", "guest_room")
new_fn("apartment", "_apartment_")
new_fn("terrace", "_terrace")

# Remove old verbose tags from each row
newarray_set = set(newarray)
hoteltags_geo['new_tags'] = hoteltags_geo['new_tags'].apply(lambda tags: tags - newarray_set)

# Rebuild tag set
tag_sum_list = []
for i in hoteltags_geo['new_tags']:
    get_tag_sum_elems(i)
tag_sum_set = set(tag_sum_list)

# Convert tags into one-hot encoded format
df_exploded = hoteltags_geo[['hotel_name', 'new_tags']].explode('new_tags')
df_onehot = pd.pivot_table(
    df_exploded, 
    index = 'hotel_name', 
    columns = 'new_tags',
    aggfunc = lambda x: 1, 
    fill_value = 0
).reset_index()
hoteltags_geo = hoteltags_geo.drop(columns = tag_sum_set, errors = 'ignore')
hoteltags_geo = hoteltags_geo.merge(df_onehot, on = 'hotel_name', how = 'left')

# Room type bar chart
index_room = ['single_room','twin_room','double_room','superior_room','king_room','executive_room',
              '_suite_','triple_room','_penthouse_','standard_room','family_room','deluxe_room',
              'guest_room','_apartment_','_terrace_']
room_count = [get_special(i).shape[0] for i in index_room]
df1 = pd.DataFrame({'Number of Hotel': room_count}, index = index_room)
fig1 = px.bar(df1, x = 'Number of Hotel', y = df1.index, orientation = 'h',
              title = 'Number of hotels by Room Types', labels = {'index': 'Room Types'})
fig1.write_html("image/room_types_plot.html")

# View feature bar chart
index_view = ['river_view','private_pool','break_fast','spa_bath','city_view',
              'sea_view','eiffel_tower_view','wheelchair_accessible']
view_count = [get_special(i).shape[0] for i in index_view]
df2 = pd.DataFrame({'Number of Hotel': view_count}, index = index_view)
fig2 = px.bar(df2, x = 'Number of Hotel', y = df2.index, orientation = 'h',
              title = 'Number of hotels with special characteristics', labels = {'index': 'Characteristics'})
fig2.write_html("image/characteristics_plot.html")

# Save map of Eiffel Tower view hotels
get_map(get_special('eiffel_tower_view'), 'image/eiffel.html')

# Compute cosine similarity
similarityDF = cosine_similarity(hoteltags_geo.iloc[:, 6:], hoteltags_geo.iloc[:, 6:])
np.save("data/tagcosine.npy", similarityDF)
hoteltags_geo.to_pickle("data/clean_hoteltag.pkl")

# Recommendation function by tag similarity
def new_recommendations_tags(name, city, cosine_similarities):
    recommended_hotels = []
    city_index = list(hoteltags_geo[hoteltags_geo.city == city].index)
    idx = hoteltags_geo[hoteltags_geo.hotel_name == name].index[0]
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending = False)
    top_10_indexes = list(score_series.index)

    for i in top_10_indexes:
        if i in city_index:
            hotel_name = hoteltags_geo.loc[i, 'hotel_name']
            if hotel_name != name and hotel_name not in recommended_hotels:
                recommended_hotels.append(hotel_name)
        if len(recommended_hotels) == 10:
            break

    hotel_location_dict = hoteltags_geo.set_index('hotel_name')[['lat_x', 'lng_x']].T.to_dict('list')
    return {i + 1: {hotel: hotel_location_dict[hotel]} for i, hotel in enumerate(recommended_hotels)}

# Plot recommendation map
def get_hotel_fn(mydict, city):
    geolocator = Nominatim(user_agent="hotel-recommender")
    loc2 = geolocator.geocode(city)
    if loc2 is None:
        print(f"City not found: {city}")
        return None
    main_map = folium.Map(location = [loc2.latitude, loc2.longitude], zoom_start = 13)
    for i in range(1, len(mydict) + 1):
        folium.Marker(
            location = list(mydict[i].values())[0],
            tooltip = list(mydict[i].keys())[0],
            popup = list(mydict[i].keys())[0],
            icon = plugins.BeautifyIcon(number = i, icon = 'bus', border_color = 'blue',
                                      border_width = 0.5, text_color = 'red',
                                      inner_icon_style = 'margin-top:0px;')
        ).add_to(main_map)
    return main_map

# Run example recommendation
result_map = get_hotel_fn(
    new_recommendations_tags('Hilton Diagonal Mar Barcelona', 'Vienna', similarityDF),
    'Vienna'
)

if result_map:
    result_map.save("image/vienna_recommendation_map.html")
