import pandas as pd
import joblib
import numpy as np
import os

print("STARTING INTELLIGENT INFERENCE ENGINE")
print("-" * 60)

# 1. LOAD ARTIFACTS from models
models_dir = '../models'

print(f"Loading model, encoders, and luxury dictionary from '{models_dir}'...")

try:
    model = joblib.load(f'{models_dir}/modelo_precio_london_v1.pkl')
    le_room = joblib.load(f'{models_dir}/encoder_room_type.pkl')
    le_neigh = joblib.load(f'{models_dir}/encoder_neighborhood.pkl')
    keywords = joblib.load(f'{models_dir}/luxury_keywords.pkl') 
except FileNotFoundError as e:
    print(f" CRITICAL ERROR: Could not find artifacts. {e}")
    exit()

# ENGINEERING FUNCTION
def calculate_luxury_score(description_text):
    if not isinstance(description_text, str):
        return 0
    text = description_text.lower()
    # Count how many words from the list appear in the text
    count = sum(1 for word in keywords if word in text)
    return count

# 2. SIMULATING INPUT DATA FROM WEB PAGE FOR EXAMPLE
web_input_data = {
    'accommodates': 4,
    'room_type': 'Entire home/apt',
    'number_of_reviews_ltm': 12,
    'review_scores_rating': 4.95,
    'review_scores_cleanliness': 5.0,
    'review_scores_location': 4.8,
    'is_superhost': 1,
    'neighbourhood_cleansed': 'Kensington',
    'description_text': """
        Spectacular penthouse with stunning views of the city. 
        Enjoy a private terrace and exclusive concierge service. 
        Fully renovated with elegant design.
    """
}

print("\n RECEIVED TEXT:")
print(f"'{web_input_data['description_text'].strip()}'")

# 3. INTERNAL PROCESSING 
calculated_score = calculate_luxury_score(web_input_data['description_text'])
print(f"   ->  AI detected {calculated_score} luxury words.")

# Prepare final DataFrame for the model
input_df = pd.DataFrame([{
    'accommodates': web_input_data['accommodates'],
    'room_type': web_input_data['room_type'],
    'number_of_reviews_ltm': web_input_data['number_of_reviews_ltm'],
    'review_scores_rating': web_input_data['review_scores_rating'],
    'review_scores_cleanliness': web_input_data['review_scores_cleanliness'],
    'review_scores_location': web_input_data['review_scores_location'],
    'luxury_word_count': calculated_score,
    'is_superhost': web_input_data['is_superhost'],
    'neighbourhood_cleansed': web_input_data['neighbourhood_cleansed']
}])

# 4. CATEGORY TRANSLATION
try:
    input_df['room_type'] = le_room.transform(input_df['room_type'])
except ValueError:
    input_df['room_type'] = 0

# Neighborhood
try:
    input_df['neighbourhood_cleansed'] = le_neigh.transform(input_df['neighbourhood_cleansed'])
except ValueError:
    # Handle unknown neighborhood
    print("Warning: Unknown neighborhood. Using default value.")
    input_df['neighbourhood_cleansed'] = 0

# 5. PREDICTION
print("\n Calculating market price...")
price = model.predict(input_df)[0]

print("-" * 60)
print(f" RECOMMENDED PRICE: ${price:.2f} USD")
print("-" * 60)