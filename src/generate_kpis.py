import pandas as pd
import joblib
import numpy as np
import os

print("GENERATING FINAL DATASET FOR TABLEAU / ZICASSO")
print("-" * 60)

# CONFIGURATION
SILVER_DATA_PATH = '../data/silver/london_luxury_analytics_NLP.csv'
MODELS_DIR = '../models'
GOLD_OUTPUT_PATH = '../data/gold/data_core_dashboard.csv'

# LOAD CLEAN DATA AND AI BRAIN
print(" Loading historical data from Silver Layer...")
if not os.path.exists(SILVER_DATA_PATH):
    print(f"❌ ERROR: Could not find {SILVER_DATA_PATH}")
    exit()

df = pd.read_csv(SILVER_DATA_PATH)

print(f" Loading AI Model and Encoders from {MODELS_DIR}...")
try:
    model = joblib.load(f'{MODELS_DIR}/modelo_precio_london_v1.pkl')
    le_room = joblib.load(f'{MODELS_DIR}/encoder_room_type.pkl')
    le_neigh = joblib.load(f'{MODELS_DIR}/encoder_neighborhood.pkl')
except FileNotFoundError as e:
    print(f"❌ CRITICAL ERROR: Artifacts not found. {e}")
    exit()

# PREPARE DATA FOR AI CONSUMPTION
X_features = df[['accommodates', 'room_type', 'number_of_reviews_ltm', 
                 'review_scores_rating', 'review_scores_cleanliness', 
                 'review_scores_location', 'luxury_word_count', 
                 'is_superhost', 'neighbourhood_cleansed']].copy()

# Ensure types are string before encoding to avoid errors
X_features['room_type'] = X_features['room_type'].astype(str)
X_features['neighbourhood_cleansed'] = X_features['neighbourhood_cleansed'].astype(str)

# ENCODING LOGIC
known_neighs = set(le_neigh.classes_)

# Helper function to handle unseen labels safely
def safe_neigh_transform(val):
    return val if val in known_neighs else list(known_neighs)[0] 

X_features['neighbourhood_cleansed'] = X_features['neighbourhood_cleansed'].apply(safe_neigh_transform)

# Transform text to numbers
print("   -> Transforming categorical features...")
X_features['room_type'] = le_room.transform(X_features['room_type'])
X_features['neighbourhood_cleansed'] = le_neigh.transform(X_features['neighbourhood_cleansed'])

# 3. GENERATE MASSIVE PREDICTIONS
print(f"AI is valuing {len(df)} properties...")
df['predicted_price'] = model.predict(X_features)

# 4. BUSINESS LOGIC
# Calculate the gap: (AI Price - Real Price)
df['price_difference'] = df['predicted_price'] - df['price']

# Opportunity Classification Logic
# if the price is negative is Bad Deal

conditions = [
    (df['price_difference'] > 50),   
    (df['price_difference'] < -50) 
]

choices = [' Hidden Gem', ' Overpriced']

df['opportunity_type'] = np.select(conditions, choices, default=' Fair Price')

# SAVE
# Ensure Gold directory exists
os.makedirs(os.path.dirname(GOLD_OUTPUT_PATH), exist_ok=True)

df.to_csv(GOLD_OUTPUT_PATH, index=False)

print("-" * 60)
print(f"✅ DATASET READY: {GOLD_OUTPUT_PATH}")
print("Columns translated and logic applied.")
