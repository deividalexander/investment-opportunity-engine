import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import os

print(" STARTING MODEL")
print("-" * 60)

# 1. LOAD DATA 
file_path = '../data/silver/london_luxury_analytics_NLP.csv'
df = pd.read_csv(file_path)

# ---------------------------------------------------------
# OUTLIER CLEANING
# ---------------------------------------------------------
print(f" Original rows: {len(df)}")
# Remove absurd prices (<$10) and the Top 5% most expensive
# This stabilizes the model and prevents negative R2 scores.
df = df[df['price'] > 10]
price_limit = df['price'].quantile(0.95)
df = df[df['price'] < price_limit]
print(f"Rows after cleaning extreme prices (Top 5% cut): {len(df)}")
print("-" * 60)

# 2. PREPARATION
features = [
    'accommodates', 'room_type', 'number_of_reviews_ltm', 
    'review_scores_rating', 'review_scores_cleanliness', 
    'review_scores_location', 'luxury_word_count', 
    'is_superhost', 'neighbourhood_cleansed'
]
target = 'price'

# Encoding
le_room = LabelEncoder()
df['room_type'] = le_room.fit_transform(df['room_type'])
le_neigh = LabelEncoder()
df['neighbourhood_cleansed'] = le_neigh.fit_transform(df['neighbourhood_cleansed'])

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(name, model):
    print(f"\n🏃 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"   R2 Score (Accuracy): {r2:.2%}")
    print(f"   Mean Error (MAE): ${mae:.2f}")
    return r2, mae

# 
# LINEAR REGRESSION (Baseline)
r2_lr, mae_lr = evaluate_model("Linear Regression", LinearRegression())

# RANDOM FOREST (The Balanced Choice)
r2_rf, mae_rf = evaluate_model("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))

# XGBOOST (The Pro Challenger)
r2_xgb, mae_xgb = evaluate_model("XGBoost", xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42))

# =========================================================
# Results
# =========================================================
print("\n" + "="*30)
print("LEADERBOARD")
print("="*30)
models = [
    ("Linear Regression", r2_lr),
    ("Random Forest    ", r2_rf),
    ("XGBoost          ", r2_xgb)
]
# Sort by R2
models.sort(key=lambda x: x[1], reverse=True)

for name, score in models:
    bar = "█" * int(score * 20) if score > 0 else ""
    print(f"{name} | {score:.1%} {bar}")

print("\n💡 CONCLUSION:")
winner = models[0][0].strip()
if winner == "XGBoost":
    print("   -> XGBoost won. Ideal for production due to speed and precision.")
elif winner == "Random Forest":
    print("   -> Random Forest won. Often better when data has noise/variance.")
else:
    print("   -> Unexpected result. Check data quality.")

print("-" * 60)
print("💾 SAVING THE MODEL...")

# Ensure 'models' directory exists
models_dir = '../models'
os.makedirs(models_dir, exist_ok=True)

# Re-train best model with ALL training data before saving
best_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)

# Save .pkl Artifacts
joblib.dump(best_model, f'{models_dir}/modelo_precio_london_v1.pkl')
print(f"✅ Model saved as: '{models_dir}/modelo_precio_london_v1.pkl'")
print("   -> This artifact is ready for cloud deployment.")

# SAVE ENCODERS
joblib.dump(le_room, f'{models_dir}/encoder_room_type.pkl')
joblib.dump(le_neigh, f'{models_dir}/encoder_neighborhood.pkl')
print("✅ Encoders saved. Ready for Production!")

#Define keyword list
luxury_keywords = [
    'luxury', 'penthouse', 'spectacular', 'views', 'concierge', 
    'elegant', 'stunning', 'spacious', 'private', 'renovated', 
    'designer', 'terrace', 'exclusive', 'premium'
]

# SAVE LIST AS ARTIFACT
joblib.dump(luxury_keywords, f'{models_dir}/luxury_keywords.pkl')
print(f"✅ Luxury keywords saved: '{models_dir}/luxury_keywords.pkl'")

print("All set! Your inference script can now read text automatically.")