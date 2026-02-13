import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# STEP 1: CONFIGURATION
# ---------------------------------------------------------
print("🚀 STARTING ETL: ")
print("-" * 70)

# Sources
files = {
    '../data/bronze/listings_marz_2025.csv.gz': '2025-03-01',
    '../data/bronze/listings_junio_2025.csv.gz': '2025-06-01',
    '../data/bronze/listings_setp_2025.csv.gz': '2025-09-01'
}

# Columns to keep 
cols_to_keep = [
    'id', 'name', 'neighbourhood_cleansed', 'room_type', 
    'price', 'number_of_reviews', 'review_scores_rating', 
    'latitude', 'longitude', 'accommodates',
    'host_is_superhost',
    'review_scores_cleanliness',
    'review_scores_location',
    'availability_365',
    'reviews_per_month',      
    'number_of_reviews_ltm',  
    'description',
    'neighborhood_overview'
]

# LUXURY KEYWORDS LIST 
luxury_keywords = [
    'luxury', 'penthouse', 'spectacular', 'views', 'concierge', 
    'elegant', 'stunning', 'spacious', 'private', 'renovated', 
    'designer', 'terrace', 'exclusive', 'premium'
]

# ADVANCED DEEP FUNCTION 
def advanced_audit(df, filename):
    print(f"\n🔬 DEEP DIAGNOSIS: {filename.upper()}")
    print(f"   -> Total rows: {len(df)}")
    print("=" * 125)
    header = f"{'COLUMN NAME':<35} | {'DATA TYPE':<10} | {'NON-NULL':<9} | {'NULLS':<8} | {'VARIABLE TYPE (INFERRED)'}"
    print(header)
    print("-" * 125)

    for col in df.columns:
        dtype = str(df[col].dtype)
        non_nulls = df[col].count()
        nulls = df[col].isnull().sum()
        unique_count = df[col].nunique()
        var_type = "Unknown"

        if pd.api.types.is_numeric_dtype(df[col]):
            if unique_count == 2: var_type = "Binary (0/1)"
            elif unique_count < 20: var_type = f"Categorical Num ({unique_count})"
            else: var_type = "Continuous Numeric"
        else:
            if 'date' in col: var_type = "Date"
            elif unique_count <= 2 and {'t', 'f'}.issubset(df[col].dropna().unique()): var_type = "Binary (t/f)"
            elif unique_count < 50: var_type = f"Categorical Txt ({unique_count})"
            else: var_type = "Free Text" if unique_count != len(df) else "Unique ID"

        print(f"{col:<35} | {dtype:<10} | {non_nulls:<9} | {nulls:<6} | {var_type}")
    print("=" * 125)
    print("\n")

# ---------------------------------------------------------
# STEP 2: INGEST AND PROCESSING
# ---------------------------------------------------------
dataframes_list = []

for filename, date_label in files.items():
    print(f"📂 Processing: {filename} ...")
    try:
        if not os.path.exists(filename):
             print(f" ERROR: File not found: {filename}")
             continue

        if 'setp' in filename:
            print(" September Detected: Running Full Radiography...")
            df_full = pd.read_csv(filename)
            advanced_audit(df_full, filename) 

            # Filter columns after inspection
            cols_existing = [c for c in cols_to_keep if c in df_full.columns]
            temp_df = df_full[cols_existing].copy()
            
        else:

            temp_df = pd.read_csv(filename, usecols=cols_to_keep)

        # Common Transformations
        temp_df['data_date'] = pd.to_datetime(date_label)
        
        # Price Cleaning
        temp_df = temp_df.dropna(subset=['price'])
        temp_df['price'] = temp_df['price'].astype(str).str.replace(r'[$,]', '', regex=True)
        
        dataframes_list.append(temp_df)
        print(f"   -> ✅ Added ({len(temp_df)} rows)")

    except Exception as e:
        print(f"   ❌ ERROR: Could not process {filename}. Reason: {e}")

# ---------------------------------------------------------
# STEP 3: MERGE AND NLP
# ---------------------------------------------------------
if dataframes_list:
    print("-" * 70)
    print("Merging 3 periods into one Master Dataset...")
    full_df = pd.concat(dataframes_list, ignore_index=True)
    
    print("Running Text Analysis (Luxury Scoring)...")
    
    # 1. Text cleaning
    full_df['description'] = full_df['description'].fillna('').astype(str).str.lower()
    full_df['neighborhood_overview'] = full_df['neighborhood_overview'].fillna('no overview available').astype(str).str.lower()
    
    # 2. Optimized count
    def count_luxury_terms(text):
        return sum(1 for word in luxury_keywords if word in text)

    # 3. Apply to dataframe
    full_df['luxury_word_count'] = full_df['description'].apply(count_luxury_terms)
    print("   -> 'luxury_word_count' calculated successfully.")


    # ---------------------------------------------------------
    # STEP 4: FINAL CLEANING AND METRICS
    # ---------------------------------------------------------
    print("Applying final data cleaning...")
    
    # Convert Price to Numeric
    full_df['price'] = pd.to_numeric(full_df['price'], errors='coerce')
    
    # Superhost (Null fix)
    full_df['host_is_superhost'] = full_df['host_is_superhost'].fillna('f')
    full_df['is_superhost'] = np.where(full_df['host_is_superhost'] == 't', 1, 0)
    
    # Fill Scores and ML Metrics
    cols_zero = ['reviews_per_month', 'number_of_reviews_ltm']
    full_df[cols_zero] = full_df[cols_zero].fillna(0)
    
    cols_avg = ['review_scores_rating', 'review_scores_cleanliness', 'review_scores_location']
    for col in cols_avg:
        full_df[col] = full_df[col].fillna(full_df[col].mean())

    # Engagement Score
    full_df['engagement_score'] = (full_df['number_of_reviews_ltm'] * full_df['review_scores_rating']) / 100

    # ---------------------------------------------------------
    # STEP 5: FINAL AUDIT
    # ---------------------------------------------------------
    print("\n DATA QUALITY ANALYSIS (FINAL MASTER DATASET):")
    print(f"TOTAL COMBINED ROWS: {len(full_df)}")
    print("-" * 75)
    print(f"{'COLUMN NAME':<35} | {'NON-NULL':<10} | {'NULLS':<10} | {'TYPE'}")
    print("-" * 75)

    for col in full_df.columns:
        non_nulls = full_df[col].count()
        nulls = full_df[col].isnull().sum()
        dtype = str(full_df[col].dtype)
        alert = " <-- ⚠️" if nulls > 0 else ""
        print(f"{col:<35} | {non_nulls:<10} | {nulls:<10} | {dtype} {alert}")
    print("-" * 75)

    # Export to Silver Layer
    output_path = '../data/silver/london_luxury_analytics_NLP.csv'
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    full_df.to_csv(output_path, index=False)
    print(f"✅ PROCESS FINISHED. File saved at: {output_path}")

else:
    print("⚠️ Could not generate the dataset (No files loaded).")