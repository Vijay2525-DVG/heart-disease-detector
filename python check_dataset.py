import pandas as pd

print("Checking your dataset...")

# Try to load the dataset
try:
    df = pd.read_csv('heart.csv')
    
    print(f"\n{'='*60}")
    print(f"DATASET INFORMATION")
    print(f"{'='*60}")
    
    print(f"\nShape: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print(f"\n{'='*60}")
    print(f"COLUMN NAMES")
    print(f"{'='*60}")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    
    print(f"\n{'='*60}")
    print(f"FIRST 5 ROWS")
    print(f"{'='*60}")
    print(df.head())
    
    print(f"\n{'='*60}")
    print(f"DATA TYPES")
    print(f"{'='*60}")
    print(df.dtypes)
    
    print(f"\n{'='*60}")
    print(f"MISSING VALUES")
    print(f"{'='*60}")
    print(df.isnull().sum())
    
    print(f"\n{'='*60}")
    print(f"BASIC STATISTICS")
    print(f"{'='*60}")
    print(df.describe())
    
    # Try to identify the target column
    print(f"\n{'='*60}")
    print(f"IDENTIFYING TARGET COLUMN")
    print(f"{'='*60}")
    
    # Common target column names
    possible_targets = ['output', 'target', 'num', 'diagnosis', 'condition', 'disease', 'class', 'label']
    
    target_col = None
    for col in df.columns:
        if col.lower() in possible_targets:
            target_col = col
            print(f"✓ Found potential target column: '{col}'")
            print(f"  Unique values: {df[col].unique()}")
            print(f"  Value counts:\n{df[col].value_counts()}")
            break
    
    if target_col is None:
        print("⚠ Could not automatically identify target column.")
        print("\nPlease check your last column (usually the target):")
        last_col = df.columns[-1]
        print(f"Last column: '{last_col}'")
        print(f"Unique values: {df[last_col].unique()}")
        print(f"Value counts:\n{df[last_col].value_counts()}")
        print(f"\nIf this is your target column, rename it to 'output' or update the training script.")
    
except FileNotFoundError:
    print("❌ Error: 'heart.csv' not found!")
    print("\nPlease ensure heart.csv is in the same folder as this script.")
    print("\nOptions:")
    print("1. Run download_dataset.py to download the dataset")
    print("2. Download manually from Kaggle:")
    print("   https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
    
except Exception as e:
    print(f"❌ Error reading dataset: {e}")
    import traceback
    traceback.print_exc()