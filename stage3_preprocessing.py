# Credit Card Fraud Detection Analysis - Stage 3: Data Preprocessing

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

print("=== Stage 3: Data Preprocessing ===")

# Load cleaned data from Stage 2
df = pd.read_csv('creditcard.csv')
df_clean = df.drop_duplicates()

print(f"Data shape after removing duplicates: {df_clean.shape}")

# 3.1 Feature Engineering
print("\n3.1 Feature engineering...")

# Add hour feature (as in EDA)
df_clean['Hour'] = df_clean['Time'] / 3600
df_clean['Hour'] = df_clean['Hour'] % 24

# Apply RobustScaler to Amount feature (based on author's method)
print("Applying RobustScaler to Amount feature...")
scaler = RobustScaler()
df_clean['Amount_scaled'] = scaler.fit_transform(df_clean['Amount'].values.reshape(-1, 1))

# Save the scaler for future use
with open('models/robust_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("RobustScaler saved to models/robust_scaler.pkl")

# 3.2 Data Splitting
print("\n3.2 Data splitting...")

# Define features and target
features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
           'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount_scaled', 'Hour']
target = 'Class'

X = df_clean[features]
y = df_clean[target]

# Split data using stratified sampling (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training set class distribution:")
print(y_train.value_counts())
print(f"Test set class distribution:")
print(y_test.value_counts())

# 3.3 Cross-Validation Setup
print("\n3.3 Cross-validation setup...")

# Set up StratifiedKFold for cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"Using {n_splits}-fold Stratified Cross-Validation")

# Show the fold distributions
fold_info = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    y_train_fold = y_train.iloc[train_idx]
    y_val_fold = y_train.iloc[val_idx]
    
    fold_info.append({
        'fold': fold + 1,
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'train_fraud': y_train_fold.sum(),
        'val_fraud': y_val_fold.sum(),
        'train_fraud_rate': y_train_fold.mean(),
        'val_fraud_rate': y_val_fold.mean()
    })

fold_df = pd.DataFrame(fold_info)
print("Cross-validation fold distribution:")
print(fold_df)

# 3.4 Save Processed Data
print("\n3.4 Saving processed data...")

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save processed datasets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

print("Training data saved to data/train_data.csv")
print("Test data saved to data/test_data.csv")

# Save feature names and other metadata
preprocessing_info = {
    'features': features,
    'target': target,
    'n_features': len(features),
    'train_shape': X_train.shape,
    'test_shape': X_test.shape,
    'n_splits': n_splits,
    'random_state': 42,
    'scaler_type': 'RobustScaler'
}

with open('models/preprocessing_info.pkl', 'wb') as f:
    pickle.dump(preprocessing_info, f)

print("Preprocessing info saved to models/preprocessing_info.pkl")

# 3.5 Baseline Model Test (Quick test to ensure data is properly prepared)
print("\n3.5 Quick baseline model test...")

# Train a simple Random Forest to test the pipeline
rf_baseline = RandomForestClassifier(
    n_estimators=10,  # Small number for quick test
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_baseline.fit(X_train, y_train)

# Quick evaluation
train_score = rf_baseline.score(X_train, y_train)
test_score = rf_baseline.score(X_test, y_test)

print(f"Baseline Random Forest - Training accuracy: {train_score:.4f}")
print(f"Baseline Random Forest - Test accuracy: {test_score:.4f}")

# Save baseline model
with open('models/rf_baseline.pkl', 'wb') as f:
    pickle.dump(rf_baseline, f)

print("Baseline model saved to models/rf_baseline.pkl")

# 3.6 Summary
print("\n3.6 Preprocessing summary...")

summary_info = {
    'original_data_shape': df.shape,
    'cleaned_data_shape': df_clean.shape,
    'duplicates_removed': df.shape[0] - df_clean.shape[0],
    'features_used': len(features),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'train_fraud_count': y_train.sum(),
    'test_fraud_count': y_test.sum(),
    'train_fraud_rate': y_train.mean(),
    'test_fraud_rate': y_test.mean(),
    'cv_folds': n_splits
}

print("Preprocessing Summary:")
for key, value in summary_info.items():
    print(f"  {key}: {value}")

# Save summary
with open('data/preprocessing_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=== Stage 3: Data Preprocessing Summary ===\n\n")
    for key, value in summary_info.items():
        f.write(f"{key}: {value}\n")

print("\n=== Stage 3 Complete ===")
print("Data preprocessing completed successfully!")
print("Files saved:")
print("  - data/train_data.csv")
print("  - data/test_data.csv") 
print("  - data/preprocessing_summary.txt")
print("  - models/robust_scaler.pkl")
print("  - models/preprocessing_info.pkl")
print("  - models/rf_baseline.pkl")