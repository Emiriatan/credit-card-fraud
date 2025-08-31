# Credit Card Fraud Detection Analysis - Stage 1: Data Overview and Initial Check

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# Create visualizations directory if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

warnings.filterwarnings('ignore')

# Set English font to avoid encoding issues
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=== Stage 1: Data Overview and Initial Check ===")

# 1.1 Data Loading and Basic Information Check
print("\n1.1 Loading dataset and checking basic information...")

# Load dataset
df = pd.read_csv('creditcard.csv')

print(f"Data shape: {df.shape}")
print(f"Number of features: {df.shape[1]}")
print(f"Number of samples: {df.shape[0]}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nData types:")
print(df.dtypes)
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())

# 1.2 Data Quality Assessment
print("\n1.2 Data quality assessment...")

# Check missing values
missing_values = df.isnull().sum()
print(f"Missing values: {missing_values[missing_values > 0]}")

# Check duplicate data
duplicate_rows = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_rows}")

# Basic statistics
print(f"\nBasic statistics:")
print(df.describe())

# Check data format consistency
print(f"\nData format consistency check:")
for col in df.columns:
    print(f"{col}: {df[col].dtype}")

# 1.3 Class Imbalance Analysis
print("\n1.3 Class imbalance analysis...")

# Class distribution
class_counts = df['Class'].value_counts()
print(f"Class distribution:")
print(class_counts)

# Class percentage
class_percentage = df['Class'].value_counts(normalize=True) * 100
print(f"\nClass percentage:")
print(class_percentage)

# Visualization of class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Class', data=df)
plt.title('Fraud vs Normal Transactions Distribution')
plt.xlabel('Class (0: Normal, 1: Fraud)')
plt.ylabel('Number of Transactions')
plt.savefig('visualizations/class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Detailed statistics
print(f"\nDetailed statistics:")
print(f"Total transactions: {len(df)}")
print(f"Normal transactions: {class_counts[0]} ({class_percentage[0]:.2f}%)")
print(f"Fraud transactions: {class_counts[1]} ({class_percentage[1]:.2f}%)")
print(f"Fraud ratio: 1:{int(class_counts[0]/class_counts[1])}")

# Save basic information to file
with open('visualizations/stage1_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=== Stage 1: Data Overview Summary ===\n\n")
    f.write(f"Data shape: {df.shape}\n")
    f.write(f"Missing values: {missing_values[missing_values > 0].to_dict()}\n")
    f.write(f"Duplicate rows: {duplicate_rows}\n")
    f.write(f"Class distribution: {class_counts.to_dict()}\n")
    f.write(f"Class percentage: {class_percentage.to_dict()}\n")
    f.write(f"Fraud ratio: 1:{int(class_counts[0]/class_counts[1])}\n")

print("\n=== Stage 1 Complete ===")
print("Results saved to visualizations/class_distribution.png and visualizations/stage1_summary.txt")