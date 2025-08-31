# Credit Card Fraud Detection Analysis - Stage 2: Exploratory Data Analysis (EDA) - Optimized

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== Stage 2: Exploratory Data Analysis ===")

# Load data
df = pd.read_csv('creditcard.csv')

# Remove duplicate rows first
print("Removing duplicate rows...")
df_clean = df.drop_duplicates()
print(f"Original shape: {df.shape}")
print(f"After removing duplicates: {df_clean.shape}")
print(f"Removed {df.shape[0] - df_clean.shape[0]} duplicate rows")

# 2.1 Time Pattern Analysis
print("\n2.1 Time pattern analysis...")

# Convert Time to hours
df_clean['Hour'] = df_clean['Time'] / 3600
df_clean['Hour'] = df_clean['Hour'] % 24  # Convert to 24-hour format

# Calculate hourly statistics
hourly_stats = df_clean.groupby('Hour').agg({
    'Class': ['count', 'mean'],
    'Amount': 'mean'
}).round(4)

print("Hourly transaction statistics:")
print(hourly_stats.head())

# 2.2 Transaction Amount Analysis
print("\n2.2 Transaction amount analysis...")

# Amount statistics by class
amount_stats = df_clean.groupby('Class')['Amount'].describe()
print("Amount statistics by class:")
print(amount_stats)

# 2.3 Feature Correlation Analysis
print("\n2.3 Feature correlation analysis...")

# Calculate correlation matrix
correlation_matrix = df_clean.corr()

# Correlation with Class
class_correlation = correlation_matrix['Class'].sort_values(ascending=False)
print("Top 10 features most correlated with Class:")
print(class_correlation.head(11))  # Including Class itself

print("\nTop 10 features most negatively correlated with Class:")
print(class_correlation.tail(10))

# 2.4 Summary Statistics
print("\n2.4 Summary statistics...")

# Update class counts after removing duplicates
updated_class_counts = df_clean['Class'].value_counts()
updated_class_percentage = df_clean['Class'].value_counts(normalize=True) * 100

print("Updated class distribution after removing duplicates:")
print(f"Normal transactions: {updated_class_counts[0]} ({updated_class_percentage[0]:.2f}%)")
print(f"Fraud transactions: {updated_class_counts[1]} ({updated_class_percentage[1]:.2f}%)")
print(f"Fraud ratio: 1:{int(updated_class_counts[0]/updated_class_counts[1])}")

# Create simplified visualizations
plt.style.use('default')

# 1. Time patterns visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
hourly_transactions = df_clean.groupby('Hour').size()
plt.plot(hourly_transactions.index, hourly_transactions.values, marker='o', markersize=3)
plt.title('Transaction Count by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Transactions')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
hourly_fraud_rate = df_clean.groupby('Hour')['Class'].mean()
plt.plot(hourly_fraud_rate.index, hourly_fraud_rate.values, marker='o', color='red', markersize=3)
plt.title('Fraud Rate by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Fraud Rate')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/time_patterns.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Amount distribution visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.boxplot(x='Class', y='Amount', data=df_clean[df_clean['Amount'] <= 500])
plt.title('Amount Distribution by Class (Amount <= 500)')
plt.xlabel('Class (0: Normal, 1: Fraud)')
plt.ylabel('Amount')

plt.subplot(1, 2, 2)
# Sample data for faster plotting
normal_sample = df_clean[df_clean['Class'] == 0].sample(10000, random_state=42)
fraud_sample = df_clean[df_clean['Class'] == 1]
plt.hist([normal_sample['Amount'], fraud_sample['Amount']], bins=30, alpha=0.7, 
         label=['Normal', 'Fraud'], color=['blue', 'red'])
plt.title('Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.legend()
plt.xlim(0, 500)

plt.tight_layout()
plt.savefig('visualizations/amount_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Correlation heatmap
plt.figure(figsize=(10, 6))
top_features = class_correlation.abs().nlargest(11).index
top_corr_matrix = df_clean[top_features].corr()
sns.heatmap(top_corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
            annot_kws={'size': 8})
plt.title('Correlation Heatmap - Top 11 Features')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Feature distributions for top correlated features
plt.figure(figsize=(15, 8))
top_features = class_correlation.abs().nlargest(6).index[1:]  # Exclude Class itself

for i, feature in enumerate(top_features):
    plt.subplot(2, 3, i+1)
    # Sample data for faster plotting
    normal_sample = df_clean[df_clean['Class'] == 0].sample(5000, random_state=42)
    fraud_sample = df_clean[df_clean['Class'] == 1]
    
    sns.kdeplot(data=normal_sample, x=feature, color='blue', label='Normal', alpha=0.5)
    sns.kdeplot(data=fraud_sample, x=feature, color='red', label='Fraud', alpha=0.7)
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.savefig('visualizations/feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# Save EDA summary
with open('visualizations/stage2_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=== Stage 2: EDA Summary ===\n\n")
    f.write(f"Data shape after removing duplicates: {df_clean.shape}\n")
    f.write(f"Removed {df.shape[0] - df_clean.shape[0]} duplicate rows\n")
    f.write(f"Updated class distribution: {updated_class_counts.to_dict()}\n")
    f.write(f"Fraud ratio: 1:{int(updated_class_counts[0]/updated_class_counts[1])}\n")
    f.write(f"Top 5 positive correlations: {class_correlation[1:6].to_dict()}\n")
    f.write(f"Top 5 negative correlations: {class_correlation[-6:-1].to_dict()}\n")
    f.write(f"Peak transaction hours: {hourly_transactions.nlargest(3).index.tolist()}\n")
    f.write(f"Peak fraud rate hours: {hourly_fraud_rate.nlargest(3).index.tolist()}\n")

print("\n=== Stage 2 Complete ===")
print("Visualizations saved to visualizations/ directory")
print("EDA summary saved to visualizations/stage2_summary.txt")