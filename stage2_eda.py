# Credit Card Fraud Detection Analysis - Stage 2: Exploratory Data Analysis (EDA)

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

# Time distribution for fraud vs normal transactions
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.histplot(data=df_clean[df_clean['Class'] == 0], x='Hour', bins=24, alpha=0.5, color='blue', label='Normal')
sns.histplot(data=df_clean[df_clean['Class'] == 1], x='Hour', bins=24, alpha=0.7, color='red', label='Fraud')
plt.title('Transaction Distribution by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Transactions')
plt.legend()

plt.subplot(1, 2, 2)
hourly_fraud_rate = df_clean.groupby('Hour')['Class'].mean()
plt.plot(hourly_fraud_rate.index, hourly_fraud_rate.values, marker='o', color='red')
plt.title('Fraud Rate by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Fraud Rate')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/time_patterns.png', dpi=300, bbox_inches='tight')
plt.show()

# 2.2 Transaction Amount Analysis
print("\n2.2 Transaction amount analysis...")

# Amount statistics by class
amount_stats = df_clean.groupby('Class')['Amount'].describe()
print("Amount statistics by class:")
print(amount_stats)

# Amount distribution visualization
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='Class', y='Amount', data=df_clean[df_clean['Amount'] <= 1000])  # Limit to reduce outlier impact
plt.title('Amount Distribution by Class (Amount <= 1000)')
plt.xlabel('Class (0: Normal, 1: Fraud)')
plt.ylabel('Amount')

plt.subplot(2, 2, 2)
sns.histplot(data=df_clean[df_clean['Class'] == 0], x='Amount', bins=50, alpha=0.5, color='blue', label='Normal')
sns.histplot(data=df_clean[df_clean['Class'] == 1], x='Amount', bins=50, alpha=0.7, color='red', label='Fraud')
plt.title('Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.legend()
plt.xlim(0, 1000)  # Limit to better visualize

plt.subplot(2, 2, 3)
sns.kdeplot(data=df_clean[df_clean['Class'] == 0], x='Amount', color='blue', label='Normal', log_scale=True)
sns.kdeplot(data=df_clean[df_clean['Class'] == 1], x='Amount', color='red', label='Fraud', log_scale=True)
plt.title('Amount Density (Log Scale)')
plt.xlabel('Amount (Log Scale)')
plt.ylabel('Density')
plt.legend()

plt.subplot(2, 2, 4)
# Average amount by hour and class
hourly_amount = df_clean.groupby(['Hour', 'Class'])['Amount'].mean().unstack()
hourly_amount.plot(figsize=(8, 4))
plt.title('Average Transaction Amount by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Average Amount')
plt.legend(['Normal', 'Fraud'])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/amount_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

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

# Heatmap of top correlations
plt.figure(figsize=(12, 8))
top_features = class_correlation.abs().nlargest(15).index
top_corr_matrix = df_clean[top_features].corr()
sns.heatmap(top_corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap - Top 15 Features')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 2.4 Feature Distribution Analysis
print("\n2.4 Feature distribution analysis...")

# Select top correlated features for detailed analysis
top_positive_features = class_correlation[1:6].index  # Top 5 positive correlations
top_negative_features = class_correlation[-6:-1].index  # Top 5 negative correlations

# Plot distributions for top features
plt.figure(figsize=(20, 12))
for i, feature in enumerate(top_positive_features):
    plt.subplot(2, 5, i+1)
    sns.kdeplot(data=df_clean[df_clean['Class'] == 0], x=feature, color='blue', label='Normal', alpha=0.5)
    sns.kdeplot(data=df_clean[df_clean['Class'] == 1], x=feature, color='red', label='Fraud', alpha=0.7)
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()

for i, feature in enumerate(top_negative_features):
    plt.subplot(2, 5, i+6)
    sns.kdeplot(data=df_clean[df_clean['Class'] == 0], x=feature, color='blue', label='Normal', alpha=0.5)
    sns.kdeplot(data=df_clean[df_clean['Class'] == 1], x=feature, color='red', label='Fraud', alpha=0.7)
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.savefig('visualizations/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 2.5 Summary Statistics
print("\n2.5 Summary statistics...")

# Update class counts after removing duplicates
updated_class_counts = df_clean['Class'].value_counts()
updated_class_percentage = df_clean['Class'].value_counts(normalize=True) * 100

print("Updated class distribution after removing duplicates:")
print(f"Normal transactions: {updated_class_counts[0]} ({updated_class_percentage[0]:.2f}%)")
print(f"Fraud transactions: {updated_class_counts[1]} ({updated_class_percentage[1]:.2f}%)")
print(f"Fraud ratio: 1:{int(updated_class_counts[0]/updated_class_counts[1])}")

# Save EDA summary
with open('visualizations/stage2_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=== Stage 2: EDA Summary ===\n\n")
    f.write(f"Data shape after removing duplicates: {df_clean.shape}\n")
    f.write(f"Removed {df.shape[0] - df_clean.shape[0]} duplicate rows\n")
    f.write(f"Updated class distribution: {updated_class_counts.to_dict()}\n")
    f.write(f"Top 5 positive correlations: {class_correlation[1:6].to_dict()}\n")
    f.write(f"Top 5 negative correlations: {class_correlation[-6:-1].to_dict()}\n")

print("\n=== Stage 2 Complete ===")
print("Visualizations saved to visualizations/ directory")
print("EDA summary saved to visualizations/stage2_summary.txt")