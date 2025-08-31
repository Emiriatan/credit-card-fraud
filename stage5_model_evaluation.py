# Credit Card Fraud Detection Analysis - Stage 5: Model Evaluation and Comparison

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

print("=== Stage 5: Model Evaluation and Comparison ===")

# Load test data and results
print("Loading data and results...")
test_data = pd.read_csv('data/test_data.csv')
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

with open('models/training_results.pkl', 'rb') as f:
    results = pickle.load(f)

with open('data/model_performance_summary.csv', 'r') as f:
    summary_df = pd.read_csv(f)

print(f"Test data shape: {X_test.shape}")
print(f"Number of models evaluated: {len(results)}")

# 5.1 Detailed Model Evaluation
print("\n5.1 Detailed model evaluation...")

detailed_results = []

for model_name, result in results.items():
    y_pred_proba = result['y_pred_proba']
    y_pred = result['y_pred']
    
    # Calculate comprehensive metrics
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # Store for later use
    result['ROC_Curve'] = (fpr, tpr)
    result['PR_Curve'] = (precision, recall)
    result['Confusion_Matrix'] = cm
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
    
    detailed_results.append({
        'Model': model_name,
        'Test_AUC': test_auc,
        'Accuracy': accuracy,
        'Precision': precision_score,
        'Recall': recall_score,
        'F1_Score': f1_score,
        'Average_Precision': avg_precision,
        'True_Negatives': tn,
        'False_Positives': fp,
        'False_Negatives': fn,
        'True_Positives': tp
    })
    
    print(f"\n{model_name}:")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision_score:.4f}")
    print(f"  Recall: {recall_score:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Create detailed results DataFrame
detailed_df = pd.DataFrame(detailed_results)
detailed_df = detailed_df.sort_values('Test_AUC', ascending=False)

print("\n=== Detailed Model Performance Summary ===")
print(detailed_df[['Model', 'Test_AUC', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'Average_Precision']].to_string(index=False))

# 5.2 ROC Curve Comparison
print("\n5.2 Creating ROC curve comparison...")

plt.figure(figsize=(12, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

for i, (model_name, result) in enumerate(results.items()):
    fpr, tpr = result['ROC_Curve']
    auc_score = result['Test_AUC']
    plt.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2, 
             label=f'{model_name} (AUC = {auc_score:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('visualizations/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("ROC curves saved to visualizations/roc_curves_comparison.png")

# 5.3 Confusion Matrix Visualization
print("\n5.3 Creating confusion matrices...")

fig, axes = plt.subplots(1, len(results), figsize=(15, 4))
if len(results) == 1:
    axes = [axes]

for i, (model_name, result) in enumerate(results.items()):
    cm = result['Confusion_Matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'{model_name}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

print("Confusion matrices saved to visualizations/confusion_matrices.png")

# 5.4 Feature Importance Visualization
print("\n5.4 Creating feature importance visualization...")

with open('models/feature_importance.pkl', 'rb') as f:
    feature_importance_dict = pickle.load(f)

if feature_importance_dict:
    # Get the best model's feature importance
    best_model_name = detailed_df.iloc[0]['Model']
    if best_model_name in feature_importance_dict:
        feature_importance = feature_importance_dict[best_model_name]
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        }).sort_values('importance', ascending=False).head(15)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top 15 Feature Importance - {best_model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance for {best_model_name} saved to visualizations/feature_importance.png")

# 5.5 Model Performance Comparison Table
print("\n5.5 Creating performance comparison table...")

# Create a comprehensive comparison table
comparison_table = detailed_df[['Model', 'Test_AUC', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'Average_Precision']]
comparison_table = comparison_table.round(4)

# Save comparison table
comparison_table.to_csv('data/model_comparison_table.csv', index=False)
print("Comparison table saved to data/model_comparison_table.csv")

# 5.6 Best Model Analysis
print("\n5.6 Best model analysis...")

best_model = detailed_df.iloc[0]
best_model_name = best_model['Model']
best_model_auc = best_model['Test_AUC']

print(f"Best Model: {best_model_name}")
print(f"Best AUC: {best_model_auc:.4f}")
print(f"Best Precision: {best_model['Precision']:.4f}")
print(f"Best Recall: {best_model['Recall']:.4f}")
print(f"Best F1 Score: {best_model['F1_Score']:.4f}")

# 5.7 Save Evaluation Results
print("\n5.7 Saving evaluation results...")

# Save detailed results
detailed_df.to_csv('data/detailed_model_evaluation.csv', index=False)
print("Detailed evaluation saved to data/detailed_model_evaluation.csv")

# Save evaluation summary
evaluation_summary = {
    'best_model': best_model_name,
    'best_auc': best_model_auc,
    'total_models': len(results),
    'evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_performances': detailed_df.to_dict('records')
}

with open('data/evaluation_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=== Stage 5: Model Evaluation Summary ===\n\n")
    f.write(f"Best Model: {evaluation_summary['best_model']}\n")
    f.write(f"Best AUC: {evaluation_summary['best_auc']:.4f}\n")
    f.write(f"Total Models Evaluated: {evaluation_summary['total_models']}\n")
    f.write(f"Evaluation Date: {evaluation_summary['evaluation_date']}\n\n")
    f.write("Detailed Performance:\n")
    f.write(detailed_df[['Model', 'Test_AUC', 'Precision', 'Recall', 'F1_Score']].to_string(index=False))

print("\n=== Stage 5 Complete ===")
print("Model evaluation completed successfully!")
print("Files saved:")
print("  - visualizations/roc_curves_comparison.png")
print("  - visualizations/confusion_matrices.png")
print("  - visualizations/feature_importance.png")
print("  - data/model_comparison_table.csv")
print("  - data/detailed_model_evaluation.csv")
print("  - data/evaluation_summary.txt")