# Credit Card Fraud Detection Analysis - Stage 4: Quick Model Training

import pandas as pd
import numpy as np
import pickle
import time
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("=== Stage 4: Quick Model Training ===")

# Load preprocessed data
print("Loading preprocessed data...")
train_data = pd.read_csv('data/train_data.csv')
test_data = pd.read_csv('data/test_data.csv')

X_train = train_data.drop('Class', axis=1)
y_train = train_data['Class']
X_test = test_data.drop('Class', axis=1)
y_test = test_data['Class']

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# 4.1 Model Definitions (simplified)
print("\n4.1 Defining models...")

models = {
    'LogisticRegression': LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=200
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=30,  # Further reduced
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        max_depth=8
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=30,  # Further reduced
        random_state=42
    )
}

print(f"Defined {len(models)} models:")
for name in models.keys():
    print(f"  - {name}")

# 4.2 Model Training (simplified - no CV for speed)
print("\n4.2 Training models...")

results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    start_time = time.time()
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predictions on test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    training_time = time.time() - start_time
    
    # Store results
    results[model_name] = {
        'test_auc': test_auc,
        'training_time': training_time,
        'model': model,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }
    
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")
    
    # Save model
    with open(f'models/{model_name.lower()}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# 4.3 Results Summary
print("\n4.3 Model performance summary...")

# Create results DataFrame
summary_data = []
for model_name, result in results.items():
    summary_data.append({
        'Model': model_name,
        'Test_AUC': result['test_auc'],
        'Training_Time': result['training_time']
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Test_AUC', ascending=False)

print("Model Performance Summary:")
print(summary_df.to_string(index=False))

# Save results
summary_df.to_csv('data/model_performance_summary.csv', index=False)
print("\nResults saved to data/model_performance_summary.csv")

# Save all results
with open('models/training_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Training results saved to models/training_results.pkl")

# 4.4 Feature Importance Analysis
print("\n4.4 Feature importance analysis...")

feature_importance_dict = {}
features = X_train.columns.tolist()

for model_name in ['RandomForest']:
    if model_name in results:
        model = results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            feature_importance_dict[model_name] = dict(zip(features, feature_importance))
            
            # Get top 10 features
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 features for {model_name}:")
            print(importance_df.head(10).to_string(index=False))

# Save feature importance
with open('models/feature_importance.pkl', 'wb') as f:
    pickle.dump(feature_importance_dict, f)

print("Feature importance saved to models/feature_importance.pkl")

# 4.5 Training Summary
print("\n4.5 Training summary...")

best_model = summary_df.iloc[0]
print(f"Best performing model: {best_model['Model']}")
print(f"Best Test AUC: {best_model['Test_AUC']:.4f}")

# Save training summary
training_summary = {
    'total_models_trained': len(models),
    'best_model': best_model['Model'],
    'best_test_auc': best_model['Test_AUC'],
    'total_training_time': sum(result['training_time'] for result in results.values()),
    'models_performance': summary_df.to_dict('records')
}

with open('data/training_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=== Stage 4: Model Training Summary ===\n\n")
    f.write(f"Total models trained: {training_summary['total_models_trained']}\n")
    f.write(f"Best model: {training_summary['best_model']}\n")
    f.write(f"Best Test AUC: {training_summary['best_test_auc']:.4f}\n")
    f.write(f"Total training time: {training_summary['total_training_time']:.2f} seconds\n\n")
    f.write("Model Performance Summary:\n")
    f.write(summary_df.to_string(index=False))

print("\n=== Stage 4 Complete ===")
print("Model training completed successfully!")
print("Files saved:")
print("  - Individual model files in models/ directory")
print("  - data/model_performance_summary.csv")
print("  - models/training_results.pkl")
print("  - models/feature_importance.pkl")
print("  - data/training_summary.txt")