# Credit Card Fraud Detection Analysis - Stage 4: Model Building and Training

import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
# import catboost as cb  # Commented out if not installed
import warnings
warnings.filterwarnings('ignore')

print("=== Stage 4: Model Building and Training ===")

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

# Load preprocessing info
with open('models/preprocessing_info.pkl', 'rb') as f:
    preprocessing_info = pickle.load(f)

features = preprocessing_info['features']
n_splits = preprocessing_info['n_splits']

# 4.1 Model Definitions
print("\n4.1 Defining models...")

models = {
    'LogisticRegression': LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ),
    'LightGBM': lgb.LGBMClassifier(
        is_unbalance=True,
        random_state=42,
        verbose=-1
    ),
    # 'CatBoost': cb.CatBoostClassifier(
    #     random_state=42,
    #     verbose=False
    # )
}

print(f"Defined {len(models)} models:")
for name in models.keys():
    print(f"  - {name}")

# 4.2 Cross-Validation Setup
print("\n4.2 Setting up cross-validation...")

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 4.3 Model Training and Cross-Validation
print("\n4.3 Training models with cross-validation...")

results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    start_time = time.time()
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=skf, scoring='roc_auc', 
        n_jobs=-1
    )
    
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
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_auc': test_auc,
        'training_time': training_time,
        'model': model,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }
    
    print(f"  CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Training time: {training_time:.2f} seconds")
    
    # Save model
    with open(f'models/{model_name.lower()}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

# 4.4 Results Summary
print("\n4.4 Model performance summary...")

# Create results DataFrame
summary_data = []
for model_name, result in results.items():
    summary_data.append({
        'Model': model_name,
        'CV_AUC_Mean': result['cv_mean'],
        'CV_AUC_Std': result['cv_std'],
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

# 4.5 Feature Importance Analysis (for tree-based models)
print("\n4.5 Feature importance analysis...")

feature_importance_dict = {}

for model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
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

# 4.6 Training Summary
print("\n4.6 Training summary...")

best_model = summary_df.iloc[0]
print(f"Best performing model: {best_model['Model']}")
print(f"Best Test AUC: {best_model['Test_AUC']:.4f}")
print(f"Best CV AUC: {best_model['CV_AUC_Mean']:.4f} (+/- {best_model['CV_AUC_Std'] * 2:.4f})")

# Save training summary
training_summary = {
    'total_models_trained': len(models),
    'best_model': best_model['Model'],
    'best_test_auc': best_model['Test_AUC'],
    'best_cv_auc': best_model['CV_AUC_Mean'],
    'total_training_time': sum(result['training_time'] for result in results.values()),
    'models_performance': summary_df.to_dict('records')
}

with open('data/training_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=== Stage 4: Model Training Summary ===\n\n")
    f.write(f"Total models trained: {training_summary['total_models_trained']}\n")
    f.write(f"Best model: {training_summary['best_model']}\n")
    f.write(f"Best Test AUC: {training_summary['best_test_auc']:.4f}\n")
    f.write(f"Best CV AUC: {training_summary['best_cv_auc']:.4f}\n")
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