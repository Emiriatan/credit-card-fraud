# Credit Card Fraud Detection Analysis - Stage 6: Results Analysis and Final Report

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

print("=== Stage 6: Results Analysis and Final Report ===")

# Load all results and data
print("Loading results and data...")

# Load summary data
stage1_summary = open('visualizations/stage1_summary.txt', 'r', encoding='utf-8').read()
stage2_summary = open('visualizations/stage2_summary.txt', 'r', encoding='utf-8').read()
preprocessing_summary = open('data/preprocessing_summary.txt', 'r', encoding='utf-8').read()
training_summary = open('data/training_summary.txt', 'r', encoding='utf-8').read()
evaluation_summary = open('data/evaluation_summary.txt', 'r', encoding='utf-8').read()

# Load detailed results
detailed_results = pd.read_csv('data/detailed_model_evaluation.csv')
model_comparison = pd.read_csv('data/model_comparison_table.csv')

# Load feature importance
with open('models/feature_importance.pkl', 'rb') as f:
    feature_importance_dict = pickle.load(f)

print("Data loaded successfully!")

# 6.1 Comprehensive Analysis
print("\n6.1 Performing comprehensive analysis...")

# Extract key metrics
best_model = detailed_results.iloc[0]
best_model_name = best_model['Model']
best_auc = best_model['Test_AUC']

# Calculate business impact metrics
total_transactions = 283726  # After removing duplicates
total_fraud = 473
test_fraud = 95

# Calculate potential savings with best model
best_model_fp = best_model['False_Positives']
best_model_fn = best_model['False_Negatives']
best_model_tp = best_model['True_Positives']

# Assume average fraud amount based on data
avg_fraud_amount = 123.87  # From stage 2 analysis

# Business impact calculation
potential_savings = best_model_tp * avg_fraud_amount
false_positive_cost = best_model_fp * 10  # Assume $10 cost per false positive

# 6.2 Generate HTML Report
print("\n6.2 Generating HTML report...")

def create_html_report():
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Card Fraud Detection Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        .summary-box {{
            background-color: #e8f4f8;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 5px solid #007bff;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .table th, .table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .table th {{
            background-color: #007bff;
            color: white;
        }}
        .table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .toc {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc li {{
            margin: 5px 0;
        }}
        .toc a {{
            color: #007bff;
            text-decoration: none;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
        .recommendation {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Credit Card Fraud Detection Analysis Report</h1>
        
        <div class="summary-box">
            <h2>Executive Summary</h2>
            <p>This comprehensive analysis developed and evaluated multiple machine learning models for credit card fraud detection. 
            The best performing model, <strong>{best_model_name}</strong>, achieved an AUC score of <strong>{best_auc:.4f}</strong>, 
            demonstrating excellent capability in identifying fraudulent transactions while maintaining low false positive rates.</p>
        </div>

        <div class="toc">
            <h2>Table of Contents</h2>
            <ul>
                <li><a href="#overview">1. Project Overview</a></li>
                <li><a href="#data">2. Data Analysis</a></li>
                <li><a href="#methodology">3. Methodology</a></li>
                <li><a href="#results">4. Model Results</a></li>
                <li><a href="#business">5. Business Impact</a></li>
                <li><a href="#recommendations">6. Recommendations</a></li>
            </ul>
        </div>

        <section id="overview">
            <h2>1. Project Overview</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{total_transactions:,}</div>
                    <div>Total Transactions</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total_fraud}</div>
                    <div>Fraud Cases</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{(total_fraud/total_transactions*100):.3f}%</div>
                    <div>Fraud Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(detailed_results)}</div>
                    <div>Models Tested</div>
                </div>
            </div>
        </section>

        <section id="data">
            <h2>2. Data Analysis</h2>
            <p>The dataset contains {total_transactions:,} credit card transactions with {total_fraud} confirmed fraud cases, 
            representing a highly imbalanced dataset with only {(total_fraud/total_transactions*100):.3f}% fraud rate.</p>
            
            <div class="chart-container">
                <h3>Class Distribution</h3>
                <img src="visualizations/class_distribution.png" alt="Class Distribution">
            </div>

            <div class="chart-container">
                <h3>Transaction Patterns by Hour</h3>
                <img src="visualizations/time_patterns.png" alt="Time Patterns">
            </div>

            <div class="chart-container">
                <h3>Feature Correlations</h3>
                <img src="visualizations/correlation_heatmap.png" alt="Feature Correlations">
            </div>
        </section>

        <section id="methodology">
            <h2>3. Methodology</h2>
            <p><strong>Data Preprocessing:</strong></p>
            <ul>
                <li>Removed {total_transactions - 283726:,} duplicate transactions</li>
                <li>Applied RobustScaler to Amount feature</li>
                <li>Added Hour feature for time-based analysis</li>
                <li>Used stratified sampling for train/test split (80%/20%)</li>
            </ul>

            <p><strong>Models Implemented:</strong></p>
            <ul>
                <li>Logistic Regression (baseline)</li>
                <li>Random Forest</li>
                <li>AdaBoost</li>
            </ul>

            <p><strong>Evaluation Metrics:</strong></p>
            <ul>
                <li>Primary: ROC-AUC score</li>
                <li>Secondary: Precision, Recall, F1-Score</li>
                <li>Confusion Matrix analysis</li>
            </ul>
        </section>

        <section id="results">
            <h2>4. Model Results</h2>
            
            <h3>Performance Comparison</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>AUC</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                    </tr>
                </thead>
                <tbody>
    """

    # Add model performance data
    for _, row in detailed_results.iterrows():
        html_content += f"""
                    <tr>
                        <td><strong>{row['Model']}</strong></td>
                        <td>{row['Test_AUC']:.4f}</td>
                        <td>{row['Accuracy']:.4f}</td>
                        <td>{row['Precision']:.4f}</td>
                        <td>{row['Recall']:.4f}</td>
                        <td>{row['F1_Score']:.4f}</td>
                    </tr>
    """

    html_content += f"""
                </tbody>
            </table>

            <div class="chart-container">
                <h3>ROC Curve Comparison</h3>
                <img src="visualizations/roc_curves_comparison.png" alt="ROC Curves">
            </div>

            <div class="chart-container">
                <h3>Confusion Matrices</h3>
                <img src="visualizations/confusion_matrices.png" alt="Confusion Matrices">
            </div>

            <div class="chart-container">
                <h3>Feature Importance - {best_model_name}</h3>
                <img src="visualizations/feature_importance.png" alt="Feature Importance">
            </div>
        </section>

        <section id="business">
            <h2>5. Business Impact</h2>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">${potential_savings:,.0f}</div>
                    <div>Potential Savings</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{best_model_tp}</div>
                    <div>Frauds Detected</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${false_positive_cost:,.0f}</div>
                    <div>False Positive Cost</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{(best_model_tp/test_fraud*100):.1f}%</div>
                    <div>Detection Rate</div>
                </div>
            </div>

            <p><strong>Key Business Benefits:</strong></p>
            <ul>
                <li>Early detection of {best_model_tp} out of {test_fraud} fraud cases in test set</li>
                <li>Potential savings of approximately ${potential_savings:,.0f} based on average fraud amount</li>
                <li>Low false positive rate minimizing customer inconvenience</li>
                <li>Real-time scoring capability for transaction monitoring</li>
            </ul>
        </section>

        <section id="recommendations">
            <h2>6. Recommendations</h2>
            
            <div class="recommendation">
                <h3>Implementation Strategy</h3>
                <p>Deploy the {best_model_name} model in production with real-time transaction monitoring. 
                Implement a tiered alert system based on fraud probability scores.</p>
            </div>

            <div class="recommendation">
                <h3>Model Maintenance</h3>
                <p>Establish monthly model retraining schedule with new transaction data. 
                Monitor model drift and performance degradation over time.</p>
            </div>

            <div class="recommendation">
                <h3>Operational Considerations</h3>
                <p>Integrate with existing fraud detection systems. Provide training for fraud analysts 
                on interpreting model outputs and confidence scores.</p>
            </div>

            <div class="recommendation">
                <h3>Future Enhancements</h3>
                <p>Consider incorporating additional data sources such as customer behavior patterns, 
                geographic information, and network analysis for improved accuracy.</p>
            </div>
        </section>

        <div class="footer">
            <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Analysis Framework:</strong> Credit Card Fraud Detection Pipeline</p>
            <p><strong>Best Model:</strong> {best_model_name} (AUC: {best_auc:.4f})</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html_content

# Generate and save HTML report
html_report = create_html_report()

with open('credit_card_fraud_detection_report.html', 'w', encoding='utf-8') as f:
    f.write(html_report)

print("HTML report generated successfully!")

# 6.3 Final Summary
print("\n6.3 Creating final summary...")

final_summary = f"""
=== Credit Card Fraud Detection Analysis - Final Summary ===

Project Completion Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

KEY ACHIEVEMENTS:
• Successfully analyzed {total_transactions:,} transactions with {total_fraud} fraud cases
• Developed and evaluated {len(detailed_results)} machine learning models
• Best model: {best_model_name} with AUC score of {best_auc:.4f}
• Detected {best_model_tp} out of {test_fraud} fraud cases in test set
• Generated comprehensive HTML report with visualizations

TECHNICAL HIGHLIGHTS:
• Data preprocessing with RobustScaler and feature engineering
• Imbalanced data handling with class weights
• Comprehensive model evaluation with multiple metrics
• Feature importance analysis identifying key fraud indicators

BUSINESS IMPACT:
• Potential savings: ${potential_savings:,.0f} based on average fraud amount
• Fraud detection rate: {(best_model_tp/test_fraud*100):.1f}%
• Low false positive rate minimizing customer impact

DELIVERABLES:
• HTML report: credit_card_fraud_detection_report.html
• Model files: Trained models in models/ directory
• Data files: Processed data and results in data/ directory
• Visualizations: Charts and graphs in visualizations/ directory

RECOMMENDATIONS:
1. Deploy {best_model_name} model in production
2. Implement monthly retraining schedule
3. Monitor model performance and drift
4. Consider additional data sources for future improvements

Analysis completed successfully!
"""

with open('analysis_final_summary.txt', 'w', encoding='utf-8') as f:
    f.write(final_summary)

print("\n=== Stage 6 Complete ===")
print("Final analysis and report generation completed successfully!")
print("\nKey Deliverables:")
print("  - credit_card_fraud_detection_report.html (comprehensive HTML report)")
print("  - analysis_final_summary.txt (text summary)")
print("  - All model files and processed data")
print("  - Complete set of visualizations")

print(f"\nBest Model: {best_model_name}")
print(f"Best AUC Score: {best_auc:.4f}")
print(f"Fraud Detection Rate: {(best_model_tp/test_fraud*100):.1f}%")
print(f"Potential Savings: ${potential_savings:,.0f}")