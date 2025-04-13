import os
import numpy as np  # Added import
import simulate_data
import clean_real_data
import train_model
import evaluate_visualize
import pandas as pd
import pickle

def generate_report():
    """
    Generate a summary report with theoretical insights and dynamic findings.
    
    Returns:
        list: Report lines.
    """
    report = [
        "IERG3050 Project: Logistic Regression Analysis\n",
        "1. THEORETICAL FOUNDATION",
        "- Sigmoid function maps linear predictors to probabilities",
        "- Cross-entropy loss optimized via gradient descent",
        "- Regularization (L1/L2) controls model complexity\n",
        "2. KEY FINDINGS"
    ]
    
    # Dynamic feature importance
    try:
        with open('outputs/reg_model_real.pkl', 'rb') as f:
            model = pickle.load(f)
        coef = model.coef_[0]
        features = ['study_hours', 'sleep_hours', 'attendance']
        top_idx = np.argmax(np.abs(coef))
        report.append(f"- Top predictor: {features[top_idx]} (coefficient: {coef[top_idx]:.3f})")
    except Exception as e:
        report.append("- Feature importance unavailable: Run train_model.py first")
    
    # Load metrics
    try:
        metrics_df = pd.read_csv('outputs/evaluation_metrics.csv')
        report.append("Model Performance Summary:\n")
        report.append(metrics_df.to_string(index=False))
        report.append("\n")
        accuracies = metrics_df['Accuracy']
        if accuracies.max() >= 0.80:
            report.append("Accuracy Goal: Achieved (>80% for at least one model).")
        else:
            report.append("Accuracy Goal: Below 80%. Consider class imbalance or feature engineering. "
                         "See F1-Score and ROC-AUC for balanced evaluation.")
        report.append("\n")
    except FileNotFoundError:
        report.append("Error: Metrics file not found. Run evaluate_visualize.py first.\n")
    
    # Visualizations
    report.append("Visualizations (see 'outputs/' directory):\n")
    report.append("- Decision Boundaries: decision_boundary_binary.png, decision_boundary_poly.png, "
                  "decision_boundary_multi.png\n")
    report.append("- ROC Curves: roc_curve_sim.png, roc_curve_real.png\n")
    report.append("- Feature Importance: feature_importance_sim.png, feature_importance_real.png, "
                  "feature_importance_sim_multi.png\n")
    report.append("- Class Distribution: class_distribution.png\n")
    report.append("- Real Data Scatter: real_scatter.png\n")
    report.append("- Multi-Class Confusion Matrix: multi_class_cm.png\n")
    report.append("- Bayesian Posterior (if available): bayesian_posterior.png\n")
    report.append("- 3D Feature Space: 3d_feature_space.png\n")
    
    # Insights
    report.append("Key Insights:\n")
    report.append("- Study hours and attendance strongly influence student success.\n")
    report.append("- SMOTE and balanced models improve performance on imbalanced data.\n")
    report.append("- Polynomial features capture non-linear relationships.\n")
    report.append("- Bayesian models provide uncertainty estimates, enhancing interpretability.\n")
    
    return report

def main():
    """Run the full project pipeline."""
    np.random.seed(42)  # Global seed for reproducibility
    try:
        print("Running simulate_data.py...")
        simulate_data.main()
    except Exception as e:
        print(f"Error in simulate_data.py: {e}")
        return
    
    try:
        print("\nRunning clean_real_data.py...")
        clean_real_data.main()
    except Exception as e:
        print(f"Error in clean_real_data.py: {e}")
        return
    
    try:
        print("\nRunning train_model.py...")
        train_model.main()
    except Exception as e:
        print(f"Error in train_model.py: {e}")
        return
    
    try:
        print("\nRunning evaluate_visualize.py...")
        evaluate_visualize.main()
    except Exception as e:
        print(f"Error in evaluate_visualize.py: {e}")
        return
    
    try:
        print("\nGenerating project report...")
        report = generate_report()
        os.makedirs('outputs', exist_ok=True)
        output_file = 'outputs/project_report.txt'
        if os.path.exists(output_file):
            print(f"Warning: Overwriting {output_file}")
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        print(f"Report saved to '{output_file}'")
    except Exception as e:
        print(f"Error generating report: {e}")
        return
    
    print("\nAll scripts completed. Check 'outputs/' for results and report.")

if __name__ == "__main__":
    main()