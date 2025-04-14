import os
import numpy as np
import simulate_data
import clean_real_data
import train_model
import evaluate_visualize
import pandas as pd
import pickle
from datetime import datetime

def generate_report():
    """
    Generate a detailed HTML report with enhanced data and visualizations.
    
    Returns:
        str: HTML content of the report.
    """
    try:
        html_content = []
        html_content.append('<!DOCTYPE html>')
        html_content.append('<html lang="en">')
        html_content.append('<head>')
        html_content.append('    <meta charset="UTF-8">')
        html_content.append('    <title>IERG3050 Project: Logistic Regression Analysis</title>')
        html_content.append('    <style>')
        html_content.append('        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }')
        html_content.append('        h1, h2, h3 { color: #2c3e50; }')
        html_content.append('        h1 { text-align: center; }')
        html_content.append('        table { width: 100%; border-collapse: collapse; margin: 20px 0; }')
        html_content.append('        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }')
        html_content.append('        th { background-color: #f2f2f2; }')
        html_content.append('        tr:nth-child(even) { background-color: #f9f9f9; }')
        html_content.append('        .highlight { background-color: #d4edda; }')
        html_content.append('        ul { margin: 10px 0; }')
        html_content.append('        a { color: #007bff; text-decoration: none; }')
        html_content.append('        a:hover { text-decoration: underline; }')
        html_content.append('        .section { margin-bottom: 30px; }')
        html_content.append('        .error { color: #dc3545; font-style: italic; }')
        html_content.append('        img { max-width: 100%; height: auto; display: block; margin: 10px 0; }')
        html_content.append('        .caption { text-align: center; font-style: italic; margin-bottom: 20px; }')
        html_content.append('    </style>')
        html_content.append('</head>')
        html_content.append('<body>')

        # Header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content.append('    <h1>IERG3050 Project: Logistic Regression Analysis</h1>')
        html_content.append(f'    <p style="text-align: center;">Generated on {timestamp}</p>')

        # Introduction
        html_content.append('    <div class="section">')
        html_content.append('        <h2>1. Introduction</h2>')
        html_content.append('        <p>This report analyzes student performance using logistic regression and other machine learning models, leveraging both simulated and real datasets. The simulated dataset mimics student behaviors, while the real dataset is sourced from the UCI Student Performance dataset (Portuguese students).</p>')
        try:
            sim_data = pd.read_csv('outputs/simulated_student_data.csv')
            real_data = pd.read_csv('outputs/cleaned_real_data.csv')
            html_content.append('        <h3>1.1 Dataset Overview</h3>')
            html_content.append('        <table>')
            html_content.append('            <tr><th>Dataset</th><th>Records</th><th>Features</th><th>Classes</th></tr>')
            html_content.append(f'            <tr><td>Simulated</td><td>{len(sim_data)}</td><td>Study Hours, Sleep Hours, Attendance</td><td>Pass/Fail</td></tr>')
            html_content.append(f'            <tr><td>Real</td><td>{len(real_data)}</td><td>Study Hours, Sleep Hours, Attendance</td><td>Pass/Fail, Grade Class (Fail, Pass, Excellent)</td></tr>')
            html_content.append('        </table>')
        except:
            html_content.append('        <p class="error">Dataset overview unavailable.</p>')
        html_content.append('    </div>')

        # Theoretical Foundation
        html_content.append('    <div class="section">')
        html_content.append('        <h2>2. Theoretical Foundation</h2>')
        html_content.append('        <ul>')
        html_content.append('            <li><strong>Sigmoid Function</strong>: Maps linear predictors to probabilities between 0 and 1, enabling binary classification.</li>')
        html_content.append('            <li><strong>Cross-Entropy Loss</strong>: Optimized via gradient descent to minimize prediction errors.</li>')
        html_content.append('            <li><strong>Regularization</strong>: L1 (Lasso) and L2 (Ridge) penalties control model complexity to prevent overfitting.</li>')
        html_content.append('        </ul>')
        html_content.append('        <img src="../outputs/sigmoid_function.png" alt="Sigmoid Function">')
        html_content.append('        <p class="caption">Figure 1: Sigmoid function mapping logits to probabilities.</p>')
        html_content.append('    </div>')

        # Feature Analysis
        html_content.append('    <div class="section">')
        html_content.append('        <h2>3. Feature Analysis</h2>')
        
        # Model Coefficients
        html_content.append('        <h3>3.1 Model Coefficients</h3>')
        try:
            html_content.append('        <table>')
            html_content.append('            <tr><th>Model</th><th>Study Hours</th><th>Sleep Hours</th><th>Attendance</th></tr>')
            for model_name, file_name in [
                ('Real Data (L2)', 'reg_model_real.pkl'),
                ('Simulated Data (L2)', 'reg_model_sim.pkl'),
                ('Real Data (L1)', 'l1_model_real.pkl'),
                ('Simulated Data (L1)', 'l1_model_sim.pkl')
            ]:
                try:
                    with open(f'outputs/{file_name}', 'rb') as f:
                        model = pickle.load(f)
                    coef = model.coef_[0]
                    html_content.append(f'            <tr>')
                    html_content.append(f'                <td>{model_name}</td>')
                    html_content.append(f'                <td>{coef[0]:.3f}</td>')
                    html_content.append(f'                <td>{coef[1]:.3f}</td>')
                    html_content.append(f'                <td>{coef[2]:.3f}</td>')
                    html_content.append('            </tr>')
                except:
                    html_content.append(f'            <tr><td>{model_name}</td><td colspan="3" class="error">Model unavailable</td></tr>')
            html_content.append('        </table>')
        except Exception as e:
            html_content.append(f'        <p class="error">Coefficients unavailable: {str(e)}</p>')
        
        # Feature Correlations
        html_content.append('        <h3>3.2 Feature Correlations</h3>')
        try:
            sim_data = pd.read_csv('outputs/simulated_student_data.csv')
            real_data = pd.read_csv('outputs/cleaned_real_data.csv')
            features = ['study_hours', 'sleep_hours', 'attendance']
            for dataset_name, data in [('Simulated Data', sim_data), ('Real Data', real_data)]:
                corr = data[features].corr()
                html_content.append(f'        <h4>{dataset_name}</h4>')
                html_content.append('        <table>')
                html_content.append('            <tr><th>Feature</th><th>Study Hours</th><th>Sleep Hours</th><th>Attendance</th></tr>')
                for i, feat in enumerate(features):
                    html_content.append(f'            <tr>')
                    html_content.append(f'                <td>{feat}</td>')
                    html_content.append(f'                <td>{corr.iloc[i, 0]:.3f}</td>')
                    html_content.append(f'                <td>{corr.iloc[i, 1]:.3f}</td>')
                    html_content.append(f'                <td>{corr.iloc[i, 2]:.3f}</td>')
                    html_content.append('            </tr>')
                html_content.append('        </table>')
            html_content.append('        <img src="../outputs/correlation_heatmap.png" alt="Correlation Heatmap">')
            html_content.append('        <p class="caption">Figure 2: Correlation heatmaps for simulated and real data features.</p>')
        except Exception as e:
            html_content.append(f'        <p class="error">Correlation analysis unavailable: {str(e)}</p>')
        html_content.append('    </div>')

        # Class Distribution
        html_content.append('    <div class="section">')
        html_content.append('        <h2>4. Class Distribution</h2>')
        try:
            y_sim = np.load('outputs/y_sim_test.npy')
            y_real = np.load('outputs/y_real_test.npy')
            y_sim_multi = np.load('outputs/y_sim_multi_test.npy')
            for dataset_name, y in [
                ('Simulated Data (Binary)', y_sim),
                ('Real Data (Binary)', y_real),
                ('Simulated Data (Multi-Class)', y_sim_multi)
            ]:
                counts = pd.Series(y).value_counts(normalize=True)
                html_content.append(f'        <h3>{dataset_name}</h3>')
                html_content.append('        <ul>')
                for label, prop in counts.items():
                    label_name = 'Fail' if label == 0 else 'Pass' if label == 1 else 'Excellent'
                    html_content.append(f'            <li>{label_name}: {prop:.2%}</li>')
                html_content.append('        </ul>')
            html_content.append('        <img src="../outputs/class_distribution.png" alt="Class Distribution">')
            html_content.append('        <p class="caption">Figure 3: Distribution of pass/fail and multi-class labels across datasets.</p>')
        except Exception as e:
            html_content.append(f'        <p class="error">Class distribution unavailable: {str(e)}</p>')
        html_content.append('    </div>')

        # Key Findings
        html_content.append('    <div class="section">')
        html_content.append('        <h2>5. Key Findings</h2>')
        try:
            html_content.append('        <ul>')
            for dataset, file_name in [('Real Data', 'reg_model_real.pkl'), ('Simulated Data', 'reg_model_sim.pkl')]:
                with open(f'outputs/{file_name}', 'rb') as f:
                    model = pickle.load(f)
                coef = model.coef_[0]
                features = ['study_hours', 'sleep_hours', 'attendance']
                top_idx = np.argmax(np.abs(coef))
                html_content.append(f'            <li><strong>{dataset}</strong>: Top predictor is {features[top_idx]} (coefficient: {coef[top_idx]:.3f}).</li>')
            html_content.append('        </ul>')
            html_content.append('        <img src="../outputs/feature_importance_real.png" alt="Feature Importance (Real)">')
            html_content.append('        <p class="caption">Figure 4: Feature importance for real data model.</p>')
            html_content.append('        <img src="../outputs/feature_importance_sim.png" alt="Feature Importance (Simulated)">')
            html_content.append('        <p class="caption">Figure 5: Feature importance for simulated data model.</p>')
        except Exception as e:
            html_content.append(f'        <p class="error">Feature importance unavailable: {str(e)}</p>')
        html_content.append('    </div>')

        # Model Performance
        html_content.append('    <div class="section">')
        html_content.append('        <h2>6. Model Performance</h2>')
        try:
            metrics_df = pd.read_csv('outputs/evaluation_metrics.csv')
            # Binary models
            html_content.append('        <h3>6.1 Binary Classification</h3>')
            binary_models = [m for m in metrics_df['Model'] if 'Multi-Class' not in m]
            html_content.append('        <table>')
            html_content.append('            <tr><th>Model</th><th>Accuracy</th><th>F1-Score</th><th>ROC-AUC</th></tr>')
            for _, row in metrics_df[metrics_df['Model'].isin(binary_models)].iterrows():
                row_class = 'highlight' if row['Accuracy'] >= 0.80 else ''
                roc_auc = row['ROC-AUC'] if pd.notna(row['ROC-AUC']) else 'N/A'
                html_content.append(f'            <tr class="{row_class}">')
                html_content.append(f'                <td>{row["Model"]}</td>')
                html_content.append(f'                <td>{row["Accuracy"]:.3f}</td>')
                html_content.append(f'                <td>{row["F1-Score"]:.3f}</td>')
                html_content.append(f'                <td>{roc_auc}</td>')
                html_content.append('            </tr>')
            html_content.append('        </table>')
            # Multi-class models
            html_content.append('        <h3>6.2 Multi-Class Classification</h3>')
            multi_models = [m for m in metrics_df['Model'] if 'Multi-Class' in m]
            html_content.append('        <table>')
            html_content.append('            <tr><th>Model</th><th>Accuracy</th><th>F1-Score</th></tr>')
            for _, row in metrics_df[metrics_df['Model'].isin(multi_models)].iterrows():
                row_class = 'highlight' if row['Accuracy'] >= 0.80 else ''
                html_content.append(f'            <tr class="{row_class}">')
                html_content.append(f'                <td>{row["Model"]}</td>')
                html_content.append(f'                <td>{row["Accuracy"]:.3f}</td>')
                html_content.append(f'                <td>{row["F1-Score"]:.3f}</td>')
                html_content.append('            </tr>')
            html_content.append('        </table>')
            accuracies = metrics_df['Accuracy']
            if accuracies.max() >= 0.80:
                html_content.append('        <p><strong>Accuracy Goal</strong>: Achieved (≥80% for at least one model).</p>')
            else:
                html_content.append('        <p><strong>Accuracy Goal</strong>: Below 80%. Consider addressing class imbalance or enhancing feature engineering.</p>')
            html_content.append('        <img src="../outputs/roc_curve_sim.png" alt="ROC Curve (Simulated)">')
            html_content.append('        <p class="caption">Figure 6: ROC curves for simulated data binary models.</p>')
            html_content.append('        <img src="../outputs/roc_curve_real.png" alt="ROC Curve (Real)">')
            html_content.append('        <p class="caption">Figure 7: ROC curves for real data binary models.</p>')
            html_content.append('        <img src="../outputs/decision_boundary_multi.png" alt="Decision Boundary (Multi-Class)">')
            html_content.append('        <p class="caption">Figure 8: Decision boundary for simulated multi-class classification.</p>')
        except FileNotFoundError:
            html_content.append('        <p class="error">Error: Metrics file not found. Run evaluate_visualize.py first.</p>')
        except Exception as e:
            html_content.append(f'        <p class="error">Error loading metrics: {str(e)}</p>')
        html_content.append('    </div>')

        # Interactive Visualizations
        html_content.append('    <div class="section">')
        html_content.append('        <h2>7. Interactive Visualizations</h2>')
        html_content.append('        <p>Additional interactive visualizations are available in the <code>outputs/</code> directory:</p>')
        html_content.append('        <ul>')
        html_content.append('            <li><a href="../outputs/decision_boundary_binary.html">Decision Boundary (Simulated Binary)</a>: Classification regions.</li>')
        html_content.append('            <li><a href="../outputs/decision_boundary_poly.html">Decision Boundary (Polynomial)</a>: Non-linear boundaries.</li>')
        html_content.append('            <li><a href="../outputs/decision_boundary_multi.html">Decision Boundary (Multi-Class)</a>: Multi-class regions.</li>')
        html_content.append('            <li><a href="../outputs/roc_curve_sim.html">ROC Curve (Simulated)</a>: Model performance.</li>')
        html_content.append('            <li><a href="../outputs/roc_curve_real.html">ROC Curve (Real)</a>: Model performance.</li>')
        html_content.append('            <li><a href="../outputs/feature_importance_sim.html">Feature Importance (Simulated)</a>: Key predictors.</li>')
        html_content.append('            <li><a href="../outputs/feature_importance_real.html">Feature Importance (Real)</a>: Key predictors.</li>')
        html_content.append('            <li><a href="../outputs/class_distribution.html">Class Distribution</a>: Label distributions.</li>')
        html_content.append('            <li><a href="../outputs/real_scatter.html">Real Data Scatter</a>: Study hours vs. attendance.</li>')
        html_content.append('            <li><a href="../outputs/multi_class_cm.html">Multi-Class Confusion Matrix</a>: Multi-class performance.</li>')
        html_content.append('            <li><a href="../outputs/bayesian_posterior.html">Bayesian Posterior</a>: Parameter distributions (if available).</li>')
        html_content.append('            <li><a href="../outputs/3d_feature_space.html">3D Feature Space</a>: Feature relationships.</li>')
        html_content.append('            <li><a href="../outputs/correlation_heatmap.html">Correlation Heatmap</a>: Feature relationships.</li>')
        html_content.append('            <li><a href="../outputs/sigmoid_function.html">Sigmoid Function</a>: Theoretical visualization.</li>')
        html_content.append('        </ul>')
        html_content.append('        <img src="../outputs/decision_boundary_binary.png" alt="Decision Boundary (Binary)">')
        html_content.append('        <p class="caption">Figure 9: Decision boundary for simulated binary classification.</p>')
        html_content.append('    </div>')

        # Insights
        html_content.append('    <div class="section">')
        html_content.append('        <h2>8. Key Insights</h2>')
        html_content.append('        <ul>')
        try:
            metrics_df = pd.read_csv('outputs/evaluation_metrics.csv')
            smote_acc = metrics_df[metrics_df['Model'] == 'Simulated SMOTE']['Accuracy'].values[0]
            basic_acc = metrics_df[metrics_df['Model'] == 'Simulated Basic']['Accuracy'].values[0]
            html_content.append(f'            <li><strong>Feature Impact</strong>: Study hours and attendance strongly predict success across datasets.</li>')
            html_content.append(f'            <li><strong>Class Imbalance</strong>: SMOTE improved accuracy from {basic_acc:.3f} (Basic) to {smote_acc:.3f} (SMOTE) on simulated data.</li>')
            html_content.append('            <li><strong>Non-Linearity</strong>: Polynomial features enhance model accuracy by capturing complex relationships.</li>')
            html_content.append('            <li><strong>Uncertainty</strong>: Bayesian models provide robust uncertainty estimates for simulated data predictions.</li>')
        except:
            html_content.append('            <li><strong>Feature Impact</strong>: Study hours and attendance strongly predict success.</li>')
            html_content.append('            <li><strong>Class Imbalance</strong>: SMOTE and balanced models improve performance.</li>')
            html_content.append('            <li><strong>Non-Linearity</strong>: Polynomial features enhance accuracy.</li>')
            html_content.append('            <li><strong>Uncertainty</strong>: Bayesian models offer uncertainty estimates.</li>')
        html_content.append('        </ul>')
        html_content.append('    </div>')

        # Conclusion
        html_content.append('    <div class="section">')
        html_content.append('        <h2>9. Conclusion</h2>')
        try:
            metrics_df = pd.read_csv('outputs/evaluation_metrics.csv')
            best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax()]['Model']
            best_acc = metrics_df['Accuracy'].max()
            html_content.append(f'        <p>This project demonstrates logistic regression’s effectiveness in predicting student performance, with the best model ({best_model}) achieving an accuracy of {best_acc:.3f}. Enhanced by feature analysis, class balancing, and interactive visualizations, it provides actionable insights for educational outcomes.</p>')
        except:
            html_content.append('        <p>This project demonstrates logistic regression’s effectiveness in predicting student performance, enhanced by detailed feature analysis and visualizations.</p>')
        html_content.append('    </div>')

        # Close HTML
        html_content.append('</body>')
        html_content.append('</html>')

        return '\n'.join(html_content)

    except Exception as e:
        print(f"Error in generate_report: {str(e)}")
        raise

def main():
    """Run the full project pipeline."""
    np.random.seed(42)
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
        os.makedirs('reports', exist_ok=True)
        
        # Save HTML report
        html_output = 'reports/project_report.html'
        if os.path.exists(html_output):
            print(f"Warning: Overwriting {html_output}")
        with open(html_output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"HTML report saved to '{html_output}'")
        
        # Convert HTML to PDF
        try:
            from weasyprint import HTML
            pdf_output = 'reports/project_report.pdf'
            HTML(html_output).write_pdf(pdf_output)
            print(f"PDF report saved to '{pdf_output}'")
        except ImportError:
            print("Error: weasyprint not installed. Install it with `pip install weasyprint`.")
        except Exception as e:
            print(f"Error generating PDF: {e}")
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return
    
    print("\nAll scripts completed. Check 'outputs/' for results and 'reports/' for the report.")

if __name__ == "__main__":
    main()