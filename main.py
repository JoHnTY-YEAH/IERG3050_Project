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
    Generate a detailed HTML report with embedded graphs and extended statistics.
    
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
        html_content.append('        body {')
        html_content.append('            font-family: Arial, sans-serif;')
        html_content.append('            margin: 40px;')
        html_content.append('            line-height: 1.6;')
        html_content.append('        }')
        html_content.append('        h1, h2, h3 {')
        html_content.append('            color: #2c3e50;')
        html_content.append('        }')
        html_content.append('        h1 {')
        html_content.append('            text-align: center;')
        html_content.append('        }')
        html_content.append('        table {')
        html_content.append('            width: 100%;')
        html_content.append('            border-collapse: collapse;')
        html_content.append('            margin: 20px 0;')
        html_content.append('        }')
        html_content.append('        th, td {')
        html_content.append('            border: 1px solid #ddd;')
        html_content.append('            padding: 8px;')
        html_content.append('            text-align: left;')
        html_content.append('        }')
        html_content.append('        th {')
        html_content.append('            background-color: #f2f2f2;')
        html_content.append('        }')
        html_content.append('        tr:nth-child(even) {')
        html_content.append('            background-color: #f9f9f9;')
        html_content.append('        }')
        html_content.append('        .highlight {')
        html_content.append('            background-color: #d4edda;')
        html_content.append('        }')
        html_content.append('        ul {')
        html_content.append('            margin: 10px 0;')
        html_content.append('        }')
        html_content.append('        a {')
        html_content.append('            color: #007bff;')
        html_content.append('            text-decoration: none;')
        html_content.append('        }')
        html_content.append('        a:hover {')
        html_content.append('            text-decoration: underline;')
        html_content.append('        }')
        html_content.append('        .section {')
        html_content.append('            margin-bottom: 30px;')
        html_content.append('        }')
        html_content.append('        .error {')
        html_content.append('            color: #dc3545;')
        html_content.append('            font-style: italic;')
        html_content.append('        }')
        html_content.append('        img {')
        html_content.append('            max-width: 100%;')
        html_content.append('            height: auto;')
        html_content.append('            display: block;')
        html_content.append('            margin: 10px 0;')
        html_content.append('        }')
        html_content.append('        .caption {')
        html_content.append('            text-align: center;')
        html_content.append('            font-style: italic;')
        html_content.append('            margin-bottom: 20px;')
        html_content.append('        }')
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
        html_content.append('        <p>This report provides a comprehensive analysis of student performance using logistic regression and other machine learning models. It includes simulated and real data analysis, model performance metrics, feature statistics, and embedded visualizations for deeper insights.</p>')
        html_content.append('    </div>')

        # Theoretical Foundation
        html_content.append('    <div class="section">')
        html_content.append('        <h2>2. Theoretical Foundation</h2>')
        html_content.append('        <ul>')
        html_content.append('            <li><strong>Sigmoid Function</strong>: Maps linear predictors to probabilities between 0 and 1, enabling binary classification.</li>')
        html_content.append('            <li><strong>Cross-Entropy Loss</strong>: Optimized via gradient descent to minimize prediction errors.</li>')
        html_content.append('            <li><strong>Regularization</strong>: L1 (Lasso) and L2 (Ridge) penalties control model complexity to prevent overfitting.</li>')
        html_content.append('        </ul>')
        html_content.append('    </div>')

        # Feature Analysis
        html_content.append('    <div class="section">')
        html_content.append('        <h2>3. Feature Analysis</h2>')
        
        # Model Coefficients
        html_content.append('        <h3>3.1 Model Coefficients</h3>')
        try:
            html_content.append('        <table>')
            html_content.append('            <tr>')
            html_content.append('                <th>Model</th>')
            html_content.append('                <th>Study Hours</th>')
            html_content.append('                <th>Sleep Hours</th>')
            html_content.append('                <th>Attendance</th>')
            html_content.append('            </tr>')
            for model_name, file_name in [('Real Data', 'reg_model_real.pkl'), ('Simulated Data', 'reg_model_sim.pkl')]:
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
        except Exception as e:
            html_content.append(f'        <p class="error">Correlation analysis unavailable: {str(e)}</p>')
        html_content.append('    </div>')

        # Class Distribution
        html_content.append('    <div class="section">')
        html_content.append('        <h2>4. Class Distribution</h2>')
        try:
            y_sim = np.load('outputs/y_sim_test.npy')
            y_real = np.load('outputs/y_real_test.npy')
            for dataset_name, y in [('Simulated Data', y_sim), ('Real Data', y_real)]:
                counts = pd.Series(y).value_counts(normalize=True)
                html_content.append(f'        <h3>{dataset_name}</h3>')
                html_content.append('        <ul>')
                for label, prop in counts.items():
                    html_content.append(f'            <li>Class {int(label)}: {prop:.2%}</li>')
                html_content.append('        </ul>')
            html_content.append('        <img src="class_distribution.png" alt="Class Distribution">')
            html_content.append('        <p class="caption">Figure 1: Distribution of pass/fail labels across datasets.</p>')
        except Exception as e:
            html_content.append(f'        <p class="error">Class distribution unavailable: {str(e)}</p>')
        html_content.append('    </div>')

        # Key Findings
        html_content.append('    <div class="section">')
        html_content.append('        <h2>5. Key Findings</h2>')
        try:
            with open('outputs/reg_model_real.pkl', 'rb') as f:
                model = pickle.load(f)
            coef = model.coef_[0]
            features = ['study_hours', 'sleep_hours', 'attendance']
            top_idx = np.argmax(np.abs(coef))
            html_content.append(f'        <p><strong>Top Predictor</strong>: {features[top_idx]} (coefficient: {coef[top_idx]:.3f})</p>')
            html_content.append('        <img src="feature_importance_real.png" alt="Feature Importance (Real)">')
            html_content.append('        <p class="caption">Figure 2: Feature importance for real data model.</p>')
        except Exception as e:
            html_content.append(f'        <p class="error">Feature importance unavailable: {str(e)}</p>')
        html_content.append('    </div>')

        # Model Performance
        html_content.append('    <div class="section">')
        html_content.append('        <h2>6. Model Performance</h2>')
        try:
            metrics_df = pd.read_csv('outputs/evaluation_metrics.csv')
            html_content.append('        <table>')
            html_content.append('            <tr>')
            html_content.append('                <th>Model</th>')
            html_content.append('                <th>Accuracy</th>')
            html_content.append('                <th>F1-Score</th>')
            html_content.append('                <th>ROC-AUC</th>')
            html_content.append('            </tr>')
            for _, row in metrics_df.iterrows():
                row_class = 'highlight' if row['Accuracy'] >= 0.80 else ''
                roc_auc = row['ROC-AUC'] if pd.notna(row['ROC-AUC']) else 'N/A'
                html_content.append(f'            <tr class="{row_class}">')
                html_content.append(f'                <td>{row["Model"]}</td>')
                html_content.append(f'                <td>{row["Accuracy"]:.3f}</td>')
                html_content.append(f'                <td>{row["F1-Score"]:.3f}</td>')
                html_content.append(f'                <td>{roc_auc}</td>')
                html_content.append('            </tr>')
            html_content.append('        </table>')
            accuracies = metrics_df['Accuracy']
            if accuracies.max() >= 0.80:
                html_content.append('        <p><strong>Accuracy Goal</strong>: Achieved (≥80% for at least one model).</p>')
            else:
                html_content.append('        <p><strong>Accuracy Goal</strong>: Below 80%. Consider addressing class imbalance or enhancing feature engineering.</p>')
            html_content.append('        <img src="roc_curve_sim.png" alt="ROC Curve (Simulated)">')
            html_content.append('        <p class="caption">Figure 3: ROC curves for simulated data models.</p>')
        except FileNotFoundError:
            html_content.append('        <p class="error">Error: Metrics file not found. Run evaluate_visualize.py first.</p>')
        except Exception as e:
            html_content.append(f'        <p class="error">Error loading metrics: {str(e)}</p>')
        html_content.append('    </div>')

        # Visualizations
        html_content.append('    <div class="section">')
        html_content.append('        <h2>7. Interactive Visualizations</h2>')
        html_content.append('        <p>Additional interactive visualizations are available in the <code>outputs/</code> directory:</p>')
        html_content.append('        <ul>')
        html_content.append('            <li><a href="decision_boundary_binary.html">Decision Boundary (Simulated Binary)</a>: Classification regions.</li>')
        html_content.append('            <li><a href="decision_boundary_poly.html">Decision Boundary (Polynomial)</a>: Non-linear boundaries.</li>')
        html_content.append('            <li><a href="decision_boundary_multi.html">Decision Boundary (Multi-Class)</a>: Multi-class regions.</li>')
        html_content.append('            <li><a href="roc_curve_sim.html">ROC Curve (Simulated)</a>: Model performance.</li>')
        html_content.append('            <li><a href="roc_curve_real.html">ROC Curve (Real)</a>: Model performance.</li>')
        html_content.append('            <li><a href="feature_importance_sim.html">Feature Importance (Simulated)</a>: Key predictors.</li>')
        html_content.append('            <li><a href="feature_importance_real.html">Feature Importance (Real)</a>: Key predictors.</li>')
        html_content.append('            <li><a href="class_distribution.html">Class Distribution</a>: Label distributions.</li>')
        html_content.append('            <li><a href="real_scatter.html">Real Data Scatter</a>: Study hours vs. attendance.</li>')
        html_content.append('            <li><a href="multi_class_cm.html">Multi-Class Confusion Matrix</a>: Multi-class performance.</li>')
        html_content.append('            <li><a href="bayesian_posterior.html">Bayesian Posterior</a>: Parameter distributions (if available).</li>')
        html_content.append('            <li><a href="3d_feature_space.html">3D Feature Space</a>: Feature relationships.</li>')
        html_content.append('        </ul>')
        html_content.append('        <img src="decision_boundary_binary.png" alt="Decision Boundary (Binary)">')
        html_content.append('        <p class="caption">Figure 4: Decision boundary for simulated binary classification.</p>')
        html_content.append('    </div>')

        # Insights
        html_content.append('    <div class="section">')
        html_content.append('        <h2>8. Key Insights</h2>')
        html_content.append('        <ul>')
        html_content.append('            <li><strong>Feature Impact</strong>: Study hours and attendance strongly predict success.</li>')
        html_content.append('            <li><strong>Class Imbalance</strong>: SMOTE and balanced models improve performance.</li>')
        html_content.append('            <li><strong>Non-Linearity</strong>: Polynomial features enhance accuracy.</li>')
        html_content.append('            <li><strong>Uncertainty</strong>: Bayesian models offer uncertainty estimates.</li>')
        html_content.append('        </ul>')
        html_content.append('    </div>')

        # Conclusion
        html_content.append('    <div class="section">')
        html_content.append('        <h2>9. Conclusion</h2>')
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
        os.makedirs('outputs', exist_ok=True)
        
        # Save HTML report (optional, for reference)
        html_output = 'outputs/project_report.html'
        if os.path.exists(html_output):
            print(f"Warning: Overwriting {html_output}")
        with open(html_output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"HTML report saved to '{html_output}'")
        
        # Convert HTML to PDF
        try:
            from weasyprint import HTML
            pdf_output = 'outputs/project_report.pdf'
            HTML(html_output).write_pdf(pdf_output)
            print(f"PDF report saved to '{pdf_output}'")
        except ImportError:
            print("Error: weasyprint not installed. Install it with `pip install weasyprint`.")
        except Exception as e:
            print(f"Error generating PDF: {e}")
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return
    
    print("\nAll scripts completed. Check 'outputs/' for results and report.")

if __name__ == "__main__":
    main()