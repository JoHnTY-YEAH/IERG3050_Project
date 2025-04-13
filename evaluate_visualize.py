import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
import pickle
import os
from matplotlib.colors import ListedColormap

def load_models_and_data():
    """Load trained models and test data."""
    try:
        with open('outputs/basic_model_sim.pkl', 'rb') as f:
            basic_model_sim = pickle.load(f)
        with open('outputs/reg_model_sim.pkl', 'rb') as f:
            reg_model_sim = pickle.load(f)
        with open('outputs/balanced_model_sim.pkl', 'rb') as f:
            balanced_model_sim = pickle.load(f)
        with open('outputs/basic_model_real.pkl', 'rb') as f:
            basic_model_real = pickle.load(f)
        with open('outputs/reg_model_real.pkl', 'rb') as f:
            reg_model_real = pickle.load(f)
        with open('outputs/balanced_model_real.pkl', 'rb') as f:
            balanced_model_real = pickle.load(f)
        X_sim_test = np.load('outputs/X_sim_test.npy')
        y_sim_test = np.load('outputs/y_sim_test.npy')
        X_real_test = np.load('outputs/X_real_test.npy')
        y_real_test = np.load('outputs/y_real_test.npy')
        real_data = pd.read_csv('outputs/cleaned_real_data.csv')  # Load for scatter plot
        return (basic_model_sim, reg_model_sim, balanced_model_sim,
                basic_model_real, reg_model_real, balanced_model_real,
                X_sim_test, y_sim_test, X_real_test, y_real_test, real_data)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

def evaluate_model(model, X_test, y_test, model_name, metrics_list):
    """Evaluate model performance with multiple metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Append metrics to list
    metrics_list.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'ROC-AUC': auc
    })
    
    return y_pred, model.predict_proba(X_test)[:, 1]

def plot_decision_boundary(model, X, y, title, filename):
    """Plot decision boundary using two features."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']))
    plt.xlabel('Study Hours (Scaled)')
    plt.ylabel('Attendance (Scaled)')
    plt.title(title)
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')
    plt.close()

def plot_roc_curve(models, X_test, y_test, labels, title, filename):
    """Plot ROC curves for multiple models."""
    plt.figure()
    for model, label in zip(models, labels):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')
    plt.close()

def plot_feature_importance(model, title, filename):
    """Plot feature importance based on model coefficients."""
    coef = model.coef_[0]
    features = ['Study Hours', 'Sleep Hours', 'Attendance']
    plt.figure()
    sns.barplot(x=coef, y=features)
    plt.title(title)
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')
    plt.close()

def plot_class_distribution(y_sim, y_real, filename):
    """Plot bar plot of pass/fail counts for simulated and real data."""
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_sim, hue=y_sim, palette=['#FF0000', '#0000FF'], legend=False)
    plt.title('Simulated Data Class Distribution')
    plt.xlabel('Pass/Fail (0=Fail, 1=Pass)')
    plt.subplot(1, 2, 2)
    sns.countplot(x=y_real, hue=y_real, palette=['#FF0000', '#0000FF'], legend=False)
    plt.title('Real Data Class Distribution')
    plt.xlabel('Pass/Fail (0=Fail, 1=Pass)')
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')
    plt.close()

def plot_real_scatter(data, filename):
    """Plot scatter of study_hours vs. attendance for real data, colored by pass_fail."""
    plt.figure()
    sns.scatterplot(data=data, x='study_hours', y='attendance', hue='pass_fail',
                    palette=['#FF0000', '#0000FF'], alpha=0.6)
    plt.title('Real Data: Study Hours vs. Attendance')
    plt.xlabel('Study Hours')
    plt.ylabel('Attendance (%)')
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')
    plt.close()

def main():
    """Main function to evaluate models and generate visualizations."""
    # Load models and data
    data = load_models_and_data()
    if data is None:
        return
    (basic_model_sim, reg_model_sim, balanced_model_sim,
     basic_model_real, reg_model_real, balanced_model_real,
     X_sim_test, y_sim_test, X_real_test, y_real_test, real_data) = data
    
    # List to store metrics
    metrics_list = []
    
    # Evaluate models
    evaluate_model(basic_model_sim, X_sim_test, y_sim_test, "Simulated Basic", metrics_list)
    evaluate_model(reg_model_sim, X_sim_test, y_sim_test, "Simulated Regularized", metrics_list)
    evaluate_model(balanced_model_sim, X_sim_test, y_sim_test, "Simulated Balanced", metrics_list)
    evaluate_model(basic_model_real, X_real_test, y_real_test, "Real Basic", metrics_list)
    evaluate_model(reg_model_real, X_real_test, y_real_test, "Real Regularized", metrics_list)
    evaluate_model(balanced_model_real, X_real_test, y_real_test, "Real Balanced", metrics_list)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv('outputs/evaluation_metrics.csv', index=False)
    print("\nMetrics saved to 'outputs/evaluation_metrics.csv'")
    
    # Train model for decision boundary (2 features: study_hours, attendance)
    X_sim_2d = X_sim_test[:, [0, 2]]  # Study hours, attendance
    model_2d = LogisticRegression().fit(X_sim_2d, y_sim_test)
    plot_decision_boundary(model_2d, X_sim_2d, y_sim_test,
                          'Decision Boundary (Simulated Data)',
                          'decision_boundary.png')
    
    # Plot ROC curves
    plot_roc_curve([basic_model_sim, reg_model_sim, balanced_model_sim],
                   X_sim_test, y_sim_test,
                   ['Basic', 'Regularized', 'Balanced'],
                   'ROC Curve (Simulated Data)',
                   'roc_curve_sim.png')
    plot_roc_curve([basic_model_real, reg_model_real, balanced_model_real],
                   X_real_test, y_real_test,
                   ['Basic', 'Regularized', 'Balanced'],
                   'ROC Curve (Real Data)',
                   'roc_curve_real.png')
    
    # Plot feature importance (use regularized models)
    plot_feature_importance(reg_model_sim,
                           'Feature Importance (Simulated Regularized)',
                           'feature_importance_sim.png')
    plot_feature_importance(reg_model_real,
                           'Feature Importance (Real Regularized)',
                           'feature_importance_real.png')
    
    # Plot class distribution
    plot_class_distribution(y_sim_test, y_real_test, 'class_distribution.png')
    
    # Plot real data scatter
    plot_real_scatter(real_data, 'real_scatter.png')
    
    print("\nVisualizations saved to 'outputs/'")

if __name__ == "__main__":
    main()