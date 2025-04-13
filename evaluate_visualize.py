import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle
import os
from matplotlib.colors import ListedColormap

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

def load_models_and_data():
    """
    Load trained models and test data.
    
    Returns:
        tuple: Models, data, and objects, or None if error.
    """
    required_files = [
        'outputs/basic_model_sim.pkl', 'outputs/reg_model_sim.pkl',
        'outputs/balanced_model_sim.pkl', 'outputs/smote_model_sim.pkl',
        'outputs/poly_model_sim.pkl', 'outputs/basic_model_real.pkl',
        'outputs/reg_model_real.pkl',
        'outputs/balanced_model_real.pkl', 'outputs/smote_model_real.pkl',
        'outputs/poly_model_real.pkl', 'outputs/poly_sim.pkl',
        'outputs/poly_real.pkl', 'outputs/basic_model_sim_multi.pkl',
        'outputs/reg_model_sim_multi.pkl', 'outputs/balanced_model_sim_multi.pkl',
        'outputs/smote_model_sim_multi.pkl', 'outputs/poly_model_sim_multi.pkl',
        'outputs/poly_sim_multi.pkl', 'outputs/X_sim_test.npy',
        'outputs/y_sim_test.npy', 'outputs/X_real_test.npy',
        'outputs/y_real_test.npy', 'outputs/X_sim_multi_test.npy',
        'outputs/y_sim_multi_test.npy', 'outputs/cleaned_real_data.csv'
    ]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Missing file '{f}'. Run train_model.py first.")
            return None
    
    try:
        with open('outputs/basic_model_sim.pkl', 'rb') as f:
            basic_model_sim = pickle.load(f)
        with open('outputs/reg_model_sim.pkl', 'rb') as f:
            reg_model_sim = pickle.load(f)
        with open('outputs/balanced_model_sim.pkl', 'rb') as f:
            balanced_model_sim = pickle.load(f)
        with open('outputs/smote_model_sim.pkl', 'rb') as f:
            smote_model_sim = pickle.load(f)
        with open('outputs/poly_model_sim.pkl', 'rb') as f:
            poly_model_sim = pickle.load(f)
        with open('outputs/basic_model_real.pkl', 'rb') as f:
            basic_model_real = pickle.load(f)
        with open('outputs/reg_model_real.pkl', 'rb') as f:
            reg_model_real = pickle.load(f)
        with open('outputs/balanced_model_real.pkl', 'rb') as f:
            balanced_model_real = pickle.load(f)
        with open('outputs/smote_model_real.pkl', 'rb') as f:
            smote_model_real = pickle.load(f)
        with open('outputs/poly_model_real.pkl', 'rb') as f:
            poly_model_real = pickle.load(f)
        with open('outputs/poly_sim.pkl', 'rb') as f:
            poly_sim = pickle.load(f)
        with open('outputs/poly_real.pkl', 'rb') as f:
            poly_real = pickle.load(f)
        with open('outputs/basic_model_sim_multi.pkl', 'rb') as f:
            basic_model_sim_multi = pickle.load(f)
        with open('outputs/reg_model_sim_multi.pkl', 'rb') as f:
            reg_model_sim_multi = pickle.load(f)
        with open('outputs/balanced_model_sim_multi.pkl', 'rb') as f:
            balanced_model_sim_multi = pickle.load(f)
        with open('outputs/smote_model_sim_multi.pkl', 'rb') as f:
            smote_model_sim_multi = pickle.load(f)
        with open('outputs/poly_model_sim_multi.pkl', 'rb') as f:
            poly_model_sim_multi = pickle.load(f)
        with open('outputs/poly_sim_multi.pkl', 'rb') as f:
            poly_sim_multi = pickle.load(f)
        bayesian_sim = None
        if os.path.exists('outputs/bayesian_sim.nc') and PYMC_AVAILABLE:
            bayesian_sim = az.from_netcdf('outputs/bayesian_sim.nc')
        dl_real = None
        if os.path.exists('outputs/dl_real.keras') and TF_AVAILABLE:
            dl_real = tf.keras.models.load_model('outputs/dl_real.keras')
        X_sim_test = np.load('outputs/X_sim_test.npy')
        y_sim_test = np.load('outputs/y_sim_test.npy')
        X_real_test = np.load('outputs/X_real_test.npy')
        y_real_test = np.load('outputs/y_real_test.npy')
        X_sim_multi_test = np.load('outputs/X_sim_multi_test.npy')
        y_sim_multi_test = np.load('outputs/y_sim_multi_test.npy')
        real_data = pd.read_csv('outputs/cleaned_real_data.csv')
        return (basic_model_sim, reg_model_sim, balanced_model_sim, smote_model_sim,
                poly_model_sim, basic_model_real, reg_model_real, balanced_model_real,
                smote_model_real, poly_model_real, basic_model_sim_multi,
                reg_model_sim_multi, balanced_model_sim_multi, smote_model_sim_multi,
                poly_model_sim_multi, bayesian_sim, dl_real, X_sim_test, y_sim_test,
                X_real_test, y_real_test, X_sim_multi_test, y_sim_multi_test,
                poly_sim, poly_real, poly_sim_multi, real_data)
    except Exception as e:
        print(f"Error loading models/data: {e}")
        return None

def evaluate_model(model, X_test, y_test, model_name, metrics_list, deep_learning=False, bayesian=False, poly=None):
    """
    Evaluate a model and append metrics to the provided list.

    Parameters:
    - model: Trained model (scikit-learn, keras, or arviz InferenceData).
    - X_test: Test features (numpy array).
    - y_test: Test labels (numpy array).
    - model_name: Name of the model (str).
    - metrics_list: List to store metrics (list of dicts).
    - deep_learning: Flag for deep learning model (bool).
    - bayesian: Flag for Bayesian model (bool).
    - poly: Polynomial transformer for polynomial models (PolynomialFeatures or None).

    Returns:
    - Updated metrics_list with evaluation results.
    """
    print(f"\n{model_name} Results:")
    
    # Apply polynomial transformation if provided
    X_test_input = poly.transform(X_test) if poly is not None else X_test
    
    if bayesian:
        # Bayesian model: Use posterior predictive samples
        posterior = model.posterior_predictive["y"].mean(("chain", "draw"))
        y_pred = (posterior > 0.5).astype(int).values.flatten()
        y_prob = posterior.values.flatten() if posterior is not None else None
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None and len(np.unique(y_test)) == 2 else 'N/A'
        cm = confusion_matrix(y_test, y_pred)
    else:
        # Non-Bayesian models (scikit-learn or deep learning)
        if deep_learning:
            # Deep learning: Handle sigmoid (1D) output
            y_prob = model.predict(X_test_input, verbose=0).flatten()  # 1D probabilities
            y_pred = (y_prob > 0.5).astype(int).flatten()  # Binary threshold
        else:
            # Scikit-learn models
            y_pred = model.predict(X_test_input)
            y_prob = model.predict_proba(X_test_input) if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Compute ROC-AUC for binary classification
        auc = 'N/A'
        if y_prob is not None and len(np.unique(y_test)) == 2:
            if y_prob.ndim == 2:  # Softmax output (e.g., scikit-learn)
                auc = roc_auc_score(y_test, y_prob[:, 1])
            else:  # Sigmoid output (deep learning)
                auc = roc_auc_score(y_test, y_prob)
        
        cm = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"ROC-AUC: {auc if auc != 'N/A' else 'N/A'}")
    print(f"Confusion Matrix:\n{cm}")

    # Append metrics to list
    metrics_list.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'ROC-AUC': auc
    })

    return metrics_list

def compare_optimizers(X_train, y_train, X_test, y_test):
    """
    Compare different optimization algorithms.
    
    Args:
        X_train, y_train, X_test, y_test: Training and test data.
    
    Returns:
        pd.DataFrame: Results.
    """
    solvers = ['lbfgs', 'newton-cg', 'sag', 'saga']
    results = []
    
    for solver in solvers:
        model = LogisticRegression(solver=solver, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        results.append({'Solver': solver, 'Accuracy': acc})
    
    df = pd.DataFrame(results)
    plt.figure()
    sns.barplot(x='Solver', y='Accuracy', data=df)
    plt.title('Optimizer Comparison (Higher is Better)')
    plt.savefig('outputs/optimizer_comparison.png')
    plt.close()
    return df

def plot_decision_boundary(model, X, y, title, filename, poly=None):
    """
    Plot decision boundary using two features.
    
    Args:
        model: Trained model.
        X: Features.
        y: Labels.
        title (str): Plot title.
        filename (str): Output file name.
        poly: Polynomial transformer, if any.
    """
    X_2d = X[:, [0, 2]]  # study_hours, attendance
    if poly is not None:
        X_2d_poly = poly.fit_transform(X_2d)
        model.fit(X_2d_poly, y)
    else:
        X_2d_poly = X_2d
        model.fit(X_2d, y)
    
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if poly is not None:
        grid = poly.transform(grid)
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    plt.figure()
    cmap = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'][:len(np.unique(y))])
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF', '#00FF00'][:len(np.unique(y))]))
    plt.xlabel('Study Hours (Scaled)')
    plt.ylabel('Attendance (Scaled)')
    plt.title(title)
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')
    plt.close()

def plot_roc_curve(models, X_test, y_test, labels, title, filename, poly=None):
    """
    Plot ROC curves for multiple models (binary only).
    
    Args:
        models: List of trained models.
        X_test: Test features.
        y_test: Test labels.
        labels: Model names.
        title (str): Plot title.
        filename (str): Output file name.
        poly: Polynomial transformer, if any.
    """
    plt.figure()
    for idx, (model, label) in enumerate(zip(models, labels)):
        # Apply polynomial transformation only for the polynomial model
        X_test_input = poly.transform(X_test) if poly is not None and idx == 4 else X_test
        y_prob = model.predict_proba(X_test_input)[:, 1]
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
    """
    Plot feature importance based on model coefficients.
    
    Args:
        model: Trained model.
        title (str): Plot title.
        filename (str): Output file name.
    """
    coef = np.mean(model.coef_, axis=0) if model.coef_.ndim > 1 else model.coef_
    features = ['Study Hours', 'Sleep Hours', 'Attendance']
    plt.figure()
    sns.barplot(x=coef, y=features)
    plt.title(title)
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')
    plt.close()

def plot_class_distribution(y_sim, y_real, y_sim_multi, filename):
    """
    Plot bar plot of class counts for simulated, real, and multi-class data.
    
    Args:
        y_sim, y_real, y_sim_multi: Labels.
        filename (str): Output file name.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    sns.countplot(x=y_sim, hue=y_sim, palette=['#FF0000', '#0000FF'], legend=False)
    plt.title('Simulated Data (Binary)')
    plt.xlabel('Pass/Fail (0=Fail, 1=Pass)')
    plt.subplot(1, 3, 2)
    sns.countplot(x=y_real, hue=y_real, palette=['#FF0000', '#0000FF'], legend=False)
    plt.title('Real Data (Binary)')
    plt.xlabel('Pass/Fail (0=Fail, 1=Pass)')
    plt.subplot(1, 3, 3)
    sns.countplot(x=y_sim_multi, hue=y_sim_multi, palette=['#FF0000', '#0000FF', '#00FF00'], legend=False)
    plt.title('Simulated Data (Multi-Class)')
    plt.xlabel('Grade (0=Fail, 1=Pass, 2=Excellent)')
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')
    plt.close()

def plot_real_scatter(data, filename):
    """
    Plot scatter of study_hours vs. attendance for real data.
    
    Args:
        data (pd.DataFrame): Real data.
        filename (str): Output file name.
    """
    plt.figure()
    sns.scatterplot(data=data, x='study_hours', y='attendance', hue='pass_fail',
                    palette=['#FF0000', '#0000FF'], alpha=0.6)
    plt.title('Real Data: Study Hours vs. Attendance')
    plt.xlabel('Study Hours')
    plt.ylabel('Attendance (%)')
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')
    plt.close()

def plot_multi_class_cm(y_true, y_pred, title, filename):
    """
    Plot confusion matrix for multi-class.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        title (str): Plot title.
        filename (str): Output file name.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fail', 'Pass', 'Excellent'],
                yticklabels=['Fail', 'Pass', 'Excellent'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')
    plt.close()

def plot_bayesian_posterior(trace, filename):
    """
    Plot posterior distributions for Bayesian model.
    
    Args:
        trace: Posterior samples.
        filename (str): Output file name.
    """
    if trace is None or not PYMC_AVAILABLE:
        print("No Bayesian posterior plot generated.")
        return
    plt.figure(figsize=(10, 6))
    for i, var in enumerate(['intercept', 'beta[0]', 'beta[1]', 'beta[2]']):
        plt.subplot(2, 2, i+1)
        sns.histplot(trace.posterior[var].values.flatten(), kde=True)
        plt.title(var)
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')
    plt.close()

def plot_3d_feature_space(X, y):
    """
    3D visualization of feature relationships.
    
    Args:
        X: Features.
        y: Labels.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']))
    ax.set_xlabel('Study Hours')
    ax.set_ylabel('Sleep Hours')
    ax.set_zlabel('Attendance')
    plt.title('3D Feature Space Coloring by Pass/Fail')
    plt.savefig('outputs/3d_feature_space.png')
    plt.close()

def main():
    """Evaluate models and generate visualizations."""
    data = load_models_and_data()
    if data is None:
        return
    (basic_model_sim, reg_model_sim, balanced_model_sim, smote_model_sim,
     poly_model_sim, basic_model_real, reg_model_real, balanced_model_real,
     smote_model_real, poly_model_real, basic_model_sim_multi,
     reg_model_sim_multi, balanced_model_sim_multi, smote_model_sim_multi,
     poly_model_sim_multi, bayesian_sim, dl_real, X_sim_test, y_sim_test,
     X_real_test, y_real_test, X_sim_multi_test, y_sim_multi_test,
     poly_sim, poly_real, poly_sim_multi, real_data) = data
    
    metrics_list = []
    
    evaluate_model(basic_model_sim, X_sim_test, y_sim_test, "Simulated Basic", metrics_list)
    evaluate_model(reg_model_sim, X_sim_test, y_sim_test, "Simulated Regularized", metrics_list)
    evaluate_model(balanced_model_sim, X_sim_test, y_sim_test, "Simulated Balanced", metrics_list)
    evaluate_model(smote_model_sim, X_sim_test, y_sim_test, "Simulated SMOTE", metrics_list)
    evaluate_model(poly_model_sim, X_sim_test, y_sim_test, "Simulated Polynomial", metrics_list, poly=poly_sim)
    if bayesian_sim is not None:
        evaluate_model(bayesian_sim, X_sim_test, y_sim_test, "Simulated Bayesian", metrics_list, bayesian=True)
    evaluate_model(basic_model_real, X_real_test, y_real_test, "Real Basic", metrics_list)
    evaluate_model(reg_model_real, X_real_test, y_real_test, "Real Regularized", metrics_list)
    evaluate_model(balanced_model_real, X_real_test, y_real_test, "Real Balanced", metrics_list)
    evaluate_model(smote_model_real, X_real_test, y_real_test, "Real SMOTE", metrics_list)
    evaluate_model(poly_model_real, X_real_test, y_real_test, "Real Polynomial", metrics_list, poly=poly_real)
    if dl_real is not None:
        evaluate_model(dl_real, X_real_test, y_real_test, "Real Deep Learning", metrics_list, deep_learning=True)
    
    evaluate_model(basic_model_sim_multi, X_sim_multi_test, y_sim_multi_test,
                   "Simulated Multi-Class Basic", metrics_list)
    evaluate_model(reg_model_sim_multi, X_sim_multi_test, y_sim_multi_test,
                   "Simulated Multi-Class Regularized", metrics_list)
    evaluate_model(balanced_model_sim_multi, X_sim_multi_test, y_sim_multi_test,
                   "Simulated Multi-Class Balanced", metrics_list)
    evaluate_model(smote_model_sim_multi, X_sim_multi_test, y_sim_multi_test,
                   "Simulated Multi-Class SMOTE", metrics_list)
    evaluate_model(poly_model_sim_multi, X_sim_multi_test, y_sim_multi_test,
                   "Simulated Multi-Class Polynomial", metrics_list, poly=poly_sim_multi)
    
    metrics_df = pd.DataFrame(metrics_list)
    output_file = 'outputs/evaluation_metrics.csv'
    if os.path.exists(output_file):
        print(f"Warning: Overwriting {output_file}")
    metrics_df.to_csv(output_file, index=False)
    print(f"\nMetrics saved to '{output_file}'")
    
    model_2d = LogisticRegression(max_iter=1000).fit(X_sim_test[:, [0, 2]], y_sim_test)
    plot_decision_boundary(model_2d, X_sim_test, y_sim_test,
                          'Decision Boundary (Simulated Binary)',
                          'decision_boundary_binary.png')
    plot_decision_boundary(LogisticRegression(max_iter=1000), X_sim_test, y_sim_test,
                          'Decision Boundary (Simulated Polynomial)',
                          'decision_boundary_poly.png', poly=PolynomialFeatures(degree=2))
    plot_decision_boundary(LogisticRegression(max_iter=1000), X_sim_multi_test, y_sim_multi_test,
                          'Decision Boundary (Simulated Multi-Class)',
                          'decision_boundary_multi.png')
    
    plot_roc_curve([basic_model_sim, reg_model_sim, balanced_model_sim, smote_model_sim, poly_model_sim],
                   X_sim_test, y_sim_test,
                   ['Basic', 'Regularized', 'Balanced', 'SMOTE', 'Polynomial'],
                   'ROC Curve (Simulated Binary)',
                   'roc_curve_sim.png', poly=poly_sim)
    plot_roc_curve([basic_model_real, reg_model_real, balanced_model_real, smote_model_real, poly_model_real],
                   X_real_test, y_real_test,
                   ['Basic', 'Regularized', 'Balanced', 'SMOTE', 'Polynomial'],
                   'ROC Curve (Real Binary)',
                   'roc_curve_real.png', poly=poly_real)
    
    plot_feature_importance(reg_model_sim,
                           'Feature Importance (Simulated Regularized)',
                           'feature_importance_sim.png')
    plot_feature_importance(reg_model_real,
                           'Feature Importance (Real Regularized)',
                           'feature_importance_real.png')
    plot_feature_importance(reg_model_sim_multi,
                           'Feature Importance (Simulated Multi-Class)',
                           'feature_importance_sim_multi.png')
    
    plot_class_distribution(y_sim_test, y_real_test, y_sim_multi_test, 'class_distribution.png')
    
    plot_real_scatter(real_data, 'real_scatter.png')
    
    plot_multi_class_cm(y_sim_multi_test, poly_model_sim_multi.predict(poly_sim_multi.transform(X_sim_multi_test)),
                        'Confusion Matrix (Simulated Multi-Class Polynomial)',
                        'multi_class_cm.png')
    
    plot_bayesian_posterior(bayesian_sim, 'bayesian_posterior.png')
    
    plot_3d_feature_space(X_real_test, y_real_test)
    
    print("\nVisualizations saved to 'outputs/'")

if __name__ == "__main__":
    main()