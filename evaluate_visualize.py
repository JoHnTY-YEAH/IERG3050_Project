import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for fallback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle
import os
from matplotlib.colors import ListedColormap
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        'outputs/basic_model_sim.pkl', 'outputs/reg_model_sim.pkl', 'outputs/l1_model_sim.pkl',
        'outputs/balanced_model_sim.pkl', 'outputs/smote_model_sim.pkl', 'outputs/poly_model_sim.pkl',
        'outputs/dt_model_sim.pkl', 'outputs/basic_model_real.pkl', 'outputs/reg_model_real.pkl',
        'outputs/l1_model_real.pkl', 'outputs/balanced_model_real.pkl', 'outputs/smote_model_real.pkl',
        'outputs/poly_model_real.pkl', 'outputs/dt_model_real.pkl', 'outputs/poly_sim.pkl',
        'outputs/poly_real.pkl', 'outputs/basic_model_sim_multi.pkl', 'outputs/reg_model_sim_multi.pkl',
        'outputs/l1_model_sim_multi.pkl', 'outputs/balanced_model_sim_multi.pkl',
        'outputs/smote_model_sim_multi.pkl', 'outputs/poly_model_sim_multi.pkl',
        'outputs/dt_model_sim_multi.pkl', 'outputs/poly_sim_multi.pkl', 'outputs/X_sim_test.npy',
        'outputs/y_sim_test.npy', 'outputs/X_real_test.npy', 'outputs/y_real_test.npy',
        'outputs/X_sim_multi_test.npy', 'outputs/y_sim_multi_test.npy', 'outputs/cleaned_real_data.csv'
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
        with open('outputs/l1_model_sim.pkl', 'rb') as f:
            l1_model_sim = pickle.load(f)
        with open('outputs/balanced_model_sim.pkl', 'rb') as f:
            balanced_model_sim = pickle.load(f)
        with open('outputs/smote_model_sim.pkl', 'rb') as f:
            smote_model_sim = pickle.load(f)
        with open('outputs/poly_model_sim.pkl', 'rb') as f:
            poly_model_sim = pickle.load(f)
        with open('outputs/dt_model_sim.pkl', 'rb') as f:
            dt_model_sim = pickle.load(f)
        with open('outputs/basic_model_real.pkl', 'rb') as f:
            basic_model_real = pickle.load(f)
        with open('outputs/reg_model_real.pkl', 'rb') as f:
            reg_model_real = pickle.load(f)
        with open('outputs/l1_model_real.pkl', 'rb') as f:
            l1_model_real = pickle.load(f)
        with open('outputs/balanced_model_real.pkl', 'rb') as f:
            balanced_model_real = pickle.load(f)
        with open('outputs/smote_model_real.pkl', 'rb') as f:
            smote_model_real = pickle.load(f)
        with open('outputs/poly_model_real.pkl', 'rb') as f:
            poly_model_real = pickle.load(f)
        with open('outputs/dt_model_real.pkl', 'rb') as f:
            dt_model_real = pickle.load(f)
        with open('outputs/poly_sim.pkl', 'rb') as f:
            poly_sim = pickle.load(f)
        with open('outputs/poly_real.pkl', 'rb') as f:
            poly_real = pickle.load(f)
        with open('outputs/basic_model_sim_multi.pkl', 'rb') as f:
            basic_model_sim_multi = pickle.load(f)
        with open('outputs/reg_model_sim_multi.pkl', 'rb') as f:
            reg_model_sim_multi = pickle.load(f)
        with open('outputs/l1_model_sim_multi.pkl', 'rb') as f:
            l1_model_sim_multi = pickle.load(f)
        with open('outputs/balanced_model_sim_multi.pkl', 'rb') as f:
            balanced_model_sim_multi = pickle.load(f)
        with open('outputs/smote_model_sim_multi.pkl', 'rb') as f:
            smote_model_sim_multi = pickle.load(f)
        with open('outputs/poly_model_sim_multi.pkl', 'rb') as f:
            poly_model_sim_multi = pickle.load(f)
        with open('outputs/dt_model_sim_multi.pkl', 'rb') as f:
            dt_model_sim_multi = pickle.load(f)
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
        return (basic_model_sim, reg_model_sim, l1_model_sim, balanced_model_sim, smote_model_sim,
                poly_model_sim, dt_model_sim, basic_model_real, reg_model_real, l1_model_real,
                balanced_model_real, smote_model_real, poly_model_real, dt_model_real,
                basic_model_sim_multi, reg_model_sim_multi, l1_model_sim_multi,
                balanced_model_sim_multi, smote_model_sim_multi, poly_model_sim_multi,
                dt_model_sim_multi, bayesian_sim, dl_real, X_sim_test, y_sim_test,
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
        try:
            if hasattr(model, 'posterior_predictive') and hasattr(model.posterior_predictive, 'y'):
                posterior = model.posterior_predictive["y"].mean(("chain", "draw"))
                y_pred = (posterior > 0.5).astype(int).values.flatten()
                y_prob = posterior.values.flatten()
            else:
                print("Warning: No posterior predictive samples found. Using posterior mean for predictions.")
                intercept = model.posterior['intercept'].mean().values
                beta = model.posterior['beta'].mean(('chain', 'draw')).values
                logits = intercept + np.dot(X_test_input, beta)
                y_prob = 1 / (1 + np.exp(-logits))
                y_pred = (y_prob > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else 'N/A'
            cm = confusion_matrix(y_test, y_pred)
        except Exception as e:
            print(f"Error evaluating Bayesian model: {e}")
            return metrics_list
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
    # Use Plotly for interactive bar plot
    fig = px.bar(df, x='Solver', y='Accuracy', title='Optimizer Comparison (Higher is Better)',
                 labels={'Accuracy': 'Accuracy'}, color='Solver')
    os.makedirs('outputs', exist_ok=True)
    fig.write_html('outputs/optimizer_comparison.html')
    return df

def plot_decision_boundary(model, X, y, title, filename, poly=None):
    """
    Plot decision boundary using two features with Plotly.
    
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
    
    # Plotly contour plot
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            colorscale=[[0, '#FFAAAA'], [0.5, '#AAAAFF'], [1, '#AAFFAA']][:len(np.unique(y))],
            opacity=0.3,
            showscale=False
        )
    )
    # Scatter plot for data points
    fig.add_trace(
        go.Scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            mode='markers',
            marker=dict(
                color=y,
                colorscale=['#FF0000', '#0000FF', '#00FF00'][:len(np.unique(y))],
                size=8
            ),
            name='Data Points'
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title='Study Hours (Scaled)',
        yaxis_title='Attendance (Scaled)',
        showlegend=False
    )
    os.makedirs('outputs', exist_ok=True)
    fig.write_html(f'outputs/{filename}')
    fig.write_image(f'outputs/{filename.replace(".html", ".png")}', width=800, height=600)

def plot_roc_curve(models, X_test, y_test, labels, title, filename, poly=None):
    """
    Plot ROC curves for multiple models (binary only) with Plotly.
    
    Args:
        models: List of trained models.
        X_test: Test features.
        y_test: Test labels.
        labels: Model names.
        title (str): Plot title.
        filename (str): Output file name.
        poly: Polynomial transformer, if any.
    """
    fig = go.Figure()
    for idx, (model, label) in enumerate(zip(models, labels)):
        # Apply polynomial transformation only for the polynomial model
        X_test_input = poly.transform(X_test) if poly is not None and 'Polynomial' in label else X_test
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_input)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'{label} (AUC = {auc:.2f})'
                )
            )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='black'),
            name='Random Guess'
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    os.makedirs('outputs', exist_ok=True)
    fig.write_html(f'outputs/{filename}')
    fig.write_image(f'outputs/{filename.replace(".html", ".png")}', width=800, height=600)

def plot_feature_importance(model, title, filename, is_decision_tree=False):
    """
    Plot feature importance based on model coefficients or decision tree importances with Plotly.
    
    Args:
        model: Trained model.
        title (str): Plot title.
        filename (str): Output file name.
        is_decision_tree (bool): Flag for decision tree model.
    """
    if is_decision_tree:
        importance = model.feature_importances_
    else:
        importance = np.mean(model.coef_, axis=0) if model.coef_.ndim > 1 else model.coef_
    features = ['Study Hours', 'Sleep Hours', 'Attendance']
    df = pd.DataFrame({'Feature': features, 'Importance': importance})
    
    fig = px.bar(df, x='Importance', y='Feature', orientation='h', title=title,
                 labels={'Importance': 'Coefficient / Importance'}, color='Feature')
    os.makedirs('outputs', exist_ok=True)
    fig.write_html(f'outputs/{filename}')
    fig.write_image(f'outputs/{filename.replace(".html", ".png")}', width=800, height=600)

def plot_class_distribution(y_sim, y_real, y_sim_multi, filename):
    """
    Plot bar plot of class counts for simulated, real, and multi-class data with Plotly.
    
    Args:
        y_sim, y_real, y_sim_multi: Labels.
        filename (str): Output file name.
    """
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=['Simulated Data (Binary)', 'Real Data (Binary)', 'Simulated Data (Multi-Class)'],
                        shared_yaxes=True)
    
    # Simulated binary
    sim_counts = pd.Series(y_sim).value_counts().reset_index()
    sim_counts.columns = ['Class', 'Count']
    fig.add_trace(
        go.Bar(x=sim_counts['Class'], y=sim_counts['Count'], marker_color=['#FF0000', '#0000FF'],
               name='Simulated Binary'),
        row=1, col=1
    )
    
    # Real binary
    real_counts = pd.Series(y_real).value_counts().reset_index()
    real_counts.columns = ['Class', 'Count']
    fig.add_trace(
        go.Bar(x=real_counts['Class'], y=real_counts['Count'], marker_color=['#FF0000', '#0000FF'],
               name='Real Binary'),
        row=1, col=2
    )
    
    # Simulated multi-class
    multi_counts = pd.Series(y_sim_multi).value_counts().reset_index()
    multi_counts.columns = ['Class', 'Count']
    fig.add_trace(
        go.Bar(x=multi_counts['Class'], y=multi_counts['Count'], marker_color=['#FF0000', '#0000FF', '#00FF00'],
               name='Simulated Multi-Class'),
        row=1, col=3
    )
    
    fig.update_layout(
        title='Class Distribution',
        showlegend=False,
        xaxis_title='Pass/Fail (0=Fail, 1=Pass)',
        xaxis2_title='Pass/Fail (0=Fail, 1=Pass)',
        xaxis3_title='Grade (0=Fail, 1=Pass, 2=Excellent)',
        yaxis_title='Count'
    )
    os.makedirs('outputs', exist_ok=True)
    fig.write_html(f'outputs/{filename}')
    fig.write_html(f'outputs/{filename}')
    fig.write_image(f'outputs/{filename.replace(".html", ".png")}', width=800, height=600)

def plot_real_scatter(data, filename):
    """
    Plot scatter of study_hours vs. attendance for real data with Plotly.
    
    Args:
        data (pd.DataFrame): Real data.
        filename (str): Output file name.
    """
    fig = px.scatter(data, x='study_hours', y='attendance', color='pass_fail',
                     title='Real Data: Study Hours vs. Attendance',
                     labels={'study_hours': 'Study Hours', 'attendance': 'Attendance (%)', 'pass_fail': 'Pass/Fail'},
                     color_discrete_map={0: '#FF0000', 1: '#0000FF'},
                     opacity=0.6)
    os.makedirs('outputs', exist_ok=True)
    fig.write_html(f'outputs/{filename}')

def plot_multi_class_cm(y_true, y_pred, title, filename):
    """
    Plot confusion matrix for multi-class with Plotly.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        title (str): Plot title.
        filename (str): Output file name.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Fail', 'Pass', 'Excellent']
    fig = px.imshow(cm, text_auto=True, labels=dict(x='Predicted', y='True', color='Count'),
                    x=labels, y=labels, title=title, color_continuous_scale='Blues')
    os.makedirs('outputs', exist_ok=True)
    fig.write_html(f'outputs/{filename}')

def plot_bayesian_posterior(trace, filename):
    """
    Plot posterior distributions for Bayesian model with Plotly.
    
    Args:
        trace: Posterior samples (arviz.InferenceData).
        filename (str): Output file name.
    """
    if trace is None or not PYMC_AVAILABLE:
        print("No Bayesian posterior plot generated.")
        return
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=['intercept', 'beta_study_hours', 'beta_sleep_hours', 'beta_attendance'])
    
    # Plot intercept
    values = trace.posterior['intercept'].values.flatten()
    fig.add_trace(
        go.Histogram(x=values, nbinsx=50, name='intercept', showlegend=False),
        row=1, col=1
    )
    
    # Plot beta coefficients
    for i, name in enumerate(['study_hours', 'sleep_hours', 'attendance']):
        values = trace.posterior['beta'].sel(beta_dim_0=i).values.flatten()
        row = (i // 2) + 1
        col = (i % 2) + 1
        fig.add_trace(
            go.Histogram(x=values, nbinsx=50, name=f'beta_{name}', showlegend=False),
            row=row, col=col
        )
    
    fig.update_layout(title='Bayesian Posterior Distributions', showlegend=False)
    os.makedirs('outputs', exist_ok=True)
    fig.write_html(f'outputs/{filename}')
    fig.write_image(f'outputs/{filename.replace(".html", ".png")}', width=800, height=600)

def plot_3d_feature_space(X, y):
    """
    3D visualization of feature relationships with Plotly.
    
    Args:
        X: Features.
        y: Labels.
    """
    fig = px.scatter_3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y.astype(str),
        title='3D Feature Space Coloring by Pass/Fail',
        labels={'x': 'Study Hours', 'y': 'Sleep Hours', 'z': 'Attendance'},
        color_discrete_map={'0': '#FF0000', '1': '#0000FF'}
    )
    os.makedirs('outputs', exist_ok=True)
    fig.write_html('outputs/3d_feature_space.html')

def plot_correlation_heatmap(sim_data, real_data, filename):
    """
    Plot correlation heatmaps for simulated and real data with Plotly.
    
    Args:
        sim_data (pd.DataFrame): Simulated data.
        real_data (pd.DataFrame): Real data.
        filename (str): Output file name.
    """
    features = ['study_hours', 'sleep_hours', 'attendance']
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Simulated Data', 'Real Data'],
                        shared_yaxes=True)
    
    # Simulated data heatmap
    sim_corr = sim_data[features].corr()
    fig.add_trace(
        go.Heatmap(
            z=sim_corr.values,
            x=features,
            y=features,
            text=sim_corr.values.round(3),
            texttemplate="%{text}",
            colorscale='Viridis',
            showscale=False
        ),
        row=1, col=1
    )
    
    # Real data heatmap
    real_corr = real_data[features].corr()
    fig.add_trace(
        go.Heatmap(
            z=real_corr.values,
            x=features,
            y=features,
            text=real_corr.values.round(3),
            texttemplate="%{text}",
            colorscale='Viridis',
            showscale=True
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Feature Correlations',
        showlegend=False
    )
    os.makedirs('outputs', exist_ok=True)
    fig.write_html(f'outputs/{filename}')
    fig.write_image(f'outputs/{filename.replace(".html", ".png")}', width=800, height=600)

def plot_sigmoid_function(filename):
    """
    Plot the sigmoid function for theoretical explanation with Plotly.
    
    Args:
        filename (str): Output file name.
    """
    x = np.linspace(-6, 6, 100)
    y = 1 / (1 + np.exp(-x))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Sigmoid Function',
            line=dict(color='blue')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0, 1],
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[-6, 6],
            y=[0.5, 0.5],
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=False
        )
    )
    fig.update_layout(
        title='Sigmoid Function',
        xaxis_title='z',
        yaxis_title='σ(z)',
        showlegend=True
    )
    os.makedirs('outputs', exist_ok=True)
    fig.write_html(f'outputs/{filename}')
    fig.write_image(f'outputs/{filename.replace(".html", ".png")}', width=800, height=600)

def main():
    """Evaluate models and generate visualizations."""
    data = load_models_and_data()
    if data is None:
        return
    (basic_model_sim, reg_model_sim, l1_model_sim, balanced_model_sim, smote_model_sim,
     poly_model_sim, dt_model_sim, basic_model_real, reg_model_real, l1_model_real,
     balanced_model_real, smote_model_real, poly_model_real, dt_model_real,
     basic_model_sim_multi, reg_model_sim_multi, l1_model_sim_multi,
     balanced_model_sim_multi, smote_model_sim_multi, poly_model_sim_multi,
     dt_model_sim_multi, bayesian_sim, dl_real, X_sim_test, y_sim_test,
     X_real_test, y_real_test, X_sim_multi_test, y_sim_multi_test,
     poly_sim, poly_real, poly_sim_multi, real_data) = data
    
    metrics_list = []
    
    # [Existing model evaluations unchanged]
    evaluate_model(basic_model_sim, X_sim_test, y_sim_test, "Simulated Basic", metrics_list)
    evaluate_model(reg_model_sim, X_sim_test, y_sim_test, "Simulated Regularized (L2)", metrics_list)
    evaluate_model(l1_model_sim, X_sim_test, y_sim_test, "Simulated L1 Regularized", metrics_list)
    evaluate_model(balanced_model_sim, X_sim_test, y_sim_test, "Simulated Balanced", metrics_list)
    evaluate_model(smote_model_sim, X_sim_test, y_sim_test, "Simulated SMOTE", metrics_list)
    evaluate_model(poly_model_sim, X_sim_test, y_sim_test, "Simulated Polynomial", metrics_list, poly=poly_sim)
    evaluate_model(dt_model_sim, X_sim_test, y_sim_test, "Simulated Decision Tree", metrics_list)
    
    if bayesian_sim is not None:
        evaluate_model(bayesian_sim, X_sim_test, y_sim_test, "Simulated Bayesian", metrics_list, bayesian=True)
    if dl_real is not None:
        evaluate_model(dl_real, X_real_test, y_real_test, "Real Deep Learning", metrics_list, deep_learning=True)
    
    evaluate_model(basic_model_real, X_real_test, y_real_test, "Real Basic", metrics_list)
    evaluate_model(reg_model_real, X_real_test, y_real_test, "Real Regularized (L2)", metrics_list)
    evaluate_model(l1_model_real, X_real_test, y_real_test, "Real L1 Regularized", metrics_list)
    evaluate_model(balanced_model_real, X_real_test, y_real_test, "Real Balanced", metrics_list)
    evaluate_model(smote_model_real, X_real_test, y_real_test, "Real SMOTE", metrics_list)
    evaluate_model(poly_model_real, X_real_test, y_real_test, "Real Polynomial", metrics_list, poly=poly_real)
    evaluate_model(dt_model_real, X_real_test, y_real_test, "Real Decision Tree", metrics_list)
    
    evaluate_model(basic_model_sim_multi, X_sim_multi_test, y_sim_multi_test,
                   "Simulated Multi-Class Basic", metrics_list)
    evaluate_model(reg_model_sim_multi, X_sim_multi_test, y_sim_multi_test,
                   "Simulated Multi-Class Regularized (L2)", metrics_list)
    evaluate_model(l1_model_sim_multi, X_sim_multi_test, y_sim_multi_test,
                   "Simulated Multi-Class L1 Regularized", metrics_list)
    evaluate_model(balanced_model_sim_multi, X_sim_multi_test, y_sim_multi_test,
                   "Simulated Multi-Class Balanced", metrics_list)
    evaluate_model(smote_model_sim_multi, X_sim_multi_test, y_sim_multi_test,
                   "Simulated Multi-Class SMOTE", metrics_list)
    evaluate_model(poly_model_sim_multi, X_sim_multi_test, y_sim_multi_test,
                   "Simulated Multi-Class Polynomial", metrics_list, poly=poly_sim_multi)
    evaluate_model(dt_model_sim_multi, X_sim_multi_test, y_sim_multi_test,
                   "Simulated Multi-Class Decision Tree", metrics_list)
    
    metrics_df = pd.DataFrame(metrics_list)
    output_file = 'outputs/evaluation_metrics.csv'
    if os.path.exists(output_file):
        print(f"Warning: Overwriting {output_file}")
    metrics_df.to_csv(output_file, index=False)
    print(f"\nMetrics saved to '{output_file}'")
    
    # [Existing visualizations unchanged]
    model_2d = LogisticRegression(max_iter=1000).fit(X_sim_test[:, [0, 2]], y_sim_test)
    plot_decision_boundary(model_2d, X_sim_test, y_sim_test,
                          'Decision Boundary (Simulated Binary)',
                          'decision_boundary_binary.html')
    plot_decision_boundary(LogisticRegression(max_iter=1000), X_sim_test, y_sim_test,
                          'Decision Boundary (Simulated Polynomial)',
                          'decision_boundary_poly.html', poly=PolynomialFeatures(degree=2))
    plot_decision_boundary(LogisticRegression(max_iter=1000), X_sim_multi_test, y_sim_multi_test,
                          'Decision Boundary (Simulated Multi-Class)',
                          'decision_boundary_multi.html')
    
    plot_roc_curve([basic_model_sim, reg_model_sim, l1_model_sim, balanced_model_sim, smote_model_sim, poly_model_sim, dt_model_sim],
                   X_sim_test, y_sim_test,
                   ['Basic', 'Regularized (L2)', 'L1 Regularized', 'Balanced', 'SMOTE', 'Polynomial', 'Decision Tree'],
                   'ROC Curve (Simulated Binary)',
                   'roc_curve_sim.html', poly=poly_sim)
    plot_roc_curve([basic_model_real, reg_model_real, l1_model_real, balanced_model_real, smote_model_real, poly_model_real, dt_model_real],
                   X_real_test, y_real_test,
                   ['Basic', 'Regularized (L2)', 'L1 Regularized', 'Balanced', 'SMOTE', 'Polynomial', 'Decision Tree'],
                   'ROC Curve (Real Binary)',
                   'roc_curve_real.html', poly=poly_real)
    
    plot_feature_importance(reg_model_sim,
                           'Feature Importance (Simulated Regularized L2)',
                           'feature_importance_sim.html')
    plot_feature_importance(l1_model_sim,
                           'Feature Importance (Simulated L1 Regularized)',
                           'feature_importance_sim_l1.html')
    plot_feature_importance(dt_model_sim,
                           'Feature Importance (Simulated Decision Tree)',
                           'feature_importance_sim_dt.html', is_decision_tree=True)
    plot_feature_importance(reg_model_real,
                           'Feature Importance (Real Regularized L2)',
                           'feature_importance_real.html')
    plot_feature_importance(l1_model_real,
                           'Feature Importance (Real L1 Regularized)',
                           'feature_importance_real_l1.html')
    plot_feature_importance(dt_model_real,
                           'Feature Importance (Real Decision Tree)',
                           'feature_importance_real_dt.html', is_decision_tree=True)
    plot_feature_importance(reg_model_sim_multi,
                           'Feature Importance (Simulated Multi-Class L2)',
                           'feature_importance_sim_multi.html')
    plot_feature_importance(l1_model_sim_multi,
                           'Feature Importance (Simulated Multi-Class L1)',
                           'feature_importance_sim_multi_l1.html')
    plot_feature_importance(dt_model_sim_multi,
                           'Feature Importance (Simulated Multi-Class Decision Tree)',
                           'feature_importance_sim_multi_dt.html', is_decision_tree=True)
    
    plot_class_distribution(y_sim_test, y_real_test, y_sim_multi_test, 'class_distribution.html')
    plot_real_scatter(real_data, 'real_scatter.html')
    plot_multi_class_cm(y_sim_multi_test, poly_model_sim_multi.predict(poly_sim_multi.transform(X_sim_multi_test)),
                        'Confusion Matrix (Simulated Multi-Class Polynomial)',
                        'multi_class_cm.html')
    plot_bayesian_posterior(bayesian_sim, 'bayesian_posterior.html')
    plot_3d_feature_space(X_real_test, y_real_test)
    
    # New visualizations
    sim_data = pd.read_csv('outputs/simulated_student_data.csv')
    plot_correlation_heatmap(sim_data, real_data, 'correlation_heatmap.html')
    plot_sigmoid_function('sigmoid_function.html')
    
    print("\nVisualizations saved to 'outputs/' as HTML files")

if __name__ == "__main__":
    main()