import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier  # Added for decision tree
import statsmodels.api as sm
import pickle
import os
from imblearn.over_sampling import SMOTE

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("Warning: pymc or arviz not installed. Bayesian model will be skipped.")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: tensorflow not installed. Deep learning model will be skipped.")

def compute_imbalance_ratio(y, dataset_name):
    """
    Compute and print class imbalance ratio for binary or multi-class data.
    
    Args:
        y: Target variable.
        dataset_name (str): Name of dataset for printing.
    
    Returns:
        pd.Series: Class counts.
    """
    counts = pd.Series(y).value_counts()
    print(f"\n{dataset_name} Class Distribution:")
    print(counts)
    if len(counts) == 2:  # Binary case
        ratio = counts[1] / counts[0] if counts[0] > 0 else float('inf')
        print(f"Imbalance Ratio (Class 1 / Class 0): {ratio:.3f}")
    return counts

def load_data(sim_file='outputs/simulated_student_data.csv', real_file='outputs/cleaned_real_data.csv'):
    """
    Load simulated and real datasets from CSV files.
    
    Args:
        sim_file (str): Path to simulated data CSV.
        real_file (str): Path to real data CSV.
    
    Returns:
        tuple: (sim_data, real_data) or (None, None) if error.
    """
    if not (os.path.exists(sim_file) and os.path.exists(real_file)):
        print(f"Error: Missing input files. Ensure '{sim_file}' and '{real_file}' exist. "
              "Run simulate_data.py and clean_real_data.py first.")
        return None, None
    try:
        sim_data = pd.read_csv(sim_file)
        real_data = pd.read_csv(real_file)
        return sim_data, real_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def prepare_data(data, dataset_name, multi_class=False):
    """
    Prepare features and target, split data, and scale features.
    
    Args:
        data (pd.DataFrame): Input data.
        dataset_name (str): Name for printing.
        multi_class (bool): If True, generate multi-class labels.
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    X = data[['study_hours', 'sleep_hours', 'attendance']]
    if multi_class and 'grade_class' in data.columns:
        y = data['grade_class']
    elif multi_class:
        # Align with clean_real_data.py's bins for consistency
        G3_sim = (X['study_hours'] * 0.4 + X['attendance'] * 0.1)  # Simplified mapping
        y = pd.cut(G3_sim, bins=[-np.inf, 3.6, 5.6, np.inf], labels=[0, 1, 2]).astype(int)
    else:
        y = data['pass_fail']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    compute_imbalance_ratio(y_train, dataset_name)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_bayesian_model(X_train, y_train):
    if not PYMC_AVAILABLE:
        print("Bayesian model skipped: pymc or arviz not available.")
        return None
    try:
        print("Training Bayesian model for simulated data...")
        with pm.Model() as model:
            beta = pm.Normal('beta', mu=[0.5, 0.1, 0.05], sigma=0.5, shape=3)
            intercept = pm.Normal('intercept', mu=-2, sigma=1)
            logits = intercept + pm.math.dot(X_train, beta)
            pm.Bernoulli('y', logit_p=logits, observed=y_train)
            trace = pm.sample(500, tune=500, target_accept=0.9, return_inferencedata=True)
            
            print("Generating posterior predictive samples...")
            with model:
                posterior_predictive = pm.sample_posterior_predictive(trace)
            
            # Add posterior predictive to trace
            trace.posterior_predictive = posterior_predictive.posterior_predictive
            
            print("Saving Bayesian model trace...")
            os.makedirs('outputs', exist_ok=True)
            az.to_netcdf(trace, 'outputs/bayesian_sim.nc')
            print("Bayesian model saved to 'outputs/bayesian_sim.nc'")
            return trace
    except Exception as e:
        print(f"Error in Bayesian model: {str(e)}")
        return None

def train_deep_learning_model(X_train, y_train):
    if not TF_AVAILABLE:
        print("Deep learning model skipped: tensorflow not available.")
        return None
    try:
        print("Training deep learning model for real data...")
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2)
        print("Deep learning model saved to 'outputs/dl_real.keras'")
        return model
    except Exception as e:
        print(f"Error in deep learning model: {e}")
        return None

def train_models(X_train, y_train, multi_class=False, bayesian=False, deep_learning=False):
    """
    Train logistic regression models, decision tree, including Bayesian and deep learning.
    
    Args:
        X_train: Scaled training features.
        y_train: Training labels.
        multi_class (bool): If True, train for multi-class.
        bayesian (bool): If True, train Bayesian model.
        deep_learning (bool): If True, train deep learning model.
    
    Returns:
        tuple: Trained models and objects.
    """
    basic_model = LogisticRegression(random_state=42, max_iter=1000)
    basic_model.fit(X_train, y_train)
    
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    reg_model = GridSearchCV(LogisticRegression(penalty='l2', solver='lbfgs', random_state=42, max_iter=1000),
                             param_grid, cv=5, scoring='accuracy')
    reg_model.fit(X_train, y_train)
    print(f"Best C for regularized model (L2): {reg_model.best_params_['C']}")
    
    # L1 regularized model with increased max_iter and adjusted tol
    l1_model = GridSearchCV(LogisticRegression(penalty='l1', solver='saga', random_state=42, max_iter=5000, tol=1e-3),
                        param_grid, cv=5, scoring='accuracy')
    l1_model.fit(X_train, y_train)
    print(f"Best C for L1 regularized model: {l1_model.best_params_['C']}")
    
    balanced_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    balanced_model.fit(X_train, y_train)
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    smote_model = LogisticRegression(random_state=42, max_iter=1000)
    smote_model.fit(X_train_smote, y_train_smote)
    
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    poly_model = LogisticRegression(random_state=42, max_iter=1000)
    poly_model.fit(X_train_poly, y_train)
    
    # Decision tree model
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    
    glm_result = None
    if not multi_class:
        X_train_with_const = sm.add_constant(X_train)
        glm_model = sm.GLM(y_train, X_train_with_const, family=sm.families.Binomial())
        glm_result = glm_model.fit()
        print("\nGLM Model Summary:")
        print(glm_result.summary())
    
    bayesian_trace = None
    if bayesian and not multi_class:
        bayesian_trace = train_bayesian_model(X_train, y_train)
    
    dl_model = None
    if deep_learning and not multi_class:
        dl_model = train_deep_learning_model(X_train, y_train)
    
    return (basic_model, reg_model.best_estimator_, l1_model.best_estimator_, balanced_model, smote_model,
            poly_model, poly, dt_model, glm_result, bayesian_trace, dl_model)

def main():
    """Load data, train models, and save outputs."""
    sim_data, real_data = load_data()
    if sim_data is None or real_data is None:
        return
    
    print("\nPreparing simulated data (binary)...")
    X_sim_train, X_sim_test, y_sim_train, y_sim_test, sim_scaler = prepare_data(sim_data, "Simulated")
    
    print("\nPreparing real data (binary)...")
    X_real_train, X_real_test, y_real_train, y_real_test, real_scaler = prepare_data(real_data, "Real")
    
    print("\nPreparing simulated data (multi-class)...")
    X_sim_multi_train, X_sim_multi_test, y_sim_multi_train, y_sim_multi_test, sim_multi_scaler = prepare_data(
        sim_data, "Simulated Multi-Class", multi_class=True)
    
    print("\nTraining models for simulated data (binary)...")
    (basic_model_sim, reg_model_sim, l1_model_sim, balanced_model_sim, smote_model_sim,
     poly_model_sim, poly_sim, dt_model_sim, glm_sim, bayesian_sim, dl_sim) = train_models(
        X_sim_train, y_sim_train, bayesian=PYMC_AVAILABLE)
    
    print("\nTraining models for real data (binary)...")
    (basic_model_real, reg_model_real, l1_model_real, balanced_model_real, smote_model_real,
     poly_model_real, poly_real, dt_model_real, glm_real, bayesian_real, dl_real) = train_models(
        X_real_train, y_real_train, deep_learning=TF_AVAILABLE)
    
    print("\nTraining models for simulated data (multi-class)...")
    (basic_model_sim_multi, reg_model_sim_multi, l1_model_sim_multi, balanced_model_sim_multi,
     smote_model_sim_multi, poly_model_sim_multi, poly_sim_multi, dt_model_sim_multi, _, _, _) = train_models(
        X_sim_multi_train, y_sim_multi_train, multi_class=True)
    
    os.makedirs('outputs', exist_ok=True)
    output_files = [
        ('basic_model_sim.pkl', basic_model_sim),
        ('reg_model_sim.pkl', reg_model_sim),
        ('l1_model_sim.pkl', l1_model_sim),  # Added
        ('balanced_model_sim.pkl', balanced_model_sim),
        ('smote_model_sim.pkl', smote_model_sim),
        ('poly_model_sim.pkl', poly_model_sim),
        ('basic_model_real.pkl', basic_model_real),
        ('reg_model_real.pkl', reg_model_real),
        ('l1_model_real.pkl', l1_model_real),  # Added
        ('balanced_model_real.pkl', balanced_model_real),
        ('smote_model_real.pkl', smote_model_real),
        ('poly_model_real.pkl', poly_model_real),
        ('sim_scaler.pkl', sim_scaler),
        ('real_scaler.pkl', real_scaler),
        ('poly_sim.pkl', poly_sim),
        ('poly_real.pkl', poly_real),
        ('glm_sim.pkl', glm_sim),
        ('glm_real.pkl', glm_real),
        ('basic_model_sim_multi.pkl', basic_model_sim_multi),
        ('reg_model_sim_multi.pkl', reg_model_sim_multi),
        ('l1_model_sim_multi.pkl', l1_model_sim_multi),  # Added
        ('balanced_model_sim_multi.pkl', balanced_model_sim_multi),
        ('smote_model_sim_multi.pkl', smote_model_sim_multi),
        ('poly_model_sim_multi.pkl', poly_model_sim_multi),
        ('poly_sim_multi.pkl', poly_sim_multi),
        ('dt_model_sim.pkl', dt_model_sim),  # Added
        ('dt_model_real.pkl', dt_model_real),  # Added
        ('dt_model_sim_multi.pkl', dt_model_sim_multi),  # Added
    ]
    for fname, obj in output_files:
        if obj is not None:
            fpath = os.path.join('outputs', fname)
            if os.path.exists(fpath):
                print(f"Warning: Overwriting {fpath}")
            with open(fpath, 'wb') as f:
                pickle.dump(obj, f)
    
    if bayesian_sim is not None:
        az.to_netcdf(bayesian_sim, 'outputs/bayesian_sim.nc')
    if dl_real is not None:
        dl_real.save('outputs/dl_real.keras')
    
    np.save('outputs/X_sim_test.npy', X_sim_test)
    np.save('outputs/y_sim_test.npy', y_sim_test)
    np.save('outputs/X_real_test.npy', X_real_test)
    np.save('outputs/y_real_test.npy', y_real_test)
    np.save('outputs/X_sim_multi_test.npy', X_sim_multi_test)
    np.save('outputs/y_sim_multi_test.npy', y_sim_multi_test)
    
    print("Models and test data saved to 'outputs/'")

if __name__ == "__main__":
    main()