import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os

def compute_imbalance_ratio(y, dataset_name):
    """Compute and print class imbalance ratio."""
    counts = pd.Series(y).value_counts()
    ratio = counts[1] / counts[0] if counts[0] > 0 else float('inf')
    print(f"\n{dataset_name} Class Imbalance Ratio (Pass/Fail): {ratio:.3f}")
    print(f"Class Distribution (0=fail, 1=pass):")
    print(counts)
    return ratio

def load_data(sim_file='outputs/simulated_student_data.csv', real_file='outputs/cleaned_real_data.csv'):
    """Load simulated and real datasets from CSV files."""
    try:
        sim_data = pd.read_csv(sim_file)
        real_data = pd.read_csv(real_file)
        return sim_data, real_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None

def prepare_data(data, dataset_name):
    """Prepare features and target, split data, and scale features."""
    X = data[['study_hours', 'sleep_hours', 'attendance']]
    y = data['pass_fail']
    # Split 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compute and print imbalance ratio
    compute_imbalance_ratio(y_train, dataset_name)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, y_train):
    """Train basic, regularized, and balanced logistic regression models."""
    # Basic model (no regularization penalty)
    basic_model = LogisticRegression(random_state=42)
    basic_model.fit(X_train, y_train)
    
    # Regularized model (L2 penalty, tune C with GridSearchCV)
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    reg_model = GridSearchCV(LogisticRegression(penalty='l2', solver='lbfgs', random_state=42),
                             param_grid, cv=5, scoring='accuracy')
    reg_model.fit(X_train, y_train)
    print(f"Best C for regularized model: {reg_model.best_params_['C']}")
    
    # Balanced model (handle class imbalance)
    balanced_model = LogisticRegression(class_weight='balanced', random_state=42)
    balanced_model.fit(X_train, y_train)
    
    return basic_model, reg_model.best_estimator_, balanced_model

def main():
    """Main function to load data, train models, and save outputs."""
    # Load data
    sim_data, real_data = load_data()
    if sim_data is None or real_data is None:
        return
    
    # Prepare simulated data
    print("\nPreparing simulated data...")
    X_sim_train, X_sim_test, y_sim_train, y_sim_test, sim_scaler = prepare_data(sim_data, "Simulated")
    
    # Prepare real data
    print("\nPreparing real data...")
    X_real_train, X_real_test, y_real_train, y_real_test, real_scaler = prepare_data(real_data, "Real")
    
    # Train models
    print("\nTraining models for simulated data...")
    basic_model_sim, reg_model_sim, balanced_model_sim = train_models(X_sim_train, y_sim_train)
    print("\nTraining models for real data...")
    basic_model_real, reg_model_real, balanced_model_real = train_models(X_real_train, y_real_train)
    
    # Save models and scalers
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/basic_model_sim.pkl', 'wb') as f:
        pickle.dump(basic_model_sim, f)
    with open('outputs/reg_model_sim.pkl', 'wb') as f:
        pickle.dump(reg_model_sim, f)
    with open('outputs/balanced_model_sim.pkl', 'wb') as f:
        pickle.dump(balanced_model_sim, f)
    with open('outputs/basic_model_real.pkl', 'wb') as f:
        pickle.dump(basic_model_real, f)
    with open('outputs/reg_model_real.pkl', 'wb') as f:
        pickle.dump(reg_model_real, f)
    with open('outputs/balanced_model_real.pkl', 'wb') as f:
        pickle.dump(balanced_model_real, f)
    with open('outputs/sim_scaler.pkl', 'wb') as f:
        pickle.dump(sim_scaler, f)
    with open('outputs/real_scaler.pkl', 'wb') as f:
        pickle.dump(real_scaler, f)
    
    # Save test data for evaluation
    np.save('outputs/X_sim_test.npy', X_sim_test)
    np.save('outputs/y_sim_test.npy', y_sim_test)
    np.save('outputs/X_real_test.npy', X_real_test)
    np.save('outputs/y_real_test.npy', y_real_test)
    
    print("Models and test data saved to 'outputs/'")

if __name__ == "__main__":
    main()