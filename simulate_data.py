import numpy as np
import pandas as pd
import os

def simulate_student_data(n=1000, seed=42):
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate 1,000 records
    study_hours = np.random.uniform(0, 10, n)
    sleep_hours = np.random.uniform(0, 10, n)
    attendance = np.random.uniform(0, 100, n)
    
    # Logistic function coefficients
    beta_0, beta_1, beta_2, beta_3 = -5, 0.5, 0.3, 0.04
    
    # Calculate log-odds
    log_odds = beta_0 + beta_1 * study_hours + beta_2 * sleep_hours + beta_3 * attendance
    
    # Convert to probability
    probabilities = 1 / (1 + np.exp(-log_odds))
    
    # Generate pass/fail labels with noise
    pass_fail = np.random.binomial(1, probabilities)
    # Add noise: flip ~5% of labels
    noise_indices = np.random.choice(n, size=int(0.05 * n), replace=False)
    pass_fail[noise_indices] = 1 - pass_fail[noise_indices]
    
    # Create DataFrame
    sim_data = pd.DataFrame({
        'study_hours': study_hours,
        'sleep_hours': sleep_hours,
        'attendance': attendance,
        'pass_fail': pass_fail
    })
    
    return sim_data

def main():
    # Simulate data
    sim_data = simulate_student_data()
    
    # Save to CSV
    os.makedirs('outputs', exist_ok=True)  # Ensure outputs folder exists
    sim_data.to_csv('outputs/simulated_student_data.csv', index=False)
    print("Simulated data saved to 'outputs/simulated_student_data.csv'")
    
    # Basic stats
    print("\nSimulated Data Summary:")
    print(sim_data.describe())
    print("\nPass/Fail Distribution:")
    print(sim_data['pass_fail'].value_counts())

if __name__ == "__main__":
    main()