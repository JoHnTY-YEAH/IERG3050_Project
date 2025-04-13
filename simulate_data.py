import numpy as np
import pandas as pd
import os

def simulate_student_data(n=1000, seed=42):
    """
    Generate simulated student data with study_hours, sleep_hours, attendance, and pass_fail labels.
    
    Args:
        n (int): Number of records to generate (default: 1000).
        seed (int): Random seed for reproducibility (default: 42).
    
    Returns:
        pd.DataFrame: Simulated data with features and labels.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate features
    study_hours = np.random.uniform(0, 10, n)
    # Use normal distribution for sleep_hours, centered at 7 hours
    sleep_hours = np.clip(np.random.normal(loc=7, scale=1, size=n), 4, 10)
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
    """Simulate data and save to CSV."""
    # Simulate data
    sim_data = simulate_student_data()
    
    # Save to CSV
    os.makedirs('outputs', exist_ok=True)
    output_file = 'outputs/simulated_student_data.csv'
    if os.path.exists(output_file):
        print(f"Warning: Overwriting {output_file}")
    sim_data.to_csv(output_file, index=False)
    print(f"Simulated data saved to '{output_file}'")
    
    # Basic stats
    print("\nSimulated Data Summary:")
    print(sim_data.describe())
    print("\nPass/Fail Distribution:")
    print(sim_data['pass_fail'].value_counts())
    # Imbalance ratio
    counts = sim_data['pass_fail'].value_counts()
    ratio = counts[1] / counts[0] if 0 in counts and counts[0] > 0 else float('inf')
    print(f"Imbalance Ratio (Pass/Fail): {ratio:.3f}")

if __name__ == "__main__":
    main()