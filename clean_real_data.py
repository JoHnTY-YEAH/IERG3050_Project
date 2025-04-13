import pandas as pd
import numpy as np
import os

def clean_real_data(input_file='student-por.csv'):
    """
    Clean and preprocess the UCI Student Performance dataset.
    
    Args:
        input_file (str): Path to input CSV file (default: 'student-por.csv').
    
    Returns:
        pd.DataFrame: Cleaned data with engineered features, or None if error occurs.
    """
    # Load data
    try:
        data = pd.read_csv(input_file, sep=';')
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found. Download from https://archive.ics.uci.edu/ml/datasets/Student+Performance")
        return None
    
    # Validate required columns
    required_cols = ['studytime', 'absences', 'G3']
    if not all(col in data.columns for col in required_cols):
        print(f"Error: Dataset missing required columns: {required_cols}")
        return None
    
    # Select relevant features
    data = data[['studytime', 'absences', 'G3']].copy()
    
    # Create multi-class labels (0=Fail, 1=Pass, 2=Excellent)
    data['grade_class'] = pd.cut(data['G3'],
                                bins=[-np.inf, 9, 14, np.inf],
                                labels=[0, 1, 2]).astype(int)
    
    # Binary pass/fail
    data['pass_fail'] = (data['G3'] >= 10).astype(int)
    
    # Feature engineering
    data['study_hours'] = data['studytime'] * 2.5
    max_absences = data['absences'].max()
    data['attendance'] = 100 - (data['absences'] / max(1, max_absences)) * 100
    data['sleep_hours'] = np.random.uniform(6, 8, len(data))
    
    # Final features
    cleaned_data = data[['study_hours', 'sleep_hours', 'attendance', 'pass_fail', 'grade_class']]
    cleaned_data = cleaned_data.dropna()
    
    if cleaned_data.empty:
        print("Error: No data remains after cleaning.")
        return None
    
    return cleaned_data

def main():
    """Clean real data and save to CSV."""
    real_data = clean_real_data()
    if real_data is None:
        return
    
    os.makedirs('outputs', exist_ok=True)
    output_file = 'outputs/cleaned_real_data.csv'
    if os.path.exists(output_file):
        print(f"Warning: Overwriting {output_file}")
    real_data.to_csv(output_file, index=False)
    print(f"Cleaned real data saved to '{output_file}'")
    
    print("\nReal Data Summary:")
    print(real_data.describe())
    print("\nPass/Fail Distribution:")
    print(real_data['pass_fail'].value_counts())
    print("\nGrade Class Distribution:")
    print(real_data['grade_class'].value_counts())

if __name__ == "__main__":
    main()