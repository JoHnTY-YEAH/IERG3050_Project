import pandas as pd
import numpy as np
import os

def clean_real_data(input_file='student-por.csv'):
    # Load data
    try:
        data = pd.read_csv(input_file, sep=';')
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found. Please download 'student-por.csv' from UCI.")
        return None
    
    # Select relevant features
    # studtime (1-4 scale), absences (0-93), G3 (final grade 0-20)
    data = data[['studytime', 'absences', 'G3']].copy()
    
    # Define pass/fail (G3 >= 10 is pass)
    data['pass_fail'] = (data['G3'] >= 10).astype(int)
    
    # Map features to match simulated data
    # Scale studytime (1-4) to study_hours (0-10)
    data['study_hours'] = data['studytime'] * 2.5
    # Convert absences to attendance (0-100%)
    max_absences = data['absences'].max()
    data['attendance'] = 100 - (data['absences'] / max_absences * 100)
    # Sleep hours: Random 6-8 hours since not in dataset
    # Justification: Typical sleep range for students (Wolfson & Carskadon, 2003, Sleep Medicine Reviews)
    data['sleep_hours'] = np.random.uniform(6, 8, len(data))
    
    # Final features
    cleaned_data = data[['study_hours', 'sleep_hours', 'attendance', 'pass_fail']]
    
    # Handle missing values
    cleaned_data = cleaned_data.dropna()
    
    # Check if data is empty
    if cleaned_data.empty:
        print("Error: No data remains after cleaning.")
        return None
    
    return cleaned_data

def main():
    # Clean data
    real_data = clean_real_data()
    if real_data is None:
        return
    
    # Save to CSV
    os.makedirs('outputs', exist_ok=True)
    real_data.to_csv('outputs/cleaned_real_data.csv', index=False)
    print("Cleaned real data saved to 'outputs/cleaned_real_data.csv'")
    
    # Basic stats
    print("\nReal Data Summary:")
    print(real_data.describe())
    print("\nPass/Fail Distribution:")
    print(real_data['pass_fail'].value_counts())

if __name__ == "__main__":
    main()