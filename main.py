import os
import simulate_data
import clean_real_data
import train_model
import evaluate_visualize

def main():
    print("Running simulate_data.py...")
    simulate_data.main()
    
    print("\nRunning clean_real_data.py...")
    clean_real_data.main()
    
    print("\nRunning train_model.py...")
    train_model.main()
    
    print("\nRunning evaluate_visualize.py...")
    evaluate_visualize.main()
    
    print("\nAll scripts completed. Check 'outputs/' for results.")

if __name__ == "__main__":
    main()