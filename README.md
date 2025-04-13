# IERG3050 Project: Predicting Student Success with Logistic Regression

## Overview
This project predicts student success (pass/fail) using logistic regression, combining simulated and real-world data. Features include study hours, sleep hours, and attendance. The pipeline includes data simulation, cleaning, model training, evaluation, and visualization, fulfilling the requirements of IERG3050.

## Project Structure
- `simulate_data.py`: Generates 1,000 simulated student records.
- `clean_real_data.py`: Processes the UCI Student Performance dataset.
- `train_model.py`: Trains basic, regularized, and balanced logistic regression models.
- `evaluate_visualize.py`: Evaluates models and generates plots (ROC curves, decision boundary, feature importance, scatter, class distribution).
- `main.py`: Orchestrates the pipeline.
- `outputs/`: Stores CSVs, models, and plots.

## Setup
1. Install dependencies: `pip3 install pandas numpy scikit-learn matplotlib seaborn`
2. Download `student-por.csv` from the UCI Machine Learning Repository and place it in the project root.
3. Run `simulate_data.py`: `python3 simulate_data.py`
4. Run `clean_real_data.py`: `python3 clean_real_data.py`
5. Run `train_model.py`: `python3 train_model.py`
6. Run `evaluate_visualize.py`: `python3 evaluate_visualize.py`
7. Run `main.py`: `python3 main.py`

## Outputs
- **Data**: `simulated_student_data.csv`, `cleaned_real_data.csv`
- **Models**: Pickled models (`basic_model_sim.pkl`, etc.)
- **Plots**: `roc_curve_sim.png`, `roc_curve_real.png`, `decision_boundary.png`, `feature_importance_sim.png`, `feature_importance_real.png`, `class_distribution.png`, `real_scatter.png`
- **Metrics**: `evaluation_metrics.csv`

## Notes
- The project assumes 6-8 hours of sleep for real data, based on Wolfson & Carskadon (2003).
- Cross-validation is used for hyperparameter tuning to ensure robustness.
- Metrics and plots support a comprehensive report or presentation.

For questions, contact the team at [your_email@example.com].
