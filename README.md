# IERG3050 Project Setup and Testing Guide

This guide provides detailed instructions to set up, install dependencies, run, and test the Python code for the IERG3050 project, "Predicting Student Success with Logistic Regression: A Simulated and Real-World Analysis." The project implements a pipeline to simulate student data, preprocess real-world data, train multiple logistic regression models (including L1, L2, balanced, SMOTE, polynomial, decision tree, Bayesian, and deep learning variants), evaluate performance, and visualize results using Plotly for interactive plots. The guide is designed for users with basic Python knowledge and covers environment setup, dependency installation, data preparation, execution, and verification.

## Project Overview
The project consists of six Python scripts:
1. **simulate_data.py**: Generates synthetic student data (1,000 records) with features (`study_hours`, `sleep_hours`, `attendance`) and a binary `pass_fail` label.
2. **clean_real_data.py**: Preprocesses the UCI Student Performance dataset (`student-por.csv`), engineering features and labels (`pass_fail`, `grade_class`).
3. **train_model.py**: Trains logistic regression models, a decision tree, and optional Bayesian and deep learning models on both datasets.
4. **evaluate_visualize.py**: Evaluates model performance (accuracy, F1-score, ROC-AUC) and generates interactive Plotly visualizations (decision boundaries, ROC curves, feature importance, etc.).
5. **interactive_demo.py**: Provides a Jupyter notebook interface to explore logistic regression decision boundaries interactively.
6. **main.py**: Orchestrates the pipeline, running all scripts and generating a summary report (`project_report.txt`).

The project requires a real-world dataset (`student-por.csv`) and several Python libraries. Outputs (data, models, plots, metrics) are saved in an `outputs/` directory.

## Prerequisites
Before setting up the project, ensure you have:
- A computer with **Python 3.8 or higher** installed (3.9 is recommended, as tested).
- Access to a terminal (e.g., Command Prompt on Windows, Terminal on macOS/Linux) or an IDE (e.g., VS Code, PyCharm).
- Internet access to download dependencies and the UCI dataset.
- Basic familiarity with Python, pip, and virtual environments.
- (Optional) Jupyter Notebook for running `interactive_demo.py`.

## Step 1: Clone or Download the Project
1. **Obtain the Code**:
   - If the project is hosted on a repository (e.g., GitHub), clone it:
     ```bash
     git clone <repository-url>
     cd <project-directory>
     ```
   - Otherwise, download the six Python scripts (`simulate_data.py`, `clean_real_data.py`, `train_model.py`, `evaluate_visualize.py`, `interactive_demo.py`, `main.py`) and place them in a project directory (e.g., `IERG3050_Project/`).
2. **Verify Files**:
   - Ensure all six `.py` files are present in the project directory.
   - Create a subdirectory for the dataset (e.g., `data/`) if preferred, though not required.

## Step 2: Download the UCI Dataset
The project uses the UCI Student Performance dataset (`student-por.csv`) for real-world analysis.
1. **Download the Dataset**:
   - Visit: https://archive.ics.uci.edu/ml/datasets/Student+Performance
   - Download the zip file (`student.zip`).
   - Extract `student-por.csv` from the zip.
2. **Place the Dataset**:
   - Copy `student-por.csv` to the project directory (e.g., `IERG3050_Project/student-por.csv`).
   - Alternatively, place it in a `data/` subdirectory and update `clean_real_data.py`’s `input_file` parameter (e.g., `input_file='data/student-por.csv'`).
3. **Verify**:
   - Open `student-por.csv` in a text editor or spreadsheet to confirm it contains columns like `studytime`, `absences`, and `G3`.
   - Ensure the file is semicolon-separated (`;`), as expected by `clean_real_data.py`.

## Step 3: Set Up a Virtual Environment
Using a virtual environment isolates project dependencies, preventing conflicts with other Python projects.
1. **Create a Virtual Environment**:
   - Open a terminal in the project directory.
   - Run:
     ```bash
     python3 -m venv ierg3050_env
     ```
   - This creates a virtual environment named `ierg3050_env`.
2. **Activate the Virtual Environment**:
   - **Windows**:
     ```bash
     ierg3050_env\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source ierg3050_env/bin/activate
     ```
   - You should see `(ierg3050_env)` in your terminal prompt, indicating the environment is active.
3. **Verify Python Version**:
   - Run:
     ```bash
     python --version
     ```
   - Confirm it’s Python 3.8 or higher (e.g., `Python 3.9.7`).
4. **Deactivate (Later)**:
   - To exit the virtual environment after setup, run:
     ```bash
     deactivate
     ```

## Step 4: Install Required Libraries
The project depends on several Python libraries for data processing, modeling, and visualization. Install them in the virtual environment.
1. **Install Libraries**:
   - With the virtual environment activated, run:
     ```bash
     pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn statsmodels plotly jupyter tensorflow pymc
     ```
   - This installs:
     - **numpy**: Array operations (used in `simulate_data.py`, etc.).
     - **pandas**: Data handling (used in all scripts).
     - **scikit-learn**: Logistic regression, decision tree, and metrics (used in `train_model.py`, `evaluate_visualize.py`).
     - **matplotlib**: Fallback plotting (used minimally, as Plotly is primary).
     - **seaborn**: Enhanced plotting (used in `evaluate_visualize.py`).
     - **imbalanced-learn**: SMOTE for class imbalance (used in `train_model.py`).
     - **statsmodels**: GLM model (used in `train_model.py`).
     - **plotly**: Interactive visualizations (used in `evaluate_visualize.py`).
     - **jupyter**: Jupyter Notebook for `interactive_demo.py`.
     - **tensorflow**: Deep learning model (optional, used in `train_model.py`).
     - **pymc**: Bayesian model (optional, used in `train_model.py`).
2. **Verify Installation**:
   - Run:
     ```bash
     python -c "import numpy, pandas, sklearn, matplotlib, seaborn, imblearn, statsmodels, plotly, tensorflow, pymc; print('All libraries imported successfully')"
     ```
   - If no errors appear, the libraries are installed correctly.
   - If `tensorflow` or `pymc` fail to import, the code will still run (with warnings about skipping deep learning or Bayesian models).
3. **Troubleshooting**:
   - **pip version error**:
     - Update pip:
       ```bash
       pip install --upgrade pip
       ```
   - **TensorFlow issues** (e.g., macOS M1/M2 compatibility):
     - Install `tensorflow-macos` instead:
       ```bash
       pip install tensorflow-macos
       ```
   - **PyMC issues**:
     - Ensure dependencies like `arviz`:
       ```bash
       pip install arviz
       ```
   - **Permission errors**:
     - Use `--user`:
       ```bash
       pip install <package> --user
       ```
   - **Version conflicts**:
     - Create a fresh virtual environment and reinstall.
4. **Optional: Save Dependencies**:
   - Generate a `requirements.txt` for reproducibility:
     ```bash
     pip freeze > requirements.txt
     ```
   - Others can install from it:
     ```bash
     pip install -r requirements.txt
     ```

## Step 5: Prepare the Environment
1. **Create Outputs Directory**:
   - The code automatically creates an `outputs/` directory, but you can manually create it:
     ```bash
     mkdir outputs
     ```
   - This will store CSVs, models, and HTML plots.
2. **Check File Structure**:
   - Your directory should look like:
     ```
     IERG3050_Project/
     ├── simulate_data.py
     ├── clean_real_data.py
     ├── train_model.py
     ├── evaluate_visualize.py
     ├── interactive_demo.py
     ├── main.py
     ├── student-por.csv
     ├── ierg3050_env/
     ├── outputs/ (created later)
     ```
3. **Clear Old Outputs (Optional)**:
   - If testing repeatedly, delete `outputs/` to start fresh:
     ```bash
     rm -rf outputs
     ```

## Step 6: Run the Pipeline
The main pipeline is executed via `main.py`, which runs all scripts in sequence.
1. **Activate Virtual Environment** (if not already active):
   - See Step 3.
2. **Run the Pipeline**:
   - In the project directory, run:
     ```bash
     python3 main.py
     ```
   - Expected output:
     - `simulate_data.py`: Generates `outputs/simulated_student_data.csv`, prints summary statistics (e.g., mean `study_hours`, `pass_fail` counts).
     - `clean_real_data.py`: Processes `student-por.csv`, saves `outputs/cleaned_real_data.csv`, logs outliers (e.g., “Warning: 5 outliers detected in study_hours”), and prints summaries.
     - `train_model.py`: Trains models, prints best `C` values (e.g., “Best C for L1 regularized model: 10”), GLM summaries, and saves models (e.g., `outputs/l1_model_sim.pkl`).
     - `evaluate_visualize.py`: Evaluates models, prints metrics (e.g., “Simulated L1 Regularized Accuracy: 0.820”), and saves `outputs/evaluation_metrics.csv` and HTML plots (e.g., `outputs/roc_curve_sim.html`).
     - `main.py`: Saves `outputs/project_report.txt` summarizing theory, findings, and visualizations.
   - Runtime: ~1-5 minutes, depending on hardware (TensorFlow/PyMC models increase time).
3. **Expected Outputs**:
   - **Files in `outputs/`**:
     - CSVs: `simulated_student_data.csv`, `cleaned_real_data.csv`, `evaluation_metrics.csv`.
     - Models: `basic_model_sim.pkl`, `l1_model_sim.pkl`, `dt_model_sim.pkl`, etc.
     - Plots: `decision_boundary_binary.html`, `roc_curve_sim.html`, `feature_importance_sim_l1.html`, etc.
     - Report: `project_report.txt`.
     - Test data: `X_sim_test.npy`, `y_sim_test.npy`, etc.
     - Optional: `bayesian_sim.nc`, `dl_real.keras` (if PyMC/TensorFlow are installed).
   - **Console Logs**:
     - Data summaries, class distributions, model metrics, and file save confirmations.
     - Possible warnings:
       - “TensorFlow not installed” or “PyMC not available” (safe to ignore; skips optional models).
       - Outlier warnings from `clean_real_data.py` (informational, no action needed).
4. **Troubleshooting**:
   - **FileNotFoundError for `student-por.csv`**:
     - Ensure `student-por.csv` is in the project directory.
     - Verify the path in `clean_real_data.py` (`input_file='student-por.csv'`).
   - **Missing output files**:
     - Run scripts individually (e.g., `python3 simulate_data.py`) to isolate errors.
   - **Memory errors** (e.g., TensorFlow/PyMC)**:
     - Skip by commenting out `bayesian=True` or `deep_learning=True` in `train_model.py`’s `main()`.
   - **Plotly rendering issues**:
     - Ensure a modern browser (e.g., Chrome, Firefox) for `.html` files.
     - Verify Plotly installation:
       ```bash
       python -c "import plotly; print(plotly.__version__)"
       ```

## Step 7: Test Individual Scripts
To verify each component works independently:
1. **simulate_data.py**:
   - Run:
     ```bash
     python3 simulate_data.py
     ```
   - Check:
     - `outputs/simulated_student_data.csv` exists.
     - Console shows summary (e.g., mean `sleep_hours` ~7, `pass_fail` counts).
     - Open CSV to confirm 1,000 rows, 4 columns (`study_hours`, `sleep_hours`, `attendance`, `pass_fail`).
2. **clean_real_data.py**:
   - Ensure `student-por.csv` is present.
   - Run:
     ```bash
     python3 clean_real_data.py
     ```
   - Check:
     - `outputs/cleaned_real_data.csv` exists.
     - Console logs outliers and summaries (e.g., `pass_fail` and `grade_class` counts).
     - CSV has columns: `study_hours`, `sleep_hours`, `attendance`, `pass_fail`, `grade_class`.
3. **train_model.py**:
   - Requires `outputs/simulated_student_data.csv` and `outputs/cleaned_real_data.csv`.
   - Run:
     ```bash
     python3 train_model.py
     ```
   - Check:
     - Model files in `outputs/` (e.g., `l1_model_sim_multi.pkl`, `dt_model_real.pkl`).
     - Console shows best `C` values, GLM summaries, and save confirmations.
     - No `ConvergenceWarning` for L1 model (due to `max_iter=5000`).
   - Note: May skip Bayesian/deep learning if dependencies are missing.
4. **evaluate_visualize.py**:
   - Requires `outputs/` files from `train_model.py`.
   - Run:
     ```bash
     python3 evaluate_visualize.py
     ```
   - Check:
     - `outputs/evaluation_metrics.csv` lists metrics for all models (e.g., “Simulated Decision Tree”, “Real L1 Regularized”).
     - HTML plots in `outputs/` (e.g., `feature_importance_sim_dt.html`).
     - Open plots in a browser to verify interactivity (e.g., zoom on `roc_curve_real.html`).
5. **interactive_demo.py**:
   - Requires Jupyter Notebook.
   - Install Jupyter if not done:
     ```bash
     pip install jupyter
     ```
   - Start Jupyter:
     ```bash
     jupyter notebook
     ```
   - Open `interactive_demo.py` in the Jupyter interface.
   - Convert to a notebook (if not already `.ipynb`):
     - Copy code into a new notebook.
     - Save as `interactive_demo.ipynb`.
   - Run all cells.
   - Check:
     - Interactive widget appears, allowing changes to `C`, `penalty`, `solver`.
     - Decision boundary plot updates dynamically (saved as `outputs/interactive_boundary.html`).
   - Note: Requires `ipywidgets`:
     ```bash
     pip install ipywidgets
     ```
6. **main.py**:
   - Already tested in Step 6.
   - Confirms all scripts work together.
   - Check `outputs/project_report.txt` for a summary (theory, findings, metrics, visualizations).

## Step 8: Verify Results
To ensure the project meets its objectives:
1. **Data Quality**:
   - **Simulated Data** (`outputs/simulated_student_data.csv`):
     - ~1,000 rows, no missing values.
     - `sleep_hours` mean ~7 (normal distribution).
     - `pass_fail` imbalance ratio printed (e.g., ~1.2).
   - **Real Data** (`outputs/cleaned_real_data.csv`):
     - Columns match simulated data, plus `grade_class`.
     - Outlier warnings logged but data intact.
2. **Model Performance**:
   - Open `outputs/evaluation_metrics.csv`.
   - Check:
     - Accuracy >80% for most models (e.g., “Simulated SMOTE”, “Real Polynomial”).
     - F1-scores reasonable (e.g., >0.75 for balanced models).
     - ROC-AUC >0.8 for binary models (except decision tree, which lacks `predict_proba`).
   - Compare L1 vs. L2 models (L1 should show sparsity in `feature_importance_sim_l1.html`).
3. **Visualizations**:
   - Open HTML files in a browser:
     - `decision_boundary_binary.html`: Shows clear class separation.
     - `roc_curve_sim.html`: Multiple curves with AUC values.
     - `feature_importance_real_dt.html`: Decision tree emphasizes key features (e.g., `study_hours`).
   - Interact with plots (zoom, hover for tooltips).
4. **Report**:
   - Read `outputs/project_report.txt`:
     - Confirms top predictor (e.g., `study_hours`).
     - Lists visualizations and metrics.
     - Notes SMOTE/balanced model improvements for imbalance.
5. **Interactive Demo**:
   - In Jupyter, adjust `C` in `interactive_demo.ipynb`.
   - Verify decision boundary changes (stronger regularization shrinks the boundary).

## Step 9: Troubleshooting Common Issues
- **ImportError**:
  - Reinstall missing library:
    ```bash
    pip install <package>
    ```
  - Ensure virtual environment is active.
- **ConvergenceWarning**:
  - L1 model should converge with `max_iter=5000` in `train_model.py`.
  - If warnings persist, increase to `max_iter=10000` or reduce `param_grid`.
- **FileNotFoundError**:
  - Verify `student-por.csv` path.
  - Run scripts in order: `simulate_data.py`, `clean_real_data.py`, `train_model.py`.
- **Plotly Plots Don’t Render**:
  - Open `.html` files in Chrome/Firefox.
  - Reinstall Plotly:
    ```bash
    pip install plotly --force-reinstall
    ```
- **TensorFlow/PyMC Skipped**:
  - Normal if not installed.
  - Install if needed (see Step 4).
- **Jupyter Widget Issues**:
  - Ensure `ipywidgets`:
    ```bash
    pip install ipywidgets
    ```
  - Restart Jupyter kernel.

## Step 10: Share with Others
To help others replicate your setup:
1. **Provide Files**:
   - Share the six `.py` files and `student-por.csv`.
   - Include `requirements.txt` (from Step 4).
2. **Instructions**:
   - Share this guide or summarize:
     - Install Python 3.9.
     - Set up virtual environment: `python3 -m venv ierg3050_env`.
     - Activate: `source ierg3050_env/bin/activate` (or Windows equivalent).
     - Install dependencies: `pip install -r requirements.txt`.
     - Place `student-por.csv` in project directory.
     - Run: `python3 main.py`.
     - For interactive demo: `jupyter notebook`, open `interactive_demo.ipynb`.
3. **Outputs**:
   - Share `outputs/` (or let them generate it).
   - Highlight key files: `evaluation_metrics.csv`, `project_report.txt`, `roc_curve_sim.html`.

## Additional Resources
- **UCI Dataset**: https://archive.ics.uci.edu/ml/datasets/Student+Performance
- **Scikit-Learn Docs**: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- **Plotly Docs**: https://plotly.com/python/
- **Jupyter Setup**: https://jupyter.org/install
- **Virtual Environments**: https://docs.python.org/3/tutorial/venv.html