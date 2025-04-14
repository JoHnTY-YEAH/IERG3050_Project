# IERG3050 Project: Student Performance Analysis

This project analyzes student performance using logistic regression and other machine learning models (e.g., decision trees, Bayesian, deep learning) on simulated and real datasets. It includes data generation, model training, evaluation, visualizations, and a comprehensive report.

- **Features**: Study hours, sleep hours, attendance.
- **Datasets**: Simulated data and UCI Student Performance (Portuguese students).
- **Outputs**: Model metrics, interactive visualizations, HTML/PDF report.

## Prerequisites

- **Operating System**: macOS
- **Tools**:
  - [Homebrew](https://brew.sh/) (package manager)
  - Python 3.11
- **System Libraries**: Required for PDF generation (`weasyprint`)

## Setup Instructions

Follow these steps to set up the project in a clean environment.

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python 3.11**:
   ```bash
   brew install python@3.11
   ```

3. **Install System Libraries for `weasyprint`**:
   ```bash
   brew install libffi pango gdk-pixbuf libxml2 glib
   ```

4. **Clone or Set Up Project Directory**:
   Ensure the following files are in `IERG3050_Project/`:
   - `main.py`
   - `simulate_data.py`
   - `clean_real_data.py`
   - `train_model.py`
   - `evaluate_visualize.py`
   - `interactive_demo.py`
   - `student-por.csv` (UCI dataset)

5. **Create and Activate Virtual Environment**:
   ```bash
   cd ~/IERG3050_Project
   rm -rf venv  # Remove old venv if exists
   python3.11 -m venv venv
   source venv/bin/activate
   ```

6. **Install Python Dependencies**:
   ```bash
   pip install numpy pandas scikit-learn matplotlib plotly pymc arviz imblearn statsmodels tensorflow weasyprint
   ```

7. **Create Output Directories**:
   ```bash
   mkdir -p outputs reports
   ```

## Running the Project

1. **Ensure Virtual Environment is Active**:
   ```bash
   source ~/IERG3050_Project/venv/bin/activate
   ```

2. **Run the Pipeline**:
   ```bash
   python3 main.py
   ```

3. **Expected Output**:
   - Terminal:
     ```
     Running simulate_data.py...
     Simulated data saved to 'outputs/simulated_student_data.csv'
     ...
     HTML report saved to 'reports/project_report.html'
     PDF report saved to 'reports/project_report.pdf'
     All scripts completed. Check 'outputs/' for results and 'reports/' for the report.
     ```
   - Files:
     - `outputs/`: Datasets (`simulated_student_data.csv`, `cleaned_real_data.csv`), models (`*.pkl`, `*.nc`), metrics (`evaluation_metrics.csv`), visualizations (`*.html`, `*.png`).
     - `reports/`: `project_report.html`, `project_report.pdf`.

4. **View Results**:
   - Open `reports/project_report.html` in a browser for the full report with interactive links.
   - Open `reports/project_report.pdf` for a static version.
   - Explore `outputs/*.html` (e.g., `roc_curve_sim.html`, `3d_feature_space.html`) for interactive plots.

## Interactive Demo

- Run `interactive_demo.py` in a Jupyter notebook to explore decision boundaries interactively:
  ```bash
  pip install jupyter
  jupyter notebook interactive_demo.py
  ```

## Directory Structure

```
IERG3050_Project/
├── main.py
├── simulate_data.py
├── clean_real_data.py
├── train_model.py
├── evaluate_visualize.py
├── interactive_demo.py
├── student-por.csv
├── venv/
├── outputs/
│   ├── simulated_student_data.csv
│   ├── cleaned_real_data.csv
│   ├── evaluation_metrics.csv
│   ├── *.pkl, *.npy, *.nc, *.keras
│   ├── *.html, *.png
├── reports/
│   ├── project_report.html
│   ├── project_report.pdf
```
