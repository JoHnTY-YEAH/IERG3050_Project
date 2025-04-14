# IERG3050 Project: Student Performance Analysis

This project analyzes student performance using logistic regression and other machine learning models (e.g., decision trees, Bayesian, deep learning) on simulated and real datasets. It includes data generation, model training, evaluation, visualizations, and a comprehensive report.

- **Features**: Study hours, sleep hours, attendance.
- **Datasets**: Simulated data and UCI Student Performance (Portuguese students).
- **Outputs**: Model metrics, interactive visualizations, HTML/PDF report.

## Prerequisites

- **Operating Systems**: macOS or Windows 10/11
- **Tools**:
  - Python 3.11
  - [Homebrew](https://brew.sh/) (macOS only, package manager)
  - Git (optional, for cloning the repository)
- **System Libraries**: Required for PDF generation (`weasyprint`)

## Setup Instructions

Follow the steps below to set up the project in a clean environment. Instructions are provided for both **macOS** and **Windows**.

### 1. Install Python 3.11

- **macOS**:
  - Install Homebrew (if not already installed):
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
  - Install Python 3.11:
    ```bash
    brew install python@3.11
    ```
  - Verify installation:
    ```bash
    python3.11 --version
    ```
    Expected output: `Python 3.11.x`.

- **Windows**:
  - Download the Python 3.11 installer from [python.org](https://www.python.org/downloads/release/python-3110/).
  - Run the installer, ensuring you check **"Add Python to PATH"**.
  - Verify installation in Command Prompt or PowerShell:
    ```cmd
    python --version
    ```
    Expected output: `Python 3.11.x`.

### 2. Install System Libraries for `weasyprint`

- **macOS**:
  - Install required libraries using Homebrew:
    ```bash
    brew install libffi pango gdk-pixbuf libxml2 glib
    ```

- **Windows**:
  - `weasyprint` requires GTK and Pango. Install them via MSYS2:
    - Download and install [MSYS2](https://www.msys2.org/).
    - Open the MSYS2 terminal and run:
      ```bash
      pacman -S mingw-w64-x86_64-gtk3 mingw-w64-x86_64-pango mingw-w64-x86_64-libffi
      ```
    - Alternatively, follow [WeasyPrint’s Windows installation guide](https://weasyprint.readthedocs.io/en/stable/install.html#windows) for precompiled GTK binaries.
  - Ensure GTK binaries are in your system PATH if required.

### 3. Clone or Set Up Project Directory

- Download or clone the project files into a directory:
  - **macOS**: e.g., `~/IERG3050_Project/`
  - **Windows**: e.g., `C:\IERG3050_Project\`
- Ensure the following files are in the project directory:
  - `main.py`
  - `simulate_data.py`
  - `clean_real_data.py`
  - `train_model.py`
  - `evaluate_visualize.py`
  - `interactive_demo.py`
  - `student-por.csv` (UCI dataset)

### 4. Create and Activate Virtual Environment

- **macOS**:
  ```bash
  cd ~/IERG3050_Project
  rm -rf venv  # Remove old venv if exists
  python3.11 -m venv venv
  source venv/bin/activate
  ```

- **Windows**:
  - Open Command Prompt or PowerShell:
    ```cmd
    cd C:\IERG3050_Project
    rmdir /S /Q venv  :: Remove old venv if exists
    python -m venv venv
    venv\Scripts\activate
    ```
  - You should see `(venv)` in your prompt.

### 5. Install Python Dependencies

- With the virtual environment active, install required packages:
  - **macOS**:
    ```bash
    pip3 install numpy pandas scikit-learn matplotlib plotly pymc arviz imblearn statsmodels tensorflow weasyprint seaborn kaleido
    ```
  - **Windows**:
    ```cmd
    pip install numpy pandas scikit-learn matplotlib plotly pymc arviz imblearn statsmodels tensorflow weasyprint seaborn kaleido
    ```

### 6. Create Output Directories

- **macOS**:
  ```bash
  mkdir -p outputs reports
  ```

- **Windows**:
  ```cmd
  mkdir outputs reports
  ```

## Running the Project

1. **Ensure Virtual Environment is Active**:
   - **macOS**:
     ```bash
     source ~/IERG3050_Project/venv/bin/activate
     ```
   - **Windows**:
     ```cmd
     venv\Scripts\activate
     ```

2. **Run the Pipeline**:
   - **macOS**:
     ```bash
     python3 main.py
     ```
   - **Windows**:
     ```cmd
     python main.py
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
   - Open `reports/project_report.html` in a web browser for the full report with interactive links.
   - Open `reports/project_report.pdf` for a static version.
   - Explore `outputs/*.html` (e.g., `roc_curve_sim.html`, `3d_feature_space.html`) for interactive plots.

## Interactive Demo

- Run `interactive_demo.py` in a Jupyter notebook to explore decision boundaries interactively:
  - Install Jupyter:
    - **macOS**:
      ```bash
      pip3 install jupyter
      ```
    - **Windows**:
      ```cmd
      pip install jupyter
      ```
  - Run the notebook:
    - **macOS**:
      ```bash
      jupyter notebook interactive_demo.py
      ```
    - **Windows**:
      ```cmd
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