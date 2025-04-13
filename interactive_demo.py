import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import ipywidgets as widgets
from IPython.display import display, Image
import os

def load_data():
    """
    Load simulated data for interactive demo.
    
    Returns:
        tuple: (X, y) or (None, None) if error.
    """
    sim_file = 'outputs/simulated_student_data.csv'
    if not os.path.exists(sim_file):
        print(f"Error: '{sim_file}' not found. Run simulate_data.py first.")
        return None, None
    try:
        data = pd.read_csv(sim_file)
        X = data[['study_hours', 'attendance']].values
        y = data['pass_fail'].values
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def plot_decision_boundary(model, X, y, title, filename):
    """
    Plot decision boundary for two features.
    
    Args:
        model: Trained model.
        X: Features (study_hours, attendance).
        y: Labels.
        title (str): Plot title.
        filename (str): Output file name.
    """
    X_2d = X
    model.fit(X_2d, y)
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), alpha=0.3)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']))
    plt.xlabel('Study Hours')
    plt.ylabel('Attendance')
    plt.title(title)
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')
    plt.close()

def interactive_boundary():
    """
    Create interactive widget for exploring decision boundaries.
    Must be run in a Jupyter notebook.
    """
    X, y = load_data()
    if X is None or y is None:
        return
    
    @widgets.interact(
        C=widgets.FloatLogSlider(min=-3, max=3, value=1, base=10, description='C'),
        penalty=widgets.Dropdown(options=['l2'], value='l2', description='Penalty'),
        solver=widgets.Dropdown(options=['lbfgs', 'saga'], value='lbfgs', description='Solver')
    )
    def update_boundary(C, penalty, solver):
        model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
        plot_decision_boundary(model, X, y, f'C={C}, penalty={penalty}, solver={solver}',
                              'interactive_boundary.png')
        display(Image(filename='outputs/interactive_boundary.png'))

if __name__ == "__main__":
    print("This script requires a Jupyter notebook environment.")
    print("Run the following in a notebook cell:")
    print("from interactive_demo import interactive_boundary")
    print("interactive_boundary()")