import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import statsmodels.api as sm
from IPython.display import display, clear_output

# Set visual style for the projector
plt.rcParams.update({'font.size': 14})
out = widgets.Output()

def run_simulation(n_obs, noise_level, seed):
    with out:
        clear_output(wait=True)
        
        # 1. Generate Data
        np.random.seed(seed)
        x = np.linspace(0, 10, n_obs)
        y = 5.0 + 1.6 * x + np.random.normal(0, noise_level, size=n_obs)

        # 2. Fit OLS
        X = sm.add_constant(x) 
        model = sm.OLS(y, X).fit()
        
        # 3. Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(x, y, color='#3498db', alpha=0.6, label='Accounting Data (Actuals)')
        ax.plot(x, model.predict(X), color='#e74c3c', linewidth=4, label='OLS Estimate (Our Guess)')
        ax.plot(x, 5.0 + 1.6 * x, color='#2ecc71', linestyle='--', label='True Cost Function')
        ax.set_title(f"Accounting Regression: Sample Size = {n_obs}", fontsize=18)
        ax.set_xlabel("Activity Level (Units)", fontsize=14)
        ax.set_ylabel("Total Cost ($)", fontsize=14)
        ax.legend(loc='upper left')
        plt.show()

        # 4. Large Font Statistical Table
        print("="*70)
        print(f"{'REGRESSION AUDIT REPORT':^70}")
        print("="*70)
        print(f"{'R-Squared (Model Fit):':<30} {model.rsquared:.4f}")
        print("-" * 70)
        print(f"{'Component':<25} | {'Estimate':<10} | {'t-Stat':<10} | {'P-Value':<10}")
        print("-" * 70)
        
        names = ['Fixed Cost (Int)', 'Variable Cost (Slope)']
        for i, name in enumerate(names):
            p = model.pvalues[i]
            star = "***" if p < 0.01 else "**" if p < 0.05 else ""
            print(f"{name:<25} | {model.params[i]:<10.3f} | {model.tvalues[i]:<10.2f} | {p:<10.4f} {star}")
        print("="*70)
        print("Notes: P-Value < 0.05 (**) means we are 95% confident the cost is not random.")

# UI Logic
n_slider = widgets.IntSlider(min=10, max=1000, step=10, value=30, description='Sample (N)')
noise_slider = widgets.FloatSlider(min=0.5, max=15.0, step=0.5, value=4.0, description='Noise')
refresh_btn = widgets.Button(description="Refresh Data", button_style='success', icon='sync')

state = type('State', (), {'seed': 42})
def on_refresh(b):
    state.seed = np.random.randint(0, 10000)
    run_simulation(n_slider.value, noise_slider.value, state.seed)
def on_change(change):
    run_simulation(n_slider.value, noise_slider.value, state.seed)

refresh_btn.on_click(on_refresh)
n_slider.observe(on_change, names='value')
noise_slider.observe(on_change, names='value')

display(widgets.VBox([widgets.HBox([n_slider, noise_slider, refresh_btn]), out]))
run_simulation(30, 4.0, 42)
