import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd

st.set_page_config(page_title="Regression Stats Lab", layout="wide")

st.title("📊 Regression Significance Lab")
st.markdown("Observe how **Noise** and **Sample Size** affect p-values and statistical significance.")

# --- SIDEBAR: Focus on Noise and N ---
with st.sidebar:
    st.header("Simulation Settings")
    n_points = st.slider("Sample Size (N)", 10, 500, 100)
    noise_level = st.slider("Amount of Noise (Standard Deviation)", 0.0, 10.0, 2.0)
    
    st.divider()
    st.info("Fixed True Relationship: y = 2x + 5")
    true_slope = 2.0
    true_intercept = 5.0

# --- DATA GENERATION ---
x = np.linspace(0, 10, n_points)
epsilon = np.random.normal(0, noise_level, size=n_points)
y = (true_slope * x) + true_intercept + epsilon

# --- STATISTICAL ANALYSIS (The "Math" Part) ---
# We add a constant to X to get the intercept (b) in the regression
X_with_constant = sm.add_constant(x)
model = sm.OLS(y, X_with_constant)
results = model.fit()

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y, alpha=0.5, label="Simulated Data", color="#3498db")
# Plot the fitted line based on the calculation
ax.plot(x, results.predict(X_with_constant), color='red', label="Fitted Regression Line")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
st.pyplot(fig)

# --- STATISTICAL SUMMARY TABLE ---
st.subheader("Regression Coefficients & Significance")

# Extracting the stats into a clean DataFrame
summary_df = pd.DataFrame({
    "Coefficient": results.params,
    "Standard Error": results.bse,
    "t-statistic": results.tvalues,
    "p-value": results.pvalues
}, index=["Intercept (b)", "Slope (m)"])

# Display the table with highlighting for significant p-values
st.table(summary_df.style.format("{:.4f}"))

# Extra Insights
col1, col2, col3 = st.columns(3)
col1.metric("R-Squared", f"{results.rsquared:.3f}")
col2.metric("Adj. R-Squared", f"{results.rsquared_adj:.3f}")
col3.metric("F-Statistic", f"{results.fvalue:.2f}")

if results.pvalues[1] < 0.05:
    st.success("✅ The Slope is statistically significant (p < 0.05)")
else:
    st.error("❌ The Slope is NOT statistically significant (p > 0.05)")
