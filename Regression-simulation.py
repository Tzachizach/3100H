import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd

# Page setup
st.set_page_config(page_title="Regression Stats Lab", layout="wide")

st.title("📊 Regression Significance Lab")

# --- SIDEBAR: Inputs ---
with st.sidebar:
    st.header("Simulation Settings")
    
    # NEW: Manual input box for Slope
    true_slope = st.number_input("Input True Slope Coefficient:", value=2.0, step=0.1)
    
    # Sliders for other variables
    true_intercept = 5.0  # Kept constant, or you can make this an input too
    n_points = st.slider("Sample Size (N)", 10, 500, 100)
    noise_level = st.slider("Amount of Noise (Standard Deviation)", 0.0, 10.0, 2.0)

# --- DATA GENERATION ---
np.random.seed(42) # For consistent results while testing
x = np.linspace(0, 10, n_points)
epsilon = np.random.normal(0, noise_level, size=n_points)
y = (true_slope * x) + true_intercept + epsilon

# --- STATISTICAL ANALYSIS ---
X_with_constant = sm.add_constant(x)
model = sm.OLS(y, X_with_constant)
results = model.fit()

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(x, y, alpha=0.5, label="Simulated Data", color="#3498db")
ax.plot(x, results.predict(X_with_constant), color='red', label="Fitted OLS Line")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
st.pyplot(fig)

# --- FORMATTED STATISTICAL TABLE ---
st.subheader("Regression Coefficients & Significance")

# Create the DataFrame
summary_df = pd.DataFrame({
    "Variable": ["Intercept (constant)", "Slope (x1)"],
    "Coefficient": results.params,
    "Std. Error": results.bse,
    "t-statistic": results.tvalues,
    "p-value": results.pvalues
})

# Centering and formatting using st.column_config
st.dataframe(
    summary_df,
    column_config={
        "Variable": st.column_config.TextColumn("Variable", width="medium"),
        "Coefficient": st.column_config.NumberColumn("Coefficient", format="%.4f"),
        "Std. Error": st.column_config.NumberColumn("Std. Error", format="%.4f"),
        "t-statistic": st.column_config.NumberColumn("t-statistic", format="%.4f"),
        "p-value": st.column_config.NumberColumn("p-value", format="%.4f"),
    },
    hide_index=True,
    use_container_width=True
)

# Custom Centered CSS for the UI
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Key Metrics
col1, col2, col3 = st.columns(3)
col1.metric("R-Squared", f"{results.rsquared:.3f}")
col2.metric("Adj. R-Squared", f"{results.rsquared_adj:.3f}")
col3.metric("F-Statistic", f"{results.fvalue:.2f}")
