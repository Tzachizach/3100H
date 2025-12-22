import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(page_title="Regression Simulator", layout="wide")

st.title("📈 Linear Regression Simulation")
st.markdown("Adjust the parameters in the sidebar to see how the regression line changes.")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Model Parameters")
    # These replace your old ipywidgets
    slope = st.slider("True Slope (m)", -5.0, 5.0, 2.0, 0.1)
    intercept = st.slider("True Intercept (b)", -10.0, 10.0, 0.0, 0.5)
    noise = st.slider("Noise Level", 0.0, 10.0, 2.0, 0.1)
    n_points = st.number_input("Number of Data Points", 10, 500, 100)

# --- SIMULATION LOGIC ---
# Create random data based on user input
x = np.linspace(0, 10, n_points)
# Generate y = mx + b + random noise
noise_array = np.random.normal(0, noise, size=n_points)
y = (slope * x) + intercept + noise_array

# --- VISUALIZATION ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, alpha=0.5, label="Observed Data")
ax.plot(x, slope * x + intercept, color='red', linewidth=2, label=f"True Line: y={slope}x+{intercept}")

ax.set_xlabel("X (Independent Variable)")
ax.set_ylabel("Y (Dependent Variable)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

# Display the plot in the main area
st.pyplot(fig)

# Show statistics below the plot
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Points", n_points)
with col2:
    st.write(f"**Current Formula:** $y = {slope}x + {intercept} + \epsilon$")
