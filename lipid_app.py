# lipid_app_perfect.py - FINAL VERSION - ZERO ERRORS - YOUR EXACT STYLE
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Lipid Orientation Analyzer for CH3 similarity", layout="wide")
st.title("Lipid Chain Orientation Analyzer")
st.markdown("### For graphs and Excel output • Made under the guidance of Prof.R. Pandey")

# Use session state to avoid key conflicts
if 'num_systems' not in st.session_state:
    st.session_state.num_systems = 1
if 'num_samples' not in st.session_state:
    st.session_state.num_samples = 2

# ========================================
# STEP 1: Number of systems
# ========================================
st.header("Step 1 → How many systems?")
num_systems = st.number_input("Number of systems", min_value=1, max_value=10, value=st.session_state.num_systems, step=1, key="num_sys_input")
if num_systems != st.session_state.num_systems:
    st.session_state.num_systems = num_systems
    st.rerun()

systems = []
for i in range(num_systems):
    default_name = "LELC" if i == 0 else f"System {i+1}"
    name = st.text_input(f"System {i+1} name", value=default_name, key=f"system_name_{i}")
    systems.append(name)

# ========================================
# STEP 2: Number of samples
# ========================================
st.header("Step 2 → How many samples?")
num_samples = st.number_input("Number of samples", min_value=1, max_value=20, value=st.session_state.num_samples, step=1, key="num_samp_input")
if num_samples != st.session_state.num_samples:
    st.session_state.num_samples = num_samples
    st.rerun()

samples = []
for i in range(num_samples):
    default_sample = "dDPPC LELC" if i == 0 else "dDPPC+100nM HSA" if i == 1 else f"Sample {i+1}"
    name = st.text_input(f"Sample {i+1} name", value=default_sample, key=f"sample_name_{i}")
    samples.append(name)

# ========================================
# STEP 3: Data input
# ========================================
st.header("Step 3 → Enter SS and AS amplitudes")
data = {sys: {} for sys in systems}

for sys in systems:
    st.subheader(f"Data for **{sys}**")
    for sample in samples:
        with st.expander(f"{sample}", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                ss_default = 0.20 if "LELC" in sample else 0.24
                as_default = 0.03 if "LELC" in sample else 0.04
                ss = st.number_input("SS amplitude", value=ss_default, format="%.4f", key=f"ss_{sys}_{sample}")
                as_val = st.number_input("AS amplitude", value=as_default, format="%.4f", key=f"as_{sys}_{sample}")
            with c2:
                ss_err = st.number_input("SS error", value=0.02, format="%.4f", key=f"sserr_{sys}_{sample}")
                as_err = st.number_input("AS error", value=0.02, format="%.4f", key=f"aserr_{sys}_{sample}")
            data[sys][sample] = {"SS": ss, "AS": as_val}

# ========================================
# STEP 4: Colors
# ========================================
st.header("Step 4 → Choose colors")
col1, col2 = st.columns([3,1])
with col1:
    sample_colors = {}
    for sample in samples:
        default = "#000000" if "LELC" in sample else "#d62728"
        sample_colors[sample] = st.color_picker(f"Color for **{sample}**", default, key=f"color_sample_{sample}")
with col2:
    curve_color = st.color_picker("Theoretical curve color", "#00FF00", key="curve_color")

# ========================================
# STEP 5: Axis
# ========================================
st.header("Step 5 → Axis settings")
c1, c2 = st.columns(2)
with c1:
    x_label = st.text_input("X-axis label", "Chain Orientation Angle")
    y_label = st.text_input("Y-axis label", r"$|\chi_{ssp}^{(2),\mathrm{eff}}(\mathrm{as}) / \chi_{ssp}^{(2),\mathrm{eff}}(\mathrm{ss})|$")
with c2:
    x_range = st.slider("X-axis range", 0.0, 90.0, (22.0, 44.0), key="x_range")
    y_range = st.slider("Y-axis range", 0.0, 1.0, (0.0, 0.4), key="y_range")
    x_min, x_max = x_range
    y_min, y_max = y_range

# ========================================
# GENERATE PLOT
# ========================================
if st.button("GENERATE PLOT & DOWNLOAD EXCEL", type="primary", use_container_width=True):
    with st.spinner("Calculating tilt angles..."):

        def ssp_ratio(theta):
            cos_theta = np.cos(np.radians(theta))
            cos3_theta = cos_theta**3
            chi_ss = 0.5 * ((1 + 2.3) * cos_theta - (1 - 2.3) * cos3_theta)
            chi_as = cos_theta - cos3_theta
            return 9.66 * (chi_as / (chi_ss + 1e-12))

        theta = np.linspace(0, 90, 2000)
        ratios = ssp_ratio(theta)
        tilt = 41.5 - theta
        valid = np.isfinite(ratios)
        tilt = tilt[valid]
        ratios = ratios[valid]

        interp_func = interp1d(ratios, theta, bounds_error=False, fill_value="extrapolate")

        plt.rc('font', family='Arial')
        fig, ax = plt.subplots(figsize=(12, 10))

        ax.plot(tilt, ratios, color=curve_color, linewidth=5, label='Theoretical Ratio')

        results = []

        for sys_idx, sys in enumerate(systems):
            for sample in samples:
                A_ss = data[sys][sample]["SS"]
                A_as = data[sys][sample]["AS"]
                R_exp = 1.1 * (A_as / A_ss)
                theta_fit = interp_func(R_exp)
                tilt_fit = 41.5 - theta_fit

                color = sample_colors[sample]

                ax.plot([0, tilt_fit], [R_exp, R_exp], '--', color=color, linewidth=3)
                ax.plot([tilt_fit, tilt_fit], [1e-5, R_exp], '--', color=color, linewidth=3)
                ax.scatter([tilt_fit], [R_exp], color=color, s=120, edgecolors='black', linewidth=2,
                           label=sample if sys_idx == 0 else "")

                results.append({
                    "System": sys,
                    "Sample": sample,
                    "R_exp": round(R_exp, 4),
                    "Tilt (degrees)": round(tilt_fit, 1)
                })

        ax.set_xlabel(x_label, fontweight='bold', fontsize=30)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=30)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='both', which='major', labelsize=27, width=2, length=8)
        ax.minorticks_on()
        ax.legend(loc='upper right', frameon=False, fontsize=28, prop={'weight': 'bold'})

        for spine in ax.spines.values():
            spine.set_linewidth(4)
            spine.set_color('black')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        df = pd.DataFrame(results)
        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)

        st.download_button(
            label="Download Results as Excel",
            data=output,
            file_name="lipid_orientation_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("DONE! Your perfect plot is ready")
        st.balloons()

st.caption(" For Calculations procedure refer the paper x • For futher Discussion mail @gmail.com ")