import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# --- 1. CORE PHYSICS ENGINE (Fully Unit-Calibrated) ---
def calculate_smr_rates(T, P_bar, molar_flows, activity=1.0):
    R = 8.314e-3 # kJ/mol*K
    F_CH4, F_H2O, F_CO, F_H2, F_CO2 = molar_flows
    F_total = max(sum(molar_flows), 1e-8)

    # Pressures evaluated directly on Bar scales to match raw literature coefficients
    P_CH4 = (F_CH4 / F_total) * P_bar
    P_H2O = (F_H2O / F_total) * P_bar
    P_CO  = (F_CO  / F_total) * P_bar
    P_H2  = max((F_H2 / F_total) * P_bar, 1e-3) # Bar scale protection cap
    P_CO2 = (F_CO2 / F_total) * P_bar

    # Arrhenius Kinetic Constants converted from kmol/kg*hr to mol/kg*s
    k1 = (4.225e15 * 1000 / 3600) * np.exp(-240.1 / (R * T))
    k2 = (1.955e6  * 1000 / 3600) * np.exp(-67.13 / (R * T))
    k3 = (1.020e15 * 1000 / 3600) * np.exp(-243.9 / (R * T))

    # Thermodynamic Equilibrium Constants
    K1 = np.exp(30.42 - 27170.0 / T)
    K2 = np.exp(-3.798 + 4160.0 / T)
    K3 = K1 * K2

    # Adsorption coefficients (Bar scale parameters)
    K_CH4 = 0.179 * np.exp(38.28 / (R * T))
    K_H2O = 0.415 * np.exp(-0.88 / (R * T))
    K_CO  = 8.23e-5 * np.exp(70.65 / (R * T))
    K_H2  = 0.612 * np.exp(-82.90 / (R * T))

    # Common Crowding Denominator (DEN)
    DEN = 1.0 + K_CO*P_CO + K_H2*np.sqrt(P_H2) + K_CH4*P_CH4 + (K_H2O*P_H2O)/P_H2

    num_r1 = (k1 / (P_H2**2.5)) * (P_CH4 * P_H2O - (P_CO * (P_H2**3)) / K1)
    num_r2 = (k2 / P_H2) * (P_CO * P_H2O - (P_CO2 * P_H2) / K2)
    num_r3 = (k3 / (P_H2**3.5)) * (P_CH4 * (F_H2O**2) - (P_CO2 * (P_H2**4)) / K3)

    r1 = (num_r1 / (DEN**2)) * activity
    r2 = (num_r2 / (DEN**2)) * activity
    r3 = (num_r3 / (DEN**2)) * activity
    return r1, r2, r3

def reactor_derivatives(z, y, P_bar, T_furnace, poison_distance):
    flows = np.clip(y[0:5], 0.0, None)

    # Target element index 5 explicitly from the solver array row safely
    T_curr = max(float(y[5]), 200.0)

    D, rho_b = 0.1, 1000.0
    A_c = np.pi * (D**2) / 4
    U_kilowatts = 0.150 # Calibrated heat transfer value for high mass load

    activity = 0.01 if z < poison_distance else 1.0
    r1, r2, r3 = calculate_smr_rates(T_curr, P_bar, flows, activity)

    # Mass Balances (mol / s * m)
    dF_CH4  = (-r1 - r3) * rho_b * A_c
    dF_H2O  = (-r1 - r2 - 2*r3) * rho_b * A_c
    dF_CO   = (r1 - r2) * rho_b * A_c
    dF_H2   = (3*r1 + r2 + 4*r3) * rho_b * A_c
    dF_CO2  = (r2 + r3) * rho_b * A_c

    # NIST Shomate Gas Heat Capacities (kJ / mol * K)
    t = T_curr / 1000.0
    cp_ch4 = (1.92 + 52.13*t + 11.97*t**2 - 11.32*t**3) * 1e-3
    cp_h2o = (32.24 + 1.92*t + 10.55*t**2 - 3.59*t**3) * 1e-3
    cp_co  = (25.56 + 6.09*t + 4.05*t**2 - 2.67*t**3) * 1e-3
    cp_h2  = (27.14 + 9.27*t - 13.81*t**2 + 7.64*t**3) * 1e-3
    cp_co2 = (24.99 + 55.18*t - 33.69*t**2 + 7.94*t**3) * 1e-3
    total_cp_kj = sum(flows[j] * [cp_ch4, cp_h2o, cp_co, cp_h2, cp_co2][j] for j in range(5))

    dH1, dH2, dH3 = 206.0, -41.0, 165.0

    Q_furnace = U_kilowatts * np.pi * D * (T_furnace - T_curr)
    Q_reaction = (r1*dH1 + r2*dH2 + r3*dH3) * rho_b * A_c

    dT = (Q_furnace - Q_reaction) / max(total_cp_kj, 1e-5)
    return [dF_CH4, dF_H2O, dF_CO, dF_H2, dF_CO2, dT]

# --- 2. STREAMLIT INTERFACE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMR Catalytic Digital Twin")
st.title("🏭 Steam Methane Reforming (SMR) Reactor Tube Digital Twin")
st.markdown("### First-Principles APM Virtual Sensor Dashboard")

st.sidebar.header("🕹️ Plant Control Settings")
T_inlet_C = st.sidebar.slider("Inlet Feed Temperature (°C)", 450.0, 650.0, 500.0)
T_furnace_C = st.sidebar.slider("External Furnace Temperature (°C)", 800.0, 1100.0, 950.0)
P_bar = st.sidebar.slider("Operating Pressure (bar)", 15.0, 40.0, 30.0)

st.sidebar.header("⚠️ Simulated Fault Injection")
sulfur_penetration = st.sidebar.slider("Sulfur Poisoning Front Penetration (meters)", 0.0, 6.0, 0.0)

# Synchronize physical units and high-throughput industrial flows
T_inlet_K = T_inlet_C + 273.15
T_furnace_K = T_furnace_C + 273.15
industrial_inlet_flows = [2.0, 6.0, 0.01, 0.10, 0.01]
y0 = industrial_inlet_flows + [T_inlet_K]

# Execute Solver Integration Routine
sol = solve_ivp(
    fun=lambda z, y: reactor_derivatives(z, y, P_bar, T_furnace_K, sulfur_penetration),
    t_span=(0, 12.0), y0=y0, method='Radau', t_eval=np.linspace(0, 12.0, 100)
)

z_m = sol.t
T_c = sol.y[5, :] - 273.15
h2_yield_exit = sol.y[3, -1]

# Re-evaluate individual point rates cleanly across the solution profile vector
r1_profile = []
r2_profile = []
for step in range(len(z_m)):
    z_curr = z_m[step]
    T_curr_K = float(sol.y[5, step])
    F_CH4 = max(float(sol.y[0, step]), 0.0)
    F_H2O = max(float(sol.y[1, step]), 0.0)
    F_CO  = max(float(sol.y[2, step]), 0.0)
    F_H2  = max(float(sol.y[3, step]), 0.0)
    F_CO2 = max(float(sol.y[4, step]), 0.0)
    flows_curr = [F_CH4, F_H2O, F_CO, F_H2, F_CO2]

    act_curr = 0.01 if z_curr < sulfur_penetration else 1.0
    r1, r2, r3 = calculate_smr_rates(T_curr_K, P_bar, flows_curr, act_curr)

    # MAGNIFIED: Multiply calculated rates by 1,000 for high-contrast graph visualization
    r1_profile.append(r1 * 1000.0)
    r2_profile.append(r2 * 1000.0)

# Dynamic Asset Protection Metric & Corrected RUL Calculation via internal array size extraction
active_zone_indices = np.where(z_m >= sulfur_penetration)
# FIXED: Access index 0 of the tuple to evaluate array size safely
if active_zone_indices[0].size > 0:
    hotspot_idx = active_zone_indices[0][np.argmin(T_c[active_zone_indices])]
    hotspot_pos = z_m[hotspot_idx]
else:
    hotspot_pos = 12.0

safety_limit_meters = 9.0
remaining_active_distance = max(0.0, safety_limit_meters - hotspot_pos)
rul_days = remaining_active_distance / 0.15

# Metric UI Display Cards
col1, col2, col3 = st.columns(3)
col1.metric(label="🚀 Exit Hydrogen Production Rate", value=f"{h2_yield_exit:.2f} mol/s")
col2.metric(label="📍 Current Endothermic Valley Position", value=f"{hotspot_pos:.2f} meters")

if rul_days > 20:
    col3.metric(label="⏳ Estimated Remaining Useful Life (RUL)", value=f"{rul_days:.1f} Days", delta="Optimal Conditions")
elif rul_days > 5:
    col3.metric(label="⚠️ WARNING: Degradation Advance", value=f"{rul_days:.1f} Days", delta="- Turnaround Required", delta_color="off")
else:
    col3.metric(label="🚨 CRITICAL BREAKTHROUGH ALARM", value=f"{rul_days:.1f} Days", delta="- Risk of Tube Rupture", delta_color="inverse")

# --- 3. GRAPHICAL VISUALIZATION LAYOUT ---
chart_col1, chart_col2 = st.columns(2)

# Chart 1: Temperature Profile
with chart_col1:
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=z_m, y=T_c, mode='lines', name='Gas Temp', line=dict(color='#ff4b4b', width=3)))
    fig_temp.add_shape(
        type="line", x0=sulfur_penetration, y0=float(min(T_c)-20), x1=sulfur_penetration, y1=float(max(T_c)+20),
        line=dict(color="orange", width=2, dash="dash")
    )
    fig_temp.update_layout(
        title="1D Spatial Temperature Profile Down Tube",
        xaxis_title="Reactor Length (z, meters)", yaxis_title="Gas Temperature (°C)",
        template="plotly_dark", height=450
    )
    st.plotly_chart(fig_temp, use_container_width=True)

# Chart 2: Independent Kinetic Reaction Rates Profile (Magnified View)
with chart_col2:
    fig_rates = go.Figure()
    fig_rates.add_trace(go.Scatter(x=z_m, y=r1_profile, mode='lines', name='r1 (Reforming Rate)', line=dict(color='#00cc96', width=2.5)))
    fig_rates.add_trace(go.Scatter(x=z_m, y=r2_profile, mode='lines', name='r2 (Water-Gas Shift Rate)', line=dict(color='#ab63fa', width=2.5)))

    # Calculate tight 5% percentage padding to maintain scaling sensitivity on magnified scales
    max_rate_found = max(max(r1_profile[1:]), max(r2_profile[1:]))
    min_rate_found = min(min(r1_profile[1:]), min(r2_profile[1:]))
    padding = max(0.05 * (max_rate_found - min_rate_found), 0.1)

    y_min_rates = min_rate_found - padding
    y_max_rates = max_rate_found + padding

    fig_rates.add_shape(
        type="line", x0=sulfur_penetration, y0=float(y_min_rates), x1=sulfur_penetration, y1=float(y_max_rates),
        line=dict(color="orange", width=2, dash="dash")
    )
    fig_rates.update_layout(
        yaxis=dict(range=[y_min_rates, y_max_rates]),
        title="1D Spatial Kinetic Reaction Rates Profile (Scaled x1,000)",
        xaxis_title="Reactor Length (z, meters)", yaxis_title="Reaction Rate (mol / kg_cat · s x 10⁻³)",
        template="plotly_dark", height=450
    )
    st.plotly_chart(fig_rates, use_container_width=True)
