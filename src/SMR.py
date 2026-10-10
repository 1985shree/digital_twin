# %%writefile app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# --- 1. CORE PHYSICS ENGINE (Calibrated to Joules/Watts) ---
def calculate_smr_rates(T, P, molar_flows, activity=1.0):
    R = 8.314e-3 # kJ/(mol*K)
    F_CH4, F_H2O, F_CO, F_H2, F_CO2 = molar_flows
    F_total = max(sum(molar_flows), 1e-8)

    P_CH4 = (F_CH4 / F_total) * P
    P_H2O = (F_H2O / F_total) * P
    P_CO  = (F_CO  / F_total) * P
    P_H2  = max((F_H2 / F_total) * P, 1e-4)
    P_CO2 = (F_CO2 / F_total) * P

    # Arrhenius Kinetic parameters
    k1 = 4.225e15 * np.exp(-240.1 / (R * T))
    k2 = 1.955e6  * np.exp(-67.13 / (R * T))
    k3 = 1.020e15 * np.exp(-243.9 / (R * T))

    K1 = np.exp(30.42 - 27170.0 / T)
    K2 = np.exp(-3.798 + 4160.0 / T)
    K3 = K1 * K2

    K_CH4 = 0.179 * np.exp(38.28 / (R * T))
    K_H2O = 0.415 * np.exp(-0.88 / (R * T))
    K_CO  = 8.23e-5 * np.exp(70.65 / (R * T))
    K_H2  = 0.612 * np.exp(-82.90 / (R * T))

    DEN = 1.0 + K_CO*P_CO + K_H2*np.sqrt(P_H2) + K_CH4*P_CH4 + (K_H2O*P_H2O)/P_H2

    num_r1 = (k1 / (P_H2**2.5)) * (P_CH4 * P_H2O - (P_CO * (P_H2**3)) / K1)
    num_r2 = (k2 / P_H2) * (P_CO * P_H2O - (P_CO2 * P_H2) / K2)
    num_r3 = (k3 / (P_H2**3.5)) * (P_CH4 * (P_H2O**2) - (P_CO2 * (P_H2**4)) / K3)

    r1 = np.clip((num_r1 / (DEN**2)) * activity, -500, 500)
    r2 = np.clip((num_r2 / (DEN**2)) * activity, -500, 500)
    r3 = np.clip((num_r3 / (DEN**2)) * activity, -500, 500)
    return r1, r2, r3

def reactor_derivatives(z, y, P, T_furnace, poison_distance):
    flows = np.clip(y[0:5], 0.0, None)
    T_curr = max(y[5], 200.0)

    # Industrial Scale Dimensions
    D, rho_b = 0.1, 1000.0
    A_c = np.pi * (D**2) / 4

    # Heat Transfer Coefficient scaled to Watts (J / s * m^2 * K)
    U_watts = 120.0

    # Step-change activity function mimicking sulfur adsorption front boundary
    activity = 0.01 if z < poison_distance else 1.0

    r1, r2, r3 = calculate_smr_rates(T_curr, P, flows, activity)

    # Mass Balances (mol / s * m)
    dF_CH4  = (-r1 - r3) * rho_b * A_c
    dF_H2O  = (-r1 - r2 - 2*r3) * rho_b * A_c
    dF_CO   = (r1 - r2) * rho_b * A_c
    dF_H2   = (3*r1 + r2 + 4*r3) * rho_b * A_c
    dF_CO2  = (r2 + r3) * rho_b * A_c

    # Shomate Gas Heat Capacities converted to Joules (J / mol * K)
    t = T_curr / 1000.0
    cp_ch4 = (1.92 + 52.13*t + 11.97*t**2 - 11.32*t**3)
    cp_h2o = (32.24 + 1.92*t + 10.55*t**2 - 3.59*t**3)
    cp_co  = (25.56 + 6.09*t + 4.05*t**2 - 2.67*t**3)
    cp_h2  = (27.14 + 9.27*t - 13.81*t**2 + 7.64*t**3)
    cp_co2 = (24.99 + 55.18*t - 33.69*t**2 + 7.94*t**3)
    total_cp_joules = sum(flows[j] * [cp_ch4, cp_h2o, cp_co, cp_h2, cp_co2][j] for j in range(5))

    # Enthalpies of Reactions scaled to Joules/mol
    dH1, dH2, dH3 = 206000.0, -41000.0, 165000.0

    # Energy Balance Derivatives evaluated in Watts
    Q_furnace = U_watts * np.pi * D * (T_furnace - T_curr)
    Q_reaction = (r1*dH1 + r2*dH2 + r3*dH3) * rho_b * A_c

    dT = (Q_furnace - Q_reaction) / max(total_cp_joules, 1e-2)
    return [dF_CH4, dF_H2O, dF_CO, dF_H2, dF_CO2, dT]

# --- 2. STREAMLIT INTERFACE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="SMR Catalytic Digital Twin")
st.title("🏭 Steam Methane Reforming (SMR) Reactor Tube Digital Twin")

st.sidebar.header("🕹️ Plant Control Settings")
T_inlet_C = st.sidebar.slider("Inlet Feed Temperature (°C)", 450.0, 600.0, 500.0)
T_furnace_C = st.sidebar.slider("External Furnace Temperature (°C)", 800.0, 1000.0, 950.0)
P_bar = st.sidebar.slider("Operating Pressure (bar)", 15.0, 40.0, 30.0)

st.sidebar.header("⚠️ Simulated Fault Injection")
sulfur_penetration = st.sidebar.slider("Sulfur Poisoning Front Penetration (meters)", 0.0, 6.0, 0.0)

# Conversions to absolute SI engine profiles
T_inlet_K = T_inlet_C + 273.15
T_furnace_K = T_furnace_C + 273.15
inlet_flows = [0.02, 0.06, 0.0, 0.001, 0.0]
y0 = inlet_flows + [T_inlet_K]

# Execute Solver Integration
sol = solve_ivp(
    fun=lambda z, y: reactor_derivatives(z, y, P_bar, T_furnace_K, sulfur_penetration),
    t_span=(0, 12.0), y0=y0, method='Radau', t_eval=np.linspace(0, 12.0, 100)
)

z_m = sol.t
T_c = sol.y[5, :] - 273.15
h2_yield_exit = sol.y[3, -1]

# Dynamic Asset Protection Metric & Corrected RUL Calculation
# Locate the precise temperature coordinate minimum inside the unpoisoned zone
active_zone_indices = np.where(z_m >= sulfur_penetration)[0]
if len(active_zone_indices) > 0:
    hotspot_idx = active_zone_indices[np.argmin(T_c[active_zone_indices])]
    hotspot_pos = z_m[hotspot_idx]
else:
    hotspot_pos = 12.0

safety_limit_meters = 8.0
remaining_active_distance = max(0.0, safety_limit_meters - hotspot_pos)
# Poisoning velocity factor: calibrated to 0.13 meters per operating day
rul_days = remaining_active_distance / 0.13

# Metric UI Display Cards
col1, col2, col3 = st.columns(3)
col1.metric(label="🚀 Exit Hydrogen Production Rate", value=f"{h2_yield_exit:.4f} mol/s")
col2.metric(label="📍 Current Endothermic Valley Position", value=f"{hotspot_pos:.2f} meters")

if rul_days > 20:
    col3.metric(label="⏳ Estimated Remaining Useful Life (RUL)", value=f"{rul_days:.1f} Days", delta="Optimal Conditions")
elif rul_days > 5:
    col3.metric(label="⚠️ WARNING: Degradation Detected", value=f"{rul_days:.1f} Days", delta="- Schedule Turnaround", delta_color="off")
else:
    col3.metric(label="🚨 CRITICAL BREAKTHROUGH ALARM", value=f"{rul_days:.1f} Days", delta="- Risk of Tube Rupture", delta_color="inverse")

# Interactive Data Plotting using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=z_m, y=T_c, mode='lines', name='Gas Temperature Profile', line=dict(color='#ff4b4b', width=3)))

# Add dynamic indicators mapping the sulfur boundary front
fig.add_shape(
    type="line", x0=sulfur_penetration, y0=float(min(T_c)-20), x1=sulfur_penetration, y1=float(max(T_c)+20),
    line=dict(color="orange", width=2, dash="dash")
)

fig.update_layout(
    title="1D Spatial Temperature Profile Down Reformer Tube Length",
    xaxis_title="Reactor Tube Length Coordinate (z, meters)", yaxis_title="Gas Temperature (°C)",
    template="plotly_dark", height=500
)
st.plotly_chart(fig, use_container_width=True)
