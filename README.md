# 🏭 First-Principles Digital Twin of an Industrial Steam Methane Reforming (SMR) Reactor Tube

## 🚀 Project Overview
This repository contains a high-fidelity, physics-informed **First-Principles Digital Twin** of an industrial **Steam Methane Reforming (SMR)** reactor tube. Operating as an Asset Performance Management (APM) software sensor, the twin tracks unmeasurable internal reactor states—such as 1D spatial gas temperature dips, multi-component gas conversions, and independent reaction velocities—by running a stiff thermodynamic numerical engine against real-world external SCADA control inputs.

The twin incorporates a localized **Fault Injection Layer** to model trace upstream sulfur (H₂S) poisoning breakthroughs. As the sulfur wave permanently coats the active Nickel (Ni) catalyst sites, the twin dynamically models the flattening of the **initial endothermic temperature dip (hotspot migration)** down the 12-meter tube coordinate and computes a live asset metric for **Remaining Useful Life (RUL)** to optimize condition-based maintenance schedules.

---

## 📁 Repository Directory Structure

```text
smr-digital-twin/
├── notebooks/
│   └── SMR reaction and sulfur poisoning effects.ipynb  # Interactive tabular analysis & numerical audits
├── src/
│   └── SMR/
│       ├── WGS.py                                       # FULL VALIDATED DIGITAL TWIN: Multi-reaction kinetics (SMR + WGS),
│       │                                                # Radau implicit solver, and magnified dual-chart dashboard interface.
│       └── SMR.py                                       # UNVALIDATED STEP BASELINE: Isolated single-reaction methane reforming
│                                                        # profile without secondary shift equilibrium verification.
└── README.md                                            # Technical asset overview & model documentation
```

---

## 🛠️ Technology Stack & Engineering Framework
* **Numerical Mathematics:** `SciPy (solve_ivp)` using the implicit **Radau** method to resolve highly stiff chemical kinetic ordinary differential equations (ODEs).
* **Data Pipelines & Structuring:** `NumPy` and `Pandas` for spatial matrix tracking and row-by-row scenario back-testing.
* **Interactive Visualization:** `Streamlit` paired with `Plotly Dark` to host a real-time, responsive operator control board.

---

## 📐 Mathematical & Kinetic Foundations (Implemented in WGS.py)

1. **Primary Reforming (SMR):**  
\[\text{CH}_4 + \text{H}_2\text{O} \rightleftharpoons \text{CO} + 3\text{H}_2 \quad (\Delta H^\circ_{298} = +206 \text{ kJ/mol})\]

2. **Water-Gas Shift (WGS):**  
\[\text{CO} + \text{H}_2\text{O} \rightleftharpoons \text{CO}_2 + \text{H}_2 \quad (\Delta H^\circ_{298} = -41 \text{ kJ/mol})\]

3. **Global Reforming:**  
\[\text{CH}_4 + 2\text{H}_2\text{O} \rightleftharpoons \text{CO}_2 + 4\text{H}_2 \quad (\Delta H^\circ_{298} = +165 \text{ kJ/mol})\]


### Intrinsic Microkinetics (Xu & Froment Model)
Instead of simplified power-law empirical approximations, the twin utilizes the globally recognized **Langmuir-Hinshelwood-Hougen-Watson (LHHW)** framework derived by Xu & Froment (1989). The reaction velocity is governed by a shared **Surface Crowding Denominator (DEN)**, modeling the physical site competition among competing species on the metal lattice:

\[\text{DEN} = 1 + K_{\text{CO}}P_{\text{CO}} + K_{\text{H}_2}P_{\text{H}_2}^{0.5} + K_{\text{CH}_4}P_{\text{CH}_4} + \frac{K_{\text{H}_2\text{O}}P_{\text{H}_2\text{O}}}{P_{\text{H}_2}}\]

The square root on the hydrogen partial pressure (\(P_{\text{H}_2}^{0.5}\)) accounts for the **dissociative adsorption** of stable H₂ gas splitting into individual atoms (\(\text{H}^*\)) to occupy two separate active Nickel sites.

---

## 📈 Real-World Anomaly Simulation: Hotspot Migration
Because methane cracking is heavily endothermic, the reaction absorbs heat violently at the inlet faster than the external furnace can supply it. This creates a distinct **initial temperature dip (endothermic valley)**.

When a trace sulfur leak enters the reactor, the twin injects a localized fault, dropping a(z) to 0.01. The chemical cooling vanishes in the poisoned zone, causing the gas temperature to climb immediately due to uninhibited furnace heat. This forces the endothermic valley to **migrate downstream**, which the twin captures to calculate structural asset degradation.

---

## 🚀 Running the Project Locally

### Prerequisites
Ensure you have the required engineering libraries installed:
```bash
pip install streamlit plotly scipy numpy pandas
```

### Deploying the Full, Validated Digital Twin Dashboard
To launch the complete multi-reaction dashboard interface (`WGS.py`) with full thermal tracking and magnified reaction velocities:
```bash
streamlit run src/SMR/WGS.py
```

### Launching the Unvalidated Step Baseline
To isolate and run the simplified, methane-only reforming profile baseline:
```bash
streamlit run src/SMR/SMR.py
```
