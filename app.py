# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st

from sim_core import (
    SimParams, GenMode, LutBundle, run_simulation, default_params, load_lut_from_df
)

st.set_page_config(page_title="Pendulum Array Energy Simulator", layout="wide")

st.title("Pendulum Array Energy Simulator")
st.caption("Six/12 chaotic double pendulums → one-way clutches → shaft → flywheel → generator. Physics-bounded (Betz cap), no unicorn watts.")

with st.sidebar:
    st.header("Array & Wind")
    num_pends = st.slider("Number of pendulums", 1, 48, 12, step=1)
    A_pend = st.slider("Swept area per pendulum (m²)", 1.0, 12.0, 4.0, step=0.5)
    Cp = st.slider("Power coefficient Cp", 0.20, 0.50, 0.35, step=0.01)
    rho = st.slider("Air density (kg/m³)", 1.0, 1.4, 1.225, step=0.005)
    v_mean = st.slider("Mean wind (m/s)", 2.0, 15.0, 6.0, step=0.1)
    v_amp = st.slider("Gust amplitude (m/s)", 0.0, 5.0, 1.5, step=0.1)
    f_gust = st.slider("Gust frequency (Hz)", 0.01, 1.0, 0.10, step=0.01)
    venturi = st.slider("Venturi multiplier", 1.00, 1.50, 1.20, step=0.01)

    st.header("Harvest Path")
    use_hinge = st.toggle("Use Hinge Damping (simple path)", value=False)
    st.caption("If off: Shaft + Flywheel + Generator model (recommended).")

    if use_hinge:
        c1 = st.slider("Upper hinge damping c1 (Nms/rad)", 0.0, 10.0, 1.8, step=0.1)
        c2 = st.slider("Lower hinge damping c2 (Nms/rad)", 0.0, 10.0, 1.5, step=0.1)
        gen_eff_h = st.slider("Hinge generator efficiency", 0.5, 0.99, 0.80, step=0.01)
    else:
        st.subheader("Shaft + Flywheel + Generator")
        G_ratio = st.slider("Gear ratio (hinge→shaft)", 1.0, 20.0, 6.0, step=0.5)
        Jf = st.slider("Flywheel inertia Jf (kg·m²)", 0.0, 5.0, 0.8, step=0.1)
        kc = st.slider("Coupler stiffness (Nms/rad)", 0.0, 20.0, 5.0, step=0.5)
        bc = st.slider("Coupler damping (Nms/rad)", 0.0, 5.0, 0.5, step=0.1)
        b_shaft = st.slider("Shaft viscous loss (Nms/rad)", 0.0, 0.2, 0.02, step=0.005)
        tau_coul = st.slider("Coulomb friction (Nm)", 0.0, 1.0, 0.0, step=0.05)

        st.subheader("Drivetrain Efficiencies")
        clutch_eff = st.slider("Clutch efficiency", 0.80, 1.00, 0.95, step=0.01)
        gear_eff   = st.slider("Gearbox efficiency", 0.80, 1.00, 0.95, step=0.01)
        alt_eff    = st.slider("Alternator efficiency (beyond windings)", 0.80, 1.00, 0.98, step=0.01)

        st.subheader("Generator Model")
        mode_choice = st.selectbox(
            "Generator mode",
            (GenMode.LUT_DIRECT, GenMode.FIXED_DC_BUS, GenMode.MPPT_RESISTIVE),
            format_func=lambda m: {
                GenMode.LUT_DIRECT: "LUT (measured curves, 1D/2D/3D)",
                GenMode.FIXED_DC_BUS: "Fixed DC Bus (diode bridge)",
                GenMode.MPPT_RESISTIVE: "MPPT (resistive proxy)",
            }[m],
        )

        Vdc = st.slider("DC bus Vdc (V)", 12.0, 96.0, 48.0, step=1.0)
        Ke  = st.slider("Back-EMF constant Ke (V/(rad/s))", 0.05, 1.50, 0.45, step=0.01)
        Rs  = st.slider("Winding+wire resistance Rs (Ω)", 0.05, 5.0, 0.9, step=0.05)
        V_diode = st.slider("Diode drop per diode (V)", 0.2, 1.2, 0.8, step=0.05)
        eta_rect = st.slider("Rectifier efficiency", 0.80, 0.99, 0.95, step=0.01)
        eta_mppt = st.slider("MPPT converter efficiency", 0.80, 0.99, 0.93, step=0.01)

        st.subheader("LUT Upload (CSV)")
        lut_file = st.file_uploader("Upload generator LUT CSV (omega_rad_s, tau_gen_Nm, P_elec_W[, Vdc][, temp_C])",
                                    type=["csv"])
        lut_tau_sign = st.selectbox("LUT torque sign (opposes motion)", (+1, -1), index=0)
        chain_eff_override = st.toggle("LUT P_elec already includes drivetrain (set CHAIN_EFF=1)", value=False)

    st.header("Run Settings")
    sim_minutes = st.slider("Simulation duration (minutes)", 1, 60, 10, step=1)
    samples = st.slider("Time samples", 2000, 60000, 20001, step=1000)
    do_plot = st.toggle("Render plots", value=True)

# Build params
P = default_params()
P.num_pends = num_pends
P.A_pend = A_pend
P.Cp = Cp
P.rho = rho
P.v_mean = v_mean
P.v_amp = v_amp
P.f_gust = f_gust
P.venturi = venturi

P.T_sec = sim_minutes * 60
P.N_samp = samples
P.do_plot = do_plot

P.use_hinge_damping = use_hinge
if use_hinge:
    P.c1 = c1; P.c2 = c2; P.gen_eff_h = gen_eff_h
else:
    P.G_ratio = G_ratio
    P.Jf = Jf
    P.kc = kc; P.bc = bc
    P.b_shaft = b_shaft; P.tau_coul = tau_coul
    P.clutch_eff = clutch_eff; P.gear_eff = gear_eff; P.alt_eff = alt_eff
    P.gen_mode = mode_choice
    P.Vdc = Vdc; P.Ke = Ke; P.Rs = Rs; P.V_diode = V_diode
    P.eta_rect = eta_rect; P.eta_mppt = eta_mppt
    if chain_eff_override:
        P.chain_eff = 1.0
    else:
        P.chain_eff = P.clutch_eff * P.gear_eff * P.alt_eff

LUT = None
if not use_hinge and lut_file is not None and P.gen_mode == GenMode.LUT_DIRECT:
    try:
        df = pd.read_csv(io.BytesIO(lut_file.getvalue()))
        LUT = load_lut_from_df(df, tau_sign=int(lut_tau_sign))
        st.success(f"LUT loaded. Mode: {LUT.mode}D; ω-range: {LUT.wmin:.2f}–{LUT.wmax:.2f} rad/s")
    except Exception as e:
        st.error(f"Failed to parse LUT: {e}")

# Run
with st.spinner("Simulating…"):
    results = run_simulation(P, LUT)

st.subheader("Executive Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Array Avg Electric Power", f"{results['avg_P_elec_array']:.1f} W")
c2.metric("Array Electric Energy", f"{results['E_elec_array']/1000:.2f} kJ")
c3.metric("Aero Cap @ Mean Wind (array)", f"{results['P_wind_cap_array']:.1f} W")

if results["violates_cap"]:
    st.error("Average electric power exceeded aero cap — check Cp/venturi/params/LUT.")
else:
    st.success("Physics check passed: average power ≤ aero cap (Betz-bounded).")

if do_plot:
    st.divider()
    st.subheader("Time Series")
    t = results["t"]
    st.line_chart(pd.DataFrame({
        "Shaft speed ws (rad/s)": results["ws"],
        "Lower hinge speed w2 (rad/s)": results["w2"],
    }, index=t))
    st.line_chart(pd.DataFrame({
        "Array Instant Electric Power (W)": results["P_elec_inst_array"]
    }, index=t))

    st.subheader("Histograms")
    colh1, colh2 = st.columns(2)
    with colh1:
        st.write("Shaft speed ws histogram")
        st.pyplot(results["fig_hist_ws"])
    with colh2:
        st.write("Instant power histogram (array)")
        st.pyplot(results["fig_hist_P"])

st.caption("Tip: Tune Jf↑ / bc↑ for smoother RPM; match Ke/Rs/Vdc to your alternator; increase A_pend or wind for higher power. Physics invoices everyone.")
