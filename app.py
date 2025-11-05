"""Streamlit UI for the Pendulum Array investor analysis dashboard."""
import importlib.util
import math
import platform
from typing import Optional

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import streamlit as st

from finance_core import (
    DEFAULT_SCENARIOS,
    annual_energy_kwh,
    build_lcoe_table_weibull,
    crf,
    expected_power_from_hist,
    expected_power_weibull,
    gamma_safe,
    size_arrays,
    weibull_ev3,
)

KALEIDO_AVAILABLE = importlib.util.find_spec("kaleido") is not None

st.set_page_config(page_title="Pendulum Array — Investor Analysis (Pro)", layout="wide")
st.title("Pendulum Array — Investor Analysis (Pro)")
st.caption(
    "Weibull/histogram AEP • LCOE • Scenario sizing • CapEx waterfall • Phase offsets & curtailment • Exports"
)


def render_png_download(label: str, fig: go.Figure, filename: str) -> None:
    """Render a download button for a Plotly figure PNG if kaleido is available."""
    if not KALEIDO_AVAILABLE:
        st.caption("Install optional 'kaleido' server-side to enable PNG downloads.")
        return
    try:
        png_bytes = fig.to_image(format="png", scale=2)
    except Exception as err:  # pragma: no cover - UI feedback only
        st.warning(f"Unable to create {label} PNG (install 'kaleido' on the server). Details: {err}")
        return
    st.download_button(label, png_bytes, filename, "image/png")


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Physics")
    rho = st.number_input("Air density ρ (kg/m³)", 1.0, 1.5, 1.225, 0.005)
    A_array = st.number_input("Swept area per array (m²)", 12.0, 400.0, 48.0, 1.0)
    Cp = st.slider("Power coefficient Cp", 0.20, 0.50, 0.35, 0.01)
    chain_eff = st.slider("Drivetrain efficiency (incl. gearbox/alt)", 0.70, 0.99, 0.88, 0.01)
    availability = st.slider("Availability (uptime fraction)", 0.50, 1.00, 0.95, 0.01)

    st.header("Wind Model")
    mode = st.radio("Energy model", ["Weibull (k, c)", "Upload wind histogram"], index=0)
    speeds = counts = None
    if mode == "Weibull (k, c)":
        k = st.slider("Weibull shape k", 1.2, 3.0, 2.0, 0.05)
        c = st.slider("Weibull scale c (m/s)", 3.0, 12.0, 7.0, 0.1)
        st.caption("Tip: k≈2 (Rayleigh-like), c is near the site's 'scale' wind.")
    else:
        up = st.file_uploader("CSV with columns: speed_mps, count (or prob)", type=["csv"])
        if up is not None:
            try:
                dfw = pd.read_csv(up)
                if "speed_mps" not in dfw.columns:
                    raise ValueError("missing speed_mps column")
                count_col: Optional[str]
                if "prob" in dfw.columns:
                    count_col = "prob"
                elif "count" in dfw.columns:
                    count_col = "count"
                else:
                    raise ValueError("missing count/prob column")
                speeds = dfw["speed_mps"].to_numpy(dtype=float)
                counts = dfw[count_col].to_numpy(dtype=float)
                st.success(f"Loaded wind histogram ({len(dfw)} bins).")
            except Exception as err:  # pragma: no cover - UI feedback only
                st.error(f"Failed to parse histogram: {err}")

    st.header("Finance")
    lifetime_yrs = st.slider("Asset life (years)", 5, 30, 15, 1)
    discount_rate = st.slider("Discount rate (WACC)", 0.02, 0.20, 0.08, 0.005)
    opex_pct = st.slider("Annual O&M (% of CapEx)", 0.00, 0.12, 0.03, 0.005)

    st.header("CapEx per array (USD)")
    mech = st.number_input("Mechanical (structure/arms/clutches)", 1000, 200000, 12000, 500)
    elec = st.number_input("Electrical (generator/rectifier/control)", 1000, 200000, 6000, 500)
    civil = st.number_input("Civil & BOS (foundations, install)", 1000, 200000, 4000, 500)
    soft = st.number_input("Soft costs (engineering, permits)", 0, 200000, 2000, 500)
    contingency_pct = st.slider("Contingency %", 0.00, 0.30, 0.10, 0.01)

    base_capex_pre_cont = mech + elec + civil + soft
    capex_array = base_capex_pre_cont * (1.0 + contingency_pct)

    st.header("Layout")
    spacing_factor = st.slider("Footprint factor (× swept area)", 1.0, 10.0, 4.0, 0.5)

    st.header("Phase Offsets / Modules")
    arrays_per_module = st.slider("Arrays per module", 1, 50, 12, 1)
    modules_count = st.slider("Modules count", 1, 500, 10, 1)
    phase_spread_deg = st.slider("Phase spread across module (°)", 0, 360, 180, 5)
    st.caption("Smoothing index ~ ripple reduction from phase staggering (doesn't increase energy).")

    st.header("Curtailment (optional)")
    inverter_limit_kW = st.number_input("Inverter AC rating per module (kW)", 0.0, 3000.0, 0.0, 1.0)
    base_peak_factor = st.slider("Unsmoothed peak / avg factor", 1.0, 5.0, 2.5, 0.1)
    st.caption("If set > 0, smoothing reduces clipping vs inverter rating.")

    with st.expander("Runtime debug"):
        st.write("Python", platform.python_version())
        st.write("Streamlit", st.__version__)
        st.write("Plotly", plotly.__version__)
        st.write("NumPy", np.__version__)
        st.write("Pandas", pd.__version__)
        st.write("Kaleido", "available" if KALEIDO_AVAILABLE else "missing")

# ---------- Core AEP ----------
if mode == "Weibull (k, c)":
    P_array_W = expected_power_weibull(rho, A_array, Cp, chain_eff, availability, k, c)
    v_mean_equiv = c * gamma_safe(1.0 + 1.0 / float(k))
    Ev3 = weibull_ev3(k, c)
else:
    if speeds is None or counts is None:
        st.stop()
    P_array_W = expected_power_from_hist(speeds, counts, rho, A_array, Cp, chain_eff, availability)
    weights_sum = float(np.sum(counts))
    if weights_sum <= 0:
        st.error("Wind histogram weights sum to zero.")
        st.stop()
    v_mean_equiv = float(np.sum(np.asarray(speeds) * counts) / weights_sum)
    Ev3 = float(np.sum(np.power(speeds, 3.0) * counts) / weights_sum)

E_array_kwh = annual_energy_kwh(P_array_W)

colA, colB, colC, colD = st.columns(4)
colA.metric("Avg Power / Array", f"{P_array_W / 1000:.3f} kW")
colB.metric("Annual Energy / Array", f"{E_array_kwh / 1000:.3f} MWh")
colC.metric("E[v^3]", f"{Ev3:.1f} (m/s)^3")
colD.metric("Equiv mean wind", f"{v_mean_equiv:.2f} m/s")

st.divider()

# ---------- Phase offsets & smoothing ----------
M = max(arrays_per_module, 1)
if M == 1 or phase_spread_deg <= 0:
    smoothing_index = 0.0
else:
    delta = math.radians(phase_spread_deg)
    phases = np.linspace(0.0, delta, M)
    cos_sum = 0.0
    pair_count = 0
    for i in range(M):
        for j in range(M):
            cos_sum += math.cos(phases[i] - phases[j])
            pair_count += 1
    pair_cos_mean = cos_sum / max(pair_count, 1)
    ripple_rel = max(0.0, min(1.0, math.sqrt(max(pair_cos_mean, 0.0))))
    smoothing_index = 1.0 - ripple_rel

P_module_kW = (P_array_W * M) / 1000.0

curtailment_note = ""
E_module_MWh = (E_array_kwh * M) / 1000.0
if inverter_limit_kW > 0:
    PF_unsmoothed = base_peak_factor
    PF_smoothed = PF_unsmoothed * (1.0 - 0.7 * smoothing_index)
    peak_unsm_kW = PF_unsmoothed * P_module_kW
    peak_smooth_kW = PF_smoothed * P_module_kW

    clip_unsm = max(0.0, peak_unsm_kW - inverter_limit_kW) / max(peak_unsm_kW, 1e-9)
    clip_smth = max(0.0, peak_smooth_kW - inverter_limit_kW) / max(peak_smooth_kW, 1e-9)

    hours_clip = 0.10
    energy_penalty_unsm = clip_unsm * hours_clip
    energy_penalty_smth = clip_smth * hours_clip

    E_module_eff_unsm = E_module_MWh * (1.0 - energy_penalty_unsm)
    E_module_eff_smth = E_module_MWh * (1.0 - energy_penalty_smth)
    curtailment_gain = E_module_eff_smth - E_module_eff_unsm

    curtailment_note = (
        f"Inverter clipping reduced by smoothing: +{curtailment_gain:.2f} MWh/yr per module (approx)."
    )
else:
    E_module_eff_smth = E_module_MWh

colS1, colS2, colS3, colS4 = st.columns(4)
colS1.metric("Arrays per module", f"{M}")
colS2.metric("Module avg power", f"{P_module_kW:.2f} kW")
colS3.metric("Smoothing index", f"{smoothing_index:.2f}")
colS4.metric("Modules count", f"{modules_count}")

total_arrays = M * max(modules_count, 1)
site_avg_power_kw = (P_array_W * total_arrays) / 1000.0
site_energy_mwh = (E_array_kwh * total_arrays) / 1000.0
site_capex = capex_array * total_arrays
site_land_m2 = total_arrays * A_array * spacing_factor

colSite1, colSite2, colSite3, colSite4 = st.columns(4)
colSite1.metric("Total arrays", f"{total_arrays}")
colSite2.metric("Site avg power", f"{site_avg_power_kw:.1f} kW")
colSite3.metric("Site annual energy", f"{site_energy_mwh:.1f} MWh")
colSite4.metric("Site CapEx", f"${site_capex:,.0f}")

st.caption(
    f"Estimated land footprint: {site_land_m2:,.0f} m² (spacing factor × swept area)."
)

if curtailment_note:
    st.info(curtailment_note)

st.divider()

# ---------- CapEx waterfall (per array) + PNG export ----------
waterfall_items = [
    ("Mechanical", mech),
    ("Electrical", elec),
    ("Civil & BOS", civil),
    ("Soft Costs", soft),
    ("Contingency", base_capex_pre_cont * contingency_pct),
]
labels = [item[0] for item in waterfall_items] + ["Total CapEx/Array"]
values = [item[1] for item in waterfall_items] + [capex_array]
measure = ["relative"] * len(waterfall_items) + ["total"]

fig_capex = go.Figure(
    go.Waterfall(
        name="CapEx",
        orientation="v",
        measure=measure,
        x=labels,
        text=[f"${value:,.0f}" for value in values],
        y=values,
        connector={"line": {"width": 1}},
    )
)
fig_capex.update_layout(title="CapEx Waterfall (per array)", showlegend=False, height=360)
st.plotly_chart(fig_capex, use_container_width=True)
render_png_download("Download CapEx Waterfall (PNG)", fig_capex, "capex_waterfall.png")

colCap1, colCap2, colCap3 = st.columns(3)
colCap1.metric("CapEx/Array (pre-cont.)", f"${base_capex_pre_cont:,.0f}")
colCap2.metric("Contingency", f"${base_capex_pre_cont * contingency_pct:,.0f}")
colCap3.metric("CapEx/Array (total)", f"${capex_array:,.0f}")

st.divider()

# ---------- LCOE sensitivity table (Weibull presets) + CSV export ----------
CRF = crf(discount_rate, lifetime_yrs)

weibull_cases = [
    ("Conservative", 1.8, 6.0),
    ("Base", 2.0, 7.0),
    ("Optimistic", 2.2, 8.0),
]
capex_options = [capex_array * factor for factor in [0.8, 1.0, 1.2]]

df_lcoe = build_lcoe_table_weibull(
    weibull_cases=weibull_cases,
    capex_list=capex_options,
    rho=rho,
    A=A_array,
    Cp=Cp,
    chain_eff=chain_eff,
    availability=availability,
    opex_pct=opex_pct,
    CRF=CRF,
)
st.subheader("LCOE sensitivity (Weibull presets)")
st.dataframe(df_lcoe, use_container_width=True)
st.download_button(
    "Download LCOE CSV",
    df_lcoe.to_csv(index=False).encode(),
    "LCOE_weibull_sensitivity.csv",
    "text/csv",
)

st.divider()

# ---------- Scenario sizing + PNG/CSV exports ----------
st.subheader("Scenario sizing (investor menu)")
st.caption("Upload your list or use defaults. Arrays target average power from your wind model.")

up_scen = st.file_uploader("Upload scenarios CSV (Scenario, Target_kW)", type=["csv"])
if up_scen is not None:
    try:
        scen_df = pd.read_csv(up_scen)[["Scenario", "Target_kW"]]
        st.success(f"Loaded scenarios: {len(scen_df)}")
    except Exception as err:  # pragma: no cover - UI feedback only
        st.error(f"Failed to parse scenarios: {err}")
        scen_df = pd.DataFrame(DEFAULT_SCENARIOS, columns=["Scenario", "Target_kW"])
else:
    scen_df = pd.DataFrame(DEFAULT_SCENARIOS, columns=["Scenario", "Target_kW"])

footprint_per_array_m2 = A_array * spacing_factor
rows = []
bars_labels, bars_values = [], []
for _, row in scen_df.iterrows():
    name = str(row["Scenario"])
    target_kW = float(row["Target_kW"])
    n_arrays = size_arrays(target_kW, P_array_W)
    total_capex = n_arrays * capex_array
    opex_year = total_capex * opex_pct
    energy_year_MWh = n_arrays * (E_array_kwh / 1000.0)
    land_m2 = n_arrays * footprint_per_array_m2

    rows.append(
        {
            "Scenario": name,
            "Target Avg Power (kW)": round(target_kW, 1),
            "Arrays Needed": int(n_arrays),
            "Total CapEx (USD)": round(total_capex, 0),
            "Annual OpEx (USD/yr)": round(opex_year, 0),
            "Annual Energy (MWh/yr)": round(energy_year_MWh, 1),
            "Est. Land Footprint (m²)": round(land_m2, 0),
        }
    )
    bars_labels.append(name)
    bars_values.append(n_arrays)

df_size = pd.DataFrame(rows)
st.dataframe(df_size, use_container_width=True)
st.download_button(
    "Download Sizing CSV",
    df_size.to_csv(index=False).encode(),
    "Scenario_sizing.csv",
    "text/csv",
)

fig_bar = go.Figure(go.Bar(x=bars_labels, y=bars_values))
fig_bar.update_layout(
    title="Arrays Needed per Scenario",
    xaxis_title="Scenario",
    yaxis_title="Arrays",
    height=420,
    bargap=0.25,
)
st.plotly_chart(fig_bar, use_container_width=True)
render_png_download("Download Arrays Bar (PNG)", fig_bar, "arrays_needed.png")

st.caption(
    "Smoothing reduces ripple and clipping but does not create energy. Energy still scales with 0.5·ρ·A·E[v³]·Cp·eff·availability."
)
