# sim_core.py
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, RegularGridInterpolator, LinearNDInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GenMode(str, Enum):
    FIXED_DC_BUS = "fixed_dc_bus"
    MPPT_RESISTIVE = "mppt_resistive"
    LUT_DIRECT = "lut_direct"


class LutBundle:
    """Holds 1D/2D/3D LUT interpolators for (omega[, Vdc][, temp]) → (tau, P)."""

    def __init__(self):
        self.ok = False
        self.mode = 0
        self.wmin = 0.0
        self.wmax = 0.0
        self.tau_fn = None
        self.p_fn = None
        self.grid_w = None
        self.grid_v = None


def load_lut_from_df(df, tau_sign=+1) -> LutBundle:
    req = ["omega_rad_s", "tau_gen_Nm", "P_elec_W"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"LUT CSV missing required column: {c}")
    have_v = "Vdc" in df.columns
    have_t = "temp_C" in df.columns
    w = np.asarray(df["omega_rad_s"], float)
    tau = np.asarray(df["tau_gen_Nm"], float) * int(tau_sign)
    p = np.asarray(df["P_elec_W"], float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    tau = np.nan_to_num(tau, nan=0.0)
    p = np.nan_to_num(p, nan=0.0)

    bundle = LutBundle()
    bundle.wmin, bundle.wmax = float(np.min(w)), float(np.max(w))

    if have_v and have_t:
        v = np.asarray(df["Vdc"], float)
        tC = np.asarray(df["temp_C"], float)
        v = np.nan_to_num(v, nan=0.0)
        tC = np.nan_to_num(tC, nan=25.0)
        pts = np.column_stack([w, v, tC])
        bundle.tau_fn = LinearNDInterpolator(pts, tau, fill_value=None)
        bundle.p_fn = LinearNDInterpolator(pts, p, fill_value=None)
        bundle.mode = 3
        bundle.ok = True
    elif have_v:
        v = np.asarray(df["Vdc"], float)
        v = np.nan_to_num(v, nan=0.0)
        w_u = np.unique(w)
        v_u = np.unique(v)
        if w_u.size * v_u.size == w.size:
            w_u = np.sort(w_u)
            v_u = np.sort(v_u)
            w_idx = {val: i for i, val in enumerate(w_u)}
            v_idx = {val: i for i, val in enumerate(v_u)}
            Tau = np.empty((w_u.size, v_u.size))
            Pow = np.empty_like(Tau)
            Tau[:] = np.nan
            Pow[:] = np.nan
            for wi, vv, tt, pp in zip(w, v, tau, p):
                Tau[w_idx[wi], v_idx[vv]] = tt
                Pow[w_idx[wi], v_idx[vv]] = pp
            Tau = np.nan_to_num(Tau, nan=np.nanmean(Tau[np.isfinite(Tau)]))
            Pow = np.nan_to_num(Pow, nan=np.nanmean(Pow[np.isfinite(Pow)]))
            bundle.tau_fn = RegularGridInterpolator((w_u, v_u), Tau, bounds_error=False, fill_value=None)
            bundle.p_fn = RegularGridInterpolator((w_u, v_u), Pow, bounds_error=False, fill_value=None)
            bundle.grid_w = w_u
            bundle.grid_v = v_u
            bundle.mode = 2
            bundle.ok = True
        else:
            pts = np.column_stack([w, v])
            bundle.tau_fn = LinearNDInterpolator(pts, tau, fill_value=None)
            bundle.p_fn = LinearNDInterpolator(pts, p, fill_value=None)
            bundle.mode = 2
            bundle.ok = True
    else:
        idx = np.argsort(w)
        w_s, tau_s, p_s = w[idx], tau[idx], p[idx]
        bundle.tau_fn = interp1d(w_s, tau_s, kind="linear", bounds_error=False, fill_value=(tau_s[0], tau_s[-1]))
        bundle.p_fn = interp1d(w_s, p_s, kind="linear", bounds_error=False, fill_value=(p_s[0], p_s[-1]))
        bundle.mode = 1
        bundle.ok = True

    return bundle


@dataclass
class SimParams:
    num_pends: int = 12
    A_pend: float = 4.0
    Cp: float = 0.35
    rho: float = 1.225
    v_mean: float = 6.0
    v_amp: float = 1.5
    f_gust: float = 0.10
    venturi: float = 1.20
    m1: float = 5.0
    m2: float = 5.0
    l1: float = 2.0
    l2: float = 2.0
    g: float = 9.81
    use_hinge_damping: bool = False
    c1: float = 1.8
    c2: float = 1.5
    gen_eff_h: float = 0.80
    G_ratio: float = 6.0
    Jf: float = 0.8
    kc: float = 5.0
    bc: float = 0.5
    b_shaft: float = 0.02
    tau_coul: float = 0.0
    clutch_eff: float = 0.95
    gear_eff: float = 0.95
    alt_eff: float = 0.98
    chain_eff: float = 0.95 * 0.95 * 0.98
    gen_mode: GenMode = GenMode.LUT_DIRECT
    Ke: float = 0.45
    Rs: float = 0.9
    V_diode: float = 0.8
    eta_rect: float = 0.95
    Vdc: float = 48.0
    eta_mppt: float = 0.93
    T_sec: int = 600
    N_samp: int = 20001
    do_plot: bool = True


def default_params() -> SimParams:
    return SimParams()


def wind_speed(t, params: SimParams):
    v = params.v_mean * params.venturi + params.v_amp * np.sin(2 * np.pi * params.f_gust * t)
    return max(v, 0.0)


def aero_power_cap_per_pend(t, params: SimParams):
    v = wind_speed(t, params)
    return 0.5 * params.rho * params.A_pend * (v**3) * params.Cp


def wind_torque_bounded(theta, omega, t, params: SimParams):
    v = wind_speed(t, params)
    v_rel = np.clip(v - omega * params.l2 * 0.5, -50.0, 50.0)
    Cd = 1.2
    force = 0.5 * params.rho * params.A_pend * Cd * v_rel * abs(v_rel)
    tau_raw = force * (params.l2 * 0.5) * np.cos(theta)
    cap = aero_power_cap_per_pend(t, params)
    omega_abs = max(abs(omega), 1e-3)
    tau_cap = cap / omega_abs
    return np.clip(tau_raw, -tau_cap, tau_cap)


def gen_fixed_dc_bus(ws, params: SimParams):
    e = params.Ke * abs(ws)
    v_drop = 2.0 * params.V_diode
    current = (e - params.Vdc - v_drop) / params.Rs
    if current <= 0:
        return 0.0, 0.0
    power_elec = params.eta_rect * current * params.Vdc
    tau_gen = -np.sign(ws) * params.Ke * current
    return tau_gen, power_elec


def gen_mppt_resistive(ws, params: SimParams):
    e = params.Ke * abs(ws)
    if e <= 1e-9:
        return 0.0, 0.0
    i_mpp = e / (2.0 * params.Rs)
    power_max = e * i_mpp - (i_mpp**2) * params.Rs
    power_elec = max(0.0, params.eta_mppt * power_max)
    tau_gen = -np.sign(ws) * params.Ke * i_mpp
    return tau_gen, power_elec


def gen_lut(ws, params: SimParams, lut: LutBundle | None):
    if lut is None or not lut.ok:
        return 0.0, 0.0
    wabs = abs(ws)
    Vdc_now = params.Vdc
    temp_now = 25.0
    if lut.mode == 3:
        tau = lut.tau_fn([wabs, Vdc_now, temp_now])
        power = lut.p_fn([wabs, Vdc_now, temp_now])
    elif lut.mode == 2:
        if isinstance(lut.tau_fn, RegularGridInterpolator):
            tau = lut.tau_fn([[wabs, Vdc_now]])
            power = lut.p_fn([[wabs, Vdc_now]])
        else:
            tau = lut.tau_fn([wabs, Vdc_now])
            power = lut.p_fn([wabs, Vdc_now])
    else:
        tau = lut.tau_fn(wabs)
        power = lut.p_fn(wabs)

    tau = float(0.0 if tau is None or np.isnan(tau) else tau)
    power = float(0.0 if power is None or np.isnan(power) else power)
    tau_signed = -np.sign(ws) * abs(tau)
    power = max(0.0, power)
    return tau_signed, power


def generator_torque_and_power(ws, params: SimParams, lut: LutBundle | None):
    if params.gen_mode == GenMode.LUT_DIRECT:
        return gen_lut(ws, params, lut)
    if params.gen_mode == GenMode.FIXED_DC_BUS:
        return gen_fixed_dc_bus(ws, params)
    if params.gen_mode == GenMode.MPPT_RESISTIVE:
        return gen_mppt_resistive(ws, params)
    return 0.0, 0.0


def rhs(t, y, params: SimParams, lut: LutBundle | None):
    th1, w1, th2, w2, phi, ws, Ew, Ewind, Eelec = y
    I1 = params.m1 * params.l1**2
    I2 = params.m2 * params.l2**2

    tau_g1 = -(params.m1 + params.m2) * params.g * params.l1 * np.sin(th1)
    tau_g2 = -params.m2 * params.g * params.l2 * np.sin(th2)
    tau_aero_lower = wind_torque_bounded(th2, w2, t, params)

    tau_u = 0.0
    tau_l = 0.0
    power_elec_out = 0.0

    if params.use_hinge_damping:
        tau_u = -np.sign(w1) * params.c1 * abs(w1)
        tau_l = -np.sign(w2) * params.c2 * abs(w2)
        power_raw = params.c1 * w1 * w1 * params.gen_eff_h + params.c2 * w2 * w2 * params.gen_eff_h
        power_cap = aero_power_cap_per_pend(t, params)
        if power_raw > power_cap:
            scale = power_cap / max(power_raw, 1e-12)
            tau_u *= scale
            tau_l *= scale
            power_elec_out = power_cap
        else:
            power_elec_out = power_raw
        dphi = ws
        dws = 0.0
    else:
        speed_err = params.G_ratio * w1 - ws
        tau_cpl_hinge = params.kc * speed_err + params.bc * speed_err
        tau_cpl_shaft = -tau_cpl_hinge * params.G_ratio
        tau_loss_shaft = -params.b_shaft * ws
        if abs(ws) > 1e-6:
            tau_loss_shaft -= params.tau_coul * np.sign(ws)
        tau_gen_g, power_elec_gen = generator_torque_and_power(ws, params, lut)
        power_chain = power_elec_gen * params.chain_eff
        tau_shaft_net = tau_cpl_shaft + tau_loss_shaft + tau_gen_g
        if params.Jf > 0.0:
            dphi = ws
            dws = tau_shaft_net / max(params.Jf, 1e-9)
        else:
            ws = params.G_ratio * w1
            dphi = ws
            dws = 0.0
            tau_cpl_hinge = 0.0
        power_cap = aero_power_cap_per_pend(t, params)
        if power_chain > power_cap:
            scale = power_cap / max(power_chain, 1e-12)
            tau_gen_g = tau_gen_g * scale
            tau_shaft_net = tau_cpl_shaft + tau_loss_shaft + tau_gen_g
            if params.Jf > 0.0:
                dws = tau_shaft_net / max(params.Jf, 1e-9)
            power_elec_out = power_cap
        else:
            power_elec_out = power_chain
        tau_u = tau_cpl_hinge

    a1 = (tau_g1 + tau_u) / I1
    a2 = (tau_g2 + tau_l + tau_aero_lower) / I2
    power_wind_in = abs(tau_aero_lower * w2)
    dEw = tau_g1 * w1 + tau_g2 * w2 + tau_u * w1 + tau_l * w2 + tau_aero_lower * w2
    dEwind = power_wind_in
    dEelec = power_elec_out

    return [w1, a1, w2, a2, dphi, dws, dEw, dEwind, dEelec]


def run_simulation(params: SimParams, lut: LutBundle | None):
    t_eval = np.linspace(0, params.T_sec, params.N_samp)
    y0 = [np.deg2rad(45.0), 0.0, np.deg2rad(45.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(lambda t, y: rhs(t, y, params, lut), [0, params.T_sec], y0, t_eval=t_eval, rtol=1e-6, atol=1e-8, max_step=0.02)
    t = sol.t
    th1, w1, th2, w2, phi, ws, Ew, Ewind, Eelec = sol.y
    E_wind_array = Ewind[-1] * params.num_pends
    E_elec_array = Eelec[-1] * params.num_pends
    avg_power_elec_array = E_elec_array / params.T_sec
    v_cap = params.v_mean * params.venturi
    P_wind_cap_array = 0.5 * params.rho * (params.num_pends * params.A_pend) * (v_cap**3) * params.Cp
    violates_cap = avg_power_elec_array > 1.05 * P_wind_cap_array
    P_inst_array = np.gradient(Eelec, t, edge_order=2) * params.num_pends

    figs = {}
    if params.do_plot:
        fig_ws = plt.figure(figsize=(5, 3))
        plt.hist(ws, bins=50)
        plt.title("Shaft speed ws histogram")
        plt.xlabel("rad/s")
        plt.ylabel("count")
        figs["fig_hist_ws"] = fig_ws

        fig_P = plt.figure(figsize=(5, 3))
        plt.hist(np.clip(P_inst_array, 0, None), bins=50)
        plt.title("Instant Electric Power (array) histogram")
        plt.xlabel("W")
        plt.ylabel("count")
        figs["fig_hist_P"] = fig_P
    else:
        figs["fig_hist_ws"] = None
        figs["fig_hist_P"] = None

    return {
        "t": t,
        "w2": w2,
        "ws": ws,
        "P_elec_inst_array": P_inst_array,
        "E_wind_array": E_wind_array,
        "E_elec_array": E_elec_array,
        "avg_P_elec_array": avg_power_elec_array,
        "P_wind_cap_array": P_wind_cap_array,
        "violates_cap": violates_cap,
        **figs,
    }
