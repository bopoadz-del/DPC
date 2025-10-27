# Pendulum Array Energy Simulator (Streamlit)

Chaotic double pendulums → one-way clutches → common shaft → flywheel → generator.
Physics-bounded (Betz limit), supports generator models:
- Fixed DC bus (diode bridge)
- MPPT (resistive proxy)
- **LUT** (measured curves, 1D/2D/3D over ω × Vdc × temp)

## Quickstart

```bash
git clone https://github.com/<your-org>/pendulum-energy-sim.git
cd pendulum-energy-sim
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
codex/setup-streamlit-pendulum-energy-simulator-kxmbq0
pip install --upgrade pip

main
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually [http://localhost:8501](http://localhost:8501)).

## Usage

* Tune **Array & Wind**: number of pendulums, swept area per pendulum, Cp, wind profile.
* Choose **Harvest Path**:

  * Hinge damping (simple)
  * Shaft + Flywheel + Generator (recommended)
* **Generator mode**:

  * `Fixed DC Bus`: requires Ke, Rs, Vdc, diode drop, rectifier efficiency.
  * `MPPT Resistive`: uses copper-limited MPP approximation with converter efficiency.
  * `LUT`: upload CSV with columns:

    * Required: `omega_rad_s`, `tau_gen_Nm`, `P_elec_W`
    * Optional: `Vdc`, `temp_C`
  * If your LUT power already includes drivetrain (post-gearbox/alt), toggle **CHAIN_EFF=1**.

## Data schema (LUT)

Example CSV is in `data/generator_curve_example.csv`.

Columns:

* `omega_rad_s`: generator shaft speed [rad/s]
* `tau_gen_Nm`: opposing torque [Nm] (if unsigned, flip “torque sign” in UI)
* `P_elec_W`: electrical output (be consistent: generator AC or DC after rectifier)
* Optional: `Vdc` [V] and `temp_C` [°C] to unlock 2D/3D interpolation

## Physics Guardrails

* Instant aerodynamic input is capped: |τ·ω| ≤ 0.5·ρ·A·v³·Cp (per pendulum).
* Average electrical output must remain ≤ array mean-wind aero cap. The app flags violations.

## Deploy

* **Local**: `streamlit run app.py`
* **Replit**: import repo, set `run = "streamlit run app.py"` in Replit’s config.
 codex/setup-streamlit-pendulum-energy-simulator-kxmbq0
* **Render**: connect the repo and Render will pick up `render.yaml` to provision a Python web service. The build step upgrades `pip` before installing the pinned dependencies (Streamlit stack + SciPy) to avoid resolver regressions. Check the Render deploy logs if you need to debug installation issues.

* **Render**: connect the repo and Render will pick up `render.yaml` to provision a Python web service.
  main
* **HF Spaces**: deploy with the same command; add a `requirements.txt`.

## License

MIT
