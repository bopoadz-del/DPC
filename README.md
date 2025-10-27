# Pendulum Array Investor Analysis (Streamlit)

Streamlit dashboard for bankability analysis of pendulum-array wind harvesters:

- Weibull and histogram wind models with automatic AEP calculation
- Module phase-offset smoothing, whole-site rollups, and optional curtailment relief modelling
- CapEx waterfall, LCOE sensitivity table, and scenario sizing toolkit
- CSV/PNG export buttons so investors can pull artefacts directly from the UI

## Quickstart

```bash
git clone https://github.com/render-examples/DPC.git
cd DPC
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open Streamlit’s URL (default [http://localhost:8501](http://localhost:8501)).

## Usage highlights

1. **Physics inputs** – set swept area, Cp, drivetrain efficiency, availability.
2. **Wind resource** – pick Weibull *(k, c)* or upload a histogram CSV (`speed_mps`, `count`/`prob`).
3. **CapEx** – break out mechanical, electrical, civil/BOS, soft costs, and contingency.
4. **Phase offsets & curtailment** – explore smoothing index vs. inverter clipping.
5. **Finance outputs** – LCOE table (Weibull presets) plus investor scenarios with land/CapEx rollups.
6. **Exports** – PNG download buttons for Plotly figures and CSV exports for tables.
7. **Whole-site metrics** – enter modules/arrays to get consolidated CapEx, energy, and land footprint numbers.

## Render deployment & debugging

Render is configured via `render.yaml`:

- **Build** – `pip install --no-cache-dir -r requirements.txt`
- **Runtime** – `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- **Python** – 3.11 (set with `PYTHON_VERSION` env var)

If a Render deploy fails:

1. Retry the deploy with **Clear build cache** – cached wheels occasionally clash with new pins.
2. Re-run `pip install --no-cache-dir -r requirements.txt` locally inside a clean virtualenv to reproduce resolver errors.
3. Once the install succeeds, commit the fixes and push. Render rebuilds automatically with the cached layer reset.

## Repository layout

```
app.py            # Streamlit UI
finance_core.py   # Energy + finance helper functions
render.yaml       # Render web service definition
requirements.txt  # Pinned Streamlit + Plotly stack
```

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
* **Render**: connect the repo and Render will pick up `render.yaml` to provision a Python web service. The build step installs the pinned dependencies with `pip install --no-cache-dir -r requirements.txt` so it works with cached layers. If a build fails, re-run the deploy with “Clear build cache” and inspect the pip output in Render’s deploy logs to spot missing wheels.
* **HF Spaces**: deploy with the same command; add a `requirements.txt`.

## Debugging Render builds

Render caches wheels between deploys. When you bump dependency pins you may need to clear the build cache and redeploy. The easiest loop is:

1. Open the failed deploy, click **Retry** → **Clear build cache**.
2. If the install still fails, copy the offending `pip` command from the logs and run it locally inside a clean virtualenv. That exposes missing system libraries or typos before you push again.
3. Once `pip install -r requirements.txt` succeeds locally, commit the updated pins and retry the Render deploy.

## License

MIT
