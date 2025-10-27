# Pendulum Array Investor Analysis (Streamlit)

Streamlit dashboard for bankability analysis of pendulum-array wind harvesters:
- Weibull and histogram wind models with automatic AEP calculation
- Module phase-offset smoothing, whole-site rollups, and optional curtailment relief modelling
- CapEx waterfall, LCOE sensitivity table, and scenario sizing toolkit
- CSV exports are built-in; add `kaleido` if you want server-side PNG downloads for Plotly charts

## Quickstart

```bash
git clone https://github.com/<your-org>/dpc.git
cd dpc
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
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

Optional PNG downloads depend on [Plotly Kaleido](https://github.com/plotly/Kaleido). To enable them on Render, add
`kaleido` to `requirements.txt` or install it via the Render shell. Without it, the app displays a notice and keeps
running normally.

If a Render deploy fails:

1. Retry the deploy with **Clear build cache** – cached wheels occasionally clash with new pins.
2. Re-run `pip install --no-cache-dir -r requirements.txt` locally inside a clean virtualenv to reproduce resolver errors.
3. If pip reports an error mentioning `<<<<<`/`>>>>>`, your `requirements.txt` still contains merge markers; remove them and re-run the install.
4. Once the install succeeds, commit the fixes and push. Render rebuilds automatically with the cached layer reset.

The sidebar **Runtime debug** panel surfaces Python, Streamlit, Plotly, NumPy, Pandas, and Kaleido status so you can
confirm versions directly on the deployed app.

## Repository layout

```
app.py            # Streamlit UI
finance_core.py   # Energy + finance helper functions
render.yaml       # Render web service definition
requirements.txt  # Pinned Streamlit + Plotly stack
```

## License

MIT
