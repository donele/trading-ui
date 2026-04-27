# Simulation Dashboard

Dash app to browse simulation heads and visualize:

- `bid`/`ask` from `log/state/*.parquet`
- order price lines from `log/order.????????.{parquet,log}` grouped by `client_order_id`

## Run

```bash
cd /home/jdlee/repos/simulation-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open `http://127.0.0.1:8050`.

## SimData Dashboard

The second dashboard variant uses `SimData` from `app2.py` and Plotly Dash:

```bash
python3 sim_dashboard/app2_dash.py
```

It defaults to `http://127.0.0.1:8051` and can be overridden with `DASH2_PORT`.

## Data roots

The app scans:

- `~/workspace/sgt/dumpsim`
- `~/workspace/sgt/livesim`
- `~/workspace/sgt/tradesim`

Each head directory must contain:

- `log/state/`
- at least one `log/order.????????.log`
