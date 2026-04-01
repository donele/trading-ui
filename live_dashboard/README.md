# Trading Dashboard

Local Plotly Dash app for visualizing order activity from Binance US strategy logs.

## What It Shows
- **Chart 1**: New orders and fills per 10-minute bucket (UTC) for the latest date found in the latest order log file.
- **Chart 2**: Cumulative filled notional per `strategy_id:symbol` key.
- **Chart 3** (drilldown): Click a point in Chart 2 to view:
  - Fill executed prices in that 10-minute bucket (points grouped by `client_order_id`).
  - Marker shape by side: `BID/BUY` = triangle-up, `ASK/SELL` = triangle-down.
  - Mid-price overlay (`(bid+ask)/2`) from state CSV files.
- **Chart 4** (client order window): Click a fill marker in Chart 3 to view:
  - X-axis window from `start_time - 1 minute` to `end_time + 1 minute`.
  - Start time = first `NEW` timestamp for `(strategy_id, client_order_id, symbol)`.
  - End time = last timestamp of any message for that same key.
  - Fill executed prices (markers) plus bid/ask step lines from state CSV files.

## Data Sources
- Order logs directory (default):
  - `/home/jdlee/workspace/sgt/livesim/binance_us/log`
  - files matching `order*.YYYYMMDD.log`
- State CSV directory (default):
  - `<ORDER_LOG_DIR>/state`
  - filename example: `ETH__USDT__SPOT__BINANCEUS.0.20260302.csv`

## Key Parsing Rules
- Timestamps are treated as UTC.
- New-order counts are deduplicated by unique `(strategy_id, client_order_id)`, then bucketed.
- Fill metrics include only rows with `order_status` in:
  - `PARTIAL_FILL`
  - `FILLED`
- Fill rows are counted only if a prior `NEW` exists for the same:
  - `(strategy_id, client_order_id, symbol)`
- IDs are normalized to strings to avoid precision/type issues in browser callbacks.

## Mid-Price Filtering
For state CSV data, the app:
- prefers `bid_price/ask_price` over raw `bid/ask` when available,
- uses `book_valid == 1` when available,
- drops non-positive prices,
- drops extreme invalid values (for example sentinel values near `9e18`).

## Requirements
- Python 3.11+
- Dependencies:
  - `dash>=3.12`
  - `pandas>=2.1`

Install:

```bash
pip install -r requirements.txt
```

## Run

```bash
python dash_app.py
```

Open:
- `http://127.0.0.1:8050`

## Environment Variables
- `ORDER_LOG_DIR`: override log directory.
- `STATE_CSV_DIR`: override state CSV directory (defaults to `<ORDER_LOG_DIR>/state`).

Example:

```bash
ORDER_LOG_DIR=/path/to/log STATE_CSV_DIR=/path/to/state python dash_app.py
```

## Notes
- App refresh interval is 30 seconds.
- If running in restricted environments, Dash debug mode can fail due to `/dev/shm` permissions.
