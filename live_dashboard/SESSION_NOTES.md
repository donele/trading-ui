# Session Notes

Last updated: 2026-03-03 (UTC)

## Scope
- Built and refined a local Dash dashboard for Binance US order logs.
- Focused on latest-day visualization from latest `order*.log` file.

## Current Dashboard
- Plot 1: New orders and fills per 10-minute bucket (UTC).
- Plot 2: Cumulative filled notional per `strategy_id:symbol` with bucket-size markers.
- Plot 3: Drilldown for clicked Plot 2 bucket/key:
  - Fill executed prices (points, colored by `client_order_id`).
  - Marker shape by side: `BID/BUY` up-triangle, `ASK/SELL` down-triangle.
  - Mid-price overlay from `log/state/*.csv` as step line (`hv`).
  - Mid-price line width reduced to improve visibility of fill-side triangle markers.
- Plot 4: Client-order window for clicked fill marker in Plot 3:
  - Triggered by click on a fill marker in Plot 3.
  - X-axis from `start_time - 1m` to `end_time + 1m`.
  - Fill executed prices as markers.
  - Bid/ask prices from state CSV as step lines (`hv`).
  - Order price from initial `NEW` shown as a horizontal line from order start to order end.

## Key Logic Decisions
- Count new orders by unique `(strategy_id, client_order_id)` first, then bucket.
- Count fills/notional only for statuses `PARTIAL_FILL` and `FILLED`.
- Fill rows are only valid if a prior NEW exists for the same:
  - `(strategy_id, client_order_id, symbol)`.
- Client-order lifecycle window key is `(strategy_id, client_order_id, symbol)`:
  - Start time: first `NEW`.
  - End time: last timestamp from any message type.
- IDs normalized as strings end-to-end to avoid JS precision/type drift issues.

## Mid-Price Data Handling
- State CSV default path: `ORDER_LOG_DIR/state` (or `STATE_CSV_DIR` override).
- CSV filename pattern used: `{symbol}.*.{YYYYMMDD}.csv`.
- Mid = `(bid + ask) / 2`, with filtering:
  - prefer `bid_price/ask_price` if present,
  - require `book_valid == 1` if available,
  - drop non-positive values,
  - drop extreme sentinel/outlier values (`>= 1e9`).

## Performance/Simplification
- Removed multi-day navigation/calendar.
- Reads latest date from latest order log file only.
- Removed top-chart rangeslider mini-plot.
- Marker size is zero for zero-value buckets where requested.

## Known Operational Notes
- Current local default is `debug=True` for development convenience.
- In restricted environments, Dash `debug=True` can fail due to `/dev/shm` permission limits.
- If that occurs, run with `debug=False`.

## Git
- Committed and pushed:
  - Commit: `0cf3f13`
  - Message: `Add client order lifecycle chart with order price overlay`

## Useful Commands
- Run app:
  - `python dash_app.py`
- Optional state override:
  - `STATE_CSV_DIR=/path/to/state python dash_app.py`
- Syntax check:
  - `python -m py_compile dash_app.py order_data.py`

## Next Ideas (optional)
- Add legend grouping/toggles for dense client traces in Plot 3.
- Add clipping/winsorization option for mid-price outlier handling.
- Add persistent runbook for restart/debug steps.
