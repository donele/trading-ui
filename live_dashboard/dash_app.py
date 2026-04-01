from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from dash import Dash, Input, Output, dcc, html
import plotly.graph_objs as go
from plotly.colors import qualitative
import pandas as pd

from order_data import DEFAULT_INTERVAL_MINUTES, load_latest_day_metrics

LOG_DIR = Path(os.getenv("ORDER_LOG_DIR", "/home/jdlee/workspace/sgt/livesim/binance_us/log"))
STATE_DIR = Path(os.getenv("STATE_CSV_DIR", str(LOG_DIR / "state")))
REFRESH_INTERVAL_MS = 30_000


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for name in candidates:
        hit = lowered.get(name.lower())
        if hit:
            return hit
    return None


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        max_abs = numeric.abs().max()
        unit = "s"
        if max_abs >= 1e17:
            unit = "ns"
        elif max_abs >= 1e14:
            unit = "us"
        elif max_abs >= 1e11:
            unit = "ms"
        return pd.to_datetime(numeric, errors="coerce", utc=True, unit=unit).dt.tz_convert(None)
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.notna().any():
        return parsed.dt.tz_convert(None)
    return pd.to_datetime(series, errors="coerce")


def _marker_symbol_for_side(side: object) -> str:
    text = str(side or "").upper()
    if text in {"BID", "BUY"}:
        return "triangle-up"
    if text in {"ASK", "SELL"}:
        return "triangle-down"
    return "circle"


@lru_cache(maxsize=32)
def _load_book_series_for_day(state_dir: str, symbol: str, day_yyyymmdd: str) -> pd.DataFrame:
    root = Path(state_dir)
    if not root.exists():
        return pd.DataFrame(columns=["event_time", "bid_price", "ask_price"])

    files = sorted(root.glob(f"{symbol}.*.{day_yyyymmdd}.csv"))
    if not files:
        return pd.DataFrame(columns=["event_time", "bid_price", "ask_price"])

    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue

        cols = header.columns.tolist()
        ts_col = _pick_column(cols, ["timestamp", "time", "ts", "event_time", "recv_time", "exchange_time"])
        # Prefer cleaned price columns first when available.
        bid_col = _pick_column(cols, ["bid_price", "best_bid_price", "bid", "best_bid"])
        ask_col = _pick_column(cols, ["ask_price", "best_ask_price", "ask", "best_ask"])
        book_valid_col = _pick_column(cols, ["book_valid"])
        if not ts_col or not bid_col or not ask_col:
            continue

        usecols = [ts_col, bid_col, ask_col]
        if book_valid_col:
            usecols.append(book_valid_col)
        try:
            df = pd.read_csv(path, usecols=usecols)
        except Exception:
            continue
        if df.empty:
            continue

        event_time = _parse_timestamp_series(df[ts_col])
        bid = pd.to_numeric(df[bid_col], errors="coerce")
        ask = pd.to_numeric(df[ask_col], errors="coerce")
        valid = bid.notna() & ask.notna()
        valid &= bid > 0
        valid &= ask > 0
        # Filter obvious sentinel/corrupt values (e.g. 9.22e18) and extreme outliers.
        valid &= bid < 1e9
        valid &= ask < 1e9
        if book_valid_col:
            book_valid = pd.to_numeric(df[book_valid_col], errors="coerce")
            valid &= book_valid == 1
        out = pd.DataFrame(
            {"event_time": event_time[valid], "bid_price": bid[valid], "ask_price": ask[valid]}
        ).dropna()
        if not out.empty:
            frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["event_time", "bid_price", "ask_price"])
    combined = pd.concat(frames, ignore_index=True).sort_values("event_time")
    return combined

app = Dash(__name__)
app.layout = html.Div(
    [
        html.H1("Latest Day Orders Dashboard", style={"marginBottom": "0.4rem"}),
        html.Div(id="date-label", style={"fontSize": "0.95rem", "marginBottom": "0.6rem"}),
        dcc.Graph(id="new-orders-graph"),
        dcc.Graph(id="cumulative-notional-graph", style={"marginTop": "0.8rem"}),
        dcc.Graph(id="bucket-fills-price-graph", style={"marginTop": "0.8rem"}),
        dcc.Graph(id="client-order-window-graph", style={"marginTop": "0.8rem"}),
        dcc.Interval(id="refresh-interval", interval=REFRESH_INTERVAL_MS, n_intervals=0),
        dcc.Store(id="metrics-store"),
        html.Div(id="log-source", style={"fontSize": "0.8rem", "color": "#666", "marginTop": "0.5rem"}),
    ],
    style={"fontFamily": "Inter, sans-serif", "maxWidth": "980px", "margin": "0 auto", "padding": "1rem"},
)


@app.callback(
    Output("metrics-store", "data"),
    Output("new-orders-graph", "figure"),
    Output("cumulative-notional-graph", "figure"),
    Output("date-label", "children"),
    Output("log-source", "children"),
    Input("refresh-interval", "n_intervals"),
)
def refresh_dashboard(_: int):
    try:
        metrics = load_latest_day_metrics(LOG_DIR, interval_minutes=DEFAULT_INTERVAL_MINUTES)
    except FileNotFoundError:
        empty = go.Figure()
        empty.update_layout(title="No data", xaxis_title="UTC time", yaxis_title="Count")
        empty2 = go.Figure()
        empty2.update_layout(title="No data", xaxis_title="UTC time", yaxis_title="Notional")
        return {}, empty, empty2, "No data available", f"No logs found in {LOG_DIR}"

    day_iso = metrics["date_iso"]
    new_series = metrics["new"]
    fill_series = metrics["fills"]
    per_key_bucket = metrics["bucket_notional_by_key"]
    per_key_cumulative = metrics["cumulative_notional_by_key"]
    new_marker_sizes = [7 if float(v) > 0 else 0 for v in new_series.values]

    top_fig = go.Figure(
        data=[
            go.Scatter(
                x=new_series.index,
                y=new_series.values,
                mode="lines+markers",
                name="New orders",
                marker=dict(size=new_marker_sizes),
            ),
            go.Scatter(
                x=fill_series.index,
                y=fill_series.values,
                mode="lines",
                name="Fills",
                line=dict(dash="dash", color="#ef553b"),
            ),
        ]
    )
    top_fig.update_layout(
        title=f"New orders and fills ({day_iso})",
        xaxis_title="UTC time",
        yaxis_title="Count per 10m",
        hovermode="x unified",
        margin=dict(t=50, b=40, l=40, r=20),
    )
    top_fig.update_xaxes(
        tickformat="%H:%M",
        rangeslider=dict(visible=False),
        showspikes=True,
        spikemode="across",
    )

    bottom_fig = go.Figure()
    for key in sorted(per_key_cumulative):
        cumulative_series = per_key_cumulative[key]
        bucket_series = per_key_bucket[key]
        bucket_values = bucket_series.values
        max_bucket = max(bucket_values) if len(bucket_values) else 0
        marker_sizes = []
        for value in bucket_values:
            if float(value) <= 0:
                marker_sizes.append(0)
            elif max_bucket <= 0:
                marker_sizes.append(6)
            else:
                # Scale markers from 6..18 based on bucket notional magnitude.
                marker_sizes.append(6 + 12 * (float(value) / float(max_bucket)))
        bottom_fig.add_trace(
            go.Scatter(
                x=cumulative_series.index,
                y=cumulative_series.values,
                mode="lines+markers",
                name=key,
                customdata=[[v] for v in bucket_series.values],
                marker=dict(size=marker_sizes, sizemode="diameter"),
                hovertemplate=(
                    "Key %{fullData.name}<br>"
                    "Bucket %{x|%Y-%m-%d %H:%M}<br>"
                    "Bucket notional %{customdata[0]:,.2f}<br>"
                    "Cumulative %{y:,.2f}<extra></extra>"
                ),
            )
        )
    bottom_fig.update_layout(
        title="Cumulative filled notional by strategy and symbol",
        xaxis_title="UTC time",
        yaxis_title="Cumulative notional",
        margin=dict(t=50, b=40, l=40, r=20),
    )
    bottom_fig.update_xaxes(tickformat="%H:%M")

    label = f"Displaying latest available date: {day_iso}"
    source = f"Source log file: {metrics['source_file']}"
    return metrics, top_fig, bottom_fig, label, source


@app.callback(
    Output("bucket-fills-price-graph", "figure"),
    Input("cumulative-notional-graph", "clickData"),
    Input("metrics-store", "data"),
)
def update_bucket_fill_prices(click_data, metrics):
    fig = go.Figure()
    if not metrics:
        fig.update_layout(
            title="Click a point in the cumulative notional chart to inspect fill prices",
            xaxis_title="UTC time",
            yaxis_title="Executed price",
        )
        return fig

    if not click_data or "points" not in click_data or not click_data["points"]:
        fig.update_layout(
            title="Click a point in the cumulative notional chart to inspect fill prices",
            xaxis_title="UTC time",
            yaxis_title="Executed price",
        )
        return fig

    point = click_data["points"][0]
    bucket_iso = pd.Timestamp(point["x"]).isoformat()
    curve_number = point.get("curveNumber")
    keys = sorted(metrics.get("cumulative_notional_by_key", {}).keys())
    if curve_number is None or curve_number >= len(keys):
        fig.update_layout(title="Unable to determine selected key", xaxis_title="UTC time", yaxis_title="Executed price")
        return fig
    selected_key = keys[curve_number]
    if not selected_key:
        fig.update_layout(title="Unable to determine selected key", xaxis_title="UTC time", yaxis_title="Executed price")
        return fig

    bucket_start = pd.Timestamp(bucket_iso)
    bucket_end = bucket_start + pd.Timedelta(minutes=DEFAULT_INTERVAL_MINUTES)
    key_symbol = selected_key.split(":", 1)[1] if ":" in selected_key else selected_key

    events = metrics.get("fill_events", [])
    filtered = []
    for event in events:
        if event.get("symbol") != key_symbol:
            continue
        if event.get("key") != selected_key:
            continue
        event_time = pd.Timestamp(event["event_time_iso"])
        if bucket_start <= event_time < bucket_end:
            filtered.append(event)

    filtered.sort(key=lambda item: item["event_time_iso"])

    day_yyyymmdd = bucket_start.strftime("%Y%m%d")
    book_day = _load_book_series_for_day(str(STATE_DIR), key_symbol, day_yyyymmdd)
    if not book_day.empty:
        mid_bucket = book_day[(book_day["event_time"] >= bucket_start) & (book_day["event_time"] < bucket_end)].copy()
        if not mid_bucket.empty:
            mid_bucket["mid_price"] = (mid_bucket["bid_price"] + mid_bucket["ask_price"]) / 2.0
            fig.add_trace(
                go.Scatter(
                    x=mid_bucket["event_time"],
                    y=mid_bucket["mid_price"],
                    mode="lines",
                    name="mid price",
                    line=dict(color="#2a9d8f", width=1.0, shape="hv"),
                    hovertemplate="Time %{x|%Y-%m-%d %H:%M:%S.%L}<br>Mid %{y:,.4f}<extra></extra>",
                )
            )

    if not filtered and fig.data:
        fig.update_layout(
            title=f"Mid price for {selected_key} in bucket {bucket_start.strftime('%Y-%m-%d %H:%M')} (no fills)",
            xaxis_title="UTC time",
            yaxis_title="Price",
            margin=dict(t=50, b=40, l=40, r=20),
        )
        return fig
    if not filtered:
        fig.update_layout(
            title=f"No fills or mid-price data for {selected_key} in bucket {bucket_start.strftime('%Y-%m-%d %H:%M')}",
            xaxis_title="UTC time",
            yaxis_title="Executed price",
            margin=dict(t=50, b=40, l=40, r=20),
        )
        return fig

    palette = qualitative.Plotly
    by_client: dict[str, list[dict]] = {}
    for event in filtered:
        client_id = event.get("client_order_id")
        client_key = str(client_id) if client_id is not None else "UNKNOWN"
        by_client.setdefault(client_key, []).append(event)

    for idx, client_key in enumerate(sorted(by_client.keys())):
        points = by_client[client_key]
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=[pd.Timestamp(p["event_time_iso"]) for p in points],
                y=[float(p["executed_price"]) for p in points],
                mode="markers",
                name=f"client {client_key}",
                marker=dict(
                    color=color,
                    size=8,
                    symbol=[_marker_symbol_for_side(p.get("side")) for p in points],
                ),
                customdata=[
                    [
                        p.get("filled_qty", 0.0),
                        p.get("client_order_id"),
                        p.get("side", ""),
                        p.get("strategy_id", ""),
                        p.get("symbol", ""),
                        p.get("client_order_key", ""),
                    ]
                    for p in points
                ],
                hovertemplate=(
                    "Time %{x|%Y-%m-%d %H:%M:%S.%L}<br>"
                    "Executed price %{y:,.4f}<br>"
                    "Filled qty %{customdata[0]:,.6f}<br>"
                    "Client order %{customdata[1]}<br>"
                    "Side %{customdata[2]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"Executed fills and mid price for {selected_key} in bucket {bucket_start.strftime('%Y-%m-%d %H:%M')}",
        xaxis_title="UTC time",
        yaxis_title="Price",
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig


@app.callback(
    Output("client-order-window-graph", "figure"),
    Input("bucket-fills-price-graph", "clickData"),
    Input("metrics-store", "data"),
)
def update_client_order_window(click_data, metrics):
    fig = go.Figure()
    if not metrics:
        fig.update_layout(
            title="Click a fill point in Chart 3 to inspect a client-order window",
            xaxis_title="UTC time",
            yaxis_title="Price",
        )
        return fig
    if not click_data or "points" not in click_data or not click_data["points"]:
        fig.update_layout(
            title="Click a fill point in Chart 3 to inspect a client-order window",
            xaxis_title="UTC time",
            yaxis_title="Price",
        )
        return fig

    point = click_data["points"][0]
    customdata = point.get("customdata")
    if not isinstance(customdata, list) or len(customdata) < 6:
        fig.update_layout(
            title="Select a fill marker in Chart 3 (not the mid-price line)",
            xaxis_title="UTC time",
            yaxis_title="Price",
        )
        return fig

    client_order_key = str(customdata[5] or "")
    if not client_order_key:
        strategy_id = str(customdata[3] or "")
        client_order_id = str(customdata[1] or "")
        symbol = str(customdata[4] or "")
        if strategy_id and client_order_id and symbol:
            client_order_key = f"{strategy_id}:{client_order_id}:{symbol}"
    windows = metrics.get("client_order_windows", {})
    window = windows.get(client_order_key)
    if not window:
        fig.update_layout(
            title=f"No lifecycle window found for {client_order_key or 'selected order'}",
            xaxis_title="UTC time",
            yaxis_title="Price",
        )
        return fig

    start_time = pd.Timestamp(window["start_time_iso"])
    end_time = pd.Timestamp(window["end_time_iso"])
    x_start = start_time - pd.Timedelta(minutes=1)
    x_end = end_time + pd.Timedelta(minutes=1)
    order_price = pd.to_numeric(window.get("order_price"), errors="coerce")
    if pd.notna(order_price) and float(order_price) > 0:
        fig.add_trace(
            go.Scatter(
                x=[start_time, end_time],
                y=[float(order_price), float(order_price)],
                mode="lines",
                name="order price",
                line=dict(color="#2a9d8f", width=2.0, dash="dot"),
                hovertemplate="Order price %{y:,.4f}<extra></extra>",
            )
        )

    symbol = client_order_key.rsplit(":", 1)[-1] if ":" in client_order_key else str(customdata[4] or "")
    day_keys = {x_start.strftime("%Y%m%d"), x_end.strftime("%Y%m%d")}
    book_frames = [_load_book_series_for_day(str(STATE_DIR), symbol, key) for key in sorted(day_keys)]
    book_frames = [frame for frame in book_frames if not frame.empty]
    if book_frames:
        book_all = pd.concat(book_frames, ignore_index=True).sort_values("event_time")
        book_window = book_all[(book_all["event_time"] >= x_start) & (book_all["event_time"] <= x_end)]
        if not book_window.empty:
            fig.add_trace(
                go.Scatter(
                    x=book_window["event_time"],
                    y=book_window["bid_price"],
                    mode="lines",
                    name="bid",
                    line=dict(color="#1f77b4", width=1.8, shape="hv"),
                    hovertemplate="Time %{x|%Y-%m-%d %H:%M:%S.%L}<br>Bid %{y:,.4f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=book_window["event_time"],
                    y=book_window["ask_price"],
                    mode="lines",
                    name="ask",
                    line=dict(color="#ef553b", width=1.8, shape="hv"),
                    hovertemplate="Time %{x|%Y-%m-%d %H:%M:%S.%L}<br>Ask %{y:,.4f}<extra></extra>",
                )
            )

    fills = []
    for event in metrics.get("fill_events", []):
        if event.get("client_order_key") != client_order_key:
            continue
        event_time = pd.Timestamp(event["event_time_iso"])
        if x_start <= event_time <= x_end:
            fills.append(event)
    fills.sort(key=lambda item: item["event_time_iso"])
    if fills:
        fig.add_trace(
            go.Scatter(
                x=[pd.Timestamp(event["event_time_iso"]) for event in fills],
                y=[float(event["executed_price"]) for event in fills],
                mode="markers",
                name="fills",
                marker=dict(
                    size=9,
                    color="#111111",
                    symbol=[_marker_symbol_for_side(event.get("side")) for event in fills],
                ),
                customdata=[[event.get("filled_qty", 0.0), event.get("side", "")] for event in fills],
                hovertemplate=(
                    "Time %{x|%Y-%m-%d %H:%M:%S.%L}<br>"
                    "Executed %{y:,.4f}<br>"
                    "Filled qty %{customdata[0]:,.6f}<br>"
                    "Side %{customdata[1]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            f"Client-order window for {client_order_key}: "
            f"{x_start.strftime('%Y-%m-%d %H:%M:%S')} to {x_end.strftime('%Y-%m-%d %H:%M:%S')}"
        ),
        xaxis_title="UTC time",
        yaxis_title="Price",
        margin=dict(t=50, b=40, l=40, r=20),
    )
    fig.update_xaxes(range=[x_start, x_end])
    return fig


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
