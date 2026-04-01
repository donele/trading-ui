#!/usr/bin/env python3
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote

import dash
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, Patch, State, dcc, html, no_update


ROOT_ORDER = ("dumpsim", "livesim", "tradesim")
ROOTS = [Path.home() / "workspace" / "sgt" / name for name in ROOT_ORDER]
ORDER_PATTERN = "order.????????.log"
BPS_COLOR = "#C19A6B"

BUY_COLORS = (
    "#003F5C",  # deep blue-teal
    "#00B4D8",  # bright cyan
    "#006D77",  # dark teal
    "#83C5BE",  # light desaturated aqua
    "#0077B6",  # vivid blue
    "#2EC4B6",  # bright turquoise
    "#264653",  # slate teal
    "#90E0EF",  # pale cyan
    "#0A9396",  # saturated teal
    "#48CAE4",  # sky cyan
)
SELL_COLORS = (
    "#7F1D1D",  # deep red
    "#F97316",  # bright orange
    "#B91C1C",  # vivid red
    "#F59E0B",  # amber
    "#DC2626",  # strong red
    "#FDBA74",  # light orange
    "#9A3412",  # burnt orange
    "#FACC15",  # yellow
    "#EF4444",  # bright red
    "#EA580C",  # orange-red
)


@dataclass(frozen=True)
class StateFile:
    symbol: str
    date: str
    path: Path


def normalize_head(head: str) -> Path | None:
    try:
        resolved = Path(head).expanduser().resolve(strict=True)
    except FileNotFoundError:
        return None
    for root in ROOTS:
        if not root.exists():
            continue
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    return None


def parse_state_filename(path: Path) -> StateFile | None:
    # Expected form: SYMBOL.0.YYYYMMDD.csv
    parts = path.name.split(".")
    if len(parts) < 4:
        return None
    if parts[-1] != "csv":
        return None
    date = parts[-2]
    if len(date) != 8 or not date.isdigit():
        return None
    symbol = ".".join(parts[:-3])
    if not symbol:
        return None
    return StateFile(symbol=symbol, date=date, path=path)


def discover_heads() -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {name: [] for name in ROOT_ORDER}
    for root_name, root in zip(ROOT_ORDER, ROOTS):
        if not root.exists():
            continue
        for log_dir in root.rglob("log"):
            if not log_dir.is_dir():
                continue
            state_dir = log_dir / "state"
            if not state_dir.is_dir():
                continue
            if not any(log_dir.glob(ORDER_PATTERN)):
                continue
            grouped[root_name].append(log_dir.parent)
        grouped[root_name].sort()
    return grouped


def state_files_for_head(head: Path) -> list[StateFile]:
    state_dir = head / "log" / "state"
    if not state_dir.is_dir():
        return []
    rows: list[StateFile] = []
    for csv_path in state_dir.glob("*.csv"):
        parsed = parse_state_filename(csv_path)
        if parsed is None:
            continue
        # Keep entries that can map to an order log date.
        if not (head / "log" / f"order.{parsed.date}.log").exists():
            continue
        rows.append(parsed)
    rows.sort(key=lambda x: (x.symbol, x.date))
    return rows


def _pick_ts_us(payload: dict, line_prefix: str) -> int | None:
    for key in (
        "our_time",
        "create_time",
        "exchange_time",
        "last_exchange_time",
        "last_cancel_time",
        "cancel_time",
    ):
        value = payload.get(key)
        if isinstance(value, int) and value > 0:
            return value
    # Fallback to log prefix wall-clock time if present.
    try:
        prefix_ts = datetime.strptime(line_prefix[:26], "%Y-%m-%d %H:%M:%S.%f")
        return int(prefix_ts.timestamp() * 1_000_000)
    except ValueError:
        return None


def _normalize_side(raw: str | None) -> str | None:
    if raw is None:
        return None
    upper = str(raw).upper()
    if upper in ("BID", "BUY"):
        return "BUY"
    if upper in ("ASK", "SELL"):
        return "SELL"
    return None


def _pick_parent_order_id(payload: dict) -> str | None:
    for key in ("parent_order_id", "parent_id", "parentOrderId"):
        value = payload.get(key)
        if value is None:
            continue
        return str(value)
    return None


def _pick_order_size(payload: dict) -> float | None:
    for key in ("qty", "order_qty", "size"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def parse_plotly_time(value) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.DatetimeIndex):
        if len(parsed) == 0:
            return None
        return parsed[0]
    return parsed


def format_interval_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    start_min = start.floor("min")
    end_min = end.floor("min")
    if end > end_min:
        end_min = end_min + timedelta(minutes=1)
    return f"{start_min.strftime('%H:%M')} - {end_min.strftime('%H:%M')}"


def first_mid_price(state_df: pd.DataFrame) -> float | None:
    if state_df.empty:
        return None
    mids = (state_df["bid"] + state_df["ask"]) / 2.0
    mids = mids.dropna()
    if mids.empty:
        return None
    first = mids.iloc[0]
    try:
        first_f = float(first)
    except (TypeError, ValueError):
        return None
    if first_f <= 0:
        return None
    return first_f


def first_reference_price(state_df: pd.DataFrame, orders: list[dict], fills: list[dict]) -> float | None:
    mid = first_mid_price(state_df)
    if mid is not None:
        return mid
    for row in orders:
        px = row.get("price")
        if px is None:
            continue
        try:
            px_f = float(px)
        except (TypeError, ValueError):
            continue
        if px_f > 0:
            return px_f
    for row in fills:
        px = row.get("price")
        if px is None:
            continue
        try:
            px_f = float(px)
        except (TypeError, ValueError):
            continue
        if px_f > 0:
            return px_f
    return None


def build_bps_ticks(price_min: float, price_max: float, base_price: float, count: int = 7) -> tuple[list[float], list[str]]:
    if base_price <= 0:
        return [], []
    if price_max <= price_min:
        return [price_min], ["0.0"]
    if count < 2:
        count = 2
    min_bps = ((price_min / base_price) - 1.0) * 10000.0
    max_bps = ((price_max / base_price) - 1.0) * 10000.0
    if max_bps < min_bps:
        min_bps, max_bps = max_bps, min_bps
    if math.isclose(max_bps, min_bps):
        return [price_min], [f"{min_bps:.0f}"]

    span = max_bps - min_bps
    raw_step = span / (count - 1)
    exp = math.floor(math.log10(raw_step)) if raw_step > 0 else 0
    base = 10 ** exp
    norm = raw_step / base
    if norm <= 1:
        step_bps = 1 * base
    elif norm <= 2:
        step_bps = 2 * base
    elif norm <= 5:
        step_bps = 5 * base
    else:
        step_bps = 10 * base

    start_bps = math.floor(min_bps / step_bps) * step_bps
    end_bps = math.ceil(max_bps / step_bps) * step_bps
    bps_ticks: list[float] = []
    current = start_bps
    max_ticks = 40
    while current <= end_bps + 1e-9 and len(bps_ticks) < max_ticks:
        bps_ticks.append(current)
        current += step_bps

    price_ticks = [base_price * (1.0 + (bps / 10000.0)) for bps in bps_ticks]
    if step_bps >= 1:
        labels = [f"{int(round(bps))}" for bps in bps_ticks]
    else:
        labels = [f"{bps:.1f}" for bps in bps_ticks]
    return price_ticks, labels


@lru_cache(maxsize=32)
def load_state_frame(state_path_str: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    state_path = Path(state_path_str)
    columns = pd.read_csv(state_path, nrows=0).columns.tolist()
    bid_col = "bid" if "bid" in columns else "bid_price"
    ask_col = "ask" if "ask" in columns else "ask_price"
    if bid_col not in columns or ask_col not in columns:
        return pd.DataFrame(columns=["time", "bid", "ask"])
    df = pd.read_csv(
        state_path,
        usecols=["time", bid_col, ask_col],
    )
    df = df.rename(columns={bid_col: "bid", ask_col: "ask"})
    df["time"] = pd.to_datetime(df["time"], unit="us", errors="coerce")
    for col in ("bid", "ask"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Filter obvious sentinel / invalid values.
        df.loc[(df[col] <= 0) | (df[col] >= 1e12), col] = pd.NA
    df = df.dropna(subset=["time"])
    return df


@lru_cache(maxsize=32)
def load_position_frame(state_path_str: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    state_path = Path(state_path_str)
    columns = pd.read_csv(state_path, nrows=0).columns.tolist()
    pos_col = "position" if "position" in columns else ("pos" if "pos" in columns else None)
    if pos_col is None:
        return pd.DataFrame(columns=["time", "position"])
    df = pd.read_csv(state_path, usecols=["time", pos_col])
    df = df.rename(columns={pos_col: "position"})
    df["time"] = pd.to_datetime(df["time"], unit="us", errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    return df


@lru_cache(maxsize=32)
def load_daily_metrics_frame(state_path_str: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    state_path = Path(state_path_str)
    columns = pd.read_csv(state_path, nrows=0).columns.tolist()

    def pick_first(*candidates: str) -> str | None:
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None

    position_col = pick_first("position", "pos")
    pnl_col = pick_first("pnl", "net_pnl", "total_pnl")
    notional_raw_col = pick_first("notional_traded")
    notional_cum_col = pick_first("cum_notional_traded", "cumulative_notional_traded")
    size_raw_col = pick_first("size_traded")
    size_cum_col = pick_first("cum_size_traded", "cumulative_size_traded")

    usecols = ["time"]
    for col in (
        position_col,
        pnl_col,
        notional_raw_col,
        notional_cum_col,
        size_raw_col,
        size_cum_col,
    ):
        if col and col not in usecols:
            usecols.append(col)

    if len(usecols) == 1:
        return pd.DataFrame(columns=["time", "position", "pnl", "cum_notional_traded", "cum_size_traded"])

    df = pd.read_csv(state_path, usecols=usecols)
    rename_map = {}
    if position_col:
        rename_map[position_col] = "position"
    if pnl_col:
        rename_map[pnl_col] = "pnl"
    if notional_raw_col:
        rename_map[notional_raw_col] = "notional_traded"
    if notional_cum_col:
        rename_map[notional_cum_col] = "cum_notional_traded"
    if size_raw_col:
        rename_map[size_raw_col] = "size_traded"
    if size_cum_col:
        rename_map[size_cum_col] = "cum_size_traded"
    df = df.rename(columns=rename_map)

    df["time"] = pd.to_datetime(df["time"], unit="us", errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")

    for col in ("position", "pnl", "notional_traded", "size_traded", "cum_notional_traded", "cum_size_traded"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "cum_notional_traded" not in df.columns and "notional_traded" in df.columns:
        df["cum_notional_traded"] = df["notional_traded"]
    if "cum_size_traded" not in df.columns and "size_traded" in df.columns:
        df["cum_size_traded"] = df["size_traded"]

    for col in ("position", "pnl", "cum_notional_traded", "cum_size_traded"):
        if col not in df.columns:
            df[col] = pd.NA
    return df


@lru_cache(maxsize=32)
def load_orders(order_log_path_str: str, symbol: str, mtime_ns: int) -> list[dict]:
    del mtime_ns
    order_log_path = Path(order_log_path_str)
    grouped: dict[str, dict] = {}
    with order_log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if symbol not in line:
                continue
            start = line.find("{")
            end = line.rfind("}")
            if start < 0 or end <= start:
                continue
            payload_raw = line[start : end + 1]
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                continue

            if payload.get("symbol") != symbol:
                continue
            client_order_id = payload.get("client_order_id")
            if client_order_id is None:
                continue

            ts_us = _pick_ts_us(payload, line)
            if ts_us is None:
                continue

            price = payload.get("price")
            if price is None:
                price = payload.get("order_price")
            try:
                price = float(price) if price is not None else None
            except (TypeError, ValueError):
                price = None

            side = _normalize_side(payload.get("side"))
            parent_order_id = _pick_parent_order_id(payload)
            size = _pick_order_size(payload)
            key = str(client_order_id)
            item = grouped.get(key)
            if item is None:
                grouped[key] = {
                    "parent_order_id": parent_order_id,
                    "client_order_id": key,
                    "side": side,
                    "price": price,
                    "size": size,
                    "start_ts_us": ts_us,
                    "end_ts_us": ts_us,
                }
                continue

            item["start_ts_us"] = min(item["start_ts_us"], ts_us)
            item["end_ts_us"] = max(item["end_ts_us"], ts_us)
            if item["parent_order_id"] is None and parent_order_id is not None:
                item["parent_order_id"] = parent_order_id
            if item["side"] is None and side is not None:
                item["side"] = side
            if item["price"] is None and price is not None:
                item["price"] = price
            if item["size"] is None and size is not None:
                item["size"] = size

    rows: list[dict] = []
    for item in grouped.values():
        if item["price"] is None:
            continue
        if item["side"] is None:
            continue
        if item["end_ts_us"] <= item["start_ts_us"]:
            item["end_ts_us"] = item["start_ts_us"] + 1
        rows.append(item)
    rows.sort(key=lambda x: x["start_ts_us"])
    return rows


@lru_cache(maxsize=32)
def load_fills(order_log_path_str: str, symbol: str, mtime_ns: int) -> list[dict]:
    del mtime_ns
    order_log_path = Path(order_log_path_str)
    rows: list[dict] = []
    with order_log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if symbol not in line:
                continue
            start = line.find("{")
            end = line.rfind("}")
            if start < 0 or end <= start:
                continue
            payload_raw = line[start : end + 1]
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                continue

            if payload.get("symbol") != symbol:
                continue
            side = _normalize_side(payload.get("side"))
            if side is None:
                continue
            ts_us = _pick_ts_us(payload, line)
            if ts_us is None:
                continue

            filled_qty = payload.get("filled_qty", 0)
            try:
                filled_qty_f = float(filled_qty)
            except (TypeError, ValueError):
                filled_qty_f = 0.0

            status = str(payload.get("order_status", "")).upper()
            is_fill = filled_qty_f > 0 or status in ("FILLED", "PARTIALLY_FILLED", "PARTIAL_FILL")
            if not is_fill:
                continue

            price = payload.get("executed_price")
            if price in (None, 0, 0.0):
                price = payload.get("average_fill_price")
            if price in (None, 0, 0.0):
                price = payload.get("order_price")
            try:
                price_f = float(price)
            except (TypeError, ValueError):
                continue
            if price_f <= 0:
                continue

            rows.append(
                {
                    "ts_us": ts_us,
                    "side": side,
                    "price": price_f,
                    "parent_order_id": _pick_parent_order_id(payload),
                    "client_order_id": str(payload.get("client_order_id", "")),
                    "filled_qty": filled_qty_f,
                }
            )
    rows.sort(key=lambda x: x["ts_us"])
    return rows


def render_index() -> html.Div:
    grouped_heads = discover_heads()
    sections: list = [
        html.H2("Simulation Heads"),
        html.P("Click a date to open daily metrics. From there, open the hourly bid/ask + order chart."),
    ]
    for root_name in ROOT_ORDER:
        heads = grouped_heads.get(root_name, [])
        sections.append(html.H3(root_name))
        if not heads:
            sections.append(html.P("(none found)"))
            continue
        for head in heads:
            state_entries = state_files_for_head(head)
            if not state_entries:
                continue
            symbol_map: dict[str, list[StateFile]] = defaultdict(list)
            for entry in state_entries:
                symbol_map[entry.symbol].append(entry)

            symbol_blocks: list = []
            for symbol, files in sorted(symbol_map.items()):
                files.sort(key=lambda x: x.date, reverse=True)
                links = []
                for state_file in files:
                    href = (
                        f"/day?head={quote(str(head))}"
                        f"&symbol={quote(state_file.symbol)}"
                        f"&date={state_file.date}"
                    )
                    links.append(
                        html.A(
                            f"{state_file.date}",
                            href=href,
                            target="_self",
                            style={"marginRight": "10px"},
                        )
                    )
                symbol_blocks.append(
                    html.Div(
                        [
                            html.Span(symbol, style={"fontWeight": "bold", "marginRight": "10px"}),
                            html.Span("dates: ", style={"marginRight": "6px"}),
                            *links,
                        ],
                        style={"marginBottom": "6px"},
                    )
                )

            sections.append(
                html.Div(
                    [
                        html.H4(str(head)),
                        html.Div(symbol_blocks),
                    ],
                    style={
                        "border": "1px solid #ddd",
                        "borderRadius": "8px",
                        "padding": "10px",
                        "marginBottom": "12px",
                    },
                )
            )
    return html.Div(sections, style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"})


def make_daily_figure(metrics_df: pd.DataFrame, symbol: str, head: Path, date: str) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Position", "PnL", "Cumulative Notional Traded", "Cumulative Size Traded"),
    )
    series = (
        ("position", "#1f77b4"),
        ("pnl", "#2ca02c"),
        ("cum_notional_traded", "#ff7f0e"),
        ("cum_size_traded", "#17a2b8"),
    )
    for row_idx, (col, color) in enumerate(series, start=1):
        y = pd.to_numeric(metrics_df[col], errors="coerce")
        fig.add_trace(
            go.Scattergl(
                x=metrics_df["time"],
                y=y,
                mode="lines",
                name=col,
                line={"color": color, "width": 1.6},
                showlegend=False,
                hovertemplate=f"{col}=%{{y}}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(title=col, row=row_idx, col=1)

    fig.update_xaxes(title="Time", row=4, col=1)
    fig.update_layout(
        template="plotly_white",
        height=980,
        title=f"{symbol} | {date} | {head} | Daily Metrics",
        margin={"l": 80, "r": 20, "t": 70, "b": 50},
        hovermode="x unified",
    )
    return fig


def render_day(search: str) -> html.Div:
    query = parse_qs(search.lstrip("?"))
    head_raw = unquote(query.get("head", [""])[0])
    symbol = unquote(query.get("symbol", [""])[0])
    date = query.get("date", [""])[0]

    head = normalize_head(head_raw)
    if head is None:
        return html.Div([html.H3("Invalid head path"), html.Pre(head_raw)])
    if not symbol or len(date) != 8 or not date.isdigit():
        return html.Div([html.H3("Missing symbol/date"), html.Pre(search)])

    state_path = head / "log" / "state" / f"{symbol}.0.{date}.csv"
    order_path = head / "log" / f"order.{date}.log"
    if not state_path.exists():
        return html.Div([html.H3("State file not found"), html.Pre(str(state_path))])

    metrics_df = load_daily_metrics_frame(str(state_path), state_path.stat().st_mtime_ns)
    if metrics_df.empty:
        return html.Div([html.H3("State file has no usable rows"), html.Pre(str(state_path))])

    orders = []
    if order_path.exists():
        orders = load_orders(str(order_path), symbol, order_path.stat().st_mtime_ns)
    available_hours = sorted({pd.to_datetime(o["start_ts_us"], unit="us").hour for o in orders})

    missing = []
    for col in ("position", "pnl", "cum_notional_traded", "cum_size_traded"):
        if metrics_df[col].dropna().empty:
            missing.append(col)

    fig = make_daily_figure(metrics_df, symbol, head, date)
    hour_options = [{"label": "all", "value": "all"}]
    hour_options.extend({"label": f"{h}h", "value": f"{h}"} for h in available_hours)
    return html.Div(
        [
            html.Div(
                [
                    html.A("Main page", href="/", target="_self", style={"marginRight": "12px"}),
                    html.Span(f"Date: {date}", style={"marginRight": "12px"}),
                    dcc.Dropdown(
                        id="day-hour-dropdown",
                        options=hour_options,
                        value="all",
                        clearable=False,
                        style={"width": "140px", "display": "inline-block", "verticalAlign": "middle"},
                    ),
                ],
                style={"margin": "8px 12px"},
            ),
            html.Div(
                f"Missing/empty metrics for this file: {', '.join(missing)}" if missing else "",
                style={"margin": "0 12px 8px 12px", "color": "#a61e4d"},
            ),
            dcc.Store(id="day-nav-meta", data={"head": str(head), "symbol": symbol, "date": date}),
            dcc.Graph(id="daily-metrics-graph", figure=fig, style={"height": "88vh"}),
        ]
    )


def make_figure(
    state_df: pd.DataFrame,
    orders: list[dict],
    fills: list[dict],
    base_mid_price: float | None,
    symbol: str,
    head: Path,
    date: str,
    initial_start: pd.Timestamp,
    initial_end: pd.Timestamp,
    show_book: bool = True,
) -> go.Figure:
    fig = go.Figure()
    if show_book:
        fig.add_trace(
            go.Scattergl(
                x=state_df["time"],
                y=state_df["bid"],
                mode="lines",
                name="Bid",
                line={"color": "#9ec5fe", "width": 1.4},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=state_df["time"],
                y=state_df["ask"],
                mode="lines",
                name="Ask",
                line={"color": "#ffc9c9", "width": 1.4},
                hoverinfo="skip",
            )
        )

    parent_color_map: dict[str, str] = {}
    buy_palette_i = 0
    sell_palette_i = 0
    for order in orders:
        side = order["side"]
        parent_order_id = str(order.get("parent_order_id") or order["client_order_id"])
        color = parent_color_map.get(parent_order_id)
        if color is None:
            if side == "BUY":
                color = BUY_COLORS[buy_palette_i % len(BUY_COLORS)]
                buy_palette_i += 1
            else:
                color = SELL_COLORS[sell_palette_i % len(SELL_COLORS)]
                sell_palette_i += 1
            parent_color_map[parent_order_id] = color

        x0 = pd.to_datetime(order["start_ts_us"], unit="us")
        x1 = pd.to_datetime(order["end_ts_us"], unit="us")
        x_mid = x0 + (x1 - x0) / 2
        y = order["price"]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y, y],
                mode="lines",
                name=f"{side} {order['client_order_id']}",
                showlegend=False,
                line={"color": color, "width": 3.4},
                customdata=[str(order.get("parent_order_id") or ""), str(order.get("parent_order_id") or "")],
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x_mid],
                y=[y],
                mode="markers",
                showlegend=False,
                marker={"size": 16, "color": "rgba(0,0,0,0)"},
                customdata=[str(order.get("parent_order_id") or "")],
                hovertemplate=(
                    f"parent_order_id={order.get('parent_order_id', '')}<br>"
                    f"client_order_id={order['client_order_id']}<br>"
                    f"side={side}<br>"
                    f"size={'' if order.get('size') is None else order.get('size')}<br>"
                    f"price={y}<extra></extra>"
                ),
            )
        )

    buy_fill_x: list = []
    buy_fill_y: list = []
    buy_fill_text: list = []
    sell_fill_x: list = []
    sell_fill_y: list = []
    sell_fill_text: list = []
    for fill in fills:
        x = pd.to_datetime(fill["ts_us"], unit="us")
        text = (
            f"parent_order_id={fill.get('parent_order_id', '')}<br>"
            f"client_order_id={fill['client_order_id']}<br>"
            f"price={fill['price']}<br>"
            f"filled_qty={fill['filled_qty']}"
        )
        if fill["side"] == "BUY":
            buy_fill_x.append(x)
            buy_fill_y.append(fill["price"])
            buy_fill_text.append(text)
        else:
            sell_fill_x.append(x)
            sell_fill_y.append(fill["price"])
            sell_fill_text.append(text)

    if buy_fill_x:
        buy_fill_parent = [str(fill.get("parent_order_id") or "") for fill in fills if fill["side"] == "BUY"]
        fig.add_trace(
            go.Scatter(
                x=buy_fill_x,
                y=buy_fill_y,
                mode="markers",
                name="Buy fills",
                showlegend=False,
                marker={"color": "#1D4ED8", "size": 10, "symbol": "arrow-up"},
                customdata=buy_fill_parent,
                text=buy_fill_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )
    if sell_fill_x:
        sell_fill_parent = [str(fill.get("parent_order_id") or "") for fill in fills if fill["side"] == "SELL"]
        fig.add_trace(
            go.Scatter(
                x=sell_fill_x,
                y=sell_fill_y,
                mode="markers",
                name="Sell fills",
                showlegend=False,
                marker={"color": "#DC2626", "size": 10, "symbol": "arrow-down"},
                customdata=sell_fill_parent,
                text=sell_fill_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    y_values = []
    if show_book:
        y_values.extend(pd.to_numeric(state_df["bid"], errors="coerce").dropna().tolist())
        y_values.extend(pd.to_numeric(state_df["ask"], errors="coerce").dropna().tolist())
    y_values.extend([float(o["price"]) for o in orders if o.get("price") is not None])
    y_values.extend([float(f["price"]) for f in fills if f.get("price") is not None])
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
    else:
        y_min = 0.0
        y_max = 1.0
    if y_max <= y_min:
        pad = max(abs(y_min) * 0.001, 1e-6)
        y_min -= pad
        y_max += pad

    bps_tickvals: list[float] = []
    bps_ticktext: list[str] = []
    if base_mid_price is not None:
        bps_tickvals, bps_ticktext = build_bps_ticks(y_min, y_max, base_mid_price, count=7)
        # Hidden anchor trace guarantees y2 is materialized by plotly.
        anchor_x = state_df["time"].iloc[0] if not state_df.empty else initial_start
        fig.add_trace(
            go.Scatter(
                x=[anchor_x],
                y=[base_mid_price],
                yaxis="y2",
                mode="markers",
                showlegend=False,
                marker={"size": 1, "color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
            )
        )

    fig.update_xaxes(
        title="Time",
        range=[initial_start, initial_end],
        rangeslider={"visible": False},
        domain=[0.09, 1.0],
        type="date",
    )
    fig.update_yaxes(
        title={"text": "Price", "standoff": 4},
        side="left",
        position=0.09,
        automargin=True,
        range=[y_min, y_max],
    )
    fig.update_layout(
        template="plotly_white",
        height=760,
        title="Orders / Fills" if not show_book else "Orders / Bid-Ask",
        hovermode="closest",
        hoverdistance=20,
        yaxis2={
            "title": {"text": "bps", "font": {"color": BPS_COLOR}},
            "overlaying": "y",
            "matches": "y",
            "side": "left",
            "position": 0.05,
            "showgrid": False,
            "zeroline": False,
            "showticklabels": True,
            "showline": True,
            "linecolor": BPS_COLOR,
            "tickcolor": BPS_COLOR,
            "tickfont": {"color": BPS_COLOR},
            "ticks": "outside",
            "tickmode": "array",
            "tickvals": bps_tickvals,
            "ticktext": bps_ticktext,
            "ticklabelposition": "outside",
            "ticklabelstandoff": 6,
            "automargin": True,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 72, "r": 20, "t": 80, "b": 50},
    )
    return fig


def make_position_figure(
    position_df: pd.DataFrame,
    symbol: str,
    head: Path,
    date: str,
    initial_start: pd.Timestamp,
    initial_end: pd.Timestamp,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=position_df["time"],
            y=position_df["position"],
            mode="lines",
            name="Position",
            line={"color": "#2a9d8f", "width": 1.8},
            hovertemplate="position=%{y}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Time", range=[initial_start, initial_end], rangeslider={"visible": False}, type="date", domain=[0.09, 1.0])
    fig.update_yaxes(title={"text": "Position", "standoff": 4}, side="left", position=0.09, automargin=True)
    fig.update_layout(
        template="plotly_white",
        height=190,
        title="Position",
        margin={"l": 72, "r": 20, "t": 60, "b": 40},
        hovermode="x unified",
    )
    return fig


def render_chart(search: str) -> html.Div:
    query = parse_qs(search.lstrip("?"))
    head_raw = unquote(query.get("head", [""])[0])
    symbol = unquote(query.get("symbol", [""])[0])
    date = query.get("date", [""])[0]
    hour_raw = unquote(query.get("hour", [""])[0])

    head = normalize_head(head_raw)
    if head is None:
        return html.Div([html.H3("Invalid head path"), html.Pre(head_raw)])
    if not symbol or len(date) != 8 or not date.isdigit():
        return html.Div([html.H3("Missing symbol/date"), html.Pre(search)])

    state_path = head / "log" / "state" / f"{symbol}.0.{date}.csv"
    order_path = head / "log" / f"order.{date}.log"
    if not state_path.exists():
        return html.Div([html.H3("State file not found"), html.Pre(str(state_path))])
    if not order_path.exists():
        return html.Div([html.H3("Order log not found"), html.Pre(str(order_path))])

    state_df = load_state_frame(str(state_path), state_path.stat().st_mtime_ns)
    position_df = load_position_frame(str(state_path), state_path.stat().st_mtime_ns)
    if state_df.empty:
        return html.Div([html.H3("State file has no usable rows"), html.Pre(str(state_path))])

    orders = load_orders(str(order_path), symbol, order_path.stat().st_mtime_ns)
    fills = load_fills(str(order_path), symbol, order_path.stat().st_mtime_ns)
    base_mid_price = first_reference_price(state_df, orders, fills)
    min_time = state_df["time"].min()
    max_time = state_df["time"].max()
    latest_start = max(min_time, max_time - timedelta(hours=1))
    requested_hour = parse_plotly_time(hour_raw) if hour_raw else None
    first_activity_time = min_time
    if orders and fills:
        first_activity_time = min(
            pd.to_datetime(orders[0]["start_ts_us"], unit="us"),
            pd.to_datetime(fills[0]["ts_us"], unit="us"),
        )
    elif orders:
        first_activity_time = pd.to_datetime(orders[0]["start_ts_us"], unit="us")
    elif fills:
        first_activity_time = pd.to_datetime(fills[0]["ts_us"], unit="us")
    if requested_hour is not None:
        window_start = max(min(requested_hour.floor("h"), latest_start), min_time)
    else:
        window_start = min(max(first_activity_time.floor("h"), min_time.floor("h")), max_time.floor("h"))
    window_end = min(window_start + timedelta(hours=1), max_time)
    if window_end <= window_start:
        window_end = window_start + timedelta(hours=1)

    fig = make_figure(state_df, orders, fills, base_mid_price, symbol, head, date, window_start, window_end, show_book=True)
    position_fig = make_position_figure(position_df, symbol, head, date, window_start, window_end)
    initial_interval_text = format_interval_label(window_start, window_end)
    return html.Div(
        [
            html.Div(
                [
                    html.A("Main page", href="/", target="_self", style={"marginRight": "12px"}),
                    html.A(
                        "Daily",
                        href=f"/day?head={quote(str(head))}&symbol={quote(symbol)}&date={date}",
                        target="_self",
                        style={"marginRight": "12px"},
                    ),
                    html.Span(f"Date: {date}", style={"marginRight": "12px"}),
                    html.Span(f"Orders: {len(orders)}", style={"marginRight": "12px"}),
                    html.Span(f"Interval: {initial_interval_text}", id="interval-label"),
                    html.Button("Prev hour", id="prev-hour-btn", n_clicks=0, style={"marginLeft": "14px"}),
                    html.Button("Next hour", id="next-hour-btn", n_clicks=0, style={"marginLeft": "8px"}),
                ],
                style={"margin": "8px 12px"},
            ),
            html.Div(
                f"{symbol} | {date} | {head}",
                style={"margin": "0 12px 6px 12px", "fontWeight": "600"},
            ),
            dcc.Store(id="hourly-nav-meta", data={"head": str(head), "symbol": symbol, "date": date}),
            dcc.Store(
                id="chart-nav-meta",
                data={
                    "min_time": min_time.isoformat(),
                    "max_time": max_time.isoformat(),
                    "window_start": window_start.isoformat(),
                    "window_minutes": 60,
                },
            ),
            dcc.Graph(id="position-graph", figure=position_fig, style={"height": "28vh"}),
            dcc.Graph(id="symbol-graph", figure=fig, style={"height": "62vh"}),
        ]
    )


def render_parent(search: str) -> html.Div:
    query = parse_qs(search.lstrip("?"))
    head_raw = unquote(query.get("head", [""])[0])
    symbol = unquote(query.get("symbol", [""])[0])
    date = query.get("date", [""])[0]
    hour_raw = unquote(query.get("hour", [""])[0])
    parent_order_id = unquote(query.get("parent_order_id", [""])[0])

    head = normalize_head(head_raw)
    if head is None:
        return html.Div([html.H3("Invalid head path"), html.Pre(head_raw)])
    if not symbol or len(date) != 8 or not date.isdigit():
        return html.Div([html.H3("Missing symbol/date"), html.Pre(search)])
    if not parent_order_id:
        return html.Div([html.H3("Missing parent_order_id"), html.Pre(search)])

    state_path = head / "log" / "state" / f"{symbol}.0.{date}.csv"
    order_path = head / "log" / f"order.{date}.log"
    if not state_path.exists():
        return html.Div([html.H3("State file not found"), html.Pre(str(state_path))])
    if not order_path.exists():
        return html.Div([html.H3("Order log not found"), html.Pre(str(order_path))])

    state_df = load_state_frame(str(state_path), state_path.stat().st_mtime_ns)
    position_df = load_position_frame(str(state_path), state_path.stat().st_mtime_ns)
    if state_df.empty:
        return html.Div([html.H3("State file has no usable rows"), html.Pre(str(state_path))])

    all_orders = load_orders(str(order_path), symbol, order_path.stat().st_mtime_ns)
    all_fills = load_fills(str(order_path), symbol, order_path.stat().st_mtime_ns)
    orders = [o for o in all_orders if str(o.get("parent_order_id") or "") == parent_order_id]
    fills = [f for f in all_fills if str(f.get("parent_order_id") or "") == parent_order_id]
    base_mid_price = first_reference_price(state_df, orders, fills)

    state_min_time = state_df["time"].min()
    state_max_time = state_df["time"].max()
    activity_times: list[pd.Timestamp] = []
    for order in orders:
        activity_times.append(pd.to_datetime(order["start_ts_us"], unit="us"))
        activity_times.append(pd.to_datetime(order["end_ts_us"], unit="us"))
    for fill in fills:
        activity_times.append(pd.to_datetime(fill["ts_us"], unit="us"))

    if activity_times:
        activity_start = min(activity_times)
        activity_end = max(activity_times)
        span = activity_end - activity_start
        if span <= timedelta(0):
            span = timedelta(minutes=1)
        margin = span * 0.1
        window_start = activity_start - margin
        window_end = activity_end + margin
    else:
        # Fallback when selected parent has no activity rows.
        window_start = state_min_time
        window_end = min(state_min_time + timedelta(hours=1), state_max_time)
        if window_end <= window_start:
            window_end = window_start + timedelta(hours=1)

    fig = make_figure(
        state_df,
        orders,
        fills,
        base_mid_price,
        symbol,
        head,
        date,
        window_start,
        window_end,
        show_book=False,
    )
    position_fig = make_position_figure(position_df, symbol, head, date, window_start, window_end)
    initial_interval_text = format_interval_label(window_start, window_end)
    hourly_href = (
        f"/chart?head={quote(str(head))}&symbol={quote(symbol)}&date={date}"
        f"&hour={quote(pd.to_datetime(window_start).floor('h').isoformat())}"
    )
    daily_href = f"/day?head={quote(str(head))}&symbol={quote(symbol)}&date={date}"
    return html.Div(
        [
            html.Div(
                [
                    html.A("Main page", href="/", target="_self", style={"marginRight": "12px"}),
                    html.A("Daily", href=daily_href, target="_self", style={"marginRight": "12px"}),
                    html.A("Hourly", href=hourly_href, target="_self", style={"marginRight": "12px"}),
                    html.Span(f"Date: {date}", style={"marginRight": "12px"}),
                    html.Span(f"Parent order: {parent_order_id}", style={"marginRight": "12px"}),
                    html.Span(
                        f"Orders: {len(orders)}/{len(all_orders)} Fills: {len(fills)}/{len(all_fills)}",
                        style={"marginRight": "12px"},
                    ),
                    html.Span(f"Interval: {initial_interval_text}", id="interval-label"),
                    html.Button("Prev hour", id="prev-hour-btn", n_clicks=0, style={"marginLeft": "14px"}),
                    html.Button("Next hour", id="next-hour-btn", n_clicks=0, style={"marginLeft": "8px"}),
                ],
                style={"margin": "8px 12px"},
            ),
            html.Div(
                f"{symbol} | {date} | {head}",
                style={"margin": "0 12px 6px 12px", "fontWeight": "600"},
            ),
            dcc.Store(id="hourly-nav-meta", data={"head": str(head), "symbol": symbol, "date": date}),
            dcc.Store(
                id="chart-nav-meta",
                data={
                    "min_time": window_start.isoformat(),
                    "max_time": window_end.isoformat(),
                    "window_start": window_start.isoformat(),
                    "window_minutes": max(1, int((window_end - window_start).total_seconds() / 60)),
                },
            ),
            dcc.Graph(id="position-graph", figure=position_fig, style={"height": "28vh"}),
            dcc.Graph(id="symbol-graph", figure=fig, style={"height": "62vh"}),
        ]
    )


app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Simulation Dashboard"
app.layout = html.Div([dcc.Location(id="url"), html.Div(id="page-content")])


@app.callback(Output("page-content", "children"), Input("url", "pathname"), Input("url", "search"))
def route(pathname: str, search: str):
    if pathname == "/day":
        return render_day(search)
    if pathname == "/parent":
        return render_parent(search)
    if pathname == "/chart":
        return render_chart(search)
    return render_index()


@app.callback(
    Output("url", "pathname"),
    Output("url", "search"),
    Input("day-hour-dropdown", "value", allow_optional=True),
    Input("symbol-graph", "clickData", allow_optional=True),
    State("url", "pathname"),
    State("day-nav-meta", "data", allow_optional=True),
    State("hourly-nav-meta", "data", allow_optional=True),
    State("url", "search"),
    prevent_initial_call=True,
)
def handle_navigation(
    hour_value: str | None,
    hourly_click_data: dict | None,
    current_pathname: str | None,
    day_nav_meta: dict | None,
    hourly_nav_meta: dict | None,
    current_search: str,
):
    if current_pathname == "/day" and hour_value and hour_value != "all":
        head_raw = ""
        symbol = ""
        date = ""
        if day_nav_meta:
            head_raw = str(day_nav_meta.get("head", ""))
            symbol = str(day_nav_meta.get("symbol", ""))
            date = str(day_nav_meta.get("date", ""))
        if (not head_raw or not symbol or len(date) != 8 or not date.isdigit()) and current_search:
            query = parse_qs(current_search.lstrip("?"))
            head_raw = unquote(query.get("head", [""])[0])
            symbol = unquote(query.get("symbol", [""])[0])
            date = query.get("date", [""])[0]
        if not head_raw or not symbol or len(date) != 8 or not date.isdigit():
            return no_update, no_update
        try:
            hour_int = int(hour_value)
        except ValueError:
            return no_update, no_update
        if hour_int < 0 or hour_int > 23:
            return no_update, no_update
        hour_start = datetime.strptime(f"{date}{hour_int:02d}", "%Y%m%d%H")
        next_search = (
            f"?head={quote(head_raw)}"
            f"&symbol={quote(symbol)}"
            f"&date={date}"
            f"&hour={quote(hour_start.isoformat())}"
        )
        return "/chart", next_search

    if current_pathname == "/chart" and hourly_click_data:
        points = hourly_click_data.get("points")
        if not isinstance(points, list) or not points:
            return no_update, no_update
        point = points[0]
        parent = point.get("customdata")
        if parent is None:
            return no_update, no_update
        parent_order_id = str(parent).strip()
        if not parent_order_id:
            return no_update, no_update
        head_raw = ""
        symbol = ""
        date = ""
        if hourly_nav_meta:
            head_raw = str(hourly_nav_meta.get("head", ""))
            symbol = str(hourly_nav_meta.get("symbol", ""))
            date = str(hourly_nav_meta.get("date", ""))
        if (not head_raw or not symbol or len(date) != 8 or not date.isdigit()) and current_search:
            query = parse_qs(current_search.lstrip("?"))
            head_raw = unquote(query.get("head", [""])[0])
            symbol = unquote(query.get("symbol", [""])[0])
            date = query.get("date", [""])[0]
        if not head_raw or not symbol or len(date) != 8 or not date.isdigit():
            return no_update, no_update
        clicked_time = parse_plotly_time(point.get("x"))
        hour_param = ""
        if clicked_time is not None:
            hour_param = f"&hour={quote(clicked_time.floor('h').isoformat())}"
        next_search = (
            f"?head={quote(head_raw)}"
            f"&symbol={quote(symbol)}"
            f"&date={date}"
            f"&parent_order_id={quote(parent_order_id)}"
            f"{hour_param}"
        )
        return "/parent", next_search

    return no_update, no_update


@app.callback(
    Output("position-graph", "figure"),
    Output("symbol-graph", "figure"),
    Output("chart-nav-meta", "data"),
    Input("prev-hour-btn", "n_clicks"),
    Input("next-hour-btn", "n_clicks"),
    State("chart-nav-meta", "data"),
    prevent_initial_call=True,
)
def step_hour(prev_clicks: int, next_clicks: int, meta: dict | None):
    del prev_clicks, next_clicks
    trigger_id = dash.callback_context.triggered_id
    if trigger_id not in ("prev-hour-btn", "next-hour-btn"):
        return no_update, no_update, no_update
    if not meta:
        return no_update, no_update, no_update

    start = pd.to_datetime(meta.get("window_start"), errors="coerce")
    min_time = pd.to_datetime(meta.get("min_time"), errors="coerce")
    max_time = pd.to_datetime(meta.get("max_time"), errors="coerce")
    window_minutes = int(meta.get("window_minutes", 10))
    if pd.isna(start) or pd.isna(min_time) or pd.isna(max_time):
        return no_update, no_update, no_update

    width = timedelta(minutes=window_minutes)
    delta = timedelta(hours=-1) if trigger_id == "prev-hour-btn" else timedelta(hours=1)
    new_start = start + delta
    latest_start = max(min_time, max_time - width)
    if new_start < min_time:
        new_start = min_time
    if new_start > latest_start:
        new_start = latest_start
    new_end = min(new_start + width, max_time)

    pos_patch = Patch()
    pos_patch["layout"]["xaxis"]["range"] = [new_start.isoformat(), new_end.isoformat()]

    patch = Patch()
    patch["layout"]["xaxis"]["range"] = [new_start.isoformat(), new_end.isoformat()]

    next_meta = dict(meta)
    next_meta["window_start"] = new_start.isoformat()
    return pos_patch, patch, next_meta


@app.callback(
    Output("interval-label", "children"),
    Input("symbol-graph", "relayoutData"),
    Input("chart-nav-meta", "data"),
    State("chart-nav-meta", "data"),
    prevent_initial_call=True,
)
def update_interval_label(relayout_data: dict | None, _meta_input: dict | None, meta: dict | None):
    del _meta_input
    start = None
    end = None
    if relayout_data:
        if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
            start = parse_plotly_time(relayout_data.get("xaxis.range[0]"))
            end = parse_plotly_time(relayout_data.get("xaxis.range[1]"))
        elif "xaxis.range" in relayout_data:
            ranges = relayout_data.get("xaxis.range")
            if isinstance(ranges, list) and len(ranges) >= 2:
                start = parse_plotly_time(ranges[0])
                end = parse_plotly_time(ranges[1])
    if (start is None or end is None) and meta:
        start = parse_plotly_time(meta.get("window_start"))
        min_time = parse_plotly_time(meta.get("min_time"))
        max_time = parse_plotly_time(meta.get("max_time"))
        window_minutes = int(meta.get("window_minutes", 60))
        if start is not None and min_time is not None and max_time is not None:
            start = max(start, min_time)
            end = min(start + timedelta(minutes=window_minutes), max_time)
    if start is None or end is None:
        return "Interval: n/a"
    return f"Interval: {format_interval_label(start, end)}"


if __name__ == "__main__":
    host = os.getenv("DASH_HOST", "127.0.0.1")
    port = int(os.getenv("DASH_PORT", "8050"))
    debug = os.getenv("DASH_DEBUG", "1").lower() in ("1", "true", "yes", "on")
    app.run(debug=debug, host=host, port=port)
