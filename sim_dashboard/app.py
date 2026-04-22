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
import pyarrow.parquet as pq


ROOT_ORDER = ("dumpsim", "livesim", "tradesim")
ROOTS = [Path.home() / "workspace" / "sgt" / name for name in ROOT_ORDER]
BPS_COLOR = "#C19A6B"
ORDER_PQ_PATTERN = "order.????????.parquet"
ORDER_LOG_PATTERN = "order.????????.log"

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
    # Expected form: SYMBOL.0.YYYYMMDD.parquet
    parts = path.name.split(".")
    if len(parts) < 4:
        return None
    if parts[-1] != "parquet":
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
            if not any(log_dir.glob(ORDER_PQ_PATTERN)) and not any(log_dir.glob(ORDER_LOG_PATTERN)):
                continue
            grouped[root_name].append(log_dir.parent)
        grouped[root_name].sort()
    return grouped


def state_files_for_head(head: Path) -> list[StateFile]:
    state_dir = head / "log" / "state"
    if not state_dir.is_dir():
        return []
    rows: list[StateFile] = []
    for pq_path in state_dir.glob("*.parquet"):
        parsed = parse_state_filename(pq_path)
        if parsed is None:
            continue
        # Keep entries that can map to an order file for the same date.
        if not (
            (head / "log" / f"order.{parsed.date}.parquet").exists()
            or (head / "log" / f"order.{parsed.date}.log").exists()
        ):
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


def _bucket_last_frame(df: pd.DataFrame, bucket_minutes: int) -> pd.DataFrame:
    if df.empty or "time" not in df.columns:
        return df
    if bucket_minutes <= 1:
        bucket_minutes = 1
    minute_df = df.copy()
    minute_df["time"] = pd.to_datetime(minute_df["time"], errors="coerce")
    minute_df = minute_df.dropna(subset=["time"]).sort_values("time")
    minute_df["time"] = minute_df["time"].dt.floor(f"{bucket_minutes}min")
    minute_df = minute_df.groupby("time", as_index=False).last()
    return minute_df


def _minute_last_frame(df: pd.DataFrame) -> pd.DataFrame:
    return _bucket_last_frame(df, 1)


def _slice_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df.empty or "time" not in df.columns:
        return df
    sliced = df.copy()
    sliced["time"] = pd.to_datetime(sliced["time"], errors="coerce")
    sliced = sliced.dropna(subset=["time"])
    return sliced[(sliced["time"] >= start) & (sliced["time"] <= end)]


def _last_row_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "time" not in df.columns:
        return df
    last = df.copy()
    last["time"] = pd.to_datetime(last["time"], errors="coerce")
    last = last.dropna(subset=["time"]).sort_values("time")
    if last.empty:
        return last
    return last.tail(1)


def _format_usd(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"${float(value):,.2f}"


def _format_num(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def _format_pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100.0:.{digits}f}%"


def _format_bps(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 10000.0:.{digits}f} bps"


def _format_ratio(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    value_f = float(value)
    if not math.isfinite(value_f):
        return "inf"
    return f"{value_f:.{digits}f}"


def _build_report_frame(metrics_df: pd.DataFrame, state_df: pd.DataFrame, bucket_minutes: int) -> pd.DataFrame:
    metrics_df = _bucket_last_frame(metrics_df, bucket_minutes)
    state_df = _bucket_last_frame(state_df, bucket_minutes)
    if metrics_df.empty or state_df.empty:
        return pd.DataFrame()

    metrics_cols = [col for col in ("time", "position", "pnl", "cum_notional_traded", "cum_size_traded") if col in metrics_df.columns]
    state_cols = [col for col in ("time", "bid", "ask") if col in state_df.columns]
    if len(metrics_cols) < 2 or len(state_cols) < 3:
        return pd.DataFrame()

    merged = pd.merge(
        metrics_df[metrics_cols],
        state_df[state_cols],
        on="time",
        how="inner",
    )
    if merged.empty:
        return merged

    merged["time"] = pd.to_datetime(merged["time"], errors="coerce")
    merged = merged.dropna(subset=["time"]).sort_values("time")
    merged["position"] = pd.to_numeric(merged.get("position"), errors="coerce")
    merged["pnl"] = pd.to_numeric(merged.get("pnl"), errors="coerce")
    merged["cum_notional_traded"] = pd.to_numeric(merged.get("cum_notional_traded"), errors="coerce")
    merged["cum_size_traded"] = pd.to_numeric(merged.get("cum_size_traded"), errors="coerce")
    merged["mid"] = (pd.to_numeric(merged["bid"], errors="coerce") + pd.to_numeric(merged["ask"], errors="coerce")) / 2.0
    merged["position_usd"] = merged["position"].abs() * merged["mid"]
    return merged.dropna(subset=["mid"])


def _report_total_notional_usd(report_df: pd.DataFrame) -> float:
    if report_df.empty or "cum_notional_traded" not in report_df.columns:
        return 0.0
    notional = pd.to_numeric(report_df["cum_notional_traded"], errors="coerce").dropna()
    if notional.empty:
        return 0.0
    return float(notional.iloc[-1])


def _return_series_stats(returns: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(returns, errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
    if clean.empty:
        return {
            "volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
        }
    mean = float(clean.mean())
    std = float(clean.std(ddof=0))
    downside = clean[clean < 0]
    downside_std = float(downside.std(ddof=0)) if not downside.empty else 0.0
    positive_sum = float(clean[clean > 0].sum())
    negative_sum = float(clean[clean < 0].sum())
    return {
        "volatility": std,
        "sharpe": mean / std if std else 0.0,
        "sortino": mean / downside_std if downside_std else 0.0,
        "profit_factor": positive_sum / abs(negative_sum) if negative_sum < 0 else (float("inf") if positive_sum > 0 else 0.0),
        "win_rate": float((clean > 0).mean()),
    }


def _report_stats(report_df: pd.DataFrame, total_fees_usd: float = 0.0, total_notional_usd: float = 0.0) -> dict[str, float]:
    if report_df.empty:
        return {
            "total_pnl": 0.0,
            "total_return": 0.0,
            "total_notional_usd": total_notional_usd,
            "total_return_bps": 0.0,
            "total_fees_usd": total_fees_usd,
            "total_fees_bps": 0.0,
            "max_drawdown": 0.0,
            "max_abs_position_usd": 0.0,
            "max_abs_position": 0.0,
        }

    pnl = pd.to_numeric(report_df["pnl"], errors="coerce").fillna(0.0)
    equity = pnl
    drawdown = equity - equity.cummax()
    position = pd.to_numeric(report_df["position"], errors="coerce").fillna(0.0)
    position_usd = pd.to_numeric(report_df["position_usd"], errors="coerce").fillna(0.0)

    total_return = float(pnl.iloc[-1] / total_notional_usd) if total_notional_usd else 0.0
    total_fees_bps = float(total_fees_usd / total_notional_usd) if total_notional_usd else 0.0
    return {
        "total_pnl": float(pnl.iloc[-1]),
        "total_return": total_return,
        "total_notional_usd": total_notional_usd,
        "total_return_bps": total_return,
        "total_fees_usd": float(total_fees_usd),
        "total_fees_bps": total_fees_bps,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "max_abs_position_usd": float(position_usd.abs().max()) if not position_usd.empty else 0.0,
        "max_abs_position": float(position.abs().max()) if not position.empty else 0.0,
    }


def _make_stats_table(title: str, rows: list[tuple[str, str]], width: str = "min(520px, 100%)") -> html.Div:
    return html.Div(
        [
            html.H4(title, style={"margin": "0 0 8px 0"}),
            html.Table(
                [
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Th(key, style={"textAlign": "left", "padding": "6px 10px", "borderBottom": "1px solid #ddd"}),
                                    html.Td(value, style={"textAlign": "right", "padding": "6px 10px", "borderBottom": "1px solid #ddd"}),
                                ]
                            )
                            for key, value in rows
                        ]
                    )
                ],
                style={"width": "100%", "borderCollapse": "collapse"},
            ),
        ],
        style={"width": width},
    )


def _make_data_table(title: str, columns: list[str], rows: list[dict[str, str]], max_height: str = "320px") -> html.Div:
    return html.Div(
        [
            html.H4(title, style={"margin": "0 0 8px 0"}),
            html.Div(
                html.Table(
                    [
                        html.Thead(html.Tr([html.Th(col, style={"padding": "6px 10px", "textAlign": "left", "borderBottom": "1px solid #ddd"}) for col in columns])),
                        html.Tbody(
                            [
                                html.Tr([html.Td(row.get(col, "-"), style={"padding": "6px 10px", "borderBottom": "1px solid #eee"}) for col in columns])
                                for row in rows
                            ]
                        ),
                    ],
                    style={"width": "100%", "borderCollapse": "collapse"},
                ),
                style={"overflowX": "auto", "maxHeight": max_height, "overflowY": "auto"},
            ),
        ],
        style={"width": "100%"},
    )


def _make_quant_summary_figure(
    x_values,
    equity_values,
    drawdown_values,
    returns_values=None,
    title="Quant Summary",
    x_title: str = "Time",
) -> go.Figure:
    rows = 3 if returns_values is not None else 2
    subplot_titles = ("Equity", "Drawdown") + (("Period PnL",) if returns_values is not None else ())
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.07, subplot_titles=subplot_titles)
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=equity_values,
            mode="lines",
            name="Equity",
            line={"color": "#2563EB", "width": 2.0},
            hovertemplate="equity=%{y:$,.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=drawdown_values,
            mode="lines",
            name="Drawdown",
            line={"color": "#C2410C", "width": 2.0},
            hovertemplate="drawdown=%{y:$,.2f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    if returns_values is not None:
        colors = ["#0F766E" if float(v) >= 0 else "#DC2626" for v in returns_values]
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=returns_values,
                marker={"color": colors},
                name="Period PnL",
                hovertemplate="pnl=%{y:$,.2f}<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )
    fig.update_yaxes(title="Equity (USD)", row=1, col=1, automargin=True)
    fig.update_yaxes(title="Drawdown (USD)", row=2, col=1, automargin=True)
    if returns_values is not None:
        fig.update_yaxes(title="Period PnL (USD)", row=3, col=1, automargin=True)
        fig.update_xaxes(title=x_title, row=3, col=1)
    else:
        fig.update_xaxes(title=x_title, row=2, col=1)
    fig.update_layout(template="plotly_white", height=420 if returns_values is None else 560, title=title, margin={"l": 18, "r": 18, "t": 60, "b": 40}, hovermode="x unified")
    return fig


def _make_report_panel(title: str, summary_rows: list[tuple[str, str]], figure: go.Figure, breakdown: html.Div | None = None) -> html.Div:
    children = [
        html.H3(title, style={"margin": "0 0 12px 0"}),
        html.Div(
            [
                html.Div(_make_stats_table("Summary", summary_rows, width="100%"), style={"flex": "0 0 340px", "minWidth": "280px"}),
                html.Div(
                    dcc.Graph(figure=figure, config={"displayModeBar": False}),
                    style={"flex": "1 1 640px", "minWidth": "0"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "alignItems": "stretch", "flexWrap": "wrap"},
        ),
    ]
    if breakdown is not None:
        children.append(html.Div(breakdown, style={"marginTop": "14px"}))
    return html.Div(
        children,
        style={
            "margin": "0 12px 14px 12px",
            "padding": "12px",
            "border": "1px solid #e5e7eb",
            "borderRadius": "10px",
            "background": "#fafafa",
        },
    )


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
    state_path = Path(state_path_str)
    for bid_col, ask_col in (("bid", "ask"), ("bid_price", "ask_price")):
        try:
            df = pd.read_parquet(state_path, columns=["time", bid_col, ask_col])
        except (KeyError, ValueError):
            continue
        df = df.rename(columns={bid_col: "bid", ask_col: "ask"})
        df["time"] = pd.to_datetime(df["time"], unit="us", errors="coerce")
        for col in ("bid", "ask"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Filter obvious sentinel / invalid values.
            df.loc[(df[col] <= 0) | (df[col] >= 1e12), col] = pd.NA
        df = df.dropna(subset=["time"])
        return df
    return pd.DataFrame(columns=["time", "bid", "ask"])


@lru_cache(maxsize=32)
def load_position_frame(state_path_str: str, mtime_ns: int) -> pd.DataFrame:
    state_path = Path(state_path_str)
    for pos_col in ("position", "pos"):
        try:
            df = pd.read_parquet(state_path, columns=["time", pos_col])
        except (KeyError, ValueError):
            continue
        df = df.rename(columns={pos_col: "position"})
        df["time"] = pd.to_datetime(df["time"], unit="us", errors="coerce")
        df["position"] = pd.to_numeric(df["position"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
        return df
    return pd.DataFrame(columns=["time", "position"])


@lru_cache(maxsize=32)
def load_daily_metrics_frame(state_path_str: str, mtime_ns: int) -> pd.DataFrame:
    state_path = Path(state_path_str)
    columns = pq.ParquetFile(state_path).schema.names

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

    df = pd.read_parquet(state_path, columns=usecols)
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


def _load_order_parquet(order_path: Path, symbol: str, mtime_ns: int) -> pd.DataFrame:
    df = load_order_table(str(order_path), mtime_ns).copy()
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    return df


@lru_cache(maxsize=32)
def load_total_fees(order_log_path_str: str, symbol: str, mtime_ns: int) -> float:
    order_log_path = Path(order_log_path_str)
    if order_log_path.suffix == ".parquet":
        df = _load_order_parquet(order_log_path, symbol, mtime_ns)
        if df.empty or "fees" not in df.columns:
            return 0.0
        fees = pd.to_numeric(df["fees"], errors="coerce").fillna(0.0)
        return float(fees.sum())

    del mtime_ns
    total = 0.0
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
            fees = payload.get("fees", 0)
            try:
                total += float(fees)
            except (TypeError, ValueError):
                continue
    return total


@lru_cache(maxsize=32)
def load_order_table(order_path_str: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    order_path = Path(order_path_str)
    columns = pq.ParquetFile(order_path).schema.names
    usecols = [
        col
        for col in (
            "symbol",
            "create_time",
            "acked_time",
            "last_update_time",
            "parent_order_id",
            "client_order_id",
            "price",
            "qty",
            "side",
            "avg_fill_price",
            "filled_qty",
            "fees",
        )
        if col in columns
    ]
    return pd.read_parquet(order_path, columns=usecols)


def _orders_from_parquet(order_path: Path, symbol: str, mtime_ns: int) -> list[dict]:
    df = _load_order_parquet(order_path, symbol, mtime_ns)
    rows: list[dict] = []
    for row in df.itertuples(index=False):
        try:
            side = _normalize_side(getattr(row, "side", None))
            price = getattr(row, "price", None)
            qty = getattr(row, "qty", None)
            client_order_id = getattr(row, "client_order_id", None)
            parent_order_id = getattr(row, "parent_order_id", None)
            start_ts_us = getattr(row, "create_time", None)
            end_ts_us = getattr(row, "last_update_time", None)
        except AttributeError:
            continue
        if client_order_id is None or side is None or price is None or start_ts_us is None or end_ts_us is None:
            continue
        try:
            start_ts_us_i = int(start_ts_us)
            end_ts_us_i = int(end_ts_us)
            price_f = float(price)
        except (TypeError, ValueError):
            continue
        if end_ts_us_i <= start_ts_us_i:
            end_ts_us_i = start_ts_us_i + 1
        rows.append(
            {
                "parent_order_id": None if parent_order_id is None else str(parent_order_id),
                "client_order_id": str(client_order_id),
                "side": side,
                "price": price_f,
                "size": None if qty is None else float(qty),
                "start_ts_us": start_ts_us_i,
                "end_ts_us": end_ts_us_i,
            }
        )
    rows.sort(key=lambda x: x["start_ts_us"])
    return rows


def _fills_from_parquet(order_path: Path, symbol: str, mtime_ns: int) -> list[dict]:
    df = _load_order_parquet(order_path, symbol, mtime_ns)
    if df.empty:
        return []
    rows: list[dict] = []
    if "filled_qty" not in df.columns:
        return []
    fill_df = df[pd.to_numeric(df["filled_qty"], errors="coerce").fillna(0) > 0].copy()
    for row in fill_df.itertuples(index=False):
        side = _normalize_side(getattr(row, "side", None))
        if side is None:
            continue
        ts_us = getattr(row, "last_update_time", None)
        if ts_us is None:
            ts_us = getattr(row, "acked_time", None)
        if ts_us is None:
            ts_us = getattr(row, "create_time", None)
        price = getattr(row, "avg_fill_price", None)
        if price in (None, 0, 0.0):
            price = getattr(row, "price", None)
        filled_qty = getattr(row, "filled_qty", None)
        if ts_us is None or price in (None, 0, 0.0) or filled_qty is None:
            continue
        try:
            rows.append(
                {
                    "ts_us": int(ts_us),
                    "side": side,
                    "price": float(price),
                    "parent_order_id": None if getattr(row, "parent_order_id", None) is None else str(getattr(row, "parent_order_id")),
                    "client_order_id": str(getattr(row, "client_order_id", "")),
                    "filled_qty": float(filled_qty),
                }
            )
        except (TypeError, ValueError):
            continue
    rows.sort(key=lambda x: x["ts_us"])
    return rows


def _aggregate_parent_orders(orders: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    for order in orders:
        parent_order_id = str(order.get("parent_order_id") or order.get("client_order_id") or "")
        side = order.get("side")
        if not parent_order_id or side not in ("BUY", "SELL"):
            continue
        key = (parent_order_id, side)
        item = grouped.get(key)
        start_ts_us = int(order["start_ts_us"])
        end_ts_us = int(order["end_ts_us"])
        price = float(order["price"])
        if item is None:
            grouped[key] = {
                "parent_order_id": parent_order_id,
                "side": side,
                "events": [(start_ts_us, 1, price), (end_ts_us, -1, price)],
                "start_ts_us": start_ts_us,
                "end_ts_us": end_ts_us,
                "order_count": 1,
                "child_order_ids": [str(order["client_order_id"])],
                "child_sizes": [order.get("size")],
            }
            continue

        item["start_ts_us"] = min(item["start_ts_us"], start_ts_us)
        item["end_ts_us"] = max(item["end_ts_us"], end_ts_us)
        item["order_count"] += 1
        item["child_order_ids"].append(str(order["client_order_id"]))
        item["child_sizes"].append(order.get("size"))
        item["events"].append((start_ts_us, 1, price))
        item["events"].append((end_ts_us, -1, price))

    rows: list[dict] = []
    for item in grouped.values():
        events = sorted(item["events"], key=lambda x: (x[0], x[1]))
        price_counts: dict[float, int] = {}
        price_points: list[tuple[int, float]] = []
        i = 0
        while i < len(events):
            ts_us = events[i][0]
            while i < len(events) and events[i][0] == ts_us and events[i][1] < 0:
                _, _, price = events[i]
                price_counts[price] = price_counts.get(price, 0) - 1
                if price_counts[price] <= 0:
                    price_counts.pop(price, None)
                i += 1
            while i < len(events) and events[i][0] == ts_us and events[i][1] > 0:
                _, _, price = events[i]
                price_counts[price] = price_counts.get(price, 0) + 1
                i += 1
            if price_counts:
                current = max(price_counts) if item["side"] == "BUY" else min(price_counts)
                price_points.append((ts_us, current))
        deduped: list[tuple[int, float]] = []
        for ts_us, price in price_points:
            if deduped and deduped[-1][1] == price:
                continue
            deduped.append((ts_us, price))
        rows.append(
            {
                "parent_order_id": item["parent_order_id"],
                "side": item["side"],
                "start_ts_us": item["start_ts_us"],
                "end_ts_us": item["end_ts_us"],
                "price_points": deduped,
                "order_count": item["order_count"],
                "child_order_ids": item["child_order_ids"],
                "child_sizes": item["child_sizes"],
            }
        )

    rows.sort(key=lambda x: (x["parent_order_id"], x["side"]))
    return rows


def _aggregate_parent_fills(fills: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict] = {}
    for fill in fills:
        parent_order_id = str(fill.get("parent_order_id") or fill.get("client_order_id") or "")
        side = fill.get("side")
        if not parent_order_id or side not in ("BUY", "SELL"):
            continue
        key = (parent_order_id, side)
        item = grouped.get(key)
        ts_us = int(fill["ts_us"])
        price = float(fill["price"])
        qty = float(fill["filled_qty"])
        if item is None:
            grouped[key] = {
                "parent_order_id": parent_order_id,
                "side": side,
                "last_ts_us": ts_us,
                "last_price": price,
                "last_qty": qty,
                "fill_count": 1,
                "total_qty": qty,
                "prices": [price],
            }
            continue

        item["fill_count"] += 1
        item["total_qty"] += qty
        item["prices"].append(price)
        if ts_us >= item["last_ts_us"]:
            item["last_ts_us"] = ts_us
            item["last_price"] = price
            item["last_qty"] = qty

    rows: list[dict] = []
    for item in grouped.values():
        rows.append(item)
    rows.sort(key=lambda x: (x["parent_order_id"], x["side"]))
    return rows


def _child_order_traces(
    orders: list[dict],
    *,
    show_child_ids: bool = True,
    line_width: float = 3.4,
) -> list[go.Scatter]:
    traces: list[go.Scatter] = []
    parent_color_map: dict[tuple[str, str], str] = {}
    buy_palette_i = 0
    sell_palette_i = 0
    for order in orders:
        side = order["side"]
        parent_order_id = str(order.get("parent_order_id") or order["client_order_id"])
        color = parent_color_map.get((parent_order_id, side))
        if color is None:
            if side == "BUY":
                color = BUY_COLORS[buy_palette_i % len(BUY_COLORS)]
                buy_palette_i += 1
            else:
                color = SELL_COLORS[sell_palette_i % len(SELL_COLORS)]
                sell_palette_i += 1
            parent_color_map[(parent_order_id, side)] = color

        x0 = pd.to_datetime(order["start_ts_us"], unit="us")
        x1 = pd.to_datetime(order["end_ts_us"], unit="us")
        y = order["price"]
        if x1 <= x0:
            order_x = [x0, x1]
        else:
            duration_seconds = max((x1 - x0).total_seconds(), 0.0)
            sample_count = max(25, min(200, int(duration_seconds) + 2))
            step = (x1 - x0) / (sample_count - 1)
            order_x = [x0 + step * i for i in range(sample_count)]
        order_y = [y] * len(order_x)
        hover_parts = [
            f"parent_order_id={order.get('parent_order_id', '')}",
            f"client_order_id={order['client_order_id']}",
            f"side={side}",
            f"size={'' if order.get('size') is None else order.get('size')}",
            f"price={y}",
        ]
        if show_child_ids:
            hover_parts.insert(2, f"child_order_id={order['client_order_id']}")
        order_hover = "<br>".join(hover_parts) + "<extra></extra>"
        traces.append(
            go.Scatter(
                x=[x0, x1] if x1 > x0 else order_x,
                y=[y, y] if x1 > x0 else order_y,
                mode="lines",
                name=f"{side} {order['client_order_id']}",
                showlegend=False,
                line={"color": color, "width": line_width},
                line_shape="hv",
                customdata=[str(order.get("parent_order_id") or "")] * (2 if x1 > x0 else len(order_x)),
                hovertemplate=order_hover,
            )
        )
        traces.append(
            go.Scatter(
                x=order_x,
                y=order_y,
                mode="markers",
                showlegend=False,
                marker={"size": 18, "color": color, "opacity": 0.001},
                customdata=[str(order.get("parent_order_id") or "")] * len(order_x),
                hovertemplate=order_hover,
            )
        )
    return traces


def _aggregated_order_traces(orders: list[dict], *, line_width: float = 2.2) -> list[go.Scatter]:
    traces: list[go.Scatter] = []
    aggregated_orders = _aggregate_parent_orders(orders)
    parent_color_map: dict[tuple[str, str], str] = {}
    buy_palette_i = 0
    sell_palette_i = 0
    for order in aggregated_orders:
        parent_order_id = str(order["parent_order_id"])
        side = order["side"]
        color = parent_color_map.get((parent_order_id, side))
        if color is None:
            if side == "BUY":
                color = BUY_COLORS[buy_palette_i % len(BUY_COLORS)]
                buy_palette_i += 1
            else:
                color = SELL_COLORS[sell_palette_i % len(SELL_COLORS)]
                sell_palette_i += 1
            parent_color_map[(parent_order_id, side)] = color

        x0 = pd.to_datetime(order["start_ts_us"], unit="us")
        x1 = pd.to_datetime(order["end_ts_us"], unit="us")
        points = order["price_points"]
        if len(points) == 1:
            points = [(order["start_ts_us"], points[0][1]), (order["end_ts_us"], points[0][1])]
        order_x = [pd.to_datetime(ts_us, unit="us") for ts_us, _ in points]
        order_y = [price for _, price in points]
        if x1 <= x0:
            order_x = [x0, x1]
            order_y = [order_y[0], order_y[0]]
        order_hover = (
            f"parent_order_id={parent_order_id}<br>"
            f"side={side}<br>"
            f"order_count={order['order_count']}<br>"
            f"price={order_y[-1]}<extra></extra>"
        )
        traces.append(
            go.Scatter(
                x=order_x,
                y=order_y,
                mode="lines",
                name=f"{side} {parent_order_id}",
                showlegend=False,
                line={"color": color, "width": line_width},
                line_shape="hv",
                customdata=[parent_order_id] * len(order_x),
                hovertemplate=order_hover,
            )
        )
    return traces


def _child_fill_traces(fills: list[dict]) -> tuple[list, list, list]:
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
    return (buy_fill_x, buy_fill_y, buy_fill_text), (sell_fill_x, sell_fill_y, sell_fill_text)


def _aggregated_fill_traces(fills: list[dict]) -> tuple[list, list, list]:
    buy_fill_x: list = []
    buy_fill_y: list = []
    buy_fill_text: list = []
    sell_fill_x: list = []
    sell_fill_y: list = []
    sell_fill_text: list = []
    aggregated_fills = _aggregate_parent_fills(fills)
    for fill in aggregated_fills:
        x = pd.to_datetime(fill["last_ts_us"], unit="us")
        text = (
            f"parent_order_id={fill.get('parent_order_id', '')}<br>"
            f"side={fill['side']}<br>"
            f"fill_count={fill['fill_count']}<br>"
            f"total_qty={fill['total_qty']}<br>"
            f"last_price={fill['last_price']}<br>"
            f"last_qty={fill['last_qty']}"
        )
        if fill["side"] == "BUY":
            buy_fill_x.append(x)
            buy_fill_y.append(fill["last_price"])
            buy_fill_text.append(text)
        else:
            sell_fill_x.append(x)
            sell_fill_y.append(fill["last_price"])
            sell_fill_text.append(text)
    return (buy_fill_x, buy_fill_y, buy_fill_text), (sell_fill_x, sell_fill_y, sell_fill_text)


@lru_cache(maxsize=32)
def load_orders(order_log_path_str: str, symbol: str, mtime_ns: int) -> list[dict]:
    order_log_path = Path(order_log_path_str)
    if order_log_path.suffix == ".parquet":
        return _orders_from_parquet(order_log_path, symbol, mtime_ns)
    del mtime_ns
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
    order_log_path = Path(order_log_path_str)
    if order_log_path.suffix == ".parquet":
        return _fills_from_parquet(order_log_path, symbol, mtime_ns)
    del mtime_ns
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
        html.P("Click a symbol for the multi-day overview, or a date for daily metrics. From there, open the hourly bid/ask + order chart."),
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
                            html.A(
                                symbol,
                                href=f"/symbol?head={quote(str(head))}&symbol={quote(symbol)}",
                                target="_self",
                                style={"fontWeight": "bold", "marginRight": "10px"},
                            ),
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


def make_daily_figure(
    metrics_df: pd.DataFrame,
    state_df: pd.DataFrame,
    symbol: str,
    head: Path,
    date: str,
    *,
    position_title: str = "Position (resets daily)",
    title_suffix: str = "Daily Metrics",
    carry_pnl_across_days: bool = False,
) -> go.Figure:
    metrics_df = _minute_last_frame(metrics_df)
    state_df = _minute_last_frame(state_df)
    merged = pd.merge(
        metrics_df[[col for col in ("time", "position", "pnl", "cum_notional_traded", "cum_size_traded") if col in metrics_df.columns]],
        state_df[[col for col in ("time", "bid", "ask") if col in state_df.columns]],
        on="time",
        how="inner",
    )
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        specs=[[{}], [{}], [{}], [{}]],
        subplot_titles=(position_title, "Mid Price", "PnL", "Cumulative Notional"),
    )
    series = (
        (1, "notional_position", "#0F766E", False, "Notional Position"),
        (3, "cum_pnl", "#2563EB", False, "Cum PnL"),
        (4, "cum_notional_traded", "#C2410C", False, "Cum Notional"),
    )
    if not merged.empty:
        merged["mid"] = (pd.to_numeric(merged["bid"], errors="coerce") + pd.to_numeric(merged["ask"], errors="coerce")) / 2.0
        merged["notional_position"] = pd.to_numeric(merged["position"], errors="coerce") * merged["mid"]
        merged["pnl"] = pd.to_numeric(merged["pnl"], errors="coerce").fillna(0.0)
        if carry_pnl_across_days:
            merged["cum_pnl"] = merged["pnl"].cumsum()
        else:
            merged["cum_pnl"] = merged["pnl"]
    for row_idx, col, color, secondary_y, label in series:
        if col == "notional_position":
            y = pd.to_numeric(merged[col], errors="coerce") if not merged.empty else pd.Series(dtype=float)
            x_values = merged["time"] if not merged.empty else []
        else:
            y = pd.to_numeric(merged[col], errors="coerce") if not merged.empty and col in merged.columns else pd.to_numeric(metrics_df[col], errors="coerce")
            x_values = merged["time"] if not merged.empty and col in merged.columns else metrics_df["time"]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y,
                mode="lines",
                name=label,
                line={"color": color, "width": 1.6},
                line_shape="hv",
                showlegend=False,
                hovertemplate=f"{label}=%{{y}}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )
    if not state_df.empty:
        mid = ((pd.to_numeric(state_df["bid"], errors="coerce") + pd.to_numeric(state_df["ask"], errors="coerce")) / 2.0).dropna()
        if not mid.empty:
            fig.add_trace(
                go.Scatter(
                    x=state_df.loc[mid.index, "time"],
                    y=mid,
                    mode="lines",
                    name="Mid Price",
                    line={"color": "#B45309", "width": 1.4},
                    line_shape="hv",
                    showlegend=False,
                    hovertemplate="Mid Price=%{y}<extra></extra>",
                ),
                row=2,
                col=1,
            )
    fig.update_yaxes(
        title={"text": "Notional Position (USD)", "font": {"color": "#0F766E"}, "standoff": 2},
        tickfont={"color": "#0F766E"},
        row=1,
        col=1,
        automargin=True,
    )
    fig.update_yaxes(
        title={"text": "Mid Price", "font": {"color": "#B45309"}, "standoff": 2},
        tickfont={"color": "#B45309"},
        row=2,
        col=1,
        automargin=True,
    )
    fig.update_yaxes(
        title={"text": "Cum PnL", "font": {"color": "#2563EB"}},
        tickfont={"color": "#2563EB"},
        row=3,
        col=1,
        automargin=True,
    )
    fig.update_yaxes(
        title={"text": "Cum Notional", "font": {"color": "#C2410C"}},
        tickfont={"color": "#C2410C"},
        row=4,
        col=1,
        automargin=True,
    )

    fig.update_xaxes(row=1, col=1, domain=[0.02, 1.0])
    fig.update_xaxes(row=2, col=1, domain=[0.02, 1.0])
    fig.update_xaxes(row=3, col=1, domain=[0.02, 1.0])
    fig.update_xaxes(title="Time", row=4, col=1, domain=[0.02, 1.0])
    fig.update_layout(
        yaxis={"side": "left", "position": 0.02, "automargin": True},
        yaxis2={"side": "left", "position": 0.02, "automargin": True},
        yaxis3={"side": "left", "position": 0.02, "automargin": True},
        yaxis4={"side": "left", "position": 0.02, "automargin": True},
    )
    fig.update_layout(
        template="plotly_white",
        height=980,
        title=f"{symbol} | {date} | {head} | {title_suffix}",
        margin={"l": 18, "r": 85, "t": 70, "b": 50},
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

    state_path = head / "log" / "state" / f"{symbol}.0.{date}.parquet"
    order_path = head / "log" / f"order.{date}.parquet"
    if not order_path.exists():
        order_path = head / "log" / f"order.{date}.log"
    if not state_path.exists():
        return html.Div([html.H3("State file not found"), html.Pre(str(state_path))])

    metrics_df = load_daily_metrics_frame(str(state_path), state_path.stat().st_mtime_ns)
    if metrics_df.empty:
        return html.Div([html.H3("State file has no usable rows"), html.Pre(str(state_path))])
    state_df = load_state_frame(str(state_path), state_path.stat().st_mtime_ns)

    orders = []
    if order_path.exists():
        orders = load_orders(str(order_path), symbol, order_path.stat().st_mtime_ns)
    total_fees_usd = load_total_fees(str(order_path), symbol, order_path.stat().st_mtime_ns) if order_path.exists() else 0.0
    available_hours = sorted({pd.to_datetime(o["start_ts_us"], unit="us").hour for o in orders})

    missing = []
    for col in ("position", "pnl", "cum_notional_traded", "cum_size_traded"):
        if metrics_df[col].dropna().empty:
            missing.append(col)

    fig = make_daily_figure(
        metrics_df,
        state_df,
        symbol,
        head,
        date,
        position_title="Position (resets daily)",
    )

    report_df = _build_report_frame(metrics_df, state_df, 1)
    report_section = None
    if not report_df.empty:
        day_total_notional_usd = _report_total_notional_usd(report_df)
        day_stats = _report_stats(report_df, total_fees_usd=total_fees_usd, total_notional_usd=day_total_notional_usd)
        pnl_series = pd.to_numeric(report_df["pnl"], errors="coerce").fillna(0.0)
        equity_values = pnl_series
        drawdown_values = equity_values - equity_values.cummax()
        returns_values = equity_values.diff().fillna(equity_values.iloc[0])
        return_stats = _return_series_stats(pnl_series.diff().fillna(pnl_series.iloc[0]) / day_total_notional_usd if day_total_notional_usd else pnl_series * 0.0)
        summary_rows = [
            ("Total Notional", _format_usd(day_stats["total_notional_usd"])),
            ("Total PnL", _format_usd(day_stats["total_pnl"])),
            ("Total Return", _format_bps(day_stats["total_return"])),
            ("Total Fees", _format_bps(day_stats["total_fees_bps"])),
            ("Max Drawdown", _format_usd(day_stats["max_drawdown"])),
            ("Sharpe (per period)", _format_ratio(return_stats["sharpe"])),
            ("Sortino (per period)", _format_ratio(return_stats["sortino"])),
            ("Profit Factor", _format_ratio(return_stats["profit_factor"])),
            ("Volatility", _format_num(return_stats["volatility"], digits=4)),
            ("Win Rate", _format_pct(return_stats["win_rate"])),
            ("Max Abs Position USD", _format_usd(day_stats["max_abs_position_usd"])),
            ("Max Abs Position", _format_num(day_stats["max_abs_position"], digits=4)),
            ("Samples", str(len(report_df))),
            ("Window", f"{report_df['time'].iloc[0]} -> {report_df['time'].iloc[-1]}"),
        ]
        report_fig = _make_quant_summary_figure(
            report_df["time"],
            equity_values,
            drawdown_values,
            returns_values,
            title=f"{symbol} | {date} | Quant Summary",
            x_title="Time",
        )
        report_section = _make_report_panel("Daily Quant Summary", summary_rows, report_fig)
    hour_options = [{"label": "Select hour", "value": "all"}]
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
            report_section if report_section is not None else html.Div(),
            dcc.Store(id="day-nav-meta", data={"head": str(head), "symbol": symbol, "date": date}),
            dcc.Graph(id="daily-metrics-graph", figure=fig, style={"height": "88vh"}),
        ]
    )


def render_symbol(search: str) -> html.Div:
    query = parse_qs(search.lstrip("?"))
    head_raw = unquote(query.get("head", [""])[0])
    symbol = unquote(query.get("symbol", [""])[0])

    head = normalize_head(head_raw)
    if head is None:
        return html.Div([html.H3("Invalid head path"), html.Pre(head_raw)])
    if not symbol:
        return html.Div([html.H3("Missing symbol"), html.Pre(search)])

    state_entries = [entry for entry in state_files_for_head(head) if entry.symbol == symbol]
    if not state_entries:
        return html.Div([html.H3("No dates found for symbol"), html.Pre(symbol)])

    state_entries.sort(key=lambda x: x.date)
    bucket_minutes = max(1, len(state_entries))

    metric_rows: list[pd.DataFrame] = []
    state_rows: list[pd.DataFrame] = []
    available_dates: list[str] = []
    for entry in state_entries:
        state_path = entry.path
        metrics_df = load_daily_metrics_frame(str(state_path), state_path.stat().st_mtime_ns)
        state_df = load_state_frame(str(state_path), state_path.stat().st_mtime_ns)
        if metrics_df.empty or state_df.empty:
            continue
        metric_bucket = _bucket_last_frame(metrics_df, bucket_minutes)
        state_bucket = _bucket_last_frame(state_df, bucket_minutes)
        if metric_bucket.empty or state_bucket.empty:
            continue
        metric_rows.append(metric_bucket)
        state_rows.append(state_bucket)
        available_dates.append(entry.date)

    if not metric_rows or not state_rows:
        return html.Div([html.H3("No usable summary data"), html.Pre(symbol)])

    date_options = [{"label": date, "value": date} for date in sorted(available_dates)]
    latest_date = available_dates[-1]

    symbol_report_frames: list[pd.DataFrame] = []
    day_reports: list[dict] = []
    total_fees_usd = 0.0
    total_notional_usd = 0.0
    cumulative_pnl_offset = 0.0
    for entry in state_entries:
        order_path = head / "log" / f"order.{entry.date}.parquet"
        if not order_path.exists():
            order_path = head / "log" / f"order.{entry.date}.log"
        metrics_df = load_daily_metrics_frame(str(entry.path), entry.path.stat().st_mtime_ns)
        state_df = load_state_frame(str(entry.path), entry.path.stat().st_mtime_ns)
        report_df = _build_report_frame(metrics_df, state_df, bucket_minutes)
        if report_df.empty:
            continue
        report_df = report_df.copy()
        day_pnl_series = pd.to_numeric(report_df["pnl"], errors="coerce").fillna(0.0)
        report_df["pnl_delta"] = day_pnl_series.diff().fillna(day_pnl_series.iloc[0])
        report_df["pnl_cumulative"] = cumulative_pnl_offset + day_pnl_series
        cumulative_pnl_offset = float(report_df["pnl_cumulative"].iloc[-1])
        symbol_report_frames.append(report_df)
        day_total_fees_usd = load_total_fees(str(order_path), symbol, order_path.stat().st_mtime_ns) if order_path.exists() else 0.0
        day_total_notional_usd = _report_total_notional_usd(report_df)
        total_fees_usd += day_total_fees_usd
        total_notional_usd += day_total_notional_usd
        day_stats = _report_stats(report_df, total_fees_usd=day_total_fees_usd, total_notional_usd=day_total_notional_usd)
        day_pnl = float(day_stats["total_pnl"])
        day_reports.append(
            {
                "date": entry.date,
                "time": pd.to_datetime(entry.date, format="%Y%m%d"),
                "pnl": day_pnl,
                "return": day_pnl / day_total_notional_usd if day_total_notional_usd else 0.0,
                "max_drawdown": float(day_stats["max_drawdown"]),
                "max_abs_position_usd": float(day_stats["max_abs_position_usd"]),
                "max_abs_position": float(day_stats["max_abs_position"]),
                "notional_usd": float(day_total_notional_usd),
                "samples": int(len(report_df)),
            }
        )

    symbol_report_section = None
    if day_reports and symbol_report_frames:
        day_report_df = pd.DataFrame(day_reports).sort_values("date").reset_index(drop=True)
        symbol_report_df = pd.concat(symbol_report_frames, ignore_index=True).sort_values("time").reset_index(drop=True)
        symbol_overview_df = symbol_report_df.copy()
        symbol_overview_df["pnl"] = pd.to_numeric(symbol_overview_df["pnl_cumulative"], errors="coerce").fillna(0.0)
        fig = make_daily_figure(
            symbol_overview_df,
            symbol_overview_df,
            symbol,
            head,
            f"{available_dates[0]} - {available_dates[-1]}",
            position_title="Position (resets daily)",
            title_suffix="Symbol Overview",
        )
        day_report_df["return"] = day_report_df["pnl"] / day_report_df["notional_usd"].replace(0, pd.NA)
        symbol_period_pnl = pd.to_numeric(symbol_report_df["pnl_delta"], errors="coerce").fillna(0.0)
        symbol_equity = pd.to_numeric(symbol_report_df["pnl_cumulative"], errors="coerce").fillna(0.0)
        symbol_drawdown = symbol_equity - symbol_equity.cummax()
        symbol_return_series = symbol_period_pnl / total_notional_usd if total_notional_usd else symbol_period_pnl * 0.0
        return_stats = _return_series_stats(symbol_return_series)

        summary_rows = [
            ("Total Notional", _format_usd(total_notional_usd)),
            ("Total PnL", _format_usd(float(day_report_df["pnl"].sum()))),
            ("Total Return", _format_bps(float(day_report_df["pnl"].sum() / total_notional_usd) if total_notional_usd else 0.0)),
            ("Total Fees", _format_bps(float(total_fees_usd / total_notional_usd) if total_notional_usd else 0.0)),
            ("Max Drawdown", _format_usd(float(symbol_drawdown.min()))),
            ("Sharpe (per period)", _format_ratio(return_stats["sharpe"])),
            ("Sortino (per period)", _format_ratio(return_stats["sortino"])),
            ("Profit Factor", _format_ratio(return_stats["profit_factor"])),
            ("Volatility", _format_num(return_stats["volatility"], digits=4)),
            ("Days", str(len(day_report_df))),
            ("Win Rate", _format_pct(float((symbol_return_series > 0).mean()))),
            ("Avg Daily Return", _format_bps(float(day_report_df["return"].mean()))),
            ("Best Day", f"{day_report_df.loc[day_report_df['pnl'].idxmax(), 'date']} ({_format_usd(float(day_report_df['pnl'].max()))})"),
            ("Worst Day", f"{day_report_df.loc[day_report_df['pnl'].idxmin(), 'date']} ({_format_usd(float(day_report_df['pnl'].min()))})"),
            ("Max Abs Position USD", _format_usd(float(day_report_df["max_abs_position_usd"].max()))),
            ("Max Abs Position", _format_num(float(day_report_df["max_abs_position"].max()), digits=4)),
        ]
        report_fig = _make_quant_summary_figure(
            symbol_report_df["time"],
            symbol_equity,
            symbol_drawdown,
            symbol_period_pnl,
            title=f"{symbol} | Multi-day Quant Summary",
            x_title="Time",
        )
        breakdown = _make_data_table(
            "Daily Breakdown",
            ["Date", "PnL", "Return", "Max DD", "Max Abs Pos USD", "Max Abs Pos", "Samples"],
            [
                {
                    "Date": row["date"],
                    "PnL": _format_usd(float(row["pnl"])),
                    "Return": _format_pct(float(row["return"])),
                    "Max DD": _format_pct(float(row["max_drawdown"])),
                    "Max Abs Pos USD": _format_usd(float(row["max_abs_position_usd"])),
                    "Max Abs Pos": _format_num(float(row["max_abs_position"]), digits=4),
                    "Samples": str(int(row["samples"])),
                }
                for _, row in day_report_df.iterrows()
            ],
        )
        symbol_report_section = _make_report_panel("Multi-day Quant Summary", summary_rows, report_fig, breakdown=breakdown)
    else:
        fig = make_daily_figure(
            pd.DataFrame(),
            pd.DataFrame(),
            symbol,
            head,
            f"{available_dates[0]} - {available_dates[-1]}",
            position_title="Position (resets daily)",
            title_suffix="Symbol Overview",
        )

    return html.Div(
        [
            html.Div(
                [
                    html.A("Main page", href="/", target="_self", style={"marginRight": "12px"}),
                    html.Span(f"Symbol: {symbol}", style={"marginRight": "12px", "fontWeight": "600"}),
                    html.Span(f"Dates: {len(available_dates)}", style={"marginRight": "12px"}),
                    dcc.Dropdown(
                        id="symbol-date-dropdown",
                        options=date_options,
                        value=None,
                        placeholder=f"Date (latest {latest_date})",
                        clearable=False,
                        style={"width": "160px", "display": "inline-block", "verticalAlign": "middle"},
                    ),
                ],
                style={"margin": "8px 12px"},
            ),
            html.Div(
                f"{symbol} | {head}",
                style={"margin": "0 12px 6px 12px", "fontWeight": "600"},
            ),
            symbol_report_section if symbol_report_section is not None else html.Div(),
            dcc.Store(id="symbol-nav-meta", data={"head": str(head), "symbol": symbol}),
            dcc.Graph(id="symbol-overview-graph", figure=fig, style={"height": "88vh"}),
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
    detailed_orders: bool = False,
    detailed_fills: bool = False,
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
                line_shape="hv",
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
                line_shape="hv",
                hoverinfo="skip",
            )
        )

    if detailed_orders:
        order_traces = _child_order_traces(orders, show_child_ids=True, line_width=3.4)
    else:
        order_traces = _aggregated_order_traces(orders, line_width=2.0)

    if detailed_fills:
        (buy_fill_x, buy_fill_y, buy_fill_text), (sell_fill_x, sell_fill_y, sell_fill_text) = _child_fill_traces(fills)
        buy_fill_parent = [str(fill.get("parent_order_id") or "") for fill in fills if fill["side"] == "BUY"]
        sell_fill_parent = [str(fill.get("parent_order_id") or "") for fill in fills if fill["side"] == "SELL"]
    else:
        aggregated_fills = _aggregate_parent_fills(fills)
        (buy_fill_x, buy_fill_y, buy_fill_text), (sell_fill_x, sell_fill_y, sell_fill_text) = _aggregated_fill_traces(fills)
        buy_fill_parent = [str(fill.get("parent_order_id") or "") for fill in aggregated_fills if fill["side"] == "BUY"]
        sell_fill_parent = [str(fill.get("parent_order_id") or "") for fill in aggregated_fills if fill["side"] == "SELL"]

    for trace in order_traces:
        fig.add_trace(trace)

    if buy_fill_x:
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
        domain=[0.05, 1.0],
        type="date",
    )
    fig.update_yaxes(
        title={"text": "Price", "standoff": 4},
        side="left",
        position=0.05,
        automargin=True,
        range=[y_min, y_max],
    )
    fig.update_layout(
        template="plotly_white",
        height=760,
        title="Orders / Fills" if not show_book else "Orders / Bid-Ask",
        clickmode="event",
        hovermode="closest",
        hoverdistance=20,
        yaxis2={
            "title": {"text": "bps", "font": {"color": BPS_COLOR}},
            "overlaying": "y",
            "matches": "y",
            "side": "left",
            "position": 0.02,
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
            "ticklabelstandoff": 4,
            "automargin": True,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 42, "r": 20, "t": 80, "b": 50},
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
        go.Scatter(
            x=position_df["time"],
            y=position_df["position"],
            mode="lines",
            name="Position",
            line={"color": "#2a9d8f", "width": 1.8},
            line_shape="hv",
            hovertemplate="position=%{y}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Time", range=[initial_start, initial_end], rangeslider={"visible": False}, type="date", domain=[0.05, 1.0])
    fig.update_yaxes(title={"text": "Position", "standoff": 4}, side="left", position=0.05, automargin=True)
    fig.update_layout(
        template="plotly_white",
        height=140,
        title="Position",
        margin={"l": 42, "r": 20, "t": 60, "b": 40},
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

    state_path = head / "log" / "state" / f"{symbol}.0.{date}.parquet"
    order_path = head / "log" / f"order.{date}.parquet"
    if not order_path.exists():
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

    state_window_df = _slice_window(state_df, window_start, window_end)
    position_window_df = _slice_window(position_df, window_start, window_end)
    orders_window = [o for o in orders if pd.to_datetime(o["end_ts_us"], unit="us") >= window_start and pd.to_datetime(o["start_ts_us"], unit="us") <= window_end]
    fills_window = [f for f in fills if window_start <= pd.to_datetime(f["ts_us"], unit="us") <= window_end]

    fig = make_figure(
        state_window_df,
        orders_window,
        fills_window,
        base_mid_price,
        symbol,
        head,
        date,
        window_start,
        window_end,
        show_book=True,
        detailed_orders=False,
        detailed_fills=False,
    )
    position_fig = make_position_figure(position_window_df, symbol, head, date, window_start, window_end)
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
            dcc.Graph(id="symbol-graph", figure=fig, style={"height": "62vh"}),
            dcc.Graph(id="position-graph", figure=position_fig, style={"height": "20vh"}),
        ]
    )


def render_parent(search: str) -> html.Div:
    query = parse_qs(search.lstrip("?"))
    head_raw = unquote(query.get("head", [""])[0])
    symbol = unquote(query.get("symbol", [""])[0])
    date = query.get("date", [""])[0]
    parent_order_id = unquote(query.get("parent_order_id", [""])[0])

    head = normalize_head(head_raw)
    if head is None:
        return html.Div([html.H3("Invalid head path"), html.Pre(head_raw)])
    if not symbol or len(date) != 8 or not date.isdigit():
        return html.Div([html.H3("Missing symbol/date"), html.Pre(search)])
    if not parent_order_id:
        return html.Div([html.H3("Missing parent_order_id"), html.Pre(search)])

    state_path = head / "log" / "state" / f"{symbol}.0.{date}.parquet"
    order_path = head / "log" / f"order.{date}.parquet"
    if not order_path.exists():
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

    state_window_df = _slice_window(state_df, window_start, window_end)
    position_window_df = _slice_window(position_df, window_start, window_end)
    orders_window = [o for o in orders if pd.to_datetime(o["end_ts_us"], unit="us") >= window_start and pd.to_datetime(o["start_ts_us"], unit="us") <= window_end]
    fills_window = [f for f in fills if window_start <= pd.to_datetime(f["ts_us"], unit="us") <= window_end]

    fig = make_figure(
        state_window_df,
        orders_window,
        fills_window,
        base_mid_price,
        symbol,
        head,
        date,
        window_start,
        window_end,
        show_book=True,
        detailed_orders=True,
        detailed_fills=True,
    )
    position_fig = make_position_figure(position_window_df, symbol, head, date, window_start, window_end)
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
            dcc.Graph(id="symbol-graph", figure=fig, style={"height": "62vh"}),
            dcc.Graph(id="position-graph", figure=position_fig, style={"height": "20vh"}),
        ]
    )


app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Simulation Dashboard"
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        html.Div(id="page-content"),
    ]
)


@app.callback(Output("page-content", "children"), Input("url", "pathname"), Input("url", "search"))
def route(pathname: str, search: str):
    if pathname == "/symbol":
        return render_symbol(search)
    if pathname == "/day":
        return render_day(search)
    if pathname == "/parent":
        return render_parent(search)
    if pathname == "/chart":
        return render_chart(search)
    return render_index()


app.clientside_callback(
    """
    function(hourValue, dayNavMeta) {
        const noUpdate = window.dash_clientside.no_update;
        if (!hourValue || hourValue === "all") {
            return noUpdate;
        }

        let head = "";
        let symbol = "";
        let date = "";
        if (dayNavMeta) {
            head = String(dayNavMeta.head || "");
            symbol = String(dayNavMeta.symbol || "");
            date = String(dayNavMeta.date || "");
        }

        const hourInt = parseInt(hourValue, 10);
        if (!head || !symbol || !/^\\d{8}$/.test(date) || Number.isNaN(hourInt) || hourInt < 0 || hourInt > 23) {
            return noUpdate;
        }

        const hourIso = date.slice(0, 4) + "-" + date.slice(4, 6) + "-" + date.slice(6, 8) +
            "T" + String(hourInt).padStart(2, "0") + ":00:00";
        const nextParams = new URLSearchParams();
        nextParams.set("head", head);
        nextParams.set("symbol", symbol);
        nextParams.set("date", date);
        nextParams.set("hour", hourIso);
        return "/chart?" + nextParams.toString();
    }
    """,
    Output("url", "href", allow_duplicate=True),
    Input("day-hour-dropdown", "value", allow_optional=True),
    State("day-nav-meta", "data", allow_optional=True),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(dateValue, symbolNavMeta) {
        const noUpdate = window.dash_clientside.no_update;
        if (!dateValue) {
            return noUpdate;
        }

        let head = "";
        let symbol = "";
        if (symbolNavMeta) {
            head = String(symbolNavMeta.head || "");
            symbol = String(symbolNavMeta.symbol || "");
        }

        if (!head || !symbol) {
            return noUpdate;
        }

        const nextParams = new URLSearchParams();
        nextParams.set("head", head);
        nextParams.set("symbol", symbol);
        nextParams.set("date", String(dateValue));
        return "/day?" + nextParams.toString();
    }
    """,
    Output("url", "href", allow_duplicate=True),
    Input("symbol-date-dropdown", "value", allow_optional=True),
    State("symbol-nav-meta", "data", allow_optional=True),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(clickData, pathname, hoverData, hourlyNavMeta) {
        const noUpdate = window.dash_clientside.no_update;
        if (pathname !== "/chart" || !clickData) {
            return noUpdate;
        }

        function extractParent(payload) {
            if (!payload || !Array.isArray(payload.points)) {
                return null;
            }
            for (const point of payload.points) {
                if (!point || point.customdata === undefined || point.customdata === null) {
                    continue;
                }
                const candidate = String(point.customdata).trim();
                if (candidate) {
                    return candidate;
                }
            }
            return null;
        }

        const parentOrderId = extractParent(clickData) || extractParent(hoverData);
        if (!parentOrderId) {
            return noUpdate;
        }

        let head = "";
        let symbol = "";
        let date = "";
        if (hourlyNavMeta) {
            head = String(hourlyNavMeta.head || "");
            symbol = String(hourlyNavMeta.symbol || "");
            date = String(hourlyNavMeta.date || "");
        }

        if (!head || !symbol || !/^\\d{8}$/.test(date)) {
            return noUpdate;
        }

        const nextParams = new URLSearchParams();
        nextParams.set("head", head);
        nextParams.set("symbol", symbol);
        nextParams.set("date", date);
        nextParams.set("parent_order_id", parentOrderId);
        return "/parent?" + nextParams.toString();
    }
    """,
    Output("url", "href", allow_duplicate=True),
    Input("symbol-graph", "clickData", allow_optional=True),
    State("url", "pathname"),
    State("symbol-graph", "hoverData", allow_optional=True),
    State("hourly-nav-meta", "data", allow_optional=True),
    prevent_initial_call=True,
)


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
