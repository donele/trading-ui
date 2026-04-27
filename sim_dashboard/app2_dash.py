#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import os
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots

try:
    from sim_dashboard.app2 import SimData
except ImportError:  # pragma: no cover
    from app2 import SimData


ROOT_ORDER = ("dumpsim", "livesim", "tradesim")
ROOTS = [Path.home() / "workspace" / "sgt" / name for name in ROOT_ORDER]


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


def parse_state_filename(path: Path):
    parts = path.name.split(".")
    if len(parts) < 4 or parts[-1] != "parquet":
        return None
    date = parts[-2]
    if len(date) != 8 or not date.isdigit():
        return None
    symbol = ".".join(parts[:-3])
    if not symbol:
        return None
    return {"symbol": symbol, "date": date, "path": path}


def state_files_for_head(head: Path):
    state_dir = head / "log" / "state"
    if not state_dir.is_dir():
        return []
    rows = []
    for pq_path in state_dir.glob("*.parquet"):
        parsed = parse_state_filename(pq_path)
        if parsed is not None:
            rows.append(parsed)
    rows.sort(key=lambda x: (x["symbol"], x["date"]))
    return rows


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
            if not any(log_dir.glob("order.????????.parquet")) and not any(log_dir.glob("order.????????.log")):
                continue
            grouped[root_name].append(log_dir.parent)
        grouped[root_name].sort()
    return grouped


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(dtype=float)


def _make_step_trace(
    x,
    y,
    name: str,
    color: str,
    showlegend: bool = False,
    legendgroup: str | None = None,
):
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line={"color": color, "width": 1.8},
        line_shape="hv",
        showlegend=showlegend,
        legendgroup=legendgroup,
        hovertemplate=f"{name}=%{{y}}<extra></extra>",
    )


def make_symbol_figure(simdata: SimData, symbol: str, sdate: str, freq: str = "5min") -> go.Figure:
    timeline = simdata.get_timeline(symbol, sdate, freq=freq)
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Size Traded", "Notional Traded", "Notional Position", "Mid", "PnL", "Fees"),
    )
    series = [
        ("size_traded", "#0F766E"),
        ("notional_traded", "#2563EB"),
        ("notional_pos", "#C2410C"),
        ("mid", "#7C3AED"),
        ("pnl", "#1D4ED8"),
        ("fees", "#DC2626"),
    ]
    for row_idx, (col, color) in enumerate(series, start=1):
        if timeline.empty or col not in timeline.columns:
            continue
        fig.add_trace(_make_step_trace(timeline.index, _safe_series(timeline, col), col, color), row=row_idx, col=1)
        fig.update_yaxes(title=col, row=row_idx, col=1, automargin=True)
    fig.update_xaxes(title="Time", row=6, col=1)
    fig.update_layout(
        template="plotly_white",
        height=980,
        title=f"{symbol} | {sdate}",
        hovermode="x unified",
        margin={"l": 18, "r": 18, "t": 70, "b": 40},
    )
    return fig


def make_portfolio_figure(simdata: SimData, sdate: str, freq: str = "5min") -> go.Figure:
    timelines = simdata.get_timelines(sdate, freq=freq)
    palette = [
        "#2563EB",
        "#0F766E",
        "#C2410C",
        "#7C3AED",
        "#0891B2",
        "#65A30D",
        "#DB2777",
        "#EA580C",
        "#4F46E5",
        "#6B7280",
    ]
    symbol_colors = {symbol: palette[idx % len(palette)] for idx, symbol in enumerate(sorted(timelines.keys()))}
    seen_symbols: set[str] = set()
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Notional Traded", "Notional Position", "PnL"),
    )
    series = ["notional_traded", "notional_pos", "pnl"]
    for row_idx, col in enumerate(series, start=1):
        total_series = []
        for symbol, tl in timelines.items():
            if tl.empty or col not in tl.columns:
                continue
            fig.add_trace(
                _make_step_trace(
                    tl.index,
                    _safe_series(tl, col),
                    symbol,
                    symbol_colors.get(symbol, "#2563EB"),
                    showlegend=symbol not in seen_symbols,
                    legendgroup=symbol,
                ),
                row=row_idx,
                col=1,
            )
            seen_symbols.add(symbol)
            total_series.append(_safe_series(tl, col))
        if total_series:
            total = pd.concat(total_series, axis=1).sum(axis=1)
            fig.add_trace(
                go.Scatter(
                    x=total.index,
                    y=total,
                    mode="lines",
                    name="total",
                    line={"color": "#111827", "width": 0.8},
                    line_shape="hv",
                    hovertemplate="total=%{y}<extra></extra>",
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
        fig.update_yaxes(title=col, row=row_idx, col=1, automargin=True)
    fig.update_xaxes(title="Time", row=3, col=1)
    fig.update_layout(
        template="plotly_white",
        height=820,
        title=f"Portfolio | {sdate}",
        hovermode="x unified",
        margin={"l": 18, "r": 18, "t": 70, "b": 40},
    )
    return fig


def _side_bucket(raw: object) -> str | None:
    if raw is None:
        return None
    value = str(raw).upper()
    if value in {"BID", "BUY"}:
        return "BUY"
    if value in {"ASK", "SELL"}:
        return "SELL"
    return None


def _make_order_segment_trace(df: pd.DataFrame, *, side: str, color: str) -> go.Scatter:
    xs = []
    ys = []
    if not df.empty:
        for row in df.itertuples(index=False):
            start_ts = getattr(row, "create_time", None)
            end_ts = getattr(row, "last_update_time", None)
            price = getattr(row, "price", None)
            if start_ts is None or end_ts is None or price is None:
                continue
            try:
                x0 = pd.to_datetime(int(start_ts), unit="us")
                x1 = pd.to_datetime(int(end_ts), unit="us")
                y = float(price)
            except (TypeError, ValueError):
                continue
            if x1 <= x0:
                x1 = x0 + timedelta(microseconds=1)
            xs.extend([x0, x1, None])
            ys.extend([y, y, None])
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        name=f"{side.lower()} order",
        line={"color": color, "width": 1.8},
        showlegend=True,
        hovertemplate=f"{side.lower()} order=%{{y}}<extra></extra>",
    )


def make_hour_figure(simdata: SimData, symbol: str, sdate: str, hour: int) -> go.Figure:
    dfo, dfbidask = simdata.get_orders_bid_ask(symbol, sdate, hour)
    t1 = pd.to_datetime(sdate, format="%Y%m%d").replace(hour=hour)
    t2 = t1 + timedelta(hours=1)
    fig = go.Figure()

    if not dfbidask.empty:
        fig.add_trace(
            go.Scatter(
                x=dfbidask.index,
                y=_safe_series(dfbidask, "bid"),
                mode="lines",
                name="bid",
                line={"color": "#2563EB", "width": 1.2},
                opacity=0.35,
                hovertemplate="bid=%{y}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dfbidask.index,
                y=_safe_series(dfbidask, "ask"),
                mode="lines",
                name="ask",
                line={"color": "#DC2626", "width": 1.2},
                opacity=0.35,
                hovertemplate="ask=%{y}<extra></extra>",
            )
        )

    if not dfo.empty:
        order_df = dfo.reset_index()
        if "side" in order_df.columns:
            order_df["side_bucket"] = order_df["side"].map(_side_bucket)
        else:
            order_df["side_bucket"] = None
        buy_df = order_df[order_df["side_bucket"] == "BUY"]
        sell_df = order_df[order_df["side_bucket"] == "SELL"]
        if not buy_df.empty:
            fig.add_trace(_make_order_segment_trace(buy_df, side="BUY", color="#0F766E"))
        if not sell_df.empty:
            fig.add_trace(_make_order_segment_trace(sell_df, side="SELL", color="#C2410C"))

        if "filled_qty" in order_df.columns:
            filled_qty = pd.to_numeric(order_df["filled_qty"], errors="coerce").fillna(0)
            fills = order_df[filled_qty > 0].copy()
        else:
            fills = pd.DataFrame(columns=order_df.columns)
        if not fills.empty:
            if "last_update_time" in fills.columns:
                fills["fill_time"] = pd.to_datetime(pd.to_numeric(fills["last_update_time"], errors="coerce"), unit="us")
            elif "create_time" in fills.columns:
                fills["fill_time"] = pd.to_datetime(pd.to_numeric(fills["create_time"], errors="coerce"), unit="us")
            else:
                fills["fill_time"] = pd.NaT
            buy_fills = fills[fills["side_bucket"] == "BUY"]
            sell_fills = fills[fills["side_bucket"] == "SELL"]
            if not buy_fills.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_fills["fill_time"],
                        y=pd.to_numeric(buy_fills["price"], errors="coerce"),
                        mode="markers",
                        name="buy fill",
                        marker={"symbol": "triangle-up", "color": "#1D4ED8", "size": 9},
                        hovertemplate="buy fill=%{y}<extra></extra>",
                    )
                )
            if not sell_fills.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_fills["fill_time"],
                        y=pd.to_numeric(sell_fills["price"], errors="coerce"),
                        mode="markers",
                        name="sell fill",
                        marker={"symbol": "triangle-down", "color": "#DC2626", "size": 9},
                        hovertemplate="sell fill=%{y}<extra></extra>",
                    )
                )

    fig.update_xaxes(range=[t1, t2], title="Time")
    fig.update_yaxes(title="Price", automargin=True)
    fig.update_layout(
        template="plotly_white",
        height=420,
        title=f"{symbol} | {sdate} | {hour:02d}:00",
        hovermode="x unified",
        margin={"l": 18, "r": 18, "t": 70, "b": 40},
    )
    return fig


def _page_shell(title: str, body) -> html.Div:
    return html.Div(
        [
            html.H2(title, style={"margin": "0 0 10px 0"}),
            body,
        ],
        style={"maxWidth": "1500px", "margin": "0 auto", "padding": "16px"},
    )


def _daily_symbol_summary(simdata: SimData, symbol: str) -> pd.DataFrame:
    rows = []
    for date in simdata.sdates:
        timeline = simdata.get_timeline(symbol, date)
        if timeline.empty:
            continue
        final = timeline.iloc[-1]
        rows.append(
            {
                "date": date,
                "pnl": float(pd.to_numeric(final.get("pnl"), errors="coerce") or 0.0),
                "notional_traded": float(pd.to_numeric(final.get("notional_traded"), errors="coerce") or 0.0),
                "notional_pos": float(pd.to_numeric(final.get("notional_pos"), errors="coerce") or 0.0),
                "fees": float(pd.to_numeric(final.get("fees"), errors="coerce") or 0.0),
                "size_traded": float(pd.to_numeric(final.get("size_traded"), errors="coerce") or 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


def _daily_portfolio_summary(simdata: SimData) -> pd.DataFrame:
    rows = []
    for date in simdata.sdates:
        timelines = simdata.get_timelines(date)
        if not timelines:
            continue
        totals: dict[str, float] = {"notional_traded": 0.0, "notional_pos": 0.0, "pnl": 0.0, "fees": 0.0, "size_traded": 0.0}
        for timeline in timelines.values():
            if timeline.empty:
                continue
            final = timeline.iloc[-1]
            for col in totals:
                value = pd.to_numeric(final.get(col), errors="coerce")
                if pd.notna(value):
                    totals[col] += float(value)
        rows.append({"date": date, **totals})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


def _stats_bucket_minutes(simdata: SimData) -> int:
    return max(5, 5 * max(1, len(simdata.sdates)))


def _combined_stats_timeline(simdata: SimData, symbol: str | None, freq_minutes: int) -> pd.DataFrame:
    frames = []
    pnl_offset = 0.0
    if symbol:
        for date in simdata.sdates:
            timeline = simdata.get_timeline(symbol, date, freq="5min")
            if not timeline.empty:
                timeline = timeline.copy()
                if "pnl" in timeline.columns:
                    pnl_series = pd.to_numeric(timeline["pnl"], errors="coerce").fillna(0.0)
                    timeline["pnl"] = pnl_offset + pnl_series
                    pnl_offset = float(timeline["pnl"].iloc[-1])
                frames.append(timeline)
    else:
        for date in simdata.sdates:
            timelines = simdata.get_timelines(date, freq="5min")
            if not timelines:
                continue
            totals = []
            for timeline in timelines.values():
                if not timeline.empty:
                    totals.append(timeline)
            if totals:
                combined_day = pd.concat(totals).sort_index()
                if combined_day.index.has_duplicates:
                    combined_day = combined_day.groupby(level=0).sum(numeric_only=True)
                if "pnl" in combined_day.columns:
                    combined_day = combined_day.copy()
                    pnl_series = pd.to_numeric(combined_day["pnl"], errors="coerce").fillna(0.0)
                    combined_day["pnl"] = pnl_offset + pnl_series
                    pnl_offset = float(combined_day["pnl"].iloc[-1])
                frames.append(combined_day)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames).sort_index()
    if combined.index.has_duplicates:
        combined = combined.groupby(level=0).last()
    freq = f"{freq_minutes}min"
    # Use the first sample in each bucket so the timestamp labels reflect
    # bucket-start values on the step chart, especially across day resets.
    combined = combined.resample(freq).first().ffill()
    return combined


def _make_stats_figure(timeline_df: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("PnL", "Notional Traded", "Notional Position"),
    )
    if not timeline_df.empty:
        x = timeline_df.index
        fig.add_trace(
            go.Scatter(
                x=x,
                y=pd.to_numeric(timeline_df["pnl"], errors="coerce"),
                mode="lines",
                name="pnl",
                line={"color": "#2563EB", "width": 1.5},
                line_shape="hv",
                hovertemplate="pnl=%{y}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=pd.to_numeric(timeline_df["notional_traded"], errors="coerce"),
                mode="lines",
                name="notional traded",
                line={"color": "#C2410C", "width": 1.5},
                line_shape="hv",
                hovertemplate="notional_traded=%{y}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=pd.to_numeric(timeline_df["notional_pos"], errors="coerce"),
                mode="lines",
                name="notional position",
                line={"color": "#0F766E", "width": 1.5},
                line_shape="hv",
                hovertemplate="notional_pos=%{y}<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )
    fig.update_layout(
        template="plotly_white",
        height=780,
        title=title,
        hovermode="x unified",
        margin={"l": 18, "r": 18, "t": 70, "b": 40},
    )
    fig.update_xaxes(title="Date", row=3, col=1)
    for row_idx in (1, 2, 3):
        fig.update_yaxes(automargin=True, row=row_idx, col=1)
    return fig


def render_stats(search: str) -> html.Div:
    query = parse_qs(search.lstrip("?"))
    head, symbol, _ = _resolve_head_symbol_date(query, require_symbol=False)
    if head is None:
        return _page_shell("Invalid head", html.Pre(unquote(query.get("head", [""])[0])))
    simdata = SimData(str(head))
    if symbol:
        summary_df = _daily_symbol_summary(simdata, symbol)
        bucket_minutes = _stats_bucket_minutes(simdata)
        chart_df = _combined_stats_timeline(simdata, symbol, bucket_minutes)
        title = f"Quant Stats | {symbol} | {head}"
    else:
        summary_df = _daily_portfolio_summary(simdata)
        bucket_minutes = _stats_bucket_minutes(simdata)
        chart_df = _combined_stats_timeline(simdata, None, bucket_minutes)
        title = f"Portfolio Quant Stats | {head}"
    if summary_df.empty:
        return _page_shell("No data", html.Pre(symbol or str(head)))
    summary_df = summary_df.copy()
    summary_df["date_ts"] = pd.to_datetime(summary_df["date"], format="%Y%m%d", errors="coerce")
    summary_df["cum_pnl"] = pd.to_numeric(summary_df["pnl"], errors="coerce").fillna(0.0).cumsum()
    summary_df["drawdown"] = summary_df["cum_pnl"] - summary_df["cum_pnl"].cummax()
    summary_df["abs_pos"] = pd.to_numeric(summary_df["notional_pos"], errors="coerce").abs()
    fig = _make_stats_figure(chart_df, title)
    total_pnl = float(pd.to_numeric(summary_df["pnl"], errors="coerce").fillna(0.0).sum())
    total_notional = float(pd.to_numeric(summary_df["notional_traded"], errors="coerce").fillna(0.0).sum())
    win_rate = float((pd.to_numeric(summary_df["pnl"], errors="coerce").fillna(0.0) > 0).mean())
    summary_line = html.Div(
        [
            html.A("Home", href="/", style={"marginRight": "12px"}),
            html.Span(f"Days: {len(summary_df)}", style={"marginRight": "12px"}),
            html.Span(f"Total PnL: {total_pnl:,.2f}", style={"marginRight": "12px"}),
            html.Span(f"Total Notional: {total_notional:,.2f}", style={"marginRight": "12px"}),
            html.Span(f"Win Rate: {win_rate:.1%}", style={"marginRight": "12px"}),
            html.Span(f"Chart Interval: {bucket_minutes} min", style={"marginRight": "12px"}),
        ],
        style={"marginBottom": "10px"},
    )
    table_rows = []
    for row in summary_df.itertuples(index=False):
        table_rows.append(
            html.Tr(
                [
                    html.Td(row.date),
                    html.Td(f"{float(row.pnl):,.2f}"),
                    html.Td(f"{float(row.notional_traded):,.2f}"),
                    html.Td(f"{float(row.notional_pos):,.2f}"),
                    html.Td(f"{float(row.fees):,.2f}"),
                ]
            )
        )
    table = html.Table(
        [
            html.Thead(html.Tr([html.Th("Date"), html.Th("PnL"), html.Th("Notional Traded"), html.Th("Notional Pos"), html.Th("Fees")])),
            html.Tbody(table_rows),
        ],
        style={"width": "100%", "borderCollapse": "collapse", "marginTop": "12px"},
    )
    return _page_shell(title, html.Div([summary_line, dcc.Graph(figure=fig), table]))


def render_index() -> html.Div:
    grouped = discover_heads()
    blocks = []
    for root_name, heads in grouped.items():
        if not heads:
            continue
        head_cards = []
        for head in heads:
            files = state_files_for_head(head)
            by_symbol: dict[str, list[str]] = defaultdict(list)
            for row in files:
                by_symbol[row["symbol"]].append(row["date"])
            all_dates = sorted({row["date"] for row in files})
            symbol_items = []
            if len(by_symbol) > 1 and all_dates:
                symbol_items.append(
                    html.Div(
                        [
                            html.A(
                                "portfolio",
                                href=f"/stats?head={quote(str(head))}",
                                style={"fontWeight": "600", "marginRight": "10px"},
                            ),
                            html.Span("dates: "),
                            *[
                                html.A(
                                    d,
                                    href=f"/portfolio?head={quote(str(head))}&date={d}",
                                    style={"marginLeft": "8px"},
                                )
                                for d in all_dates
                            ],
                        ],
                        style={"marginBottom": "8px"},
                    )
                )
            for symbol, dates in sorted(by_symbol.items()):
                dates = sorted(set(dates))
                latest_date = dates[-1]
                links = [
                    html.A(
                        symbol,
                        href=f"/stats?head={quote(str(head))}&symbol={quote(symbol)}",
                        style={"fontWeight": "600", "marginRight": "10px"},
                    )
                ]
                symbol_items.append(
                    html.Div(
                        links
                        + [
                            html.Span("dates: "),
                            *[
                                html.A(
                                    d,
                                    href=f"/symbol?head={quote(str(head))}&symbol={quote(symbol)}&date={d}",
                                    style={"marginLeft": "8px"},
                                )
                                for d in dates
                            ],
                        ],
                        style={"marginBottom": "6px"},
                    )
                )
            head_cards.append(
                html.Div(
                    [
                        html.H4(str(head), style={"margin": "0 0 8px 0"}),
                        html.Div(symbol_items),
                    ],
                    style={"border": "1px solid #e5e7eb", "borderRadius": "10px", "padding": "12px", "marginBottom": "12px"},
                )
            )
        blocks.append(html.Div([html.H3(root_name), *head_cards]))
    return _page_shell("Sim Dashboard v2", html.Div(blocks))


def _resolve_head_symbol_date(query: dict[str, list[str]], require_symbol: bool = True):
    head_raw = unquote(query.get("head", [""])[0])
    symbol = unquote(query.get("symbol", [""])[0])
    date = query.get("date", [""])[0]
    head = normalize_head(head_raw)
    if head is None:
        return None, symbol, date
    if require_symbol and not symbol:
        return head, symbol, date
    return head, symbol, date


def render_portfolio(search: str) -> html.Div:
    query = parse_qs(search.lstrip("?"))
    head, _, date = _resolve_head_symbol_date(query, require_symbol=False)
    if head is None:
        return _page_shell("Invalid head", html.Pre(unquote(query.get("head", [""])[0])))
    simdata = SimData(str(head))
    if not date:
        if not simdata.sdates:
            return _page_shell("No dates", html.Pre(str(head)))
        date = simdata.sdates[-1]
    fig = make_portfolio_figure(simdata, date)
    links = [html.A("Home", href="/", style={"marginRight": "12px"})]
    links.extend(
        html.A(d, href=f"/portfolio?head={quote(str(head))}&date={d}", style={"marginRight": "10px"})
        for d in simdata.sdates
    )
    return _page_shell(
        f"Portfolio | {head}",
        html.Div(
            [
                html.Div([html.Span("Dates: "), *links], style={"marginBottom": "10px"}),
                dcc.Graph(figure=fig),
            ]
        ),
    )


def render_symbol(search: str) -> html.Div:
    query = parse_qs(search.lstrip("?"))
    head, symbol, date = _resolve_head_symbol_date(query)
    if head is None:
        return _page_shell("Invalid head", html.Pre(unquote(query.get("head", [""])[0])))
    if not symbol:
        return _page_shell("Missing symbol", html.Pre(search))
    state_entries = [row for row in state_files_for_head(head) if row["symbol"] == symbol]
    if not state_entries:
        return _page_shell("No dates found", html.Pre(symbol))
    available_dates = sorted({row["date"] for row in state_entries})
    if not date:
        date = available_dates[-1]
    simdata = SimData(str(head))
    symbol_fig = make_symbol_figure(simdata, symbol, date)
    hour = query.get("hour", [""])[0]
    if hour and hour.isdigit():
        hour_i = max(0, min(23, int(hour)))
    else:
        hour_i = 0
    hour_fig = make_hour_figure(simdata, symbol, date, hour_i)
    date_links = [
        html.A(
            d,
            href=f"/symbol?head={quote(str(head))}&symbol={quote(symbol)}&date={d}",
            style={"marginRight": "10px"},
        )
        for d in available_dates
    ]
    hour_links = [
        html.A(
            f"{h:02d}",
            href=f"/symbol?head={quote(str(head))}&symbol={quote(symbol)}&date={date}&hour={h}",
            style={"marginRight": "8px"},
        )
        for h in range(24)
    ]
    body = html.Div(
        [
            html.Div(
                [
                    html.A("Home", href="/", style={"marginRight": "12px"}),
                    html.A("Portfolio", href=f"/stats?head={quote(str(head))}&symbol={quote(symbol)}", style={"marginRight": "12px"}),
                    html.Span(f"Symbol: {symbol}", style={"fontWeight": "600", "marginRight": "12px"}),
                    html.Span("Dates: "),
                    *date_links,
                ],
                style={"marginBottom": "10px"},
            ),
            dcc.Graph(figure=symbol_fig, style={"height": "86vh"}),
            html.Div(
                [
                    html.H4("Hourly View", style={"margin": "10px 0 8px 0"}),
                    html.Div([html.Span("Hours: "), *hour_links], style={"marginBottom": "8px"}),
                    dcc.Graph(figure=hour_fig, style={"height": "40vh"}),
                ],
                style={"marginTop": "12px"},
            ),
        ]
    )
    return _page_shell(f"{symbol} | {date} | {head}", body)


def render_chart(search: str) -> html.Div:
    query = parse_qs(search.lstrip("?"))
    head, symbol, date = _resolve_head_symbol_date(query)
    if head is None:
        return _page_shell("Invalid head", html.Pre(unquote(query.get("head", [""])[0])))
    if not symbol or not date:
        return _page_shell("Missing symbol/date", html.Pre(search))
    hour = query.get("hour", [""])[0]
    hour_i = int(hour) if hour.isdigit() else 0
    hour_i = max(0, min(23, hour_i))
    simdata = SimData(str(head))
    fig = make_hour_figure(simdata, symbol, date, hour_i)
    return _page_shell(
        f"{symbol} | {date} | {hour_i:02d}:00 | {head}",
        html.Div(
            [
                html.Div(
                    [
                        html.A("Home", href="/", style={"marginRight": "12px"}),
                        html.A("Symbol", href=f"/symbol?head={quote(str(head))}&symbol={quote(symbol)}&date={date}", style={"marginRight": "12px"}),
                        html.A("Portfolio", href=f"/stats?head={quote(str(head))}&symbol={quote(symbol)}", style={"marginRight": "12px"}),
                    ],
                    style={"marginBottom": "10px"},
                ),
                dcc.Graph(figure=fig, style={"height": "72vh"}),
            ]
        ),
    )


app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Sim Dashboard v2"
app.layout = html.Div([dcc.Location(id="url"), html.Div(id="page-content")])


@app.callback(Output("page-content", "children"), Input("url", "pathname"), Input("url", "search"))
def route(pathname: str | None, search: str | None):
    search = search or ""
    if pathname in (None, "/", ""):
        return render_index()
    if pathname == "/stats":
        return render_stats(search)
    if pathname == "/portfolio":
        return render_portfolio(search)
    if pathname == "/symbol":
        return render_symbol(search)
    if pathname == "/chart":
        return render_chart(search)
    return render_index()


if __name__ == "__main__":
    host = "127.0.0.1"
    port = int(os.getenv("DASH2_PORT", "8051"))
    debug = os.getenv("DASH2_DEBUG", "1").lower() in ("1", "true", "yes", "on")
    app.run(debug=debug, host=host, port=port)
