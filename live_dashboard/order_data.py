from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re
from typing import Optional

import pandas as pd

ORDER_FILE_PATTERN = re.compile(r"^orders?\.(\d{8})\.log$")
DEFAULT_INTERVAL_MINUTES = 10


def _normalize_id(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _list_order_files(log_dir: Path) -> list[Path]:
    if not log_dir.exists():
        return []
    return [path for path in log_dir.iterdir() if path.is_file() and ORDER_FILE_PATTERN.match(path.name)]


def _latest_log_file(log_dir: Path) -> Optional[Path]:
    files = _list_order_files(log_dir)
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _extract_timestamp(line: str) -> Optional[datetime]:
    parts = line.split(" ", 2)
    if len(parts) < 2:
        return None
    ts_str = f"{parts[0]} {parts[1]}"
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None


def _parse_payload(line: str) -> dict:
    start = line.find("{")
    end = line.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(line[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _iter_lines_reverse(path: Path, chunk_size: int = 1024 * 1024):
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        buffer = b""
        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            data = handle.read(read_size)
            buffer = data + buffer
            lines = buffer.split(b"\n")
            buffer = lines[0]
            for raw in reversed(lines[1:]):
                yield raw.decode("utf-8", errors="ignore")
        if buffer:
            yield buffer.decode("utf-8", errors="ignore")


def _latest_day_lines(path: Path) -> tuple[Optional[str], list[str]]:
    latest_day: Optional[str] = None
    collected: list[str] = []

    for line in _iter_lines_reverse(path):
        ts = _extract_timestamp(line)
        if ts is None:
            continue
        day_iso = pd.Timestamp(ts.date()).isoformat()
        if latest_day is None:
            latest_day = day_iso
        if day_iso == latest_day:
            collected.append(line)
        elif day_iso < latest_day:
            break

    if latest_day is None:
        return None, []

    collected.reverse()
    return latest_day, collected


def _day_index(day_iso: str, interval_minutes: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(day_iso)
    periods = 24 * 60 // interval_minutes
    return pd.date_range(start=start, periods=periods, freq=f"{interval_minutes}min")


def load_latest_day_metrics(
    log_dir: Path, *, interval_minutes: int = DEFAULT_INTERVAL_MINUTES
) -> dict:
    latest = _latest_log_file(log_dir)
    if latest is None:
        raise FileNotFoundError(f"No order logs found in {log_dir}")

    day_iso, lines = _latest_day_lines(latest)
    if day_iso is None:
        raise FileNotFoundError(f"No parseable timestamps in {latest}")

    new_by_key: dict[str, datetime] = {}
    new_by_key_symbol: dict[str, datetime] = {}
    fill_counts: dict[pd.Timestamp, int] = {}
    notional_by_bucket: dict[pd.Timestamp, float] = {}
    notional_by_key_bucket: dict[str, dict[pd.Timestamp, float]] = {}
    fill_events: list[dict[str, object]] = []
    client_order_windows: dict[str, dict[str, object | None]] = {}

    for line in lines:
        ts = _extract_timestamp(line)
        if ts is None:
            continue
        payload = _parse_payload(line)
        if not payload:
            continue

        bucket = pd.Timestamp(ts).floor(f"{interval_minutes}min")

        action = payload.get("order_action")
        client_id = _normalize_id(payload.get("client_order_id"))
        strategy_id = _normalize_id(payload.get("strategy_id"))
        symbol = str(payload.get("symbol") or "UNKNOWN")
        if client_id is not None and strategy_id is not None:
            window_key = f"{strategy_id}:{client_id}:{symbol}"
            window = client_order_windows.setdefault(window_key, {"start": None, "end": None, "order_price": None})
            if action == "NEW":
                start = window["start"]
                order_price_raw = payload.get("price")
                order_price: float | None
                try:
                    order_price = float(order_price_raw) if order_price_raw is not None else None
                except (TypeError, ValueError):
                    order_price = None
                if start is None or ts < start:
                    window["start"] = ts
                    window["order_price"] = order_price
                elif ts == start and window["order_price"] is None and order_price is not None:
                    window["order_price"] = order_price
            end = window["end"]
            if end is None or ts > end:
                window["end"] = ts
        if action == "NEW" and client_id is not None and strategy_id is not None:
            key = f"{strategy_id}:{client_id}"
            current = new_by_key.get(key)
            if current is None or ts < current:
                new_by_key[key] = ts
            key_symbol = f"{strategy_id}:{client_id}:{symbol}"
            current_symbol = new_by_key_symbol.get(key_symbol)
            if current_symbol is None or ts < current_symbol:
                new_by_key_symbol[key_symbol] = ts

        status = payload.get("order_status")
        if status in {"PARTIAL_FILL", "FILLED"}:
            executed_price = payload.get("executed_price")
            filled_qty = payload.get("filled_qty")
            if executed_price is None or filled_qty is None or strategy_id is None or client_id is None:
                continue

            symbol = str(payload.get("symbol") or "UNKNOWN")
            client_symbol_key = f"{strategy_id}:{client_id}:{symbol}"
            first_new_symbol_ts = new_by_key_symbol.get(client_symbol_key)
            # Symbol-level NEW validation is sufficient; strategy/client NEW is implied.
            if first_new_symbol_ts is None or ts < first_new_symbol_ts:
                continue

            filled_qty_val = float(filled_qty)
            executed_price_val = float(executed_price)
            if filled_qty_val <= 0 or executed_price_val <= 0:
                continue

            fill_counts[bucket] = fill_counts.get(bucket, 0) + 1
            notional = executed_price_val * filled_qty_val
            notional_by_bucket[bucket] = notional_by_bucket.get(bucket, 0.0) + notional
            key = f"{strategy_id}:{symbol}"
            by_bucket = notional_by_key_bucket.setdefault(key, {})
            by_bucket[bucket] = by_bucket.get(bucket, 0.0) + notional
            fill_events.append(
                {
                    "bucket_iso": bucket.isoformat(),
                    "event_time_iso": pd.Timestamp(ts).isoformat(),
                    "symbol": symbol,
                    "strategy_id": strategy_id,
                    # Keep ids as strings to avoid JS precision loss and type drift.
                    "client_order_id": client_id,
                    "client_order_key": client_symbol_key,
                    "side": str(payload.get("side") or payload.get("order_side") or "").upper(),
                    "executed_price": executed_price_val,
                    "filled_qty": filled_qty_val,
                    "key": key,
                }
            )

    new_counts: dict[pd.Timestamp, int] = {}
    for first_ts in new_by_key.values():
        bucket = pd.Timestamp(first_ts).floor(f"{interval_minutes}min")
        new_counts[bucket] = new_counts.get(bucket, 0) + 1

    index = _day_index(day_iso, interval_minutes)
    new_series = pd.Series(new_counts, dtype=float).reindex(index, fill_value=0)
    fill_series = pd.Series(fill_counts, dtype=float).reindex(index, fill_value=0)
    notional_series = pd.Series(notional_by_bucket, dtype=float).reindex(index, fill_value=0)
    cumulative_notional = notional_series.cumsum()
    cumulative_notional_by_key: dict[str, pd.Series] = {}
    bucket_notional_by_key: dict[str, pd.Series] = {}
    for key, buckets in notional_by_key_bucket.items():
        bucket_series = pd.Series(buckets, dtype=float).reindex(index, fill_value=0)
        bucket_notional_by_key[key] = bucket_series
        cumulative_notional_by_key[key] = bucket_series.cumsum()
    client_order_windows_out = {}
    for key, window in client_order_windows.items():
        start = window["start"]
        end = window["end"]
        if start is None or end is None:
            continue
        client_order_windows_out[key] = {
            "start_time_iso": pd.Timestamp(start).isoformat(),
            "end_time_iso": pd.Timestamp(end).isoformat(),
        }
        order_price = window.get("order_price")
        if isinstance(order_price, float):
            client_order_windows_out[key]["order_price"] = order_price

    return {
        "date_iso": day_iso,
        "source_file": latest.name,
        "new": new_series,
        "fills": fill_series,
        "notional": notional_series,
        "cumulative_notional": cumulative_notional,
        "bucket_notional_by_key": bucket_notional_by_key,
        "cumulative_notional_by_key": cumulative_notional_by_key,
        "fill_events": fill_events,
        "client_order_windows": client_order_windows_out,
    }
