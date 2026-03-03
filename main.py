from __future__ import annotations

import os


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import io
import json
import math
from pathlib import Path
import tempfile
import time
from threading import Lock, Thread
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout

from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import yfinance as yf

try:
    from requests_cache import CachedSession
    from requests_ratelimiter import LimiterAdapter
except ImportError:
    CachedSession = None
    LimiterAdapter = None


warnings.filterwarnings("ignore")


APP_TITLE = "ALPHA-QUANT"
BENCHMARK_TICKER = "^NSEI"
UNIVERSE_CACHE_TTL_SECONDS = 900
UNIVERSE_BATCH_SIZE = 8
UNIVERSE_FETCH_DEADLINE_SECONDS = 8
UNIVERSE_RATE_LIMIT_BACKOFF_SECONDS = 300
MARKET_DATA_CACHE_TTL_SECONDS = 1800
BACKTEST_FETCH_DEADLINE_SECONDS = 18
CHART_REQUEST_MIN_INTERVAL_SECONDS = 1.1
MAX_UNIVERSE_SELECTION = 20
DEFAULT_REBALANCE_FREQUENCY = 21
DEFAULT_MAX_HOLDINGS = 5
DEFAULT_TRANSACTION_COST_BPS = 10
MODEL_TRAINING_WINDOW = 252
SEED_COVERAGE_TOLERANCE_DAYS = 4


@dataclass(frozen=True)
class AppModeConfig:
    key: str
    label: str
    description: str
    prefer_seed_market_data: bool
    allow_live_training: bool
    universe_timeout_ms: int
    backtest_timeout_ms: int


APP_MODES = {
    "render_snapshot": AppModeConfig(
        key="render_snapshot",
        label="Render Snapshot Mode",
        description="Deployment-safe mode with cached market snapshots and bounded execution for small-host reliability.",
        prefer_seed_market_data=True,
        allow_live_training=False,
        universe_timeout_ms=30000,
        backtest_timeout_ms=80000,
    ),
    "full_research": AppModeConfig(
        key="full_research",
        label="Full Research Mode",
        description="Full historical research mode with live market retrieval and walk-forward model training.",
        prefer_seed_market_data=False,
        allow_live_training=True,
        universe_timeout_ms=12000,
        backtest_timeout_ms=45000,
    ),
}


def resolve_app_mode() -> AppModeConfig:
    raw_value = os.getenv("APP_MODE", "").strip().lower()
    if raw_value in APP_MODES:
        return APP_MODES[raw_value]
    if os.getenv("RENDER", "").strip():
        return APP_MODES["render_snapshot"]
    return APP_MODES["full_research"]


APP_MODE = resolve_app_mode()
PREFER_SEED_MARKET_DATA = APP_MODE.prefer_seed_market_data
FAST_DEPLOYMENT_MODE = not APP_MODE.allow_live_training

CORE_LIVE_STOCKS = (
    "RELIANCE.NS",
    "INFY.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "LT.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
)
SEED_DATA_PATH = Path(__file__).resolve().parent / "data" / "core_market_seed.json"


SORT_OPTIONS = {
    "ticker": {"label": "Ticker", "reverse": False},
    "last_price": {"label": "Last Price", "reverse": True},
    "momentum_1m": {"label": "1M Momentum", "reverse": True},
    "momentum_3m": {"label": "3M Momentum", "reverse": True},
    "momentum_6m": {"label": "6M Momentum", "reverse": True},
    "volatility_1m": {"label": "1M Volatility", "reverse": False},
    "drawdown_6m": {"label": "6M Drawdown", "reverse": True},
    "avg_turnover": {"label": "Average Turnover", "reverse": True},
    "rsi_14": {"label": "RSI 14", "reverse": True},
}


FACTOR_COLUMNS = (
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "trend_gap_100d",
    "drawdown_6m",
    "volatility_1m",
    "rsi_14",
)


FACTOR_DIRECTIONS = {
    "momentum_1m": 1.0,
    "momentum_3m": 1.0,
    "momentum_6m": 0.8,
    "trend_gap_100d": 0.8,
    "drawdown_6m": 0.6,
    "volatility_1m": -0.8,
    "rsi_14": 0.3,
}


NIFTY_50_STOCKS = (
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BPCL.NS",
    "BHARTIARTL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DIVISLAB.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "ITC.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LTIM.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NTPC.NS",
    "NESTLEIND.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SHREECEM.NS",
    "SBIN.NS",
    "SUNPHARMA.NS",
    "TATAMOTORS.NS",
    "TCS.NS",
    "TATACONSUM.NS",
    "TATASTEEL.NS",
    "TECHM.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS",
)


UNIVERSE_CACHE: dict[str, object] = {
    "updated_at": 0.0,
    "rows": [],
    "as_of": None,
    "tape": {},
    "warning": None,
}
UNIVERSE_CACHE_LOCK = Lock()
UNIVERSE_FETCH_STATE: dict[str, object] = {
    "backoff_until": 0.0,
    "in_progress": False,
}
MARKET_DATA_CACHE_LOCK = Lock()
PRICE_BUNDLE_CACHE: dict[tuple[tuple[str, ...], str, str], dict[str, object]] = {}
BENCHMARK_CACHE: dict[tuple[str, str], dict[str, object]] = {}
CHART_REQUEST_LOCK = Lock()
CHART_REQUEST_STATE: dict[str, float] = {"last_started_at": 0.0}


class MarketDataRateLimitedError(RuntimeError):
    pass


class MarketDataTimeoutError(RuntimeError):
    pass


class MarketDataUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class BacktestConfig:
    start_date: date
    end_date: date
    selected_stocks: tuple[str, ...]
    rebalance_frequency: int
    max_holdings: int
    transaction_cost_bps: int


def _safe_float(value):
    if value is None:
        return None
    try:
        if pd.isna(value) or not np.isfinite(float(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def get_yfinance_session():
    if CachedSession is not None and LimiterAdapter is not None:
        cache_path = os.path.join(tempfile.gettempdir(), "alpha_quant_yf_http.cache")
        session = CachedSession(
            cache_name=cache_path,
            backend="sqlite",
            expire_after=timedelta(hours=6),
            stale_if_error=True,
        )
        adapter = LimiterAdapter(per_second=0.35, burst=1)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
    else:
        session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
    )
    return session


def get_chart_session():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return session


def _price_bundle_cache_key(tickers: tuple[str, ...], start_date: date, end_date: date):
    return tuple(sorted(tickers)), start_date.isoformat(), end_date.isoformat()


def _get_cached_price_bundle(tickers: tuple[str, ...], start_date: date, end_date: date):
    cache_key = _price_bundle_cache_key(tickers, start_date, end_date)
    now = time.monotonic()
    with MARKET_DATA_CACHE_LOCK:
        entry = PRICE_BUNDLE_CACHE.get(cache_key)
        if not entry:
            return None
        if now - float(entry["updated_at"]) >= MARKET_DATA_CACHE_TTL_SECONDS:
            PRICE_BUNDLE_CACHE.pop(cache_key, None)
            return None
        return entry["close"].copy(), entry["volume"].copy()


def _store_cached_price_bundle(tickers: tuple[str, ...], start_date: date, end_date: date, close: pd.DataFrame, volume: pd.DataFrame):
    cache_key = _price_bundle_cache_key(tickers, start_date, end_date)
    with MARKET_DATA_CACHE_LOCK:
        PRICE_BUNDLE_CACHE[cache_key] = {
            "updated_at": time.monotonic(),
            "close": close.copy(),
            "volume": volume.copy(),
        }


def _get_cached_benchmark(start_date: date, end_date: date):
    cache_key = (start_date.isoformat(), end_date.isoformat())
    now = time.monotonic()
    with MARKET_DATA_CACHE_LOCK:
        entry = BENCHMARK_CACHE.get(cache_key)
        if not entry:
            return None
        if now - float(entry["updated_at"]) >= MARKET_DATA_CACHE_TTL_SECONDS:
            BENCHMARK_CACHE.pop(cache_key, None)
            return None
        return entry["series"].copy()


def _store_cached_benchmark(start_date: date, end_date: date, series: pd.Series):
    cache_key = (start_date.isoformat(), end_date.isoformat())
    with MARKET_DATA_CACHE_LOCK:
        BENCHMARK_CACHE[cache_key] = {
            "updated_at": time.monotonic(),
            "series": series.copy(),
        }


@lru_cache(maxsize=1)
def load_seed_market_data():
    if not SEED_DATA_PATH.exists():
        return None
    with SEED_DATA_PATH.open("r", encoding="utf-8") as seed_file:
        return json.load(seed_file)


def _seed_series_from_payload(payload: dict[str, object], start_date: date, end_date: date):
    dates = payload.get("dates") or []
    close_values = payload.get("close") or []
    volume_values = payload.get("volume") or []

    if not dates or not close_values:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    index = pd.to_datetime(dates)
    close = pd.Series(pd.to_numeric(close_values, errors="coerce"), index=index, dtype=float)
    volume = pd.Series(pd.to_numeric(volume_values, errors="coerce"), index=index, dtype=float).fillna(0.0)
    mask = (close.index.date >= start_date) & (close.index.date <= end_date)
    close = close.loc[mask].dropna()
    volume = volume.reindex(close.index, fill_value=0.0)
    return close, volume


def get_seed_price_bundle(tickers: tuple[str, ...], start_date: date, end_date: date):
    seed_data = load_seed_market_data()
    if not seed_data:
        return None

    tickers_payload = seed_data.get("tickers") or {}
    close_map = {}
    volume_map = {}
    for ticker in tickers:
        payload = tickers_payload.get(ticker)
        if not payload:
            continue
        close_series, volume_series = _seed_series_from_payload(payload, start_date, end_date)
        if close_series.empty:
            continue
        close_map[ticker] = close_series
        volume_map[ticker] = volume_series

    if not close_map:
        return None
    close = sanitize_price_frame(pd.DataFrame(close_map).sort_index())
    volume = pd.DataFrame(volume_map).reindex(index=close.index, columns=close.columns).fillna(0.0)
    if close.empty:
        return None
    return close, volume


def get_seed_benchmark(start_date: date, end_date: date):
    seed_data = load_seed_market_data()
    if not seed_data:
        return None

    payload = seed_data.get("benchmark")
    if not payload:
        return None
    close_series, _ = _seed_series_from_payload(payload, start_date, end_date)
    if close_series.empty:
        return None
    return close_series


def _seed_payload_covers_window(payload: dict[str, object] | None, start_date: date, end_date: date) -> bool:
    if not payload:
        return False
    dates = payload.get("dates") or []
    if not dates:
        return False
    try:
        first_available = date.fromisoformat(dates[0])
        last_available = date.fromisoformat(dates[-1])
    except (TypeError, ValueError):
        return False
    return (
        first_available <= start_date + timedelta(days=SEED_COVERAGE_TOLERANCE_DAYS)
        and last_available >= end_date - timedelta(days=SEED_COVERAGE_TOLERANCE_DAYS)
    )


def seed_price_bundle_covers_request(tickers: tuple[str, ...], start_date: date, end_date: date) -> bool:
    seed_data = load_seed_market_data()
    if not seed_data:
        return False
    tickers_payload = seed_data.get("tickers") or {}
    return all(_seed_payload_covers_window(tickers_payload.get(ticker), start_date, end_date) for ticker in tickers)


def seed_price_bundle_supports_end_date(tickers: tuple[str, ...], end_date: date) -> bool:
    seed_data = load_seed_market_data()
    if not seed_data:
        return False
    tickers_payload = seed_data.get("tickers") or {}
    return all(_seed_payload_covers_window(tickers_payload.get(ticker), end_date, end_date) for ticker in tickers)


def seed_benchmark_covers_request(start_date: date, end_date: date) -> bool:
    seed_data = load_seed_market_data()
    if not seed_data:
        return False
    return _seed_payload_covers_window(seed_data.get("benchmark"), start_date, end_date)


def seed_benchmark_supports_end_date(end_date: date) -> bool:
    seed_data = load_seed_market_data()
    if not seed_data:
        return False
    return _seed_payload_covers_window(seed_data.get("benchmark"), end_date, end_date)


def get_seed_supported_window():
    seed_data = load_seed_market_data()
    if not seed_data:
        return None

    tickers_payload = seed_data.get("tickers") or {}
    first_dates = []
    last_dates = []
    for ticker in CORE_LIVE_STOCKS:
        payload = tickers_payload.get(ticker)
        if not payload:
            return None
        dates = payload.get("dates") or []
        if not dates:
            return None
        try:
            first_dates.append(date.fromisoformat(dates[0]))
            last_dates.append(date.fromisoformat(dates[-1]))
        except (TypeError, ValueError):
            return None

    benchmark_payload = seed_data.get("benchmark") or {}
    benchmark_dates = benchmark_payload.get("dates") or []
    if not benchmark_dates:
        return None
    try:
        first_dates.append(date.fromisoformat(benchmark_dates[0]))
        last_dates.append(date.fromisoformat(benchmark_dates[-1]))
    except (TypeError, ValueError):
        return None

    return {
        "start": max(first_dates).isoformat(),
        "end": min(last_dates).isoformat(),
    }


def numeric_mean(values):
    numeric_values = []
    for value in values:
        safe_value = _safe_float(value)
        if safe_value is not None:
            numeric_values.append(safe_value)
    if not numeric_values:
        return None
    return _safe_float(sum(numeric_values) / len(numeric_values))


def build_core_monitor_rows(close: pd.DataFrame, volume: pd.DataFrame):
    if close.empty:
        return []

    returns = close.pct_change()
    turnover = (close * volume).rolling(21).mean()
    rows = []

    for ticker in close.columns:
        prices = close[ticker].dropna()
        if len(prices) < 126:
            continue

        turnover_series = turnover[ticker].dropna() if ticker in turnover else pd.Series(dtype=float)
        rows.append(
            {
                "ticker": ticker,
                "display_ticker": ticker.replace(".NS", ""),
                "last_price": _safe_float(prices.iloc[-1]),
                "momentum_1m": ratio_change(prices, 21),
                "momentum_3m": ratio_change(prices, 63),
                "momentum_6m": ratio_change(prices, 126),
                "volatility_1m": rolling_volatility(returns[ticker], 21),
                "drawdown_6m": drawdown_over_window(prices, 126),
                "avg_turnover": _safe_float(turnover_series.iloc[-1]) if not turnover_series.empty else None,
                "rsi_14": _safe_float(series_rsi(prices).dropna().iloc[-1]),
                "selectable": True,
            }
        )

    return rows


def merge_universe_rows(core_rows):
    core_rows_by_ticker = {row["ticker"]: row for row in core_rows}
    return [core_rows_by_ticker.get(ticker, default_universe_row(ticker)) for ticker in NIFTY_50_STOCKS]


def build_universe_tape(rows):
    top_momentum = max(
        (row for row in rows if row.get("momentum_3m") is not None),
        key=lambda row: row["momentum_3m"],
        default=None,
    )
    return {
        "top_momentum": top_momentum["display_ticker"] if top_momentum else "N/A",
        "avg_3m_momentum": numeric_mean(row.get("momentum_3m") for row in rows),
        "avg_1m_volatility": numeric_mean(row.get("volatility_1m") for row in rows),
    }


def build_seed_universe_rows():
    seed_data = load_seed_market_data()
    if not seed_data:
        return [], None

    tickers_payload = seed_data.get("tickers") or {}
    available_dates = []
    for ticker in CORE_LIVE_STOCKS:
        payload = tickers_payload.get(ticker) or {}
        dates = payload.get("dates") or []
        if dates:
            available_dates.append(date.fromisoformat(dates[-1]))

    if not available_dates:
        return [], None

    as_of = min(available_dates)
    start_date = as_of - timedelta(days=220)
    seeded_bundle = get_seed_price_bundle(CORE_LIVE_STOCKS, start_date, as_of)
    if seeded_bundle is None:
        return [], None

    close, volume = seeded_bundle
    return build_core_monitor_rows(close, volume), as_of.isoformat()


@lru_cache(maxsize=2048)
def calculate_rsi(prices_tuple, period=14):
    prices = pd.Series(prices_tuple, dtype=float)
    delta = prices.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period).mean()
    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(50.0)


def series_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    clean_prices = prices.astype(float)
    rsi_values = calculate_rsi(tuple(clean_prices.values), period)
    return pd.Series(rsi_values.values, index=prices.index, dtype=float)


def sanitize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    clean = frame.copy().sort_index()
    first_valid = {column: clean[column].first_valid_index() for column in clean.columns}
    clean = clean.ffill()

    for column, first_valid_index in first_valid.items():
        if first_valid_index is None:
            clean.drop(columns=[column], inplace=True)
            continue
        clean.loc[clean.index < first_valid_index, column] = np.nan

    clean = clean.loc[:, clean.notna().sum() > 1]
    return clean


def extract_yfinance_field(raw: pd.DataFrame, tickers: tuple[str, ...], field_name: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(index=pd.Index([], dtype="datetime64[ns]"))

    if not isinstance(raw.columns, pd.MultiIndex):
        if field_name not in raw.columns:
            return pd.DataFrame(index=raw.index)
        return pd.DataFrame({tickers[0]: pd.to_numeric(raw[field_name], errors="coerce")}, index=raw.index)

    extracted = {}
    top_level = raw.columns.get_level_values(0)
    for ticker in tickers:
        if ticker not in top_level:
            continue
        ticker_frame = raw[ticker]
        if field_name in ticker_frame.columns:
            extracted[ticker] = pd.to_numeric(ticker_frame[field_name], errors="coerce")
    return pd.DataFrame(extracted, index=raw.index)


def chunked_tickers(tickers: tuple[str, ...], chunk_size: int):
    for index in range(0, len(tickers), chunk_size):
        yield tickers[index : index + chunk_size]


def _chart_request_params(start_date: date | None = None, end_date: date | None = None, range_period: str | None = None):
    params = {
        "interval": "1d",
        "includePrePost": "false",
        "events": "div,splits",
    }
    if range_period is not None:
        params["range"] = range_period
        return params

    if start_date is None or end_date is None:
        raise ValueError("Either range_period or both start_date and end_date must be provided.")

    period1 = int(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    period2 = int(datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp())
    params["period1"] = period1
    params["period2"] = period2
    return params


def fetch_chart_result(
    ticker: str,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    range_period: str | None = None,
    timeout_seconds: float = 10.0,
):
    session = get_chart_session()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = _chart_request_params(start_date=start_date, end_date=end_date, range_period=range_period)

    with CHART_REQUEST_LOCK:
        wait_seconds = CHART_REQUEST_MIN_INTERVAL_SECONDS - (time.monotonic() - CHART_REQUEST_STATE["last_started_at"])
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        CHART_REQUEST_STATE["last_started_at"] = time.monotonic()

    try:
        response = session.get(url, params=params, timeout=max(3, int(math.ceil(timeout_seconds))))
    except requests.Timeout as exc:
        raise MarketDataTimeoutError(f"Timed out fetching chart data for {ticker}.") from exc
    except requests.RequestException as exc:
        raise MarketDataUnavailableError(f"Unable to fetch chart data for {ticker}.") from exc

    if response.status_code == 429:
        raise MarketDataRateLimitedError(f"Yahoo chart endpoint rate-limited {ticker}.")
    if response.status_code >= 400:
        raise MarketDataUnavailableError(f"Yahoo chart endpoint returned {response.status_code} for {ticker}.")

    try:
        payload = response.json()
    except ValueError as exc:
        raise MarketDataUnavailableError(f"Yahoo chart response for {ticker} was not valid JSON.") from exc

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        description = str(error.get("description") or "")
        if "Too Many Requests" in description:
            raise MarketDataRateLimitedError(f"Yahoo chart endpoint rate-limited {ticker}.")
        raise MarketDataUnavailableError(description or f"Yahoo chart endpoint returned an error for {ticker}.")

    results = chart.get("result") or []
    if not results:
        raise MarketDataUnavailableError(f"Yahoo chart endpoint returned no data for {ticker}.")
    return results[0]


def parse_chart_series(result):
    timestamps = result.get("timestamp") or []
    if not timestamps:
        return pd.Series(dtype=float), pd.Series(dtype=float), result.get("meta", {})

    index = pd.to_datetime(timestamps, unit="s").normalize()
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjclose_block = (result.get("indicators", {}).get("adjclose") or [{}])[0]
    close_values = adjclose_block.get("adjclose") or quote.get("close") or []
    volume_values = quote.get("volume") or []

    close = pd.Series(pd.to_numeric(close_values, errors="coerce"), index=index, dtype=float)
    volume = pd.Series(pd.to_numeric(volume_values, errors="coerce"), index=index, dtype=float).fillna(0.0)
    close = close[~close.index.duplicated(keep="last")].dropna()
    volume = volume[~volume.index.duplicated(keep="last")].reindex(close.index, fill_value=0.0)
    return close, volume, result.get("meta", {})


def download_price_chunk(tickers: tuple[str, ...], start_date: date, end_date: date, timeout_seconds: float):
    captured_output = io.StringIO()
    with redirect_stdout(captured_output), redirect_stderr(captured_output):
        raw = yf.download(
            tickers=list(tickers),
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            session=get_yfinance_session(),
            threads=False,
            timeout=max(3, int(math.ceil(timeout_seconds))),
        )

    provider_output = captured_output.getvalue()
    rate_limited = "YFRateLimitError" in provider_output or "Too Many Requests" in provider_output
    close = sanitize_price_frame(extract_yfinance_field(raw, tickers, "Close"))
    volume = extract_yfinance_field(raw, tickers, "Volume").reindex(index=close.index, columns=close.columns).fillna(0.0)
    return close, volume, rate_limited


def download_price_bundle(
    tickers: tuple[str, ...],
    start_date: date,
    end_date: date,
    stop_on_rate_limit: bool = False,
    deadline_seconds: float | None = None,
):
    cached = _get_cached_price_bundle(tickers, start_date, end_date)
    if cached is not None:
        close, volume = cached
        return close, volume, False

    if PREFER_SEED_MARKET_DATA and seed_price_bundle_supports_end_date(tickers, end_date):
        seeded = get_seed_price_bundle(tickers, start_date, end_date)
        if seeded is not None:
            seed_close, seed_volume = seeded
            _store_cached_price_bundle(tickers, start_date, end_date, seed_close, seed_volume)
            return seed_close, seed_volume, False

    close_series_by_ticker = {}
    volume_series_by_ticker = {}
    saw_rate_limit = False
    started_at = time.monotonic()

    for ticker in tickers:
        if deadline_seconds is not None:
            remaining = deadline_seconds - (time.monotonic() - started_at)
            if remaining <= 0:
                break
        else:
            remaining = 20.0

        try:
            result = fetch_chart_result(
                ticker,
                start_date=start_date,
                end_date=end_date,
                timeout_seconds=remaining,
            )
            close_series, volume_series, _ = parse_chart_series(result)
        except MarketDataRateLimitedError:
            saw_rate_limit = True
            if stop_on_rate_limit:
                break
            continue
        except MarketDataTimeoutError:
            if deadline_seconds is not None and time.monotonic() - started_at >= deadline_seconds:
                break
            continue
        except MarketDataUnavailableError:
            continue

        if close_series.empty:
            continue

        close_series_by_ticker[ticker] = close_series
        volume_series_by_ticker[ticker] = volume_series

        if saw_rate_limit and stop_on_rate_limit:
            break
        if deadline_seconds is not None and time.monotonic() - started_at >= deadline_seconds:
            break

    if close_series_by_ticker:
        close = sanitize_price_frame(pd.DataFrame(close_series_by_ticker).sort_index())
        volume = pd.DataFrame(volume_series_by_ticker).reindex(index=close.index, columns=close.columns).fillna(0.0)
    else:
        close = pd.DataFrame()
        volume = pd.DataFrame()

    if close.empty:
        if saw_rate_limit:
            seeded = get_seed_price_bundle(tickers, start_date, end_date)
            if seeded is not None:
                seed_close, seed_volume = seeded
                _store_cached_price_bundle(tickers, start_date, end_date, seed_close, seed_volume)
                return seed_close, seed_volume, True
            raise MarketDataRateLimitedError("Live market data provider is rate-limited.")
        if deadline_seconds is not None and time.monotonic() - started_at >= deadline_seconds:
            seeded = get_seed_price_bundle(tickers, start_date, end_date)
            if seeded is not None:
                seed_close, seed_volume = seeded
                _store_cached_price_bundle(tickers, start_date, end_date, seed_close, seed_volume)
                return seed_close, seed_volume, False
            raise MarketDataTimeoutError("Live market data provider timed out.")
        seeded = get_seed_price_bundle(tickers, start_date, end_date)
        if seeded is not None:
            seed_close, seed_volume = seeded
            _store_cached_price_bundle(tickers, start_date, end_date, seed_close, seed_volume)
            return seed_close, seed_volume, False
        raise MarketDataUnavailableError("Unable to fetch enough price history for the selected securities.")
    _store_cached_price_bundle(tickers, start_date, end_date, close, volume)
    return close, volume, saw_rate_limit


def download_benchmark(start_date: date, end_date: date, timeout_seconds: float = 20.0) -> pd.Series:
    cached = _get_cached_benchmark(start_date, end_date)
    if cached is not None:
        return cached

    if PREFER_SEED_MARKET_DATA and seed_benchmark_supports_end_date(end_date):
        seeded = get_seed_benchmark(start_date, end_date)
        if seeded is not None:
            _store_cached_benchmark(start_date, end_date, seeded)
            return seeded

    try:
        result = fetch_chart_result(
            BENCHMARK_TICKER,
            start_date=start_date,
            end_date=end_date,
            timeout_seconds=timeout_seconds,
        )
        benchmark, _, _ = parse_chart_series(result)
        if benchmark.empty:
            raise MarketDataUnavailableError("Unable to fetch benchmark data.")
    except (MarketDataRateLimitedError, MarketDataTimeoutError, MarketDataUnavailableError):
        seeded = get_seed_benchmark(start_date, end_date)
        if seeded is not None:
            _store_cached_benchmark(start_date, end_date, seeded)
            return seeded
        raise

    _store_cached_benchmark(start_date, end_date, benchmark)
    return benchmark


def parse_sort_field(sort_by: str | None) -> str:
    if sort_by in SORT_OPTIONS:
        return sort_by
    return "momentum_3m"


def normalize_ticker(raw_ticker: str) -> str:
    ticker = raw_ticker.strip().upper()
    if not ticker:
        return ticker
    if "." not in ticker and not ticker.startswith("^"):
        ticker = f"{ticker}.NS"
    return ticker


def parse_backtest_request(payload) -> BacktestConfig:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    start_date = parse_iso_date(payload.get("start_date"), "start_date")
    end_date = parse_iso_date(payload.get("end_date"), "end_date")

    if start_date >= end_date:
        raise ValueError("Start date must be earlier than end date.")
    if end_date > date.today():
        raise ValueError("End date cannot be in the future.")
    if (end_date - start_date).days < 120:
        raise ValueError("Choose at least 120 calendar days so the strategy has enough context.")
    if (end_date - start_date).days > 365 * 6:
        raise ValueError("Backtests are capped at 6 years on this deployment.")

    selected_raw = payload.get("selected_stocks", [])
    if not isinstance(selected_raw, list):
        raise ValueError("selected_stocks must be a list.")

    selected = []
    seen = set()
    for item in selected_raw:
        if not isinstance(item, str):
            continue
        ticker = normalize_ticker(item)
        if ticker and ticker not in seen:
            selected.append(ticker)
            seen.add(ticker)
        if len(selected) >= MAX_UNIVERSE_SELECTION:
            break

    if len(selected) < 3:
        raise ValueError("Select at least 3 stocks to build a diversified portfolio.")

    rebalance_frequency = clamp_int(
        payload.get("rebalance_frequency", DEFAULT_REBALANCE_FREQUENCY),
        "rebalance_frequency",
        minimum=10,
        maximum=63,
    )
    max_holdings = clamp_int(
        payload.get("max_stocks", payload.get("max_holdings", DEFAULT_MAX_HOLDINGS)),
        "max_holdings",
        minimum=3,
        maximum=min(12, len(selected)),
    )
    transaction_cost_bps = clamp_int(
        payload.get("transaction_cost_bps", DEFAULT_TRANSACTION_COST_BPS),
        "transaction_cost_bps",
        minimum=0,
        maximum=100,
    )

    if max_holdings > len(selected):
        raise ValueError("Max holdings cannot exceed the number of selected stocks.")

    return BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        selected_stocks=tuple(selected),
        rebalance_frequency=rebalance_frequency,
        max_holdings=max_holdings,
        transaction_cost_bps=transaction_cost_bps,
    )


def clamp_int(value, field_name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}.")
    return parsed


def parse_iso_date(value, field_name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be formatted as YYYY-MM-DD.")
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{field_name} must be formatted as YYYY-MM-DD.") from exc


def ratio_change(prices: pd.Series, periods: int):
    series = prices.dropna()
    if len(series) <= periods:
        return None
    return _safe_float(series.iloc[-1] / series.iloc[-periods - 1] - 1.0)


def rolling_volatility(returns: pd.Series, periods: int):
    series = returns.dropna()
    if len(series) < periods:
        return None
    return _safe_float(series.tail(periods).std() * math.sqrt(252))


def drawdown_over_window(prices: pd.Series, periods: int):
    series = prices.dropna()
    if len(series) < periods:
        return None
    window = series.tail(periods)
    return _safe_float(window.iloc[-1] / window.max() - 1.0)


def default_universe_row(ticker: str):
    return {
        "ticker": ticker,
        "display_ticker": ticker.replace(".NS", ""),
        "last_price": None,
        "momentum_1m": None,
        "momentum_3m": None,
        "momentum_6m": None,
        "volatility_1m": None,
        "drawdown_6m": None,
        "avg_turnover": None,
        "rsi_14": None,
        "selectable": False,
    }


def default_universe_rows():
    return [default_universe_row(ticker) for ticker in NIFTY_50_STOCKS]


def build_universe_payload(
    rows,
    sort_by: str,
    as_of,
    tape,
    is_stale: bool = False,
    is_fallback: bool = False,
    warning: str | None = None,
    supported_window: dict[str, str] | None = None,
):
    sorted_rows = list(rows)
    sort_config = SORT_OPTIONS[sort_by]
    sorted_rows.sort(
        key=lambda row: row.get(sort_by) if row.get(sort_by) is not None else (-math.inf if sort_config["reverse"] else math.inf),
        reverse=sort_config["reverse"],
    )
    return {
        "as_of": as_of,
        "is_stale": is_stale,
        "is_fallback": is_fallback,
        "warning": warning,
        "sort_by": sort_by,
        "sort_options": [{"value": key, "label": value["label"]} for key, value in SORT_OPTIONS.items()],
        "tape": tape,
        "stocks": sorted_rows,
        "supported_window": supported_window,
    }


def build_fallback_universe_payload(
    sort_by: str,
    warning: str = "Using the latest cached Yahoo snapshot for the core 10-stock monitor while live refresh runs. The full NIFTY 50 list remains visible below.",
):
    seed_rows, seeded_as_of = build_seed_universe_rows()
    rows = merge_universe_rows(seed_rows) if seed_rows else default_universe_rows()
    return build_universe_payload(
        rows=rows,
        sort_by=sort_by,
        as_of=seeded_as_of or date.today().isoformat(),
        tape=build_universe_tape(rows),
        is_stale=True,
        is_fallback=True,
        warning=warning,
        supported_window=get_seed_supported_window(),
    )


def refresh_universe_cache():
    now = time.monotonic()
    try:
        start_date = date.today() - timedelta(days=220)
        end_date = date.today()
        close, volume, saw_rate_limit = download_price_bundle(
            CORE_LIVE_STOCKS,
            start_date,
            end_date,
            stop_on_rate_limit=True,
            deadline_seconds=UNIVERSE_FETCH_DEADLINE_SECONDS,
        )
        core_rows = build_core_monitor_rows(close, volume)
        rows = merge_universe_rows(core_rows)
        missing_tickers = [ticker for ticker in NIFTY_50_STOCKS if ticker not in {row["ticker"] for row in core_rows}]
        warning = None
        if saw_rate_limit and missing_tickers:
            warning = "Yahoo Finance is rate-limiting some requests. Live metrics are shown for the core 10-stock monitor; remaining names stay listed with placeholder values."
        elif missing_tickers:
            warning = "Live metrics are shown for the core 10-stock monitor. Remaining names stay listed with placeholder values."
        with UNIVERSE_CACHE_LOCK:
            UNIVERSE_CACHE["updated_at"] = now
            UNIVERSE_CACHE["rows"] = rows
            UNIVERSE_CACHE["as_of"] = close.index.max().strftime("%Y-%m-%d")
            UNIVERSE_CACHE["tape"] = build_universe_tape(rows)
            UNIVERSE_CACHE["warning"] = warning
            UNIVERSE_FETCH_STATE["backoff_until"] = now + UNIVERSE_RATE_LIMIT_BACKOFF_SECONDS if saw_rate_limit else 0.0
    except (MarketDataRateLimitedError, MarketDataTimeoutError):
        with UNIVERSE_CACHE_LOCK:
            UNIVERSE_FETCH_STATE["backoff_until"] = now + UNIVERSE_RATE_LIMIT_BACKOFF_SECONDS
    except Exception:
        traceback.print_exc()
        with UNIVERSE_CACHE_LOCK:
            UNIVERSE_FETCH_STATE["backoff_until"] = now + 60
    finally:
        with UNIVERSE_CACHE_LOCK:
            UNIVERSE_FETCH_STATE["in_progress"] = False


def maybe_start_universe_refresh(now: float):
    with UNIVERSE_CACHE_LOCK:
        if UNIVERSE_FETCH_STATE["in_progress"] or now < float(UNIVERSE_FETCH_STATE["backoff_until"]):
            return False
        UNIVERSE_FETCH_STATE["in_progress"] = True

    Thread(target=refresh_universe_cache, daemon=True).start()
    return True


def load_universe_snapshot(sort_by: str):
    if PREFER_SEED_MARKET_DATA:
        return build_fallback_universe_payload(
            sort_by,
            warning="Using the deployed market snapshot for the core 10-stock monitor. This keeps the app responsive on the current hosting tier.",
        )

    now = time.monotonic()
    with UNIVERSE_CACHE_LOCK:
        has_cache = bool(UNIVERSE_CACHE["rows"])
        cache_is_fresh = (
            has_cache
            and isinstance(UNIVERSE_CACHE["updated_at"], float)
            and now - UNIVERSE_CACHE["updated_at"] < UNIVERSE_CACHE_TTL_SECONDS
        )
        fetch_in_progress = bool(UNIVERSE_FETCH_STATE["in_progress"])
        backoff_until = float(UNIVERSE_FETCH_STATE["backoff_until"])

    if cache_is_fresh:
        return build_sorted_universe_payload(sort_by)

    if now >= backoff_until and not fetch_in_progress:
        maybe_start_universe_refresh(now)

    if has_cache:
        return build_sorted_universe_payload(sort_by, is_stale=True)

    if now < backoff_until:
        return build_fallback_universe_payload(sort_by)

    return build_fallback_universe_payload(
        sort_by,
        warning="Loading a recent cached snapshot for the core 10-stock monitor now. A live Yahoo refresh is running in the background.",
    )


def build_sorted_universe_payload(sort_by: str, is_stale: bool = False):
    with UNIVERSE_CACHE_LOCK:
        rows = list(UNIVERSE_CACHE["rows"])
        as_of = UNIVERSE_CACHE["as_of"]
        tape = dict(UNIVERSE_CACHE["tape"])
        warning = UNIVERSE_CACHE["warning"]
    return build_universe_payload(
        rows,
        sort_by,
        as_of,
        tape,
        is_stale=is_stale,
        warning=warning,
        supported_window=get_seed_supported_window() if PREFER_SEED_MARKET_DATA else None,
    )


class QuantResearchEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(self):
        # Keep enough lookback for factor construction, but let the model
        # fall back to heuristics early instead of overfetching history.
        buffer_days = max(190, self.config.rebalance_frequency * 4)
        fetch_start = self.config.start_date - timedelta(days=buffer_days)
        fetch_started_at = time.monotonic()

        close, _, _ = download_price_bundle(
            self.config.selected_stocks,
            fetch_start,
            self.config.end_date,
            deadline_seconds=BACKTEST_FETCH_DEADLINE_SECONDS,
        )
        remaining_time = BACKTEST_FETCH_DEADLINE_SECONDS - (time.monotonic() - fetch_started_at)
        if remaining_time <= 0:
            raise MarketDataTimeoutError("Backtest market data provider timed out.")
        benchmark_close = download_benchmark(fetch_start, self.config.end_date, timeout_seconds=remaining_time)
        if time.monotonic() - fetch_started_at >= BACKTEST_FETCH_DEADLINE_SECONDS:
            raise MarketDataTimeoutError("Backtest market data provider timed out.")

        available_tickers = tuple(ticker for ticker in self.config.selected_stocks if ticker in close.columns)
        dropped_tickers = [ticker for ticker in self.config.selected_stocks if ticker not in available_tickers]
        if len(available_tickers) < 3:
            raise ValueError("Fewer than 3 selected stocks have usable history in the requested window.")

        close = close.loc[:, available_tickers]
        benchmark_close = benchmark_close.reindex(close.index).ffill().dropna()
        close = close.reindex(benchmark_close.index)

        trade_close = close.loc[self.config.start_date.isoformat():self.config.end_date.isoformat()]
        benchmark_trade = benchmark_close.reindex(trade_close.index).ffill().dropna()
        trade_close = trade_close.reindex(benchmark_trade.index)
        if trade_close.shape[0] < self.config.rebalance_frequency * 2:
            raise ValueError("The selected date range is too short for the chosen rebalance frequency.")

        factor_book = self._build_factor_book(close)
        history_index = close.index
        trade_index = trade_close.index
        forward_returns = close.shift(-self.config.rebalance_frequency) / close - 1.0
        trade_returns = trade_close.pct_change().fillna(0.0)
        benchmark_returns = benchmark_trade.pct_change().fillna(0.0)

        rebalance_positions = list(range(0, len(trade_index), self.config.rebalance_frequency))
        if rebalance_positions[-1] != len(trade_index) - 1:
            rebalance_positions.append(len(trade_index) - 1)

        portfolio_daily_returns = pd.Series(0.0, index=trade_index, dtype=float)
        last_weights = pd.Series(dtype=float)
        trade_log = []
        portfolio_snapshots = []
        rebalance_summary = []
        latest_signal_table = []
        latest_allocation = []
        latest_feature_importances = self._default_feature_importances()
        turnover_values = []

        for position_index in range(len(rebalance_positions) - 1):
            rebalance_position = rebalance_positions[position_index]
            next_position = rebalance_positions[position_index + 1]
            rebalance_date = trade_index[rebalance_position]
            next_rebalance_date = trade_index[next_position]

            panel = self._build_panel(factor_book, forward_returns, rebalance_date)
            panel = panel.dropna(subset=list(FACTOR_COLUMNS))
            if panel.empty:
                rebalance_summary.append(
                    {
                        "date": rebalance_date.strftime("%Y-%m-%d"),
                        "eligible_count": 0,
                        "selected_count": 0,
                        "turnover": 0.0,
                        "avg_signal": 0.0,
                    }
                )
                continue

            training_dates = self._training_dates(history_index, rebalance_date)
            model, scaler, feature_importances = self._train_model(factor_book, forward_returns, training_dates)
            if feature_importances:
                latest_feature_importances = feature_importances

            signal_scores, expected_returns = self._score_panel(panel, model, scaler)
            panel["signal_score"] = signal_scores
            panel["expected_return"] = expected_returns
            panel = panel.dropna(subset=["signal_score", "expected_return"])

            selected_panel = panel[panel["signal_score"] > 0].sort_values("signal_score", ascending=False).head(
                self.config.max_holdings
            )
            current_weights = self._build_weights(selected_panel, trade_returns.loc[:rebalance_date])

            turnover = self._portfolio_turnover(current_weights, last_weights)
            turnover_values.append(turnover)

            period_mask = (trade_index > rebalance_date) & (trade_index <= next_rebalance_date)
            if current_weights.any() and period_mask.any():
                weighted_returns = trade_returns.loc[period_mask, current_weights.index].fillna(0.0).dot(current_weights)
                if not weighted_returns.empty:
                    weighted_returns.iloc[0] -= turnover * (self.config.transaction_cost_bps / 10000.0)
                    portfolio_daily_returns.loc[period_mask] = weighted_returns

            selected_rows = []
            period_return_pct = trade_close.loc[next_rebalance_date] / trade_close.loc[rebalance_date] - 1.0
            for ticker, weight in current_weights.items():
                trade_log.append(
                    {
                        "ticker": ticker.replace(".NS", ""),
                        "entry_date": rebalance_date.strftime("%Y-%m-%d"),
                        "exit_date": next_rebalance_date.strftime("%Y-%m-%d"),
                        "weight": _safe_float(weight),
                        "expected_return": _safe_float(panel.loc[ticker, "expected_return"]),
                        "realized_return": _safe_float(period_return_pct.get(ticker)),
                        "signal_score": _safe_float(panel.loc[ticker, "signal_score"]),
                    }
                )
                selected_rows.append(
                    {
                        "ticker": ticker,
                        "display_ticker": ticker.replace(".NS", ""),
                        "weight": _safe_float(weight),
                        "expected_return": _safe_float(panel.loc[ticker, "expected_return"]),
                        "signal_score": _safe_float(panel.loc[ticker, "signal_score"]),
                    }
                )

            rebalance_summary.append(
                {
                    "date": rebalance_date.strftime("%Y-%m-%d"),
                    "eligible_count": int(panel.shape[0]),
                    "selected_count": int(selected_panel.shape[0]),
                    "turnover": _safe_float(turnover),
                    "avg_signal": _safe_float(selected_panel["signal_score"].mean()) if not selected_panel.empty else 0.0,
                }
            )
            portfolio_snapshots.append({"date": rebalance_date.strftime("%Y-%m-%d"), "holdings": selected_rows})
            latest_signal_table = self._build_signal_table(panel, current_weights.index)
            latest_allocation = selected_rows
            last_weights = current_weights

        portfolio_values = (1.0 + portfolio_daily_returns).cumprod()
        benchmark_values = (1.0 + benchmark_returns).cumprod()
        drawdown_values = portfolio_values / portfolio_values.cummax() - 1.0

        metrics = self._calculate_metrics(
            portfolio_daily_returns=portfolio_daily_returns,
            benchmark_returns=benchmark_returns,
            portfolio_values=portfolio_values,
            benchmark_values=benchmark_values,
            trade_log=trade_log,
            average_turnover=float(np.nanmean(turnover_values)) if turnover_values else 0.0,
        )

        summary = self._build_summary(metrics)
        return {
            "meta": {
                "benchmark": BENCHMARK_TICKER,
                "available_tickers": [ticker.replace(".NS", "") for ticker in available_tickers],
                "dropped_tickers": [ticker.replace(".NS", "") for ticker in dropped_tickers],
                "rebalance_frequency": self.config.rebalance_frequency,
                "max_holdings": self.config.max_holdings,
            },
            "summary": summary,
            "metrics": metrics,
            "performance": {
                "dates": portfolio_values.index.strftime("%Y-%m-%d").tolist(),
                "portfolio_values": [_safe_float(value) for value in portfolio_values.values],
                "benchmark_values": [_safe_float(value) for value in benchmark_values.values],
                "drawdown_values": [_safe_float(value) for value in drawdown_values.values],
            },
            "feature_importances": latest_feature_importances,
            "latest_allocation": latest_allocation,
            "latest_signal_table": latest_signal_table,
            "portfolio_snapshots": portfolio_snapshots,
            "rebalance_summary": rebalance_summary,
            "trade_log": trade_log[-80:],
        }

    def _build_factor_book(self, close: pd.DataFrame):
        returns = close.pct_change()
        return {
            "momentum_1m": close.pct_change(21),
            "momentum_3m": close.pct_change(63),
            "momentum_6m": close.pct_change(126),
            "trend_gap_100d": close / close.rolling(100).mean() - 1.0,
            "drawdown_6m": close / close.rolling(126).max() - 1.0,
            "volatility_1m": returns.rolling(21).std() * math.sqrt(252),
            "rsi_14": close.apply(series_rsi),
        }

    def _build_panel(self, factor_book, forward_returns, as_of_date):
        panel = pd.DataFrame(index=forward_returns.columns)
        for factor_name in FACTOR_COLUMNS:
            factor_frame = factor_book[factor_name]
            if as_of_date in factor_frame.index:
                panel[factor_name] = factor_frame.loc[as_of_date]
            else:
                panel[factor_name] = np.nan
        panel["target"] = forward_returns.loc[as_of_date] if as_of_date in forward_returns.index else np.nan
        panel = panel.replace([np.inf, -np.inf], np.nan)
        return panel

    def _training_dates(self, history_index, current_date):
        current_position = history_index.get_loc(current_date)
        if isinstance(current_position, slice):
            current_position = current_position.start

        max_position = current_position - self.config.rebalance_frequency
        if max_position <= 0:
            return []

        min_position = max(0, max_position - MODEL_TRAINING_WINDOW)
        return list(history_index[min_position : max_position + 1])

    def _train_model(self, factor_book, forward_returns, training_dates):
        if FAST_DEPLOYMENT_MODE:
            return None, None, self._default_feature_importances()

        if len(training_dates) < 4:
            return None, None, self._default_feature_importances()

        feature_frames = []
        target_frames = []
        for training_date in training_dates:
            panel = self._build_panel(factor_book, forward_returns, training_date)
            panel = panel.dropna(subset=list(FACTOR_COLUMNS) + ["target"])
            if panel.shape[0] < 4:
                continue
            feature_frames.append(self._winsorize(panel.loc[:, FACTOR_COLUMNS]))
            target_frames.append(panel["target"])

        if not feature_frames:
            return None, None, self._default_feature_importances()

        X = pd.concat(feature_frames)
        y = pd.concat(target_frames)
        if X.shape[0] < 40:
            return None, None, self._default_feature_importances()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Ridge(alpha=4.0)
        model.fit(X_scaled, y.values)

        feature_importances = [
            {"feature": feature, "coefficient": _safe_float(coefficient)}
            for feature, coefficient in zip(FACTOR_COLUMNS, model.coef_)
        ]
        feature_importances.sort(key=lambda row: abs(row["coefficient"]), reverse=True)
        return model, scaler, feature_importances

    def _score_panel(self, panel, model, scaler):
        working = self._winsorize(panel.loc[:, FACTOR_COLUMNS].copy())
        heuristic = self._heuristic_signal(working)

        if model is None or scaler is None:
            expected_returns = heuristic / 100.0
            return heuristic, expected_returns

        scaled = scaler.transform(working)
        predicted = pd.Series(model.predict(scaled), index=working.index, dtype=float)
        blended_signal = 0.65 * self._zscore(predicted) + 0.35 * heuristic
        return blended_signal, predicted

    def _heuristic_signal(self, feature_frame):
        components = []
        for factor_name in FACTOR_COLUMNS:
            zscore = self._zscore(feature_frame[factor_name])
            components.append(zscore * FACTOR_DIRECTIONS[factor_name])
        return pd.concat(components, axis=1).mean(axis=1)

    def _build_weights(self, selected_panel, returns_history):
        if selected_panel.empty:
            return pd.Series(dtype=float)

        raw_weights = selected_panel["signal_score"].clip(lower=0.05) / selected_panel["volatility_1m"].clip(lower=0.12)
        correlation_returns = returns_history.loc[:, selected_panel.index].tail(63).dropna(how="all")
        if correlation_returns.shape[0] >= 20 and selected_panel.shape[0] > 1:
            correlation_matrix = correlation_returns.corr().fillna(0.0)
            average_correlation = (correlation_matrix.sum() - 1.0) / max(selected_panel.shape[0] - 1, 1)
            penalty = (1.0 - average_correlation).clip(lower=0.25)
            raw_weights = raw_weights * penalty

        weights = raw_weights / raw_weights.sum()
        max_weight = min(0.35, max(0.18, 1.8 / max(selected_panel.shape[0], 1)))
        return self._cap_weights(weights, max_weight).sort_values(ascending=False)

    def _build_signal_table(self, panel, selected_index):
        visible = panel.sort_values("signal_score", ascending=False).head(12)
        rows = []
        for ticker, row in visible.iterrows():
            rows.append(
                {
                    "ticker": ticker.replace(".NS", ""),
                    "selected": ticker in selected_index,
                    "signal_score": _safe_float(row["signal_score"]),
                    "expected_return": _safe_float(row["expected_return"]),
                    "momentum_3m": _safe_float(row["momentum_3m"]),
                    "volatility_1m": _safe_float(row["volatility_1m"]),
                    "drawdown_6m": _safe_float(row["drawdown_6m"]),
                    "rsi_14": _safe_float(row["rsi_14"]),
                }
            )
        return rows

    def _calculate_metrics(self, portfolio_daily_returns, benchmark_returns, portfolio_values, benchmark_values, trade_log, average_turnover):
        daily_mean = portfolio_daily_returns.mean()
        daily_volatility = portfolio_daily_returns.std()
        annualized_volatility = daily_volatility * math.sqrt(252)
        annualized_return = ((portfolio_values.iloc[-1] ** (252 / max(len(portfolio_values), 1))) - 1.0) if not portfolio_values.empty else 0.0
        downside = portfolio_daily_returns[portfolio_daily_returns < 0]
        downside_volatility = downside.std() * math.sqrt(252) if not downside.empty else 0.0
        max_drawdown = _safe_float((portfolio_values / portfolio_values.cummax() - 1.0).min()) if not portfolio_values.empty else 0.0
        active_returns = portfolio_daily_returns - benchmark_returns
        tracking_error = active_returns.std() * math.sqrt(252)
        info_ratio = (active_returns.mean() * 252 / tracking_error) if tracking_error > 0 else 0.0

        if len(portfolio_daily_returns) > 1 and len(benchmark_returns) > 1:
            X = benchmark_returns.values.reshape(-1, 1)
            y = portfolio_daily_returns.values
            regression = LinearRegression().fit(X, y)
            beta = float(regression.coef_[0])
            alpha = float((y.mean() - beta * X.mean()) * 252)
        else:
            alpha, beta = 0.0, 0.0

        sharpe = (daily_mean * 252 / annualized_volatility) if annualized_volatility > 0 else 0.0
        sortino = (daily_mean * 252 / downside_volatility) if downside_volatility > 0 else 0.0
        calmar = (annualized_return / abs(max_drawdown)) if max_drawdown and max_drawdown < 0 else 0.0
        benchmark_total_return = benchmark_values.iloc[-1] - 1.0 if not benchmark_values.empty else 0.0
        hit_rate = (
            sum(1 for trade in trade_log if (trade.get("realized_return") or 0.0) > 0) / len(trade_log)
            if trade_log
            else 0.0
        )

        return {
            "total_return": _safe_float(portfolio_values.iloc[-1] - 1.0) if not portfolio_values.empty else 0.0,
            "benchmark_total_return": _safe_float(benchmark_total_return),
            "annualized_return": _safe_float(annualized_return),
            "annualized_volatility": _safe_float(annualized_volatility),
            "sharpe_ratio": _safe_float(sharpe),
            "sortino_ratio": _safe_float(sortino),
            "calmar_ratio": _safe_float(calmar),
            "alpha": _safe_float(alpha),
            "beta": _safe_float(beta),
            "information_ratio": _safe_float(info_ratio),
            "tracking_error": _safe_float(tracking_error),
            "max_drawdown": _safe_float(max_drawdown),
            "trade_hit_rate": _safe_float(hit_rate),
            "average_turnover": _safe_float(average_turnover),
        }

    def _build_summary(self, metrics):
        total = metrics["total_return"] or 0.0
        benchmark = metrics["benchmark_total_return"] or 0.0
        excess = total - benchmark
        drawdown = abs(metrics["max_drawdown"] or 0.0)
        sharpe = metrics["sharpe_ratio"] or 0.0
        turnover = metrics["average_turnover"] or 0.0

        overview = (
            f"Over the selected period, the strategy returned {total * 100:.1f}% versus {benchmark * 100:.1f}% for the NIFTY benchmark. "
            f"Its worst peak-to-trough decline was {drawdown * 100:.1f}%, which frames the depth of the strategy's loss during adverse periods."
        )

        diagnostics = (
            f"Excess return finished at {excess * 100:.1f}%, with Sharpe {sharpe:.2f}, information ratio {metrics['information_ratio'] or 0.0:.2f}, "
            f"and average one-way turnover of {turnover * 100:.1f}% per rebalance. This is a medium-frequency long-only signal stack rather than a high-churn stat-arb profile."
        )

        if excess > 0 and drawdown < 0.15:
            assessment = "The profile is constructive on the surface, but it still needs sensitivity checks across rebalance frequency, universe choice, and transaction costs."
        elif excess > 0:
            assessment = "Returns outpaced the benchmark, but drawdowns were material. Position sizing and signal filters should be reviewed before relying on the profile."
        else:
            assessment = "This run did not beat the benchmark. The next review should focus on regime dependence, signal decay, and whether the selected basket is too narrow."

        return {
            "overview": overview,
            "diagnostics": diagnostics,
            "assessment": assessment,
        }

    @staticmethod
    def _portfolio_turnover(current_weights, previous_weights):
        full_index = current_weights.index.union(previous_weights.index)
        if full_index.empty:
            return 0.0
        current = current_weights.reindex(full_index, fill_value=0.0)
        previous = previous_weights.reindex(full_index, fill_value=0.0)
        return float((current - previous).abs().sum())

    @staticmethod
    def _winsorize(frame):
        if frame.empty:
            return frame
        lower = frame.quantile(0.05)
        upper = frame.quantile(0.95)
        return frame.clip(lower=lower, upper=upper, axis=1)

    @staticmethod
    def _zscore(series):
        series = series.replace([np.inf, -np.inf], np.nan)
        centered = series - series.mean()
        std = series.std()
        if std is None or pd.isna(std) or std == 0:
            return pd.Series(0.0, index=series.index, dtype=float)
        return centered / std

    @staticmethod
    def _cap_weights(weights, max_weight):
        if weights.empty:
            return weights

        capped = weights.copy()
        iterations = 0
        while capped.max() > max_weight and iterations < 10:
            iterations += 1
            overweight = capped[capped > max_weight]
            excess = (overweight - max_weight).sum()
            capped.loc[overweight.index] = max_weight
            underweight = capped[capped < max_weight]
            if underweight.empty or underweight.sum() == 0:
                break
            capped.loc[underweight.index] += excess * (underweight / underweight.sum())

        return capped / capped.sum()

    @staticmethod
    def _default_feature_importances():
        rows = [{"feature": factor, "coefficient": weight} for factor, weight in FACTOR_DIRECTIONS.items()]
        rows.sort(key=lambda row: abs(row["coefficient"]), reverse=True)
        return rows


app = Flask(__name__)


@app.errorhandler(404)
def handle_not_found(exc):
    if request.path.startswith("/api/"):
        return jsonify({"error": "The requested API route was not found."}), 404
    return exc


@app.errorhandler(405)
def handle_method_not_allowed(exc):
    if request.path.startswith("/api/"):
        return jsonify({"error": "The request method is not allowed for this API route."}), 405
    return exc


@app.errorhandler(500)
def handle_internal_server_error(exc):
    if request.path.startswith("/api/"):
        return jsonify({"error": "An unexpected server error occurred while handling the API request."}), 500
    return exc


@app.route("/")
def index():
    return render_template(
        "index.html",
        app_title=APP_TITLE,
        app_runtime={
            "mode": APP_MODE.key,
            "label": APP_MODE.label,
            "description": APP_MODE.description,
            "universeTimeoutMs": APP_MODE.universe_timeout_ms,
            "backtestTimeoutMs": APP_MODE.backtest_timeout_ms,
        },
    )


@app.get("/healthz")
def healthcheck():
    return jsonify({"status": "ok", "service": APP_TITLE, "mode": APP_MODE.key})


@app.get("/api/universe")
def universe_snapshot():
    try:
        sort_by = parse_sort_field(request.args.get("sort_by"))
        return jsonify(load_universe_snapshot(sort_by))
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Unable to load the universe snapshot right now."}), 500


@app.get("/api/nifty50_stocks")
def universe_snapshot_legacy():
    try:
        sort_by = parse_sort_field(request.args.get("sort_by"))
        payload = load_universe_snapshot(sort_by)
        return jsonify(payload["stocks"])
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Unable to load stock data right now."}), 500


@app.post("/api/backtest")
def run_backtest_api():
    try:
        config = parse_backtest_request(request.get_json(silent=True))
        engine = QuantResearchEngine(config)
        started_at = time.perf_counter()
        result = engine.run()
        result["meta"]["runtime_ms"] = int((time.perf_counter() - started_at) * 1000)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except MarketDataRateLimitedError:
        return jsonify({"error": "Live market data is temporarily rate-limited. Wait a few minutes and try the backtest again."}), 503
    except MarketDataTimeoutError:
        return jsonify({"error": "Live market data took too long to respond. Try again shortly or narrow the stock basket/date range."}), 503
    except MarketDataUnavailableError:
        return jsonify({"error": "Live market data was unavailable for the selected basket. Try again shortly or choose a different stock set."}), 503
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "An unexpected error occurred while running the backtest."}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG") == "1",
    )
