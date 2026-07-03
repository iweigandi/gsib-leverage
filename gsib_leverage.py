from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time
from typing import Iterable
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

START_DATE = "2006-01-01"
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / "data"
CHART_DIR = OUTPUT_DIR / "chart"
SOURCE_DATA_DIR = OUTPUT_DIR / "source_data"
OFFICIAL_BALANCE_SHEET_PATH = SOURCE_DATA_DIR / "official_balance_sheets.csv"
CACHE_DIR = Path(os.environ.get("TEMP", str(OUTPUT_DIR / ".cache"))) / "yfinance-cache-gsib"
SOURCE_CACHE_DIR = OUTPUT_DIR / ".cache" / "source-data"
REQUEST_SLEEP_SECONDS = float(os.environ.get("YF_REQUEST_SLEEP_SECONDS", "2"))
YF_TIMEOUT_SECONDS = float(os.environ.get("YF_TIMEOUT_SECONDS", "20"))
USE_HISTORICAL_SHARES = os.environ.get("GSIB_USE_HISTORICAL_SHARES", "0") == "1"
MIN_BANKS = int(os.environ.get("GSIB_MIN_BANKS", "15"))
PCA_COMPONENTS = int(os.environ.get("GSIB_PCA_COMPONENTS", "3"))
GIV_WEIGHT_THRESHOLD = float(os.environ.get("GSIB_GIV_WEIGHT_THRESHOLD", "0.005"))

PALETTE = [
    "#00466F",
    "#F38C10",
    "#3297DB",
    "#037E73",
    "#C62828",
    "#FEBD00",
    "#41B01E",
    "#E84C3D",
    "#3D3D3D",
]


@dataclass(frozen=True)
class Bank:
    name: str
    ticker: str | None
    country: str
    region: str
    currency: str = "USD"
    fx_ticker: str | None = None
    fx_operation: str | None = None
    price_scale: float = 1.0
    public_listing: bool = True
    note: str = ""


BANKS: list[Bank] = [
    Bank("JPMorgan Chase", "JPM", "United States", "North America"),
    Bank("Bank of America", "BAC", "United States", "North America"),
    Bank("Citigroup", "C", "United States", "North America"),
    Bank("Goldman Sachs", "GS", "United States", "North America"),
    Bank("Morgan Stanley", "MS", "United States", "North America"),
    Bank("Wells Fargo", "WFC", "United States", "North America"),
    Bank("Bank of New York Mellon", "BK", "United States", "North America"),
    Bank("State Street", "STT", "United States", "North America"),
    Bank("Royal Bank of Canada", "RY.TO", "Canada", "North America", "CAD", "USDCAD=X", "div"),
    Bank("Toronto-Dominion Bank", "TD.TO", "Canada", "North America", "CAD", "USDCAD=X", "div"),
    Bank("HSBC", "HSBA.L", "United Kingdom", "Europe", "GBP", "GBPUSD=X", "mult", price_scale=0.01),
    Bank("Barclays", "BARC.L", "United Kingdom", "Europe", "GBP", "GBPUSD=X", "mult", price_scale=0.01),
    Bank("Standard Chartered", "STAN.L", "United Kingdom", "Europe", "GBP", "GBPUSD=X", "mult", price_scale=0.01),
    Bank("BNP Paribas", "BNP.PA", "France", "Europe", "EUR", "EURUSD=X", "mult"),
    Bank("Credit Agricole", "ACA.PA", "France", "Europe", "EUR", "EURUSD=X", "mult"),
    Bank("Societe Generale", "GLE.PA", "France", "Europe", "EUR", "EURUSD=X", "mult"),
    Bank("Deutsche Bank", "DBK.DE", "Germany", "Europe", "EUR", "EURUSD=X", "mult"),
    Bank("ING Groep", "INGA.AS", "Netherlands", "Europe", "EUR", "EURUSD=X", "mult"),
    Bank("Banco Santander", "SAN.MC", "Spain", "Europe", "EUR", "EURUSD=X", "mult"),
    Bank("UBS Group", "UBSG.SW", "Switzerland", "Europe", "CHF", "USDCHF=X", "div"),
    Bank("Mitsubishi UFJ Financial Group", "8306.T", "Japan", "Asia", "JPY", "USDJPY=X", "div"),
    Bank("Sumitomo Mitsui Financial Group", "8316.T", "Japan", "Asia", "JPY", "USDJPY=X", "div"),
    Bank("Mizuho Financial Group", "8411.T", "Japan", "Asia", "JPY", "USDJPY=X", "div"),
    Bank("Industrial and Commercial Bank of China", "1398.HK", "China", "Asia", "HKD", "USDHKD=X", "div"),
    Bank("China Construction Bank", "0939.HK", "China", "Asia", "HKD", "USDHKD=X", "div"),
    Bank("Bank of China", "3988.HK", "China", "Asia", "HKD", "USDHKD=X", "div"),
    Bank("Agricultural Bank of China", "1288.HK", "China", "Asia", "HKD", "USDHKD=X", "div"),
    Bank("Bank of Communications", "3328.HK", "China", "Asia", "HKD", "USDHKD=X", "div"),
    Bank("Groupe BPCE", None, "France", "Europe", public_listing=False, note="Excluded because the group is not publicly listed."),
    Bank("Credit Mutuel", None, "France", "Europe", public_listing=False, note="Excluded because the group is not publicly listed."),
]

ASSET_LABELS = ["Total Assets", "TotalAssets"]
EQUITY_LABELS = [
    "Stockholders Equity",
    "Total Equity Gross Minority Interest",
    "Common Stock Equity",
    "Total Stockholder Equity",
    "Shareholders Equity",
]


def set_custom_style() -> list[str]:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "black",
            "axes.linewidth": 1,
            "axes.grid": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "font.size": 9,
            "lines.linewidth": 1.5,
            "figure.dpi": 300,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
        }
    )
    return PALETTE


def _normalise_label(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _pick_row(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series | None:
    if frame.empty:
        return None
    lookup = {_normalise_label(idx): idx for idx in frame.index}
    for candidate in candidates:
        key = _normalise_label(candidate)
        if key in lookup:
            return pd.to_numeric(frame.loc[lookup[key]], errors="coerce")
    return None


def _cache_key(ticker: str) -> str:
    return ticker.replace(".", "_").replace("-", "_").replace("=", "_")


def _read_cached_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        out = pd.read_csv(path, index_col=0, parse_dates=True)
        out.index = pd.to_datetime(out.index).tz_localize(None)
        return out
    except Exception:
        return pd.DataFrame()


def _write_cached_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def _download_with_retries(tickers: list[str] | str, **kwargs) -> pd.DataFrame:
    frame = pd.DataFrame()
    for attempt in range(4):
        try:
            frame = yf.download(
                tickers,
                start=START_DATE,
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=YF_TIMEOUT_SECONDS,
                **kwargs,
            )
        except Exception:
            frame = pd.DataFrame()
        if frame is not None and not frame.empty:
            return frame
        time.sleep(REQUEST_SLEEP_SECONDS * (attempt + 1))
    return pd.DataFrame()


def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if raw.empty:
        return pd.Series(dtype=float)
    close = pd.Series(dtype=float)
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0):
                close = raw[(ticker, "Close")]
            elif "Close" in raw.columns.get_level_values(0):
                close = raw["Close"][ticker]
        elif "Close" in raw:
            close = raw["Close"]
    except Exception:
        close = pd.Series(dtype=float)
    if close is None or close.empty:
        return pd.Series(dtype=float)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return pd.to_numeric(close, errors="coerce").dropna()


def get_prices(banks: list[Bank]) -> dict[str, pd.Series]:
    listed = [bank for bank in banks if bank.public_listing and bank.ticker]
    tickers = [bank.ticker for bank in listed if bank.ticker]
    cache_path = SOURCE_CACHE_DIR / "stock_prices.csv"
    cached = _read_cached_frame(cache_path)
    price_frame = pd.DataFrame()
    if not cached.empty and all(ticker in cached for ticker in tickers):
        price_frame = cached[tickers]
    else:
        raw = _download_with_retries(tickers, group_by="ticker")
        for bank in listed:
            assert bank.ticker is not None
            close = _extract_close(raw, bank.ticker)
            if not close.empty:
                price_frame[bank.ticker] = close
        if not price_frame.empty:
            _write_cached_frame(price_frame, cache_path)
    prices: dict[str, pd.Series] = {}
    for bank in listed:
        assert bank.ticker is not None
        if bank.ticker in price_frame:
            prices[bank.name] = pd.to_numeric(price_frame[bank.ticker], errors="coerce").dropna() * bank.price_scale
    return prices


def get_fx_rates(banks: list[Bank]) -> dict[str, pd.Series]:
    tickers = sorted({bank.fx_ticker for bank in banks if bank.fx_ticker})
    if not tickers:
        return {}
    cache_path = SOURCE_CACHE_DIR / "fx_rates.csv"
    cached = _read_cached_frame(cache_path)
    fx_frame = pd.DataFrame()
    if not cached.empty and all(ticker in cached for ticker in tickers):
        fx_frame = cached[tickers]
    else:
        raw = _download_with_retries(tickers, group_by="ticker")
        for ticker in tickers:
            close = _extract_close(raw, ticker)
            if not close.empty:
                fx_frame[ticker] = close
        if not fx_frame.empty:
            _write_cached_frame(fx_frame, cache_path)
    return {ticker: fx_frame[ticker].dropna() for ticker in fx_frame}


def convert_to_usd(series: pd.Series, bank: Bank, fx_rates: dict[str, pd.Series]) -> pd.Series:
    out = series.astype(float).copy()
    if bank.currency == "USD" or not bank.fx_ticker:
        return out
    fx = fx_rates.get(bank.fx_ticker, pd.Series(dtype=float))
    if fx.empty:
        return pd.Series(index=out.index, dtype=float)
    fx = fx.reindex(out.index.union(fx.index)).sort_index().ffill().reindex(out.index)
    if bank.fx_operation == "mult":
        return out * fx
    if bank.fx_operation == "div":
        return out / fx
    return out


def get_official_balance_sheet(bank: Bank) -> pd.DataFrame:
    if not OFFICIAL_BALANCE_SHEET_PATH.exists():
        return pd.DataFrame()
    data = pd.read_csv(OFFICIAL_BALANCE_SHEET_PATH, parse_dates=["period_end"])
    data = data[data["bank"].eq(bank.name)].copy()
    if data.empty:
        return pd.DataFrame()
    data = data.sort_values("period_end").drop_duplicates("period_end", keep="last")
    out = data.set_index("period_end")[["total_assets", "book_equity"]].astype(float)
    out = out[(out["total_assets"] > 0) & (out["book_equity"] > 0)]
    out["book_debt"] = out["total_assets"] - out["book_equity"]
    out["balance_sheet_source"] = "official_filings"
    return out


def get_yfinance_balance_sheet(bank: Bank) -> pd.DataFrame:
    assert bank.ticker is not None
    cache_path = SOURCE_CACHE_DIR / f"balance_sheet_{_cache_key(bank.ticker)}.csv"
    cached = _read_cached_frame(cache_path)
    if not cached.empty:
        return cached
    ticker = yf.Ticker(bank.ticker)
    frames = []
    for freq in ["quarterly", "yearly"]:
        try:
            frame = ticker.get_balance_sheet(freq=freq)
        except Exception:
            frame = pd.DataFrame()
        if frame is None or frame.empty:
            continue
        assets = _pick_row(frame, ASSET_LABELS)
        equity = _pick_row(frame, EQUITY_LABELS)
        if assets is None or equity is None:
            continue
        out = pd.DataFrame({"total_assets": assets, "book_equity": equity})
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.sort_index()
        out = out[(out["total_assets"] > 0) & (out["book_equity"] > 0)]
        out["book_debt"] = out["total_assets"] - out["book_equity"]
        out["balance_sheet_source"] = f"yfinance_{freq}"
        frames.append(out)
        if freq == "quarterly":
            break
    if not frames:
        return pd.DataFrame()
    _write_cached_frame(frames[0], cache_path)
    return frames[0]


def get_balance_sheet(bank: Bank) -> pd.DataFrame:
    official = get_official_balance_sheet(bank)
    if not official.empty:
        return official
    return get_yfinance_balance_sheet(bank)


def get_shares(bank: Bank, dates: pd.DatetimeIndex) -> tuple[pd.Series, str]:
    assert bank.ticker is not None
    cache_path = SOURCE_CACHE_DIR / f"shares_{_cache_key(bank.ticker)}.csv"
    cached = _read_cached_frame(cache_path)
    if not cached.empty and "shares" in cached:
        shares = cached["shares"].dropna()
        out = shares.reindex(dates.union(shares.index)).sort_index().ffill().reindex(dates)
        return out, "cached_yfinance"
    ticker = yf.Ticker(bank.ticker)
    shares = pd.Series(dtype=float)
    source = "unavailable"
    if USE_HISTORICAL_SHARES:
        try:
            shares = ticker.get_shares_full(start=START_DATE, timeout=YF_TIMEOUT_SECONDS)
        except Exception:
            shares = pd.Series(dtype=float)
    if shares is not None and len(shares) > 0:
        shares = pd.Series(shares)
        shares.index = pd.to_datetime(shares.index).tz_localize(None)
        shares = pd.to_numeric(shares, errors="coerce").dropna().sort_index()
        source = "historical_yfinance"
    else:
        value = np.nan
        try:
            value = ticker.fast_info.get("shares", np.nan)
        except Exception:
            value = np.nan
        if not np.isfinite(value):
            try:
                value = ticker.info.get("sharesOutstanding", np.nan)
            except Exception:
                value = np.nan
        if np.isfinite(value) and value > 0:
            shares = pd.Series(value, index=[dates.min()])
            source = "latest_yfinance"
    if shares is None or shares.empty:
        return pd.Series(index=dates, dtype=float), source
    _write_cached_frame(pd.DataFrame({"shares": shares.sort_index()}), cache_path)
    out = shares.reindex(dates.union(shares.index)).sort_index().ffill().reindex(dates)
    return out, source


def _entity_demean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(float).sub(frame.mean(axis=0), axis=1)


def _pca_decomposition(frame: pd.DataFrame, n_components: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = frame.dropna(axis=0, how="any").dropna(axis=1, how="any")
    if data.shape[0] < 10 or data.shape[1] < 3:
        empty = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)
        return empty, empty, pd.DataFrame()
    centered = data.sub(data.mean(axis=0), axis=1)
    x = centered.to_numpy(dtype=float)
    n_components = min(n_components, x.shape[0], x.shape[1])
    u, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    factors = u[:, :n_components] * singular_values[:n_components]
    loadings = vt[:n_components, :].T
    common = factors @ loadings.T
    common = pd.DataFrame(common, index=data.index, columns=data.columns).add(data.mean(axis=0), axis=1)
    factor_frame = pd.DataFrame(factors, index=data.index, columns=[f"factor_{i + 1}" for i in range(n_components)])
    loading_frame = pd.DataFrame(loadings, index=data.columns, columns=[f"loading_factor_{i + 1}" for i in range(n_components)])
    weighted_change = data.mean(axis=1)
    weighted_common = common.mean(axis=1)
    if weighted_change.corr(weighted_common) < 0:
        common = -common
        factor_frame = -factor_frame
        loading_frame = -loading_frame
    return common, factor_frame, loading_frame


def build_market_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    DATA_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)
    SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    prices = get_prices(BANKS)
    fx_rates = get_fx_rates(BANKS)
    leverage_by_bank: dict[str, pd.Series] = {}
    assets_by_bank: dict[str, pd.Series] = {}
    coverage_rows = []

    for bank in BANKS:
        if not bank.public_listing or not bank.ticker:
            coverage_rows.append({
                "bank": bank.name,
                "ticker": "",
                "country": bank.country,
                "region": bank.region,
                "included": False,
                "first_date": "",
                "last_date": "",
                "observations": 0,
                "shares_source": "",
                "balance_sheet_source": "",
                "note": bank.note,
            })
            continue
        print(f"Processing {bank.name} ({bank.ticker})")
        time.sleep(REQUEST_SLEEP_SECONDS)
        price = prices.get(bank.name, pd.Series(dtype=float))
        bs = get_balance_sheet(bank)
        shares, shares_source = get_shares(bank, price.index) if not price.empty else (pd.Series(dtype=float), "")
        if price.empty or bs.empty or shares.dropna().empty:
            missing = []
            if price.empty:
                missing.append("price")
            if bs.empty:
                missing.append("balance sheet")
            if shares.dropna().empty:
                missing.append("shares")
            coverage_rows.append({
                "bank": bank.name,
                "ticker": bank.ticker,
                "country": bank.country,
                "region": bank.region,
                "included": False,
                "first_date": "",
                "last_date": "",
                "observations": 0,
                "shares_source": shares_source,
                "balance_sheet_source": "",
                "note": "Missing " + ", ".join(missing),
            })
            continue
        market_equity_local = price * shares
        book_debt_local = bs["book_debt"].reindex(price.index.union(bs.index)).sort_index().ffill().reindex(price.index)
        market_equity = convert_to_usd(market_equity_local, bank, fx_rates)
        book_debt = convert_to_usd(book_debt_local, bank, fx_rates)
        market_assets = book_debt + market_equity
        leverage = (market_assets / market_equity).replace([np.inf, -np.inf], np.nan).dropna()
        leverage = leverage[(leverage > 1.0) & (leverage < 100.0)]
        market_assets = market_assets.reindex(leverage.index)
        if leverage.empty:
            coverage_rows.append({
                "bank": bank.name,
                "ticker": bank.ticker,
                "country": bank.country,
                "region": bank.region,
                "included": False,
                "first_date": "",
                "last_date": "",
                "observations": 0,
                "shares_source": shares_source,
                "balance_sheet_source": str(bs.get("balance_sheet_source", pd.Series([""])).iloc[0]) if "balance_sheet_source" in bs else "",
                "note": "No valid market leverage after alignment.",
            })
            continue
        leverage_by_bank[bank.name] = leverage
        assets_by_bank[bank.name] = market_assets
        coverage_rows.append({
            "bank": bank.name,
            "ticker": bank.ticker,
            "country": bank.country,
            "region": bank.region,
            "included": True,
            "first_date": leverage.index.min().date().isoformat(),
            "last_date": leverage.index.max().date().isoformat(),
            "observations": int(leverage.shape[0]),
            "shares_source": shares_source,
            "balance_sheet_source": str(bs.get("balance_sheet_source", pd.Series([""])).iloc[0]) if "balance_sheet_source" in bs else "",
            "note": "",
        })

    leverage_panel = pd.DataFrame(leverage_by_bank).sort_index()
    asset_panel = pd.DataFrame(assets_by_bank).sort_index()
    valid_counts = leverage_panel.notna().sum(axis=1).combine(asset_panel.notna().sum(axis=1), min)
    min_count = max(5, min(MIN_BANKS, int(valid_counts.max()))) if not valid_counts.empty else 0
    base_index = valid_counts[valid_counts >= min_count].index
    factor_leverage = leverage_panel.reindex(base_index).ffill()
    factor_assets = asset_panel.reindex(base_index).ffill()
    if factor_leverage.empty:
        coverage = pd.DataFrame(coverage_rows)
        return pd.DataFrame(), pd.DataFrame(), coverage

    factor_start = base_index.min()
    stable_cols = [
        col for col in factor_leverage.columns
        if pd.notna(factor_leverage.loc[factor_start, col])
        and pd.notna(factor_assets.loc[factor_start, col])
        and factor_leverage[col].notna().mean() >= 0.95
        and factor_assets[col].notna().mean() >= 0.95
    ]
    if len(stable_cols) < 3:
        availability = (
            factor_leverage.notna().mean()
            .combine(factor_assets.notna().mean(), min)
            .sort_values(ascending=False)
        )
        stable_cols = availability.head(max(3, min_count)).index.tolist()

    factor_leverage = factor_leverage[stable_cols].ffill().dropna(axis=0, how="any")
    factor_assets = factor_assets[stable_cols].reindex(factor_leverage.index).ffill().dropna(axis=0, how="any")
    factor_leverage = factor_leverage.reindex(factor_assets.index).dropna(axis=0, how="any")
    factor_assets = factor_assets.reindex(factor_leverage.index)
    weights = factor_assets.div(factor_assets.sum(axis=1), axis=0)
    mean_weights = weights.mean(axis=0)
    mean_weights = mean_weights.div(mean_weights.sum())
    market_leverage = (factor_leverage * mean_weights).sum(axis=1).rename("market_leverage")

    leverage_changes = factor_leverage.diff().replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    common_changes, factor_frame, loadings = _pca_decomposition(leverage_changes, PCA_COMPONENTS)
    common_changes = common_changes.loc[leverage_changes.index, leverage_changes.columns]
    idiosyncratic_changes = leverage_changes.loc[common_changes.index, common_changes.columns] - common_changes
    component_weights = pd.DataFrame(
        np.tile(mean_weights.loc[common_changes.columns].to_numpy(dtype=float), (len(common_changes.index), 1)),
        index=common_changes.index,
        columns=common_changes.columns,
    )

    common_change = (common_changes * component_weights).sum(axis=1)
    idiosyncratic_change = (idiosyncratic_changes * component_weights).sum(axis=1)
    initial_leverage = float(market_leverage.iloc[0])

    output = pd.DataFrame(index=market_leverage.index)
    output["initial_leverage"] = initial_leverage
    output["common_factor_change"] = common_change.reindex(output.index).fillna(0.0)
    output["idiosyncratic_change"] = idiosyncratic_change.reindex(output.index).fillna(0.0)
    output["common_factor_component"] = output["common_factor_change"].cumsum()
    output["idiosyncratic_component"] = output["idiosyncratic_change"].cumsum()
    output["market_leverage"] = market_leverage.reindex(output.index)
    output["reconstructed_market_leverage"] = (
        output["initial_leverage"]
        + output["common_factor_component"]
        + output["idiosyncratic_component"]
    )
    output["reconstruction_error"] = output["market_leverage"] - output["reconstructed_market_leverage"]
    output["common_factor_index"] = _normalise_for_plot(output["common_factor_component"])
    output["idiosyncratic_component_index"] = _normalise_for_plot(output["idiosyncratic_component"])
    output["n_banks"] = len(stable_cols)
    output.index.name = "date"
    avg_weights = mean_weights.sort_values(ascending=False)
    exposure = pd.DataFrame({"bank": avg_weights.index})
    exposure["average_asset_share"] = exposure["bank"].map(avg_weights)
    if not loadings.empty:
        for column in loadings.columns:
            exposure[column] = exposure["bank"].map(loadings[column])
    exposure = exposure.sort_values("average_asset_share", ascending=False)
    coverage = pd.DataFrame(coverage_rows)
    return output, exposure, coverage


def _normalise_for_plot(series: pd.Series) -> pd.Series:
    data = series.dropna().astype(float)
    out = pd.Series(index=series.index, dtype=float)
    if data.empty:
        return out
    centered = data - data.iloc[0]
    scale = centered.std()
    if not np.isfinite(scale) or scale == 0:
        scale = data.diff().std()
    if not np.isfinite(scale) or scale == 0:
        out.loc[data.index] = centered
    else:
        out.loc[data.index] = centered / scale
    return out


def plot_market_decomposition(panel: pd.DataFrame, exposure: pd.DataFrame) -> None:
    palette = set_custom_style()
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    plot_data = panel.dropna(subset=["market_leverage"]).copy()

    initial = plot_data["initial_leverage"]
    common_top = initial + plot_data["common_factor_component"]
    reconstructed = plot_data["reconstructed_market_leverage"]

    ax.fill_between(
        plot_data.index,
        initial,
        common_top,
        color="#F8C98F",
        alpha=1.0,
        linewidth=0,
        antialiased=False,
        interpolate=True,
        label="Common factor",
        zorder=2,
    )
    ax.fill_between(
        plot_data.index,
        common_top,
        reconstructed,
        color="#94C9C3",
        alpha=1.0,
        linewidth=0,
        antialiased=False,
        interpolate=True,
        label="Idiosyncratic component",
        zorder=3,
    )
    ax.plot(
        plot_data.index,
        plot_data["market_leverage"],
        color=palette[0],
        linewidth=1.55,
        label="Market leverage",
        zorder=5,
    )
    ax.axhline(float(initial.iloc[0]), color=palette[8], linewidth=0.6, alpha=0.45, zorder=1)

    ax.set_title("G-SIB Market Leverage")
    ax.set_ylabel("Assets / market equity")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.margins(x=0.01)
    low_series = pd.concat([plot_data["market_leverage"], common_top, reconstructed], axis=1).min(axis=1)
    high_series = pd.concat([plot_data["market_leverage"], common_top, reconstructed], axis=1).max(axis=1)
    ymin = min(float(low_series.min()) - 2, float(initial.iloc[0]) - 18)
    ymax = float(high_series.max()) + 4
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="upper right", ncol=1, handlelength=1.6, borderaxespad=0.45, labelspacing=0.35)

    note = (
        "Source: Author's calculations using public filings, SEC Company Facts, Yahoo Finance market data, and FX rates.\n"
        "Components cumulate bank-level market-leverage changes from the initial value using mean asset weights."
    )
    fig.text(0.10, 0.060, note, ha="left", va="bottom", fontsize=6.2, color=palette[8], linespacing=1.15)
    plt.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.18)
    fig.savefig(CHART_DIR / "gsib_market_leverage_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(CACHE_DIR))
        yf.cache.set_cache_location(str(CACHE_DIR))
    except Exception:
        pass
    panel, exposure, coverage = build_market_panel()
    if panel.empty:
        coverage.to_csv(DATA_DIR / "gsib_market_coverage.csv", index=False)
        raise RuntimeError("No market leverage series was produced.")
    panel.to_csv(DATA_DIR / "gsib_market_leverage.csv", float_format="%.8f")
    plot_market_decomposition(panel, exposure)
    print(f"Wrote {DATA_DIR / 'gsib_market_leverage.csv'}")
    print(f"Wrote {CHART_DIR / 'gsib_market_leverage_decomposition.png'}")


if __name__ == "__main__":
    main()



