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
CACHE_DIR = Path(os.environ.get("TEMP", str(OUTPUT_DIR / ".cache"))) / "yfinance-cache-gsib"
SOURCE_CACHE_DIR = OUTPUT_DIR / ".cache" / "source-data"
REQUEST_SLEEP_SECONDS = float(os.environ.get("YF_REQUEST_SLEEP_SECONDS", "2"))
YF_TIMEOUT_SECONDS = float(os.environ.get("YF_TIMEOUT_SECONDS", "20"))
USE_HISTORICAL_SHARES = os.environ.get("GSIB_USE_HISTORICAL_SHARES", "0") == "1"

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
    Bank("Royal Bank of Canada", "RY.TO", "Canada", "North America"),
    Bank("Toronto-Dominion Bank", "TD.TO", "Canada", "North America"),
    Bank("HSBC", "HSBA.L", "United Kingdom", "Europe", price_scale=0.01),
    Bank("Barclays", "BARC.L", "United Kingdom", "Europe", price_scale=0.01),
    Bank("Standard Chartered", "STAN.L", "United Kingdom", "Europe", price_scale=0.01),
    Bank("BNP Paribas", "BNP.PA", "France", "Europe"),
    Bank("Credit Agricole", "ACA.PA", "France", "Europe"),
    Bank("Societe Generale", "GLE.PA", "France", "Europe"),
    Bank("Deutsche Bank", "DBK.DE", "Germany", "Europe"),
    Bank("ING Groep", "INGA.AS", "Netherlands", "Europe"),
    Bank("Banco Santander", "SAN.MC", "Spain", "Europe"),
    Bank("UBS Group", "UBSG.SW", "Switzerland", "Europe"),
    Bank("Mitsubishi UFJ Financial Group", "8306.T", "Japan", "Asia"),
    Bank("Sumitomo Mitsui Financial Group", "8316.T", "Japan", "Asia"),
    Bank("Mizuho Financial Group", "8411.T", "Japan", "Asia"),
    Bank("Industrial and Commercial Bank of China", "1398.HK", "China", "Asia"),
    Bank("China Construction Bank", "0939.HK", "China", "Asia"),
    Bank("Bank of China", "3988.HK", "China", "Asia"),
    Bank("Agricultural Bank of China", "1288.HK", "China", "Asia"),
    Bank("Bank of Communications", "3328.HK", "China", "Asia"),
    Bank(
        "Groupe BPCE",
        None,
        "France",
        "Europe",
        public_listing=False,
        note="Excluded because the group is not publicly listed.",
    ),
    Bank(
        "Credit Mutuel",
        None,
        "France",
        "Europe",
        public_listing=False,
        note="Excluded because the group is not publicly listed.",
    ),
]


ASSET_LABELS = [
    "Total Assets",
    "TotalAssets",
]
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
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "black",
            "axes.linewidth": 1,
            "axes.grid": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "font.size": 10,
            "lines.linewidth": 1.5,
            "figure.figsize": (6, 4.2),
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
    return ticker.replace(".", "_").replace("-", "_")


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


def get_prices(banks: list[Bank]) -> dict[str, pd.Series]:
    listed = [bank for bank in banks if bank.public_listing and bank.ticker]
    tickers = [bank.ticker for bank in listed if bank.ticker]
    raw = _download_with_retries(tickers, group_by="ticker")
    prices: dict[str, pd.Series] = {}
    if raw.empty:
        return prices
    for bank in listed:
        assert bank.ticker is not None
        close = pd.Series(dtype=float)
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if bank.ticker in raw.columns.get_level_values(0):
                    close = raw[(bank.ticker, "Close")]
                elif "Close" in raw.columns.get_level_values(0):
                    close = raw["Close"][bank.ticker]
            elif "Close" in raw:
                close = raw["Close"]
        except Exception:
            close = pd.Series(dtype=float)
        if close is None or close.empty:
            continue
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = pd.to_numeric(close, errors="coerce").dropna()
        if not close.empty:
            prices[bank.name] = close * bank.price_scale
    return prices

def get_balance_sheet(bank: Bank) -> pd.DataFrame:
    assert bank.ticker is not None
    cache_path = SOURCE_CACHE_DIR / f"balance_sheet_{_cache_key(bank.ticker)}.csv"
    cached = _read_cached_frame(cache_path)
    if not cached.empty:
        return cached
    ticker = yf.Ticker(bank.ticker)
    statements = []
    for attr in ["quarterly_balance_sheet", "balance_sheet"]:
        try:
            freq = "quarterly" if attr.startswith("quarterly") else "yearly"
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
        out["statement_frequency"] = "quarterly" if attr.startswith("quarterly") else "annual"
        statements.append(out)
        if attr.startswith("quarterly"):
            break
    if not statements:
        return pd.DataFrame()
    out = statements[0].dropna(subset=["total_assets", "book_equity"])
    out = out[(out["total_assets"] > 0) & (out["book_equity"] > 0)]
    out["book_debt"] = out["total_assets"] - out["book_equity"]
    out["book_leverage"] = out["total_assets"] / out["book_equity"]
    _write_cached_frame(out, cache_path)
    return out



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


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    DATA_DIR.mkdir(exist_ok=True)
    CHART_DIR.mkdir(exist_ok=True)

    daily_market_lev: dict[str, pd.Series] = {}
    daily_market_equity: dict[str, pd.Series] = {}
    daily_book_debt: dict[str, pd.Series] = {}
    quarterly_book_lev: dict[str, pd.Series] = {}
    quarterly_assets: dict[str, pd.Series] = {}
    quarterly_equity: dict[str, pd.Series] = {}
    coverage_rows = []
    prices = get_prices(BANKS)

    for bank in BANKS:
        if not bank.public_listing:
            coverage_rows.append(
                {
                    "bank": bank.name,
                    "ticker": "",
                    "country": bank.country,
                    "region": bank.region,
                    "included_market_leverage": False,
                    "included_book_leverage": False,
                    "first_market_date": "",
                    "last_market_date": "",
                    "book_observations": 0,
                    "market_observations": 0,
                    "shares_source": "",
                    "note": bank.note,
                }
            )
            continue

        print(f"Processing {bank.name} ({bank.ticker})")
        time.sleep(REQUEST_SLEEP_SECONDS)
        try:
            bs = get_balance_sheet(bank)
            price = prices.get(bank.name, pd.Series(dtype=float))
        except Exception as exc:
            coverage_rows.append(
                {
                    "bank": bank.name,
                    "ticker": bank.ticker,
                    "country": bank.country,
                    "region": bank.region,
                    "included_market_leverage": False,
                    "included_book_leverage": False,
                    "first_market_date": "",
                    "last_market_date": "",
                    "book_observations": 0,
                    "market_observations": 0,
                    "shares_source": "",
                    "note": f"Download failed: {exc}",
                }
            )
            continue

        has_book = not bs.empty
        has_price = not price.empty
        shares, share_source = get_shares(bank, price.index) if has_price else (pd.Series(dtype=float), "")

        if has_book:
            quarterly_book_lev[bank.name] = bs["book_leverage"]
            quarterly_assets[bank.name] = bs["total_assets"]
            quarterly_equity[bank.name] = bs["book_equity"]

        has_market = has_book and has_price and shares.notna().any()
        market_obs = 0
        first_market = ""
        last_market = ""
        note = ""
        if has_market:
            market_equity = price * shares
            debt_daily = bs["book_debt"].reindex(price.index.union(bs.index)).sort_index().ffill().reindex(price.index)
            leverage = (debt_daily + market_equity) / market_equity
            leverage = leverage.replace([np.inf, -np.inf], np.nan).dropna()
            market_equity = market_equity.reindex(leverage.index)
            debt_daily = debt_daily.reindex(leverage.index)
            if not leverage.empty:
                daily_market_lev[bank.name] = leverage
                daily_market_equity[bank.name] = market_equity
                daily_book_debt[bank.name] = debt_daily
                market_obs = int(leverage.shape[0])
                first_market = leverage.index.min().date().isoformat()
                last_market = leverage.index.max().date().isoformat()
            else:
                has_market = False
                note = "Market leverage could not be computed after aligning price, shares, and balance-sheet data."
        else:
            missing = []
            if not has_book:
                missing.append("balance-sheet data")
            if not has_price:
                missing.append("price data")
            if has_price and not shares.notna().any():
                missing.append("shares outstanding")
            note = "Missing " + ", ".join(missing) + "."

        coverage_rows.append(
            {
                "bank": bank.name,
                "ticker": bank.ticker,
                "country": bank.country,
                "region": bank.region,
                "included_market_leverage": bool(has_market and market_obs > 0),
                "included_book_leverage": bool(has_book),
                "first_market_date": first_market,
                "last_market_date": last_market,
                "book_observations": int(bs.shape[0]) if has_book else 0,
                "market_observations": market_obs,
                "shares_source": share_source,
                "note": note,
            }
        )

    daily = pd.DataFrame(daily_market_lev).sort_index()
    debt = pd.DataFrame(daily_book_debt).sort_index()
    market_equity = pd.DataFrame(daily_market_equity).sort_index()
    if not daily.empty:
        daily["aggregate_market_leverage"] = (debt.sum(axis=1, min_count=1) + market_equity.sum(axis=1, min_count=1)) / market_equity.sum(axis=1, min_count=1)
        daily.index.name = "date"

    quarterly = pd.DataFrame(quarterly_book_lev).sort_index()
    assets = pd.DataFrame(quarterly_assets).sort_index()
    equity = pd.DataFrame(quarterly_equity).sort_index()
    if not quarterly.empty:
        quarterly["aggregate_book_leverage"] = assets.sum(axis=1, min_count=1) / equity.sum(axis=1, min_count=1)
        quarterly.index.name = "date"

    coverage = pd.DataFrame(coverage_rows)
    return daily, quarterly, coverage


def plot_overview(daily: pd.DataFrame, quarterly: pd.DataFrame) -> None:
    palette = set_custom_style()
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)

    if "aggregate_market_leverage" in daily:
        ax.plot(
            daily.index,
            daily["aggregate_market_leverage"],
            color=palette[0],
            label="Daily market leverage",
            zorder=2,
        )
    if "aggregate_book_leverage" in quarterly:
        ax.plot(
            quarterly.index,
            quarterly["aggregate_book_leverage"],
            color=palette[1],
            label="Quarterly book leverage",
            zorder=3,
        )

    ax.set_title("Public G-SIB Leverage Proxies")
    ax.set_ylabel("Assets / equity")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.axhline(0, color=palette[8], lw=0.8, alpha=0.6)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    note = (
        "Source: Author's calculations using Yahoo Finance market data and financial statements. "
        "The sample includes publicly listed G-SIB groups with sufficient price, shares, and balance-sheet data."
    )
    fig.text(0.10, 0.035, note, ha="left", va="bottom", fontsize=6.5, color=palette[8], wrap=True)
    plt.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.18)
    fig.savefig(CHART_DIR / "gsib_leverage_overview.png", bbox_inches="tight")
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
    daily, quarterly, coverage = build_panel()
    if daily.empty or quarterly.empty:
        coverage.to_csv(DATA_DIR / "gsib_coverage.csv", index=False)
        raise RuntimeError("No leverage series were produced. Check data-source availability or rate limits; empty output files were not written.")
    daily.to_csv(DATA_DIR / "gsib_market_leverage_daily.csv", float_format="%.6f")
    quarterly.to_csv(DATA_DIR / "gsib_book_leverage_quarterly.csv", float_format="%.6f")
    coverage.to_csv(DATA_DIR / "gsib_coverage.csv", index=False)
    plot_overview(daily, quarterly)
    print(f"Wrote {DATA_DIR / 'gsib_market_leverage_daily.csv'}")
    print(f"Wrote {DATA_DIR / 'gsib_book_leverage_quarterly.csv'}")
    print(f"Wrote {DATA_DIR / 'gsib_coverage.csv'}")
    print(f"Wrote {CHART_DIR / 'gsib_leverage_overview.png'}")


if __name__ == "__main__":
    main()
