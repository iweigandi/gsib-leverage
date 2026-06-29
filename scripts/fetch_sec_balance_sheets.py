from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_BALANCE_SHEETS = ROOT / "source_data" / "official_balance_sheets.csv"
SEC_MAPPING = ROOT / "source_data" / "sec_reporting_gsibs.csv"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
USER_AGENT = os.environ.get("SEC_USER_AGENT", "iweigandi gsib-public-leverage research iweigandi@users.noreply.github.com")

SEC_CIK_FALLBACK = {
    "JPM": "0000019617",
    "BAC": "0000070858",
    "C": "0000831001",
    "GS": "0000886982",
    "MS": "0000895421",
    "WFC": "0000072971",
    "BK": "0001390777",
    "STT": "0000093751",
}
SEC_TICKERS = [
    ("JPMorgan Chase", "JPM"),
    ("Bank of America", "BAC"),
    ("Citigroup", "C"),
    ("Goldman Sachs", "GS"),
    ("Morgan Stanley", "MS"),
    ("Wells Fargo", "WFC"),
    ("Bank of New York Mellon", "BK"),
    ("State Street", "STT"),
]
ASSET_TAGS = ["Assets"]
EQUITY_TAGS = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
FORMS = {"10-K", "10-Q"}
OFFICIAL_COLUMNS = [
    "bank", "ticker", "period_end", "total_assets", "book_equity", "currency",
    "statement_frequency", "source_document", "source_url", "source_page", "notes",
]


def fetch_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json", "Accept-Encoding": "identity"})
    with urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def facts_for_tag(companyfacts: dict, tags: list[str]) -> tuple[pd.DataFrame, str]:
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        units = facts.get(tag, {}).get("units", {})
        if "USD" not in units:
            continue
        rows = pd.DataFrame(units["USD"])
        if rows.empty or "form" not in rows or "end" not in rows or "val" not in rows:
            continue
        rows = rows[rows["form"].isin(FORMS)].copy()
        rows["period_end"] = pd.to_datetime(rows["end"], errors="coerce")
        rows["value"] = pd.to_numeric(rows["val"], errors="coerce")
        rows["filed_date"] = pd.to_datetime(rows.get("filed"), errors="coerce")
        rows = rows.dropna(subset=["period_end", "value"])
        rows = rows.sort_values(["period_end", "filed_date"]).drop_duplicates("period_end", keep="last")
        if not rows.empty:
            return rows[["period_end", "value", "form", "filed", "fy", "fp"]], tag
    return pd.DataFrame(), ""


def infer_frequency(fp: object, form: object) -> str:
    fp = str(fp or "").upper()
    form = str(form or "").upper()
    return "annual" if form == "10-K" or fp == "FY" else "quarterly"


def build_sec_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    output_rows = []
    mapping_rows = []
    for bank, ticker in SEC_TICKERS:
        cik = SEC_CIK_FALLBACK[ticker]
        url = SEC_COMPANYFACTS_URL.format(cik=cik)
        map_row = {"bank": bank, "ticker": ticker, "cik": cik, "sec_scope": "US 10-K/10-Q filer", "notes": ""}
        try:
            companyfacts = fetch_json(url)
        except Exception as exc:
            map_row["notes"] = f"companyfacts download failed: {exc}"
            mapping_rows.append(map_row)
            continue
        assets, asset_tag = facts_for_tag(companyfacts, ASSET_TAGS)
        equity, equity_tag = facts_for_tag(companyfacts, EQUITY_TAGS)
        if assets.empty or equity.empty:
            map_row["notes"] = "missing Assets or equity tags"
            mapping_rows.append(map_row)
            continue
        merged = assets.merge(equity, on="period_end", suffixes=("_assets", "_equity"))
        merged = merged[(merged["value_assets"] > 0) & (merged["value_equity"] > 0)]
        map_row["notes"] = f"rows={len(merged)}; assets={asset_tag}; equity={equity_tag}"
        mapping_rows.append(map_row)
        for _, row in merged.iterrows():
            output_rows.append({
                "bank": bank,
                "ticker": ticker,
                "period_end": row["period_end"].date().isoformat(),
                "total_assets": int(row["value_assets"]),
                "book_equity": int(row["value_equity"]),
                "currency": "USD",
                "statement_frequency": infer_frequency(row.get("fp_assets"), row.get("form_assets")),
                "source_document": f"SEC Company Facts; assets={asset_tag}; equity={equity_tag}",
                "source_url": url,
                "source_page": "",
                "notes": f"form={row.get('form_assets')}; filed={row.get('filed_assets')}",
            })
    return pd.DataFrame(output_rows, columns=OFFICIAL_COLUMNS), pd.DataFrame(mapping_rows)


def write_official_table(sec_rows: pd.DataFrame) -> None:
    OFFICIAL_BALANCE_SHEETS.parent.mkdir(parents=True, exist_ok=True)
    if OFFICIAL_BALANCE_SHEETS.exists():
        existing = pd.read_csv(OFFICIAL_BALANCE_SHEETS)
    else:
        existing = pd.DataFrame(columns=OFFICIAL_COLUMNS)
    if not existing.empty and "source_document" in existing:
        existing = existing[~existing["source_document"].astype(str).str.contains("SEC Company Facts", na=False)]
    combined = pd.concat([existing, sec_rows], ignore_index=True).reindex(columns=OFFICIAL_COLUMNS)
    combined = combined.sort_values(["bank", "period_end"])
    combined.to_csv(OFFICIAL_BALANCE_SHEETS, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SEC Company Facts balance-sheet observations for US G-SIBs.")
    parser.add_argument("--write", action="store_true", help="Write rows into source_data/official_balance_sheets.csv")
    args = parser.parse_args()
    sec_rows, mapping = build_sec_rows()
    SEC_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(SEC_MAPPING, index=False)
    print(f"SEC rows: {len(sec_rows)}")
    if not sec_rows.empty:
        print(sec_rows.groupby("bank")["period_end"].agg(["min", "max", "count"]).to_string())
    if args.write:
        if sec_rows.empty:
            raise SystemExit("No SEC rows were fetched; official_balance_sheets.csv was not modified.")
        write_official_table(sec_rows)
        print(f"Wrote {OFFICIAL_BALANCE_SHEETS}")
    else:
        print("Dry run only. Use --write to update official_balance_sheets.csv.")


if __name__ == "__main__":
    main()
