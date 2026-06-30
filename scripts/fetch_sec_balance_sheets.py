from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_BALANCE_SHEETS = ROOT / "source_data" / "official_balance_sheets.csv"
SEC_MAPPING = ROOT / "source_data" / "sec_reporting_gsibs.csv"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
USER_AGENT = os.environ.get("SEC_USER_AGENT", "iweigandi gsib-public-leverage research iweigandi@users.noreply.github.com")

# SEC scope is broader than US domestic filers. The first group files 10-K/10-Q
# using US GAAP; the rest are SEC registrants/ADR issuers that generally file
# 20-F, 40-F, or 6-K and may use IFRS or local reporting tags.
SEC_REPORTING_GSIBS = [
    {"bank": "JPMorgan Chase", "ticker": "JPM", "cik": "0000019617", "sec_scope": "US domestic 10-K/10-Q filer"},
    {"bank": "Bank of America", "ticker": "BAC", "cik": "0000070858", "sec_scope": "US domestic 10-K/10-Q filer"},
    {"bank": "Citigroup", "ticker": "C", "cik": "0000831001", "sec_scope": "US domestic 10-K/10-Q filer"},
    {"bank": "Goldman Sachs", "ticker": "GS", "cik": "0000886982", "sec_scope": "US domestic 10-K/10-Q filer"},
    {"bank": "Morgan Stanley", "ticker": "MS", "cik": "0000895421", "sec_scope": "US domestic 10-K/10-Q filer"},
    {"bank": "Wells Fargo", "ticker": "WFC", "cik": "0000072971", "sec_scope": "US domestic 10-K/10-Q filer"},
    {"bank": "Bank of New York Mellon", "ticker": "BK", "cik": "0001390777", "sec_scope": "US domestic 10-K/10-Q filer"},
    {"bank": "State Street", "ticker": "STT", "cik": "0000093751", "sec_scope": "US domestic 10-K/10-Q filer"},
    {"bank": "Royal Bank of Canada", "ticker": "RY", "cik": "0001000275", "sec_scope": "Canadian SEC registrant / 40-F"},
    {"bank": "Toronto-Dominion Bank", "ticker": "TD", "cik": "0000947263", "sec_scope": "Canadian SEC registrant / 40-F"},
    {"bank": "HSBC", "ticker": "HSBC", "cik": "0001089113", "sec_scope": "Foreign private issuer / 20-F"},
    {"bank": "Barclays", "ticker": "BCS", "cik": "0000312069", "sec_scope": "Foreign private issuer / 20-F"},
    {"bank": "Deutsche Bank", "ticker": "DB", "cik": "0001159508", "sec_scope": "Foreign private issuer / 20-F"},
    {"bank": "ING Groep", "ticker": "ING", "cik": "0001039765", "sec_scope": "Foreign private issuer / 20-F"},
    {"bank": "Banco Santander", "ticker": "SAN", "cik": "0000891478", "sec_scope": "Foreign private issuer / 20-F"},
    {"bank": "UBS Group", "ticker": "UBS", "cik": "0001610520", "sec_scope": "Foreign private issuer / 20-F"},
    {"bank": "Mitsubishi UFJ Financial Group", "ticker": "MUFG", "cik": "0000067088", "sec_scope": "Foreign private issuer / 20-F"},
    {"bank": "Sumitomo Mitsui Financial Group", "ticker": "SMFG", "cik": "0001022837", "sec_scope": "Foreign private issuer / 20-F"},
    {"bank": "Mizuho Financial Group", "ticker": "MFG", "cik": "0001335730", "sec_scope": "Foreign private issuer / 20-F"},
]

ASSET_TAGS = ["Assets"]
EQUITY_TAGS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "Equity",
    "EquityAttributableToOwnersOfParent",
    "EquityAttributableToEquityHoldersOfParent",
]
FORMS = {"10-K", "10-Q", "20-F", "40-F", "6-K"}
OFFICIAL_COLUMNS = [
    "bank", "ticker", "period_end", "total_assets", "book_equity", "currency",
    "statement_frequency", "source_document", "source_url", "source_page", "notes",
]


def fetch_json(url: str) -> dict:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Encoding": "identity",
    }
    response = requests.get(url, headers=headers, timeout=45)
    response.raise_for_status()
    return response.json()


def _available_facts(companyfacts: dict, tag: str) -> list[dict]:
    out = []
    for taxonomy, facts in companyfacts.get("facts", {}).items():
        tag_data = facts.get(tag, {})
        for unit, rows in tag_data.get("units", {}).items():
            frame = pd.DataFrame(rows)
            if frame.empty or "form" not in frame or "end" not in frame or "val" not in frame:
                continue
            frame = frame[frame["form"].isin(FORMS)].copy()
            if frame.empty:
                continue
            frame["period_end"] = pd.to_datetime(frame["end"], errors="coerce")
            frame["value"] = pd.to_numeric(frame["val"], errors="coerce")
            frame["filed_date"] = pd.to_datetime(frame.get("filed"), errors="coerce")
            frame = frame.dropna(subset=["period_end", "value"])
            frame = frame.sort_values(["period_end", "filed_date"]).drop_duplicates("period_end", keep="last")
            if not frame.empty:
                out.append({"taxonomy": taxonomy, "tag": tag, "unit": unit, "rows": frame})
    return out


def facts_for_tag(companyfacts: dict, tags: list[str]) -> tuple[pd.DataFrame, str, str, str]:
    for tag in tags:
        matches = _available_facts(companyfacts, tag)
        if not matches:
            continue
        # Prefer monetary reporting units over shares or pure counts. Company Facts
        # usually exposes one monetary unit per issuer for these balance-sheet tags.
        monetary = [m for m in matches if not str(m["unit"]).lower().endswith("shares")]
        chosen = max(monetary or matches, key=lambda m: len(m["rows"]))
        rows = chosen["rows"]
        return rows[["period_end", "value", "form", "filed", "fy", "fp"]], chosen["taxonomy"], chosen["tag"], chosen["unit"]
    return pd.DataFrame(), "", "", ""


def infer_frequency(fp: object, form: object) -> str:
    fp = str(fp or "").upper()
    form = str(form or "").upper()
    return "annual" if form in {"10-K", "20-F", "40-F"} or fp == "FY" else "quarterly_or_interim"


def build_sec_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    output_rows = []
    mapping_rows = []
    for item in SEC_REPORTING_GSIBS:
        bank = item["bank"]
        ticker = item["ticker"]
        cik = item["cik"]
        url = SEC_COMPANYFACTS_URL.format(cik=cik)
        map_row = {"bank": bank, "ticker": ticker, "cik": cik, "sec_scope": item["sec_scope"], "notes": ""}
        try:
            companyfacts = fetch_json(url)
        except Exception as exc:
            map_row["notes"] = f"companyfacts download failed: {exc}"
            mapping_rows.append(map_row)
            continue
        assets, asset_taxonomy, asset_tag, asset_unit = facts_for_tag(companyfacts, ASSET_TAGS)
        equity, equity_taxonomy, equity_tag, equity_unit = facts_for_tag(companyfacts, EQUITY_TAGS)
        if assets.empty or equity.empty:
            map_row["notes"] = "missing compatible Assets or equity tags"
            mapping_rows.append(map_row)
            continue
        if asset_unit != equity_unit:
            map_row["notes"] = f"asset/equity units differ: assets={asset_unit}; equity={equity_unit}"
            mapping_rows.append(map_row)
            continue
        merged = assets.merge(equity, on="period_end", suffixes=("_assets", "_equity"))
        merged = merged[(merged["value_assets"] > 0) & (merged["value_equity"] > 0)]
        map_row["notes"] = (
            f"rows={len(merged)}; assets={asset_taxonomy}:{asset_tag}; "
            f"equity={equity_taxonomy}:{equity_tag}; unit={asset_unit}"
        )
        mapping_rows.append(map_row)
        for _, row in merged.iterrows():
            output_rows.append({
                "bank": bank,
                "ticker": ticker,
                "period_end": row["period_end"].date().isoformat(),
                "total_assets": int(row["value_assets"]),
                "book_equity": int(row["value_equity"]),
                "currency": asset_unit,
                "statement_frequency": infer_frequency(row.get("fp_assets"), row.get("form_assets")),
                "source_document": f"SEC Company Facts; assets={asset_taxonomy}:{asset_tag}; equity={equity_taxonomy}:{equity_tag}",
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
    parser = argparse.ArgumentParser(description="Fetch SEC Company Facts balance-sheet observations for SEC-reporting G-SIBs.")
    parser.add_argument("--write", action="store_true", help="Write rows into source_data/official_balance_sheets.csv")
    args = parser.parse_args()
    sec_rows, mapping = build_sec_rows()
    SEC_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(SEC_MAPPING, index=False)
    print(f"SEC-reporting banks checked: {len(SEC_REPORTING_GSIBS)}")
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
