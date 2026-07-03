# Public Data Sources

This project uses public data only. It does not include commercial bank-accounting data or raw financial-statement PDFs.

## Automated Sources

- SEC Company Facts API: used by `scripts/fetch_sec_balance_sheets.py` for SEC-reporting G-SIBs listed in `source_data/sec_reporting_gsibs.csv`.
- Yahoo Finance via `yfinance`: used for listed equity prices, shares outstanding, and FX rates.

## Curated Public Filing Sources

- `source_data/official_balance_sheets.csv` contains the curated public balance-sheet panel used by the script. Each row includes source metadata where available.
- `source_data/financial_statement_links.csv` documents older public filing links used during manual collection for non-SEC or historically incomplete observations.
- Raw filing PDFs and ZIPs are intentionally excluded from the repository to keep the public project lightweight and to avoid redistributing source documents.

## Excluded Sources

Private or commercial datasets used for local validation are not included and are not required to run the project.
