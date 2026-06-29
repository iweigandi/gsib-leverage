# Public G-SIB Leverage Proxies

This repository estimates public-data leverage proxies for global systemically important banks (G-SIBs) with listed equity. The accounting layer combines machine-readable market data with audited balance-sheet observations curated from official filings where free fundamentals APIs are too shallow.

- **Quarterly book leverage:** total assets divided by book equity, using quarterly or interim balance-sheet data.
- **Daily market leverage:** book debt plus daily market equity, divided by daily market equity.
- **Leverage factors:** aggregate leverage innovations and a PCA-adjusted granular instrumental-variable series constructed from bank-level leverage innovations.

The daily measure captures high-frequency changes in the market value of bank capital, while the book measure follows the slower-moving accounting balance sheet. The sample follows the Financial Stability Board G-SIB universe and includes banks with publicly listed equity and sufficient data availability.

## Outputs

- [`data/gsib_market_leverage_daily.csv`](data/gsib_market_leverage_daily.csv): bank-level and aggregate daily market leverage.
- [`data/gsib_book_leverage_quarterly.csv`](data/gsib_book_leverage_quarterly.csv): bank-level and aggregate quarterly book leverage.
- [`data/gsib_leverage_factors.csv`](data/gsib_leverage_factors.csv): aggregate leverage, AR(1) shock, and PCA-adjusted granular IV series.
- [`data/gsib_coverage.csv`](data/gsib_coverage.csv): listing status, data coverage, and balance-sheet source by institution.
- [`source_data/official_balance_sheets.csv`](source_data/official_balance_sheets.csv): curated official-filing balance-sheet observations. Rows in this file override Yahoo Finance fundamentals.
- [`source_data/official_filing_sources.csv`](source_data/official_filing_sources.csv): official filing/source pages for non-SEC G-SIBs targeted for manual extraction.
- [`chart/gsib_leverage_overview.png`](chart/gsib_leverage_overview.png): aggregate market and book leverage chart.
- [`chart/gsib_leverage_factors.png`](chart/gsib_leverage_factors.png): aggregate leverage, cumulative PCA-adjusted GIV, and aggregate leverage shock chart.
- [`Methodology.pdf`](Methodology.pdf): short methodological note.

## Method

For bank \(i\) at balance-sheet date \(q\), book leverage is

\[
BL_{i,q} = \frac{A_{i,q}}{E^{book}_{i,q}},
\]

where \(A_{i,q}\) is total assets and \(E^{book}_{i,q}\) is book equity. Daily market leverage is

\[
ML_{i,t} = \frac{D^{book}_{i,q(t)} + E^{mkt}_{i,t}}{E^{mkt}_{i,t}},
\]

where \(D^{book}_{i,q(t)} = A_{i,q(t)} - E^{book}_{i,q(t)}\) is book debt from the latest available balance-sheet observation and \(E^{mkt}_{i,t}\) is daily market capitalization. Balance-sheet observations are read first from the curated official-filing table and then from Yahoo Finance when no curated observation is available. Non-USD balance-sheet and market-equity components are converted to US dollars before aggregation.

These measures are proxies. They are not regulatory leverage ratios and should not be read as substitutes for supervisory capital metrics.

## Reproduce

```bash
pip install -r requirements.txt
python gsib_leverage.py
```

To compile the methodology note:

```bash
cd methodology
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The GitHub Action refreshes the data and chart monthly. The script batches price downloads, caches source data across runs, and keeps curated official-filing inputs under version control. If Yahoo Finance rate-limits a run, the script fails before committing empty leverage files.
