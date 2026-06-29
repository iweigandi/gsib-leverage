# Public G-SIB Leverage Proxies

This repository estimates two public-data leverage proxies for global systemically important banks (G-SIBs) with listed equity:

- **Quarterly book leverage:** total assets divided by book equity, using quarterly balance-sheet data.
- **Daily market leverage:** book debt plus daily market equity, divided by daily market equity.

The daily measure is designed to capture high-frequency changes in the market value of bank capital, while the quarterly measure follows the slower-moving accounting balance sheet. The sample follows the Financial Stability Board G-SIB universe and includes banks with publicly listed equity and sufficient data availability.

## Outputs

- [`data/gsib_market_leverage_daily.csv`](data/gsib_market_leverage_daily.csv): bank-level and aggregate daily market leverage.
- [`data/gsib_book_leverage_quarterly.csv`](data/gsib_book_leverage_quarterly.csv): bank-level and aggregate quarterly book leverage.
- [`data/gsib_coverage.csv`](data/gsib_coverage.csv): coverage, listing status, and data availability by institution.
- [`chart/gsib_leverage_overview.png`](chart/gsib_leverage_overview.png): aggregate market and book leverage chart.
- [`Methodology.pdf`](Methodology.pdf): short methodological note.

## Method

For bank \(i\) at date \(t\), book leverage is

\[
BL_{i,t} = \frac{A_{i,t}}{E^{book}_{i,t}},
\]

where \(A_{i,t}\) is total assets and \(E^{book}_{i,t}\) is book equity. Daily market leverage is

\[
ML_{i,t} = \frac{D^{book}_{i,q(t)} + E^{mkt}_{i,t}}{E^{mkt}_{i,t}},
\]

where \(D^{book}_{i,q(t)} = A_{i,q(t)} - E^{book}_{i,q(t)}\) is book debt from the latest available balance-sheet observation and \(E^{mkt}_{i,t}\) is daily market capitalization. Non-USD balance-sheet and market-equity components are converted to US dollars before aggregation. Aggregate leverage is computed from aggregate assets, book equity, book debt, and market equity rather than as an equal-weighted average across banks.

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

The GitHub Action is configured to refresh the data and chart monthly. The script batches price downloads, caches statement and shares data across runs, and uses a request delay for the remaining Yahoo Finance calls. If the upstream provider still rate-limits a run, the script fails before committing empty leverage files; in that case, rerun the workflow later.
