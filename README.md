# Regime Monitor

A local dashboard for US markets in five sections (switch in the sidebar):

* **Summary**: one page with key levels, a regime map of every model, the plain-English read and what's stretched.
* **Cross-asset**: growth × inflation quadrant, Indicator lab (tests which indicators drive the quadrant), market-implied macro regime, "what happened next" event study.
* **Equities**: trend and Markov regimes, driver attribution, breadth and rotation, volatility, mega-cap leaders.
* **Fixed income**: Treasury curve and curve regimes, what's moving the 10Y, Fed pricing proxy, credit, rate vol.
* **Crypto**: Bitcoin/Ether level ladders, drawdown and volatility, macro correlations, altcoin breadth, spot-ETF activity.

## Run it

```bash
cd ~/"Documents/Analysis/Macro Fund/Systematic Macro/equity_dashboard"
FRED_API_KEY=your_fred_key python3 run.py
```

The FRED key makes the rates, credit and macro downloads reliable (free key: https://fredaccount.stlouisfed.org/apikeys).
Without it the app still runs but falls back to FRED's slower public download, which often times out.
Alternatively save the key in a file `fred_api_key.txt` in this folder.

Use `python3` instead of `python` if that's how Python is called on your machine (macOS usually).
The first run installs any missing packages and downloads about 12 years of daily data (~20 s).
Your browser then opens at http://localhost:8501. Stop the app with **Ctrl+C**.

* `python3 run.py --demo` runs on synthetic data with no internet connection.
* Press **Refresh data** in the sidebar after the US close to pull the latest session.

Requires Python 3.9+. To keep packages separate from your system Python, create a virtual environment first
(optional):

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

## Files

| File | What's in it |
|---|---|
| `run.py` | Launcher: installs dependencies, starts Streamlit |
| `app.py` | The dashboard layout and charts |
| `models.py` | Every model, in plain pandas/numpy |
| `data.py` | Ticker universe, Yahoo download, daily cache, demo data |
| `macro.py` | Growth × inflation quadrant: FRED download, model, base rates |
| `fixed_income.py` | Curve, curve regimes, 10Y splits, Fed pricing proxy, credit, rate vol |
| `fred.py` | FRED client (API key or public CSV) |
| `ui.py` | Shared colours, cards and chart layout |
| `crypto.py` | Level ladder, drawdown/vol, macro links, alt breadth, ETF activity |
| `indicators.py` | Candidate macro indicators, transforms, real-time composites |
| `lab.py` | Indicator lab tests (scorecard, leave-one-out, walk-forward, asset relevance) |
| `lab_config.json` | Your applied indicator selection (created by the lab; delete to reset) |
| `data_cache/` | Today's downloaded closes (created automatically) |

To add or remove tickers, edit the dictionaries at the top of `data.py`.

## Data notes

* Yahoo Finance (via `yfinance`) is free but unofficial. It can rate-limit, change format, or have bad prints.
  If a download fails, the app falls back to the last cache and labels it STALE.
* Macro data for the quadrant comes from FRED (St. Louis Fed), monthly, free, no API key.
* Option skew, dealer gamma and CFTC positioning are not included, because they need paid or non-daily data.

Educational tool: not investment advice.
