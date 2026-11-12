"""
Data layer: which tickers we use, how we download them, and an offline demo set.

Live data comes from Yahoo Finance (free, via the `yfinance` package), daily closes.
Downloads are cached in ./data_cache so the dashboard opens instantly the second time
the same day. Use the "Refresh data" button in the sidebar to force a new download.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent / "data_cache"
HISTORY_YEARS = 12  # long enough for 3y z-scores + event studies with decent samples

# ---------------------------------------------------------------------------
# Universe. key = name used in the dashboard, value = Yahoo ticker
# ---------------------------------------------------------------------------
INDICES = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Russell 2000": "^RUT",
    "Dow Jones": "^DJI",
    "S&P 500 Equal Wt (RSP)": "RSP",
}
SECTORS = {
    "Tech (XLK)": "XLK", "Comm Svcs (XLC)": "XLC", "Cons Disc (XLY)": "XLY",
    "Financials (XLF)": "XLF", "Industrials (XLI)": "XLI", "Health (XLV)": "XLV",
    "Staples (XLP)": "XLP", "Utilities (XLU)": "XLU", "Energy (XLE)": "XLE",
    "Materials (XLB)": "XLB", "Real Estate (XLRE)": "XLRE", "Semis (SMH)": "SMH",
}
STYLE = {  # building blocks for style / breadth ratios
    "SPY": "SPY", "MTUM": "MTUM", "SPHB": "SPHB", "SPLV": "SPLV", "IWM": "IWM",
}
DRIVERS = {  # cross-asset drivers used by SBC + attribution
    "US 10Y yield": "^TNX",
    "Dollar (DXY)": "DX-Y.NYB",
    "High Yield (HYG)": "HYG",
    "Long Bond (TLT)": "TLT",
}
VOL = {"VIX": "^VIX", "VIX9D": "^VIX9D", "VIX3M": "^VIX3M", "VXN": "^VXN"}
LEADERS = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Amazon": "AMZN",
    "Alphabet": "GOOGL", "Meta": "META", "Tesla": "TSLA", "Broadcom": "AVGO",
    "Netflix": "NFLX", "Palantir": "PLTR",
}

FI_YAHOO = {  # fixed income prices / vol from Yahoo (yields themselves come from FRED)
    "MOVE (rate vol)": "^MOVE",
    "Short Tsy (SHY)": "SHY", "Interm. Tsy (IEF)": "IEF", "TIPS (TIP)": "TIP", "IG Corp (LQD)": "LQD",
}

OTHER = {"Gold": "GC=F"}  # used by crypto (macro links) and the Indicator lab

CRYPTO = {"Bitcoin": "BTC-USD", "Ether": "ETH-USD", "Solana": "SOL-USD", "XRP": "XRP-USD",
          "BNB": "BNB-USD", "Dogecoin": "DOGE-USD", "Cardano": "ADA-USD", "Chainlink": "LINK-USD"}
CRYPTO_ETF = {"Bitcoin ETF (IBIT)": "IBIT", "Bitcoin ETF (FBTC)": "FBTC", "Ether ETF (ETHA)": "ETHA"}
CRYPTO_EQ = {"Coinbase (COIN)": "COIN", "Strategy (MSTR)": "MSTR"}
VOLUME_OF = CRYPTO_ETF  # also download trading volume for these (ETF activity proxy)

ALL = {**INDICES, **SECTORS, **STYLE, **DRIVERS, **VOL, **LEADERS, **FI_YAHOO, **OTHER,
       **CRYPTO, **CRYPTO_ETF, **CRYPTO_EQ}

# Fixed income series from FRED (all quoted in percent; changes shown in basis points)
FI_FRED = {
    "UST 3M": "DGS3MO", "UST 6M": "DGS6MO", "UST 1Y": "DGS1", "UST 2Y": "DGS2", "UST 3Y": "DGS3",
    "UST 5Y": "DGS5", "UST 7Y": "DGS7", "UST 10Y": "DGS10", "UST 20Y": "DGS20", "UST 30Y": "DGS30",
    "10Y real yield (TIPS)": "DFII10", "5Y real yield (TIPS)": "DFII5",
    "10Y breakeven inflation": "T10YIE", "5Y breakeven inflation": "T5YIE",
    "Fed funds (EFFR)": "EFFR",
    "IG credit spread (OAS)": "BAMLC0A0CM", "HY credit spread (OAS)": "BAMLH0A0HYM2",
    "10Y term premium (Kim-Wright)": "THREEFYTP10",
}
FI_DERIVED = ["2s10s curve", "3m10y curve", "5s30s curve", "2Y minus Fed funds", "3M minus Fed funds"]

YIELD_SERIES = {"US 10Y yield"} | set(FI_FRED) | set(FI_DERIVED)  # changes measured in bp, not %
LEVEL_SERIES = YIELD_SERIES | set(VOL) | {"MOVE (rate vol)"}  # "points" series (not prices)


# ---------------------------------------------------------------------------
def _cache_file() -> Path:
    return CACHE_DIR / f"closes_{dt.date.today():%Y%m%d}.csv"


def load_closes(refresh: bool = False, demo: bool = False) -> tuple[pd.DataFrame, str]:
    """Return (wide DataFrame of daily closes with dashboard names as columns, source label)."""
    if demo:
        return make_demo_data(), "DEMO (synthetic data – not real prices)"

    f = _cache_file()
    if f.exists() and not refresh:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        # a cache written by an older version of the dashboard may lack newer series -> download again
        if len(set(ALL) - set(df.columns)) <= 5:
            stamp = dt.datetime.fromtimestamp(f.stat().st_mtime)
            return df, f"Yahoo Finance (cached today {stamp:%H:%M})"

    try:
        df = _download_yahoo()
    except Exception as exc:  # network blocked, Yahoo down, etc.
        old = sorted(CACHE_DIR.glob("closes_*.csv"))
        if old:
            df = pd.read_csv(old[-1], index_col=0, parse_dates=True)
            return df, f"STALE cache {old[-1].stem[7:]} (download failed: {exc.__class__.__name__})"
        return make_demo_data(), f"DEMO – download failed ({exc.__class__.__name__}); showing synthetic data"

    CACHE_DIR.mkdir(exist_ok=True)
    for old in CACHE_DIR.glob("closes_*.csv"):
        old.unlink()
    df.to_csv(f)
    return df, "Yahoo Finance (fresh download)"


def _download_yahoo() -> pd.DataFrame:
    import yfinance as yf

    start = dt.date.today() - dt.timedelta(days=365 * HISTORY_YEARS)
    raw = yf.download(list(ALL.values()), start=start.isoformat(), auto_adjust=True,
                      progress=False, threads=True)
    if raw is None or raw.empty:
        raise ConnectionError("Yahoo returned no data")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    inv = {v: k for k, v in ALL.items()}
    close = close.rename(columns=inv)
    close = close[[k for k in ALL if k in close.columns]]
    if isinstance(raw.columns, pd.MultiIndex) and "Volume" in raw.columns.get_level_values(0):
        vol = raw["Volume"]
        for name, t in VOLUME_OF.items():
            if t in vol:
                close[f"{name} volume"] = vol[t]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    # Yahoo has at times quoted ^TNX as yield x10; normalise to percent
    for c in YIELD_SERIES & set(close.columns):
        if close[c].median() > 20:
            close[c] = close[c] / 10
    close = close.dropna(how="all")
    # keep only US trading days (rows where the S&P printed), forward-fill small gaps
    if "S&P 500" in close:
        close = close[close["S&P 500"].notna()]
    # futures can print zero or negative (WTI, 20 Apr 2020): percentage maths breaks, so treat as missing
    for c in set(OTHER) & set(close.columns):
        close.loc[close[c] <= 0, c] = np.nan
    close = close.ffill(limit=5)
    if close.shape[0] < 300:
        raise ValueError("too little history downloaded")
    return close


# ---------------------------------------------------------------------------
# Synthetic demo data (used for offline testing / when Yahoo is unreachable)
# ---------------------------------------------------------------------------
def make_demo_data(seed: int = 7) -> pd.DataFrame:
    """Plausible fake history: one market factor with vol clustering, a rates factor,
    a dollar factor, and assets loading on them. Ends with a momentum/tech sell-off so
    every panel has something to show. NOT real prices."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * HISTORY_YEARS)
    n = len(idx)

    # GARCH-ish market vol
    vol = np.empty(n); vol[0] = 0.010
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        vol[t] = np.sqrt(0.000002 + 0.10 * (vol[t-1] * shocks[t-1]) ** 2 + 0.88 * vol[t-1] ** 2)
    mkt = 0.00035 + vol * shocks
    # final 6 sessions: AI / momentum unwind
    mkt[-6:] += np.array([-0.004, -0.012, 0.003, -0.015, -0.009, 0.004])
    rates = 0.04 * rng.standard_normal(n) + 0.25 * mkt / 0.01 * 0.03   # bp/100 per day, positive stock-bond link
    rates[-6:] = [-0.03, -0.06, 0.01, -0.07, -0.05, 0.0]
    usd = 0.003 * rng.standard_normal(n) - 0.1 * mkt
    credit = 0.35 * mkt + 0.002 * rng.standard_normal(n)
    tech = 0.006 * rng.standard_normal(n)
    tech[-6:] -= 0.012
    mom = 0.004 * rng.standard_normal(n)
    mom[-6:] -= 0.010

    def walk(start, beta_m=1.0, beta_t=0.0, beta_mom=0.0, idio=0.006, drift=0.0):
        r = beta_m * mkt + beta_t * tech + beta_mom * mom + idio * rng.standard_normal(n) + drift
        return start * np.exp(np.cumsum(r) - np.cumsum(r)[-1])  # ends at `start`

    out = {}
    out["S&P 500"] = walk(6800, 1.0, 0.3, 0.1, 0.001)
    out["SPY"] = out["S&P 500"] / 10.3
    out["Nasdaq 100"] = walk(24500, 1.2, 0.8, 0.2, 0.002)
    out["Russell 2000"] = walk(2400, 1.15, -0.1, -0.1, 0.006)
    out["IWM"] = out["Russell 2000"] / 10.05
    out["Dow Jones"] = walk(46500, 0.9, 0.0, 0.0, 0.003)
    out["S&P 500 Equal Wt (RSP)"] = walk(190, 0.95, -0.15, -0.15, 0.002)
    secs = {"Tech (XLK)": (1.2, 1.0), "Comm Svcs (XLC)": (1.0, 0.5), "Cons Disc (XLY)": (1.15, 0.3),
            "Financials (XLF)": (1.05, -0.2), "Industrials (XLI)": (1.0, -0.1), "Health (XLV)": (0.7, -0.3),
            "Staples (XLP)": (0.55, -0.4), "Utilities (XLU)": (0.55, -0.3), "Energy (XLE)": (0.9, -0.3),
            "Materials (XLB)": (1.0, -0.2), "Real Estate (XLRE)": (0.85, -0.3), "Semis (SMH)": (1.5, 1.6)}
    starts = [280, 115, 230, 53, 150, 140, 80, 85, 90, 90, 42, 340]
    for (k, (bm, bt)), s in zip(secs.items(), starts):
        out[k] = walk(s, bm, bt, 0.0, 0.006)
    out["Staples (XLP)"][-6:] *= np.linspace(1.0, 1.02, 6)
    out["MTUM"] = walk(240, 1.1, 0.4, 1.0, 0.003)
    out["SPHB"] = walk(105, 1.5, 0.5, 0.3, 0.004)
    out["SPLV"] = walk(72, 0.6, -0.3, -0.2, 0.003)
    y10 = 4.10 + np.cumsum(rates) - np.cumsum(rates)[-1]
    out["US 10Y yield"] = np.clip(y10, 0.5, None)
    out["Dollar (DXY)"] = 99.5 * np.exp(np.cumsum(usd) - np.cumsum(usd)[-1])
    out["High Yield (HYG)"] = 80 * np.exp(np.cumsum(credit + 0.0001) - np.cumsum(credit + 0.0001)[-1])
    out["Long Bond (TLT)"] = 90 * np.exp(np.cumsum(-0.17 * rates) - np.cumsum(-0.17 * rates)[-1])
    vix = np.clip(vol * np.sqrt(252) * 100 * 1.15 + 1.5 * rng.standard_normal(n).cumsum() * 0.02, 9, 80)
    vix[-6:] = vix[-6:] + np.array([2, 5, 4, 8, 9, 7])
    out["VIX"] = vix
    out["VIX9D"] = vix * (1 + 0.6 * (vix - np.median(vix)) / 40) - 0.6
    out["VIX3M"] = vix * 0.82 + 3.8
    out["VXN"] = vix * 1.2 + 1
    names = {"Apple": (1.0, 0.6, 270), "Microsoft": (1.0, 0.8, 500), "Nvidia": (1.7, 1.8, 185),
             "Amazon": (1.2, 0.8, 240), "Alphabet": (1.1, 0.8, 280), "Meta": (1.3, 0.9, 620),
             "Tesla": (1.9, 0.9, 430), "Broadcom": (1.5, 1.6, 340), "Netflix": (1.0, 0.4, 1100),
             "Palantir": (2.0, 1.2, 170)}
    for k, (bm, bt, s) in names.items():
        out[k] = walk(s, bm, bt, 0.6, 0.016)
    out["MOVE (rate vol)"] = np.clip(95 + 0.6 * (vix - np.median(vix)) * 3 + 8 * rng.standard_normal(n).cumsum() * 0.05, 50, 220)
    for k, (bt, s0) in {"Short Tsy (SHY)": (-0.019, 82.5), "Interm. Tsy (IEF)": (-0.075, 96),
                        "TIPS (TIP)": (-0.06, 110), "IG Corp (LQD)": (-0.08, 109)}.items():
        r = bt * rates + (0.12 * credit if "LQD" in k else 0) + 0.0005 * rng.standard_normal(n)
        out[k] = s0 * np.exp(np.cumsum(r) - np.cumsum(r)[-1])
    usd = np.cumsum(0.0035 * rng.standard_normal(n))
    r = -0.1 * mkt - 0.6 * np.diff(np.r_[0, usd]) + 0.010 * rng.standard_normal(n) + 0.0003
    out["Gold"] = 4100 * np.exp(np.cumsum(r) - np.cumsum(r)[-1])
    # crypto: high-beta to risk, own big cycles
    cyc = np.cumsum(0.025 * rng.standard_normal(n))
    btc_r = 1.6 * mkt + np.diff(np.r_[0, cyc]) + 0.0012
    btc_r[-6:] += np.array([-0.03, -0.045, 0.01, -0.05, -0.02, 0.015])
    out["Bitcoin"] = 98000 * np.exp(np.cumsum(btc_r) - np.cumsum(btc_r)[-1])
    for k, (lvl, b, sd) in {"Ether": (3400, 1.2, 0.025), "Solana": (160, 1.5, 0.04), "XRP": (2.3, 1.1, 0.04),
                            "BNB": (950, 0.8, 0.025), "Dogecoin": (0.17, 1.4, 0.05), "Cardano": (0.55, 1.2, 0.04),
                            "Chainlink": (15, 1.3, 0.04)}.items():
        r = b * btc_r + sd * rng.standard_normal(n) - 0.0004
        out[k] = lvl * np.exp(np.cumsum(r) - np.cumsum(r)[-1])
    for k, (lvl, src, b) in {"Bitcoin ETF (IBIT)": (56, "Bitcoin", 1.0), "Bitcoin ETF (FBTC)": (86, "Bitcoin", 1.0),
                             "Ether ETF (ETHA)": (26, "Ether", 1.0), "Coinbase (COIN)": (290, "Bitcoin", 1.4),
                             "Strategy (MSTR)": (230, "Bitcoin", 1.8)}.items():
        r = b * np.diff(np.log(out[src]), prepend=np.log(out[src][0])) + 0.008 * rng.standard_normal(n)
        v = lvl * np.exp(np.cumsum(r) - np.cumsum(r)[-1])
        if "ETF" in k:
            v[: n - 470] = np.nan  # ETFs launched in 2024
        out[k] = v
        if k in VOLUME_OF:
            base = np.abs(rng.standard_normal(n)) * 0.4 + 1
            base[-6:] *= np.array([1.5, 2.4, 1.3, 2.8, 1.9, 1.2])
            out[f"{k} volume"] = np.where(np.isnan(v), np.nan, base * (4e7 if "IBIT" in k else 6e6))
    return pd.DataFrame(out, index=idx)
