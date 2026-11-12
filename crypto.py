"""
Crypto models.

  1. Level ladder ........... the reference levels traders watch for Bitcoin/Ether (moving averages, 52-week and
                              all-time highs/lows, year open, round numbers) sorted around today's price
  2. Drawdown & volatility .. distance from the all-time high, 30d/90d realised volatility and their percentiles
  3. Macro links ............ rolling correlation and beta with the Nasdaq, gold, the dollar, real yields
  4. Alt breadth ............ ETH/BTC and how many large altcoins are beating Bitcoin (risk appetite inside crypto)
  5. ETF activity ........... spot ETF dollar trading volume vs its own average (an activity proxy, NOT net flows)

Note: the dashboard keeps US trading days only, so weekend crypto moves show up in Monday's change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import CRYPTO, CRYPTO_ETF


def ladder(s: pd.Series) -> pd.DataFrame:
    s = s.dropna()
    p = s.iloc[-1]
    lv = {
        "20-day average": s.rolling(20).mean().iloc[-1],
        "50-day average": s.rolling(50).mean().iloc[-1],
        "100-day average": s.rolling(100).mean().iloc[-1],
        "200-day average": s.rolling(200).mean().iloc[-1],
        "52-week high": s.iloc[-252:].max(), "52-week low": s.iloc[-252:].min(),
        "All-time high (in data)": s.max(),
        "Year open": s[s.index < pd.Timestamp(s.index[-1].year, 1, 1)].iloc[-1]
        if (s.index < pd.Timestamp(s.index[-1].year, 1, 1)).any() else np.nan,
        "20-day high": s.iloc[-20:].max(), "20-day low": s.iloc[-20:].min(),
    }
    step = 10 ** np.floor(np.log10(p)) / (2 if p / 10 ** np.floor(np.log10(p)) < 3 else 1)
    lv["Round number above"] = np.ceil(p / step) * step
    lv["Round number below"] = np.floor(p / step) * step
    df = pd.DataFrame({"Level": lv}).dropna()
    df["Distance %"] = (df["Level"] / p - 1) * 100
    df["Role"] = np.where(df["Level"] > p, "resistance", "support")
    return df.sort_values("Level", ascending=False)


def drawdown(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return (s / s.cummax() - 1) * 100


def vol_stats(s: pd.Series) -> dict:
    r = np.log(s.dropna()).diff()
    out = {}
    for w in (30, 90):
        rv = r.rolling(w).std() * np.sqrt(252) * 100   # trading-day annualisation (dashboard uses US trading days)
        hist = rv.dropna().iloc[-756:]
        out[w] = {"now": rv.iloc[-1], "pct": (hist.iloc[:-1] < hist.iloc[-1]).mean() * 100, "series": rv}
    return out


def macro_corr(px: pd.DataFrame, fi: pd.DataFrame | None, coin: str = "Bitcoin", window: int = 60) -> pd.DataFrame:
    r = np.log(px[coin]).diff()
    drivers = {"Nasdaq 100": np.log(px["Nasdaq 100"]).diff(), "S&P 500": np.log(px["S&P 500"]).diff(),
               "Gold": np.log(px["Gold"]).diff() if "Gold" in px else None,
               "Dollar (DXY)": np.log(px["Dollar (DXY)"]).diff()}
    if fi is not None and "10Y real yield (TIPS)" in fi:
        drivers["10Y real yield"] = fi["10Y real yield (TIPS)"].diff().reindex(px.index)
    rows = []
    for k, d in drivers.items():
        if d is None:
            continue
        both = pd.concat([r, d], axis=1).dropna().iloc[-window:]
        c = both.iloc[:, 0].corr(both.iloc[:, 1])
        beta = both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var()
        c1y = pd.concat([r, d], axis=1).dropna().iloc[-252:].corr().iloc[0, 1]
        rows.append({"Driver": k, f"Correlation {window}d": c, "Correlation 1y": c1y, f"Beta {window}d": beta})
    return pd.DataFrame(rows).set_index("Driver")


def rolling_corr(px: pd.DataFrame, a: str, b: str, window: int = 60) -> pd.Series:
    return np.log(px[a]).diff().rolling(window).corr(np.log(px[b]).diff())


def alt_breadth(px: pd.DataFrame, n: int = 20) -> tuple[pd.DataFrame, float]:
    alts = [c for c in CRYPTO if c != "Bitcoin" and c in px]
    rows = []
    for a in alts:
        s = px[a].dropna()
        if len(s) <= n:
            continue
        ra = (s.iloc[-1] / s.iloc[-1 - n] - 1) * 100
        b = px["Bitcoin"].reindex(s.index)
        rb = (b.iloc[-1] / b.iloc[-1 - n] - 1) * 100
        rows.append({"Coin": a, f"{n}d %": ra, "vs Bitcoin pp": ra - rb})
    df = pd.DataFrame(rows).set_index("Coin").sort_values("vs Bitcoin pp", ascending=False)
    share = (df["vs Bitcoin pp"] > 0).mean() * 100 if len(df) else np.nan
    return df, share


def alt_breadth_series(px: pd.DataFrame, n: int = 20) -> pd.Series:
    alts = [c for c in CRYPTO if c != "Bitcoin" and c in px]
    rel = pd.DataFrame({a: px[a].pct_change(n) - px["Bitcoin"].pct_change(n) for a in alts})
    cnt = rel.notna().sum(axis=1)
    return ((rel > 0).sum(axis=1) / cnt * 100).where(cnt >= 3)


def etf_activity(px: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in CRYPTO_ETF:
        vcol = f"{name} volume"
        if name not in px or vcol not in px:
            continue
        dv = (px[name] * px[vcol]).dropna()
        if len(dv) < 60:
            continue
        avg = dv.rolling(50).mean()
        r5 = (px[name].dropna().iloc[-1] / px[name].dropna().iloc[-6] - 1) * 100
        rows.append({"ETF": name, "Dollar volume today $m": dv.iloc[-1] / 1e6,
                     "5-day avg $m": dv.iloc[-5:].mean() / 1e6, "50-day avg $m": avg.iloc[-1] / 1e6,
                     "5-day vs 50-day": dv.iloc[-5:].mean() / avg.iloc[-1], "Price 5d %": r5,
                     "Read": ("HEAVY " + ("BUYING?" if r5 > 0 else "SELLING?")) if dv.iloc[-5:].mean() / avg.iloc[-1] > 1.5
                     else "NORMAL" if dv.iloc[-5:].mean() / avg.iloc[-1] > 0.7 else "QUIET"})
    return pd.DataFrame(rows).set_index("ETF") if rows else pd.DataFrame()
