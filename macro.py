"""
Growth x inflation quadrant.

The question is not "is growth high?" but "is growth SPEEDING UP or SLOWING DOWN?", and the same for
inflation. Markets react to changes in direction more than to levels.

For each monthly series:
    acceleration = 3-month annualised growth rate  -  12-month growth rate      (percentage points)
Positive = the recent pace is faster than the past year's pace -> accelerating.

Growth composite = average of the accelerations of four activity series, each first divided by its own
typical size (standard deviation over the last 10 years) so a noisy series can't dominate:
    PAYEMS    nonfarm payrolls (jobs)
    INDPRO    industrial production
    RRSFS     real retail & food-service sales
    W875RX1   real personal income excluding government transfers
Inflation = acceleration of core CPI (CPILFESL), in percentage points.

Source: FRED (Federal Reserve Bank of St. Louis), free, no API key, monthly.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import fred
from data import CACHE_DIR

GROWTH_SERIES = {
    "PAYEMS": "Nonfarm payrolls",
    "INDPRO": "Industrial production",
    "RRSFS": "Real retail sales",
    "W875RX1": "Real personal income ex transfers",
}
INFLATION_SERIES = {"CPILFESL": "Core CPI"}
START = "1990-01-01"

# Quadrant names. Order: growth sign, inflation sign
QUADRANTS = {
    (+1, -1): ("Goldilocks", "Growth accelerating, inflation decelerating",
               "Usually the best backdrop for equities: earnings improve while rate pressure eases. "
               "Cyclicals and growth stocks tend to lead."),
    (+1, +1): ("Reflation", "Growth and inflation both accelerating",
               "Earnings are strong but central banks lean hawkish and yields rise. Equities usually still "
               "rise; energy, materials, financials and value tend to beat long-duration growth stocks."),
    (-1, +1): ("Stagflation", "Growth decelerating, inflation accelerating",
               "Historically the hardest regime for both stocks and bonds: slowing earnings and no help from "
               "the central bank. Defensives and energy tend to hold up best."),
    (-1, -1): ("Disinflationary slowdown", "Growth and inflation both decelerating",
               "Growth worries dominate but rate cuts come into view. Bonds tend to do well; within equities, "
               "defensives and quality lead, cyclicals lag."),
}
QUADRANT_COLOR = {"Goldilocks": "#2a78d6", "Reflation": "#1baf7a",
                  "Stagflation": "#e34948", "Disinflationary slowdown": "#eda100"}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_macro(refresh: bool = False, demo: bool = False) -> tuple[pd.DataFrame, str]:
    if demo:
        return make_demo_macro(), "DEMO (synthetic macro data)"
    f = CACHE_DIR / f"macro_{dt.date.today():%Y%m%d}.csv"
    if f.exists() and not refresh:
        return pd.read_csv(f, index_col=0, parse_dates=True), "FRED (cached today)"
    try:
        ids = list(GROWTH_SERIES) + list(INFLATION_SERIES)
        df = fred.many(ids, START)
        df.index = df.index.to_period("M").to_timestamp()  # first-of-month stamps
    except Exception as exc:
        old = sorted(CACHE_DIR.glob("macro_*.csv"))
        if old:
            return pd.read_csv(old[-1], index_col=0, parse_dates=True), \
                f"STALE FRED cache {old[-1].stem[6:]} (download failed: {exc.__class__.__name__})"
        return make_demo_macro(), f"DEMO – FRED download failed ({exc.__class__.__name__}) via {fred.mode()}"
    CACHE_DIR.mkdir(exist_ok=True)
    for old in CACHE_DIR.glob("macro_*.csv"):
        old.unlink()
    df.to_csv(f)
    return df, f"FRED (fresh download, {fred.mode()})"


def make_demo_macro(seed: int = 11) -> pd.DataFrame:
    """Synthetic monthly macro levels with slow cycles. NOT real data."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, periods=(pd.Timestamp.today().to_period("M") - pd.Period(START, "M")).n - 1, freq="MS")
    n = len(idx)
    t = np.arange(n)
    cyc_g = 0.0016 * np.sin(2 * np.pi * t / 54) + 0.0008 * np.sin(2 * np.pi * t / 23)
    cyc_i = 0.0010 * np.sin(2 * np.pi * (t - 14) / 60) + 0.0004 * np.sin(2 * np.pi * t / 19)
    out = {}
    for k, (drift, noise, beta) in {"PAYEMS": (0.0012, 0.0010, 0.8), "INDPRO": (0.0010, 0.0060, 1.6),
                                    "RRSFS": (0.0015, 0.0080, 1.3), "W875RX1": (0.0020, 0.0040, 1.0)}.items():
        r = drift + beta * cyc_g + noise * rng.standard_normal(n)
        out[k] = 100 * np.exp(np.cumsum(r))
    out["CPILFESL"] = 100 * np.exp(np.cumsum(0.0022 + cyc_i + 0.0012 * rng.standard_normal(n)))
    return pd.DataFrame(out, index=idx)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def acceleration(level: pd.Series) -> pd.DataFrame:
    """3m annualised rate, 12m rate and their difference, all in %."""
    g3 = ((level / level.shift(3)) ** 4 - 1) * 100
    g12 = (level / level.shift(12) - 1) * 100
    return pd.DataFrame({"3m ann %": g3, "12m %": g12, "accel pp": g3 - g12})


def quadrant_frame(macro: pd.DataFrame, z_years: int = 10) -> pd.DataFrame:
    acc = {k: acceleration(macro[k].dropna())["accel pp"] for k in GROWTH_SERIES if k in macro}
    acc = pd.DataFrame(acc)
    win = 12 * z_years
    z = acc / acc.rolling(win, min_periods=36).std()
    growth = z.mean(axis=1, skipna=True).where(z.notna().sum(axis=1) >= 3)  # σ units; need 3 of 4 series
    infl = acceleration(macro["CPILFESL"].dropna())["accel pp"]
    q = pd.DataFrame({"growth": growth, "inflation": infl}).dropna()
    gs = np.where(q["growth"] >= 0, 1, -1)
    is_ = np.where(q["inflation"] >= 0, 1, -1)
    q["quadrant"] = [QUADRANTS[(a, b)][0] for a, b in zip(gs, is_)]
    q["months"] = q.groupby((q["quadrant"] != q["quadrant"].shift()).cumsum()).cumcount() + 1
    # when could you have known? Monthly data for month M is mostly out by the end of month M+1
    # (personal income is the last, ~4 weeks after month-end). Stamp each reading at that date.
    q["known_at"] = q.index + pd.offsets.MonthEnd(2)
    return q


def components_table(macro: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, label in {**GROWTH_SERIES, **INFLATION_SERIES}.items():
        if k not in macro:
            continue
        a = acceleration(macro[k].dropna()).dropna()
        last = a.iloc[-1]
        rows.append({"Series": label, "FRED id": k, "Latest month": a.index[-1].strftime("%b %Y"),
                     "3m annualised %": last["3m ann %"], "12m %": last["12m %"],
                     "Acceleration pp": last["accel pp"],
                     "Direction": "accelerating" if last["accel pp"] > 0 else "decelerating",
                     "Role": "Inflation" if k in INFLATION_SERIES else "Growth"})
    return pd.DataFrame(rows).set_index("Series")


def daily_quadrant(q: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """The quadrant you would have known on each trading day (no look-ahead)."""
    s = pd.Series(q["quadrant"].values, index=q["known_at"].values).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(index.union(s.index)).ffill().reindex(index)


def quadrant_returns(px: pd.DataFrame, q: pd.DataFrame, sectors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly S&P and sector returns, grouped by the quadrant known at the START of each month."""
    m = px.resample("ME").last()
    ret = m.pct_change() * 100
    known = daily_quadrant(q, m.index)        # quadrant known at each month-end ...
    regime_for_next = known.shift(1)          # ... applies to the following month's return
    df = ret.assign(quadrant=regime_for_next).dropna(subset=["quadrant", "S&P 500"])
    g = df.groupby("quadrant")
    spx = pd.DataFrame({
        "months": g.size(),
        "avg monthly %": g["S&P 500"].mean(),
        "annualised %": g["S&P 500"].mean() * 12,
        "% months up": g["S&P 500"].apply(lambda x: (x > 0).mean() * 100),
        "worst month %": g["S&P 500"].min(),
    }).reindex([v[0] for v in QUADRANTS.values()])
    secs = [s for s in sectors if s in df]
    rel = g[secs].mean().sub(g["S&P 500"].mean(), axis=0)  # avg monthly excess return vs S&P (pp)
    rel = rel.reindex([v[0] for v in QUADRANTS.values()])
    return spx, rel


def describe(q: pd.DataFrame) -> dict:
    last = q.iloc[-1]
    name = last["quadrant"]
    key = next(k for k, v in QUADRANTS.items() if v[0] == name)
    prev = q["quadrant"].iloc[-int(last["months"]) - 1] if len(q) > last["months"] else None
    return {"name": name, "subtitle": QUADRANTS[key][1], "meaning": QUADRANTS[key][2],
            "months": int(last["months"]), "growth": last["growth"], "inflation": last["inflation"],
            "as_of": q.index[-1], "known_at": last["known_at"], "previous": prev}
