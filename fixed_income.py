"""
Fixed income models (US).

  1. Curve snapshot ........ today's Treasury curve vs 1m / 3m / 1y ago
  2. Curve regime .......... bull/bear x steepener/flattener from 20-day changes in 2Y and 10Y
  3. 10Y split (a) ......... nominal = real yield (TIPS) + breakeven inflation
  4. 10Y split (b) ......... Fed path vs term premium: beta to the 2Y + residual, cross-checked with Kim-Wright
  5. Fed pricing ........... bill yields minus fed funds = average policy rate priced over each horizon
  6. Credit ................ IG / HY option-adjusted spreads, and whether credit confirms the equity move
  7. Rate volatility ....... MOVE index vs realised volatility of the 10Y yield

Daily data from FRED (yields, real yields, breakevens, fed funds, credit spreads, term premium).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import fred
from data import CACHE_DIR, FI_FRED, HISTORY_YEARS

CURVE = ["UST 3M", "UST 6M", "UST 1Y", "UST 2Y", "UST 3Y", "UST 5Y", "UST 7Y", "UST 10Y", "UST 20Y", "UST 30Y"]
CURVE_YEARS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def add_derived(fi: pd.DataFrame) -> pd.DataFrame:
    fi = fi.copy()
    fi["2s10s curve"] = fi["UST 10Y"] - fi["UST 2Y"]
    fi["3m10y curve"] = fi["UST 10Y"] - fi["UST 3M"]
    fi["5s30s curve"] = fi["UST 30Y"] - fi["UST 5Y"]
    fi["2Y minus Fed funds"] = fi["UST 2Y"] - fi["Fed funds (EFFR)"]
    fi["3M minus Fed funds"] = fi["UST 3M"] - fi["Fed funds (EFFR)"]
    return fi


def load_fi(px: pd.DataFrame, refresh: bool = False, demo: bool = False) -> tuple[pd.DataFrame, str]:
    if demo:
        return add_derived(make_demo_fi(px)), "DEMO (synthetic rates data)"
    f = CACHE_DIR / f"fi_{dt.date.today():%Y%m%d}.csv"
    if f.exists() and not refresh:
        return add_derived(pd.read_csv(f, index_col=0, parse_dates=True)), "FRED (cached today)"
    start = (dt.date.today() - dt.timedelta(days=365 * HISTORY_YEARS)).isoformat()
    try:
        raw = fred.many(list(FI_FRED.values()), start)
        df = raw.rename(columns={v: k for k, v in FI_FRED.items()})
        df = df[df.index.dayofweek < 5].sort_index().ffill(limit=5)
    except Exception as exc:
        old = sorted(CACHE_DIR.glob("fi_*.csv"))
        if old:
            return add_derived(pd.read_csv(old[-1], index_col=0, parse_dates=True)), \
                f"STALE FRED cache {old[-1].stem[3:]} (download failed: {exc.__class__.__name__})"
        return add_derived(make_demo_fi(px)), f"DEMO – FRED download failed ({exc.__class__.__name__}) via {fred.mode()}"
    CACHE_DIR.mkdir(exist_ok=True)
    for old in CACHE_DIR.glob("fi_*.csv"):
        old.unlink()
    df.to_csv(f)
    return add_derived(df), f"FRED (fresh download, {fred.mode()})"


def make_demo_fi(px: pd.DataFrame, seed: int = 5) -> pd.DataFrame:
    """Synthetic curve built around the (demo or real) 10Y yield. NOT real data."""
    rng = np.random.default_rng(seed)
    idx = px.index
    n = len(idx)
    y10 = px["US 10Y yield"].ffill().bfill().values
    slope = np.cumsum(0.012 * rng.standard_normal(n)); slope = 0.5 + slope - slope[-1]   # 10Y - 2Y
    slope[-20:] += np.linspace(0, 0.12, 20)                                              # recent steepening
    ff = np.clip(y10 - slope - 0.35 + np.cumsum(0.004 * rng.standard_normal(n)), 0.05, None)
    ff = np.round(ff * 4) / 4 + 0.08                                                      # step-like policy rate
    y2 = y10 - slope
    out = {"Fed funds (EFFR)": ff}
    for name, t in zip(CURVE, CURVE_YEARS):
        w = np.clip(np.log1p(t) / np.log1p(10), 0, 1.3)
        base = ff + (y2 - ff) * min(t / 2, 1) if t <= 2 else y2 + (y10 - y2) * (np.log(t / 2) / np.log(5))
        out[name] = base + 0.01 * w * rng.standard_normal(n)
    out["UST 10Y"] = y10
    be = 2.3 + np.cumsum(0.008 * rng.standard_normal(n)); be = be - be[-1] + 2.28
    out["10Y breakeven inflation"] = be
    out["10Y real yield (TIPS)"] = y10 - be
    out["5Y breakeven inflation"] = be + 0.05 + 0.02 * rng.standard_normal(n)
    out["5Y real yield (TIPS)"] = out["UST 5Y"] - out["5Y breakeven inflation"]
    lhy = np.log(px["High Yield (HYG)"])
    dev = (lhy - lhy.rolling(250, min_periods=20).mean()).fillna(0).values   # mean-reverting credit cycle
    out["HY credit spread (OAS)"] = np.clip(3.4 - 25 * dev, 2.5, 9)
    out["IG credit spread (OAS)"] = np.clip(0.18 + 0.25 * out["HY credit spread (OAS)"], 0.6, 3)
    tp = np.cumsum(0.01 * rng.standard_normal(n)); out["10Y term premium (Kim-Wright)"] = tp - tp[-1] + 0.45
    return pd.DataFrame(out, index=idx)


# ---------------------------------------------------------------------------
# 1-2. curve
# ---------------------------------------------------------------------------
def curve_snapshots(fi: pd.DataFrame) -> pd.DataFrame:
    cur = fi[CURVE].dropna(how="all")
    last = cur.index[-1]
    out = {}
    for lab, off in [("Today", None), ("1 month ago", pd.DateOffset(months=1)),
                     ("3 months ago", pd.DateOffset(months=3)), ("1 year ago", pd.DateOffset(years=1))]:
        d = last if off is None else cur.index[cur.index.get_indexer([last - off], method="pad")[0]]
        out[f"{lab} ({d:%d %b %y})"] = cur.loc[d]
    return pd.DataFrame(out)


CURVE_REGIMES = {
    "BULL STEEPENER": "Yields falling, short end falling faster. The market is pricing rate cuts: classic early-easing "
                      "or growth-scare-with-Fed-response move.",
    "BULL FLATTENER": "Yields falling, long end falling faster. Flight to quality / growth worries with the Fed seen "
                      "as slow to respond. Often a risk-off signal.",
    "BEAR STEEPENER": "Yields rising, long end rising faster. Term premium, inflation worries or heavy bond supply. "
                      "Hard on long-duration assets (growth stocks, long bonds).",
    "BEAR FLATTENER": "Yields rising, short end rising faster. The market is pricing a more hawkish Fed: "
                      "typical of a hiking cycle or strong data.",
    "RANGE-BOUND": "Neither yields nor the curve have moved much over the last 20 sessions.",
}


def curve_regime(fi: pd.DataFrame, n: int = 20, lvl_thr: float = 5.0, slope_thr: float = 3.0) -> pd.DataFrame:
    """Direction = average 20d change of 2Y and 10Y (bp). Shape = 20d change in 10Y-2Y (bp)."""
    d2 = (fi["UST 2Y"] - fi["UST 2Y"].shift(n)) * 100
    d10 = (fi["UST 10Y"] - fi["UST 10Y"].shift(n)) * 100
    lvl = (d2 + d10) / 2
    slope = d10 - d2
    st = np.where((lvl.abs() < lvl_thr) & (slope.abs() < slope_thr), "RANGE-BOUND",
         np.where(lvl < 0, np.where(slope > 0, "BULL STEEPENER", "BULL FLATTENER"),
                  np.where(slope > 0, "BEAR STEEPENER", "BEAR FLATTENER")))
    df = pd.DataFrame({"d2Y bp": d2, "d10Y bp": d10, "level bp": lvl, "slope bp": slope, "state": st}).dropna()
    df["days"] = df.groupby((df["state"] != df["state"].shift()).cumsum()).cumcount() + 1
    return df


# ---------------------------------------------------------------------------
# 3-4. what's moving the 10Y
# ---------------------------------------------------------------------------
def real_vs_breakeven(fi: pd.DataFrame, n: int = 20) -> dict:
    nom = (fi["UST 10Y"].iloc[-1] - fi["UST 10Y"].iloc[-1 - n]) * 100
    real = (fi["10Y real yield (TIPS)"].iloc[-1] - fi["10Y real yield (TIPS)"].iloc[-1 - n]) * 100
    be = (fi["10Y breakeven inflation"].iloc[-1] - fi["10Y breakeven inflation"].iloc[-1 - n]) * 100
    driver = ("REAL YIELDS" if abs(real) > 2 * abs(be) else "INFLATION EXPECTATIONS" if abs(be) > 2 * abs(real)
              else "BOTH")
    return {"nominal": nom, "real": real, "breakeven": be, "driver": driver}


def fed_vs_term_premium(fi: pd.DataFrame, n: int = 20, window: int = 250) -> pd.DataFrame:
    """Rolling regression of daily 10Y changes on daily 2Y changes (bp).
       Over the last n days: Fed-path part = beta x 2Y move; term-premium/other part = the rest."""
    d10 = fi["UST 10Y"].diff() * 100
    d2 = fi["UST 2Y"].diff() * 100
    beta = d10.rolling(window).cov(d2) / d2.rolling(window).var()
    m10 = (fi["UST 10Y"] - fi["UST 10Y"].shift(n)) * 100
    m2 = (fi["UST 2Y"] - fi["UST 2Y"].shift(n)) * 100
    fedpart = beta * m2
    tp = (fi["10Y term premium (Kim-Wright)"] - fi["10Y term premium (Kim-Wright)"].shift(n)) * 100
    df = pd.DataFrame({"10Y move bp": m10, "beta to 2Y": beta, "Fed path bp": fedpart,
                       "Term premium & other bp": m10 - fedpart, "Kim-Wright TP move bp": tp}).dropna()
    share = df["Fed path bp"].abs() / (df["Fed path bp"].abs() + df["Term premium & other bp"].abs())
    df["verdict"] = np.where(share >= 2 / 3, "FED-DRIVEN", np.where(share <= 1 / 3, "TERM-PREMIUM-DRIVEN", "SHARED"))
    return df


# ---------------------------------------------------------------------------
# 5. Fed pricing (bill-based proxy)
# ---------------------------------------------------------------------------
def fed_pricing(fi: pd.DataFrame) -> pd.DataFrame:
    """Each bill/note yield is roughly the average policy rate expected over its life (plus a small premium).
       So yield minus today's fed funds ~ cuts (negative) or hikes (positive) priced on average over that horizon."""
    last = fi.iloc[-1]
    ff = last["Fed funds (EFFR)"]
    rows = []
    for name, months in [("UST 3M", 3), ("UST 6M", 6), ("UST 1Y", 12), ("UST 2Y", 24)]:
        diff = (last[name] - ff) * 100
        rows.append({"Horizon": name.replace("UST ", ""), "months": months, "Yield %": last[name],
                     "vs Fed funds bp": diff, "≈ 25bp moves (avg over horizon)": diff / 25})
    return pd.DataFrame(rows).set_index("Horizon")


def fed_path(fi: pd.DataFrame, when=-1) -> pd.DataFrame:
    """Forward rates between bill maturities: the rate the curve implies for each future window.
       f(a,b) = (y_b * b - y_a * a) / (b - a). A rough stand-in for a futures-implied policy path."""
    r = fi.iloc[when]
    pts = [("now", 0, None), ("0-3m", 0.25, "UST 3M"), ("3-6m", 0.5, "UST 6M"), ("6-12m", 1.0, "UST 1Y"),
           ("1-2y", 2.0, "UST 2Y")]
    rows, prev_t, prev_y = [], 0.0, None
    for lab, t, col in pts:
        if col is None:
            rows.append({"window": "Fed funds today", "implied rate %": r["Fed funds (EFFR)"]})
            continue
        y = r[col]
        f = y if prev_y is None else (y * t - prev_y * prev_t) / (t - prev_t)
        rows.append({"window": lab, "implied rate %": f})
        prev_t, prev_y = t, y
    df = pd.DataFrame(rows).set_index("window")
    df["vs today bp"] = (df["implied rate %"] - r["Fed funds (EFFR)"]) * 100
    return df


# ---------------------------------------------------------------------------
# 6. credit
# ---------------------------------------------------------------------------
def credit_confirmation(px: pd.DataFrame, fi: pd.DataFrame, n: int = 20) -> dict:
    """Is credit telling the same story as equities over the last n sessions?
       Compares the S&P move and the HY spread move, each in units of their own typical n-day move (z)."""
    spx = px["S&P 500"]
    hy = fi["HY credit spread (OAS)"].dropna() * 100
    def z(x, is_spread):
        c = (x - x.shift(n)) if is_spread else (x / x.shift(n) - 1) * 100
        return ((c - c.rolling(756, min_periods=200).mean()) / c.rolling(756, min_periods=200).std()).iloc[-1], c.iloc[-1]
    z_eq, m_eq = z(spx, False)
    z_hy, m_hy = z(hy, True)
    if np.isnan(z_hy):
        verdict = "NOT ENOUGH HISTORY"
    elif z_eq < -1 and z_hy > 1:
        verdict = "CONFIRMING: stocks falling and spreads widening together"
    elif z_eq < -1 and z_hy <= 0.5:
        verdict = "NOT CONFIRMING: stocks falling but credit calm (equity-specific or early)"
    elif z_eq > 1 and z_hy < -1:
        verdict = "CONFIRMING: stocks rising and spreads tightening"
    elif z_eq > 0.5 and z_hy > 1:
        verdict = "WARNING: stocks up but spreads widening (credit leading lower?)"
    else:
        verdict = "NEUTRAL: no strong move in either"
    return {"eq_move": m_eq, "eq_z": z_eq, "hy_move_bp": m_hy, "hy_z": z_hy, "verdict": verdict}


# ---------------------------------------------------------------------------
# 7. rate volatility
# ---------------------------------------------------------------------------
def rate_vol(px: pd.DataFrame, fi: pd.DataFrame) -> pd.DataFrame:
    """MOVE = implied Treasury volatility in bp per year. Realised = std of daily 10Y changes (bp) x sqrt(252)."""
    rv = (fi["UST 10Y"].diff() * 100).rolling(21).std() * np.sqrt(252)
    d = pd.DataFrame({"10Y realised vol (bp/yr)": rv})
    if "MOVE (rate vol)" in px:
        d = d.join(px["MOVE (rate vol)"].rename("MOVE"), how="outer")
        d["MOVE minus realised"] = d["MOVE"] - d["10Y realised vol (bp/yr)"]
        if "VIX" in px:
            d = d.join((px["MOVE (rate vol)"] / px["VIX"]).rename("MOVE / VIX"), how="left")
    return d.ffill(limit=3)
