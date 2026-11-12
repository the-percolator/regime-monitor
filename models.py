"""
The models. Every function is plain pandas/numpy so you can read exactly what it does.

  1. Dashboard stats ........ level, changes, 5d z-score, 1y percentile
  2. Trend regime ........... 50/200-day moving-average state (slow)
  3. Trend x vol state ...... price vs 21d EMA  x  realised vol vs its median (fast), with persistence odds
  4. Market-implied regime .. sign of 20d vol-scaled moves in stocks and 10Y yields (+ dollar as context)
  5. Stock-bond correlation . are bonds hedging equities?
  6. Factor attribution ..... rolling ridge regression of S&P daily returns on drivers
  7. Breadth & rotation ..... style ratios vs their 100d average
  8. Volatility ............. VIX term structure, vol risk premium, realised-vol cone
  9. Leaders ................ momentum z, beta, drawdown for the mega-caps
 10. Event study ............ "what happened next" base rates, with independent-episode count
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import LEVEL_SERIES, YIELD_SERIES

Z_WINDOW = 756      # 3 years of sessions for z-scores
PCT_WINDOW = 252    # 1 year for percentiles


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def change(s: pd.Series, n: int, name: str | None = None) -> pd.Series:
    """n-day change in the series' natural unit: % for prices, bp for yields, points for vol."""
    name = name or s.name
    if name in YIELD_SERIES:
        return (s - s.shift(n)) * 100.0          # percent -> basis points
    if name in LEVEL_SERIES:
        return s - s.shift(n)                    # vol index points
    return (s / s.shift(n) - 1.0) * 100.0        # percent


def unit(name: str) -> str:
    return "bp" if name in YIELD_SERIES else ("pt" if name in LEVEL_SERIES else "%")


def zscore_of_change(s: pd.Series, n: int = 5, window: int = Z_WINDOW) -> pd.Series:
    """How unusual is this n-day move versus the series' own last-3-years of n-day moves."""
    c = change(s, n)
    mu = c.rolling(window, min_periods=250).mean()
    sd = c.rolling(window, min_periods=250).std()
    return (c - mu) / sd


def pct_rank(s: pd.Series, window: int = PCT_WINDOW) -> pd.Series:
    """Where today's level sits in its trailing 1-year range, 0..100."""
    return s.rolling(window, min_periods=window // 2).apply(
        lambda a: (a[:-1] < a[-1]).mean() * 100 if len(a) > 1 else np.nan, raw=True)


def days_in_state(state: pd.Series) -> pd.Series:
    """Consecutive sessions the series has carried its current label."""
    grp = (state != state.shift()).cumsum()
    return state.groupby(grp).cumcount() + 1


def realised_vol(s: pd.Series, n: int = 21) -> pd.Series:
    """Annualised realised volatility (%) from daily log returns."""
    r = np.log(s).diff()
    return r.rolling(n).std() * np.sqrt(252) * 100


# ---------------------------------------------------------------------------
# 1. dashboard statistics
# ---------------------------------------------------------------------------
def dashboard_table(px: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = []
    for n in names:
        if n not in px or px[n].dropna().shape[0] < 300:
            continue
        s = px[n].dropna()
        ytd_base = s[s.index < pd.Timestamp(s.index[-1].year, 1, 1)]
        ytd = np.nan
        if len(ytd_base):
            b = ytd_base.iloc[-1]
            ytd = ((s.iloc[-1] - b) * (100 if n in YIELD_SERIES else 1) if n in LEVEL_SERIES
                   else (s.iloc[-1] / b - 1) * 100)
        tr = trend_regime(s)
        mk = markov_states(s)
        rows.append({
            "Name": n,
            "Level": s.iloc[-1],
            "1d": change(s, 1, n).iloc[-1],
            "5d": change(s, 5, n).iloc[-1],
            "20d": change(s, 20, n).iloc[-1],
            "YTD": ytd,
            "Unit": unit(n),
            "5d z": zscore_of_change(s, 5).iloc[-1],
            "1y %ile": (s.iloc[-PCT_WINDOW:-1] < s.iloc[-1]).mean() * 100,
            "Trend": tr["state"].iloc[-1],
            "Trend days": int(tr["days"].iloc[-1]),
            "Trend × vol": mk["state"].iloc[-1] if mk is not None else "",
        })
    return pd.DataFrame(rows).set_index("Name")


def ordinal(x: float) -> str:
    n = int(round(x))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def stretched(table: pd.DataFrame, z: float = 1.5) -> list[str]:
    """Plain-English list of anything unusual: big weekly moves, range extremes, fresh trend flips."""
    out = []
    for name, r in table.iterrows():
        if pd.notna(r["5d z"]) and abs(r["5d z"]) >= z:
            out.append((abs(r["5d z"]), f"**{name}** – 5-day move of {r['5d']:+.1f}{r['Unit']} is "
                        f"{abs(r['5d z']):.1f} standard deviations {'below' if r['5d z'] < 0 else 'above'} normal"))
        if pd.notna(r["1y %ile"]) and (r["1y %ile"] >= 95 or r["1y %ile"] <= 5):
            out.append((1.0, f"**{name}** – level at the {ordinal(r['1y %ile'])} percentile of its 1-year range"))
        if r["Trend days"] <= 5:
            out.append((0.9, f"**{name}** – trend state changed {r['Trend days']} session(s) ago → now {r['Trend']}"))
    return [t for _, t in sorted(out, key=lambda x: -x[0])]


# ---------------------------------------------------------------------------
# 2. trend regime (50 / 200 day)
# ---------------------------------------------------------------------------
def trend_regime(s: pd.Series) -> pd.DataFrame:
    """UPTREND  : price > 50d > 200d
       DOWNTREND: price < 50d < 200d
       MIXED    : anything else (the averages disagree with each other or with price)"""
    s = s.dropna()
    m50, m200 = s.rolling(50).mean(), s.rolling(200).mean()
    st = np.where((s > m50) & (m50 > m200), "UPTREND",
         np.where((s < m50) & (m50 < m200), "DOWNTREND", "MIXED"))
    st = pd.Series(st, index=s.index).where(m200.notna())
    return pd.DataFrame({"px": s, "ma50": m50, "ma200": m200, "state": st, "days": days_in_state(st)})


def trend_flip_text(tr: pd.DataFrame) -> str:
    r = tr.iloc[-1]
    f = (lambda v: f"{v:,.2f}") if abs(r.ma200) < 100 else (lambda v: f"{v:,.1f}")
    d50 = (r.px / r.ma50 - 1) * 100
    d200 = (r.px / r.ma200 - 1) * 100
    if r.state == "UPTREND":
        return f"Closes below the 50-day ({f(r.ma50)}, {d50:+.1f}% away) → MIXED."
    if r.state == "DOWNTREND":
        return f"Closes above the 50-day ({f(r.ma50)}, {d50:+.1f}% away) → MIXED."
    if r.ma50 > r.ma200:
        return (f"Averages still point up (50d above 200d). Reclaiming the 50-day ({f(r.ma50)}, "
                f"{d50:+.1f}% away) → UPTREND. 200-day sits at {f(r.ma200)} ({d200:+.1f}%).")
    return (f"Averages point down (50d below 200d). A close under the 50-day ({f(r.ma50)}, "
            f"{d50:+.1f}% away) → DOWNTREND. 200-day at {f(r.ma200)} ({d200:+.1f}%).")


# ---------------------------------------------------------------------------
# 3. Trend x volatility state (persistence measured with an empirical Markov transition matrix)
# ---------------------------------------------------------------------------
MARKOV_STATES = ["Up · Low-vol", "Up · High-vol", "Down · Low-vol", "Down · High-vol"]


def markov_states(s: pd.Series, ema: int = 21, rv: int = 20, med: int = 252) -> pd.DataFrame | None:
    """Direction: price above (Up) / below (Down) its 21-day EMA.
       Volatility: 20d realised vol above (High-vol) / below (Low-vol) its own 1-year median."""
    s = s.dropna()
    if len(s) < med + rv + 10:
        return None
    e = s.ewm(span=ema, adjust=False).mean()
    if s.name in YIELD_SERIES or s.name in LEVEL_SERIES:
        vol = s.diff().rolling(rv).std()
    else:
        vol = np.log(s).diff().rolling(rv).std()
    vmed = vol.rolling(med, min_periods=med // 2).median()
    direction = np.where(s > e, "Up", "Down")
    temp = np.where(vol > vmed, "High-vol", "Low-vol")
    st = pd.Series([f"{a} · {b}" for a, b in zip(direction, temp)], index=s.index).where(vmed.notna())
    return pd.DataFrame({"px": s, "ema": e, "vol": vol, "vol_med": vmed, "state": st,
                         "days": days_in_state(st)})


def transition_matrix(states: pd.Series, lookback: int = Z_WINDOW) -> pd.DataFrame:
    """Row = today's state, column = tomorrow's state, values = empirical probability."""
    st = states.dropna().iloc[-lookback:]
    m = pd.crosstab(st.iloc[:-1].values, st.iloc[1:].values, normalize="index")
    return m.reindex(index=MARKOV_STATES, columns=MARKOV_STATES).fillna(0.0)


def markov_summary(s: pd.Series) -> dict | None:
    mk = markov_states(s)
    if mk is None:
        return None
    cur = mk["state"].iloc[-1]
    tm = transition_matrix(mk["state"])
    p = float(tm.loc[cur, cur])
    exp_len = 1 / (1 - p) if p < 1 else np.inf
    return {"state": cur, "days": int(mk["days"].iloc[-1]), "p_stay": p, "exp_len": exp_len,
            "style": "persistent → momentum" if exp_len >= 6 else "short-lived → mean reversion",
            "matrix": tm, "frame": mk}


# ---------------------------------------------------------------------------
# 4. SBC regime: Stocks, Bonds (10Y yield), Currency (DXY)
# ---------------------------------------------------------------------------
SBC_LEGS = {"Stocks": "S&P 500", "Bonds (10Y yield)": "US 10Y yield", "Dollar": "Dollar (DXY)"}
NO_DRIVER_THRESHOLD = 0.25  # |vol-scaled 20d move| below this on stocks or yields = too small to read
                            # (0.5 would label about half of all days "no clear driver"; 0.25 about a third)

# Market-implied macro regime: which kind of news dominates, read from the SIGN of joint stock and yield moves
# (the identification logic of Cieslak & Pang, "Common shocks in stocks and bonds", 2021).
# The dollar is reported next to it as a plain fact, not used to name the regime.
REGIMES = {
    # (stocks sign, yields sign): (name, driver group, bonds hedge stocks?)
    (+1, +1): ("Growth up", "GROWTH", True),
    (-1, -1): ("Growth down / risk-off", "GROWTH", True),
    (+1, -1): ("Rates easing", "RATES", False),
    (-1, +1): ("Rates tightening", "RATES", False),
}
NO_DRIVER = "No clear driver"
REGIME_ORDER = ["Growth up", "Growth down / risk-off", "Rates easing", "Rates tightening", NO_DRIVER]
REGIME_COLOR = {"Growth up": "#2a78d6", "Growth down / risk-off": "#e34948", "Rates easing": "#86b6ef",
                "Rates tightening": "#f0a3a2", NO_DRIVER: "#898781"}

SBC_EXPLAIN = {
    "Growth up": "Stocks and yields are rising **together**: news about growth is driving markets. In this world "
                 "government bonds rally when stocks fall, so **bonds hedge an equity portfolio**.",
    "Growth down / risk-off": "Stocks and yields are falling **together**: either worse growth news or a flight to "
                              "safety (the signs alone can't tell which). Treasuries are rallying while stocks fall, "
                              "so **bonds are hedging**.",
    "Rates easing": "Stocks up while yields fall: lower expected rates or inflation are lifting valuations. Supportive, "
                    "but stocks and bonds are rallying together, so bonds are **not** acting as a hedge.",
    "Rates tightening": "Stocks down while yields rise: higher expected rates or inflation are hurting both. "
                        "**Bonds stop hedging**; this is the regime that hurt 60/40 portfolios in 2022.",
    NO_DRIVER: "The stock or the yield move over the last 20 days is too small to read a driver from. "
               "Any label here would flip on one day's news.",
}


def vol_scaled_move(s: pd.Series, n: int = 20) -> pd.Series:
    """n-day move divided by the typical n-day move implied by recent daily volatility.
       ~ a t-statistic: |value| > 1 is a meaningful move."""
    if s.name in YIELD_SERIES:
        d = s.diff()
    else:
        d = np.log(s).diff()
    move = d.rolling(n).sum()
    scale = d.rolling(60).std() * np.sqrt(n)
    return move / scale


def sbc_regime(px: pd.DataFrame) -> pd.DataFrame:
    sig = pd.DataFrame({k: vol_scaled_move(px[v].rename(v)) for k, v in SBC_LEGS.items()}).dropna()
    st, yl = sig["Stocks"], sig["Bonds (10Y yield)"]
    names, groups, hedge = [], [], []
    for a, b in zip(st, yl):
        if min(abs(a), abs(b)) < NO_DRIVER_THRESHOLD:
            names.append(NO_DRIVER); groups.append("NONE"); hedge.append(None)
        else:
            n, g, h = REGIMES[(1 if a > 0 else -1, 1 if b > 0 else -1)]
            names.append(n); groups.append(g); hedge.append(h)
    sig["name"] = names
    sig["driver"] = groups
    sig["bonds_hedge"] = hedge
    sig["dollar"] = np.where(sig["Dollar"] > 0, "dollar up", "dollar down")
    sig["label"] = [f"{n} · {d}" if n != NO_DRIVER else n for n, d in zip(sig["name"], sig["dollar"])]
    sig["days"] = days_in_state(sig["name"])
    sig["conviction"] = sig[["Stocks", "Bonds (10Y yield)"]].abs().min(axis=1)
    return sig


def sbc_forward_returns(px: pd.DataFrame, sbc: pd.DataFrame, horizon: int = 20,
                        lookback_years: int = 10) -> pd.DataFrame:
    """Base rates: what the S&P did over the next `horizon` sessions in each regime (with dollar split)."""
    spx = px["S&P 500"]
    fwd = (spx.shift(-horizon) / spx - 1) * 100
    df = pd.DataFrame({"name": sbc["name"], "dollar": sbc["dollar"], "fwd": fwd.reindex(sbc.index)}).dropna()
    df = df[df.index >= df.index[-1] - pd.DateOffset(years=lookback_years)]
    df.loc[df["name"] == NO_DRIVER, "dollar"] = "–"
    g = df.groupby(["name", "dollar"])["fwd"]
    out = pd.DataFrame({"sessions": g.size(), "median fwd %": g.median(),
                        "% positive": g.apply(lambda x: (x > 0).mean() * 100), "worst %": g.min()})
    order = [(n, d) for n in REGIME_ORDER for d in (["dollar up", "dollar down"] if n != NO_DRIVER else ["–"])]
    return out.reindex([o for o in order if o in out.index])


def stock_bond_corr(px: pd.DataFrame, window: int = 60) -> pd.Series:
    """Rolling correlation of S&P daily returns with daily changes in the 10Y YIELD.
       Positive: stocks fall on days yields fall, i.e. bond PRICES rise -> bonds HEDGE equities.
       Negative: stocks fall on days yields rise -> stocks and bonds lose together -> no hedge.
       (Many sources quote the stock/bond PRICE correlation, which is this number with the sign flipped.)"""
    r = np.log(px["S&P 500"]).diff()
    dy = px["US 10Y yield"].diff()
    return r.rolling(window).corr(dy)


# ---------------------------------------------------------------------------
# 6. factor attribution (rolling ridge)
# ---------------------------------------------------------------------------
FACTORS = {"Rates (10Y, per bp)": "US 10Y yield", "Dollar (DXY %)": "Dollar (DXY)",
           "Credit (HYG %)": "High Yield (HYG)", "Vol (VIX pt)": "VIX"}


def factor_inputs(px: pd.DataFrame, target: str, use_vix: bool) -> tuple[pd.Series, pd.DataFrame]:
    y = px[target].pct_change() * 100
    X = pd.DataFrame({k: (px[v].diff() * 100 if v in YIELD_SERIES else
                          px[v].diff() if v in LEVEL_SERIES else px[v].pct_change() * 100)
                      for k, v in FACTORS.items() if use_vix or v != "VIX"})
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    return d["y"], d.drop(columns="y")


def ridge(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    mu_x, mu_y = X.mean(0), y.mean()
    sd = X.std(0); sd[sd == 0] = 1
    Z = (X - mu_x) / sd
    b = np.linalg.solve(Z.T @ Z + lam * len(y) * np.eye(Z.shape[1]), Z.T @ (y - mu_y))
    beta = b / sd
    return beta, mu_y - mu_x @ beta


def attribution(px: pd.DataFrame, target: str = "S&P 500", window: int = 500, days: int = 60,
                lam: float = 0.05, use_vix: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each of the last `days` sessions: fit betas on the previous `window` sessions (no look-ahead),
       then split that day's return into  sum(beta_k * factor_move_k) + idiosyncratic.
       Returns (daily contributions, latest betas)."""
    y, X = factor_inputs(px, target, use_vix)
    rows, last_beta = [], None
    for t in range(len(y) - days, len(y)):
        beta, a = ridge(X.values[t - window:t], y.values[t - window:t], lam)
        contrib = beta * X.values[t]
        rows.append(np.r_[contrib, y.values[t] - contrib.sum(), y.values[t]])
        last_beta = beta
    cols = list(X.columns) + ["Idiosyncratic", "Actual"]
    contrib = pd.DataFrame(rows, index=y.index[-days:], columns=cols)
    betas = pd.DataFrame({"beta": last_beta}, index=X.columns)
    return contrib, betas


def variance_share(contrib: pd.DataFrame) -> pd.Series:
    """Share of the target's variance explained by each factor over the window.
       cov(contribution_k, actual) / var(actual) -> the shares add up to exactly 100%."""
    act = contrib["Actual"]
    v = act.var()
    parts = contrib.drop(columns="Actual")
    return parts.apply(lambda c: np.cov(c, act)[0, 1] / v * 100)


# ---------------------------------------------------------------------------
# 7. breadth & rotation
# ---------------------------------------------------------------------------
RATIOS = {
    "Equal-weight vs cap-weight (RSP/SPY)": ("S&P 500 Equal Wt (RSP)", "SPY",
        "Up = the average stock beats the giants (healthy breadth). Down = gains concentrated in a few mega-caps."),
    "Small vs large (IWM/SPY)": ("IWM", "SPY",
        "Up = risk appetite and domestic growth optimism. Down = flight to size and quality."),
    "Momentum vs market (MTUM/SPY)": ("MTUM", "SPY",
        "Up = last year's winners keep winning. A sharp drop = a momentum unwind (crowded leaders being sold)."),
    "High beta vs low vol (SPHB/SPLV)": ("SPHB", "SPLV",
        "Up = investors reach for risk. Down = rotation to defensive, low-volatility stocks."),
    "Discretionary vs staples (XLY/XLP)": ("Cons Disc (XLY)", "Staples (XLP)",
        "Up = confidence in the consumer / cycle. Down = defensive rotation."),
    "Semis vs market (SMH/SPY)": ("Semis (SMH)", "SPY",
        "Up = AI / semiconductor leadership intact. Down = the AI trade is unwinding."),
}


def ratio_table(px: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, (a, b, _) in RATIOS.items():
        if a not in px or b not in px:
            continue
        r = (px[a] / px[b]).dropna().rename(label)
        ma = r.rolling(100).mean()
        above = pd.Series(np.where(r > ma, "ABOVE", "BELOW"), index=r.index).where(ma.notna())
        rows.append({"Ratio": label, "5d %": change(r, 5, "x").iloc[-1], "20d %": change(r, 20, "x").iloc[-1],
                     "5d z": zscore_of_change(r, 5).iloc[-1], "vs 100d avg": above.iloc[-1],
                     "days": int(days_in_state(above).iloc[-1]),
                     "gap to 100d %": (r.iloc[-1] / ma.iloc[-1] - 1) * 100})
    return pd.DataFrame(rows).set_index("Ratio")


def pct_above_ma(px: pd.DataFrame, names: list[str], n: int = 50) -> pd.Series:
    sub = px[[c for c in names if c in px]]
    return (sub > sub.rolling(n).mean()).sum(axis=1) / sub.shape[1] * 100


# ---------------------------------------------------------------------------
# 8. volatility
# ---------------------------------------------------------------------------
def vol_frame(px: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=px.index)
    d["VIX"] = px["VIX"]
    d["RV21 S&P"] = realised_vol(px["S&P 500"], 21)
    d["VRP (VIX − RV)"] = d["VIX"] - d["RV21 S&P"]
    if "VIX9D" in px:
        d["VIX9D / VIX"] = px["VIX9D"] / px["VIX"]
    if "VIX3M" in px:
        d["VIX / VIX3M"] = px["VIX"] / px["VIX3M"]
    if "VXN" in px:
        d["VXN"] = px["VXN"]
        d["RV21 Nasdaq"] = realised_vol(px["Nasdaq 100"], 21)
    return d


def vol_cone(s: pd.Series, windows=(10, 21, 63, 126, 252), lookback: int = Z_WINDOW) -> pd.DataFrame:
    """For each window: distribution of rolling realised vol over the last 3y, plus today's value."""
    rows = []
    for w in windows:
        rv = realised_vol(s, w).dropna().iloc[-lookback:]
        rows.append({"window": w, "min": rv.min(), "p25": rv.quantile(.25), "median": rv.median(),
                     "p75": rv.quantile(.75), "max": rv.max(), "current": rv.iloc[-1]})
    return pd.DataFrame(rows).set_index("window")


# ---------------------------------------------------------------------------
# 9. leaders
# ---------------------------------------------------------------------------
def leaders_table(px: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    spx = np.log(px["S&P 500"]).diff()
    rows = []
    for n in names:
        if n not in px:
            continue
        s = px[n].dropna()
        r = np.log(s).diff()
        rv20 = r.rolling(20).std() * np.sqrt(252)
        ret20 = s.iloc[-1] / s.iloc[-21] - 1
        mom_z = ret20 / (rv20.iloc[-1] * np.sqrt(20 / 252))
        both = pd.concat([r, spx], axis=1).dropna().iloc[-63:]
        beta = np.cov(both.iloc[:, 0], both.iloc[:, 1])[0, 1] / both.iloc[:, 1].var()
        hi = s.iloc[-252:].max()
        ybase = s[s.index < pd.Timestamp(s.index[-1].year, 1, 1)]
        rows.append({"Name": n, "Price": s.iloc[-1], "1d %": change(s, 1, n).iloc[-1],
                     "5d %": change(s, 5, n).iloc[-1], "20d %": ret20 * 100,
                     "YTD %": (s.iloc[-1] / ybase.iloc[-1] - 1) * 100 if len(ybase) else np.nan,
                     "Momentum z": mom_z, "Beta (63d)": beta, "Vol 20d %": rv20.iloc[-1] * 100,
                     "From 52w high %": (s.iloc[-1] / hi - 1) * 100,
                     "Trend": trend_regime(s)["state"].iloc[-1]})
    return pd.DataFrame(rows).set_index("Name").sort_values("Momentum z", ascending=False)


# ---------------------------------------------------------------------------
# 10. event study
# ---------------------------------------------------------------------------
def episodes(mask: pd.Series, gap: int = 5) -> int:
    """Count independent episodes: signal days closer than `gap` sessions apart belong to one episode."""
    pos = np.flatnonzero(mask.values)
    if len(pos) == 0:
        return 0
    return int(1 + (np.diff(pos) > gap).sum())


def event_study(s: pd.Series, mask: pd.Series, horizons=(5, 20), exclude_last: int = 0) -> dict:
    """For every past session where `mask` was true, look at the forward change in `s`."""
    name = s.name
    res = {"n": int(mask.sum()), "episodes": episodes(mask)}
    for h in horizons:
        fwd = (change(s, h, name).shift(-h))[mask].dropna()
        if len(fwd) == 0:
            res[h] = None
            continue
        med = fwd.median()
        sign = np.sign(med) if med != 0 else 1
        res[h] = {"median": med, "mean": fwd.mean(),
                  "hit": (np.sign(fwd) == sign).mean() * 100,
                  "pos": (fwd > 0).mean() * 100,
                  "worst": fwd.min() if sign > 0 else fwd.max(),
                  "best": fwd.max() if sign > 0 else fwd.min(),
                  "values": fwd, "n": len(fwd)}
    # unconditional base rate for comparison
    for h in horizons:
        allf = change(s, h, name).shift(-h).dropna()
        res[f"base_{h}"] = {"median": allf.median(), "pos": (allf > 0).mean() * 100}
    return res
