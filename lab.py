"""
Indicator lab: tests which indicators belong in the growth and inflation composites.

Everything is evaluated in REAL TIME: an indicator's value at month t is what was published by the end of month t
(publication lag applied; optionally first-release values from ALFRED). It is compared with an after-the-fact
"truth" built from today's fully revised data.

Per indicator
  nowcast corr ..... correlation of the value known at the end of month t with the truth for month t
                     (so slow publication is penalised: a 1-month lag means you only know last month's reading)
  best lead ........ months ahead (0-6) at which the indicator correlates best with the truth, and that correlation
  sign hit rate .... % of months where indicator and truth agree on accelerating vs decelerating
  turning points ... of the truth's sign changes (that last >= 3 months), % the indicator also flipped within
                     -6..+3 months, and its average timing (positive = early)
  false flips/yr ... indicator sign changes with no truth turn within 3 months
  revision noise ... correlation between the signal built from first-release vs latest data
Composite level
  leave-one-out .... change in composite sign hit rate when each indicator is dropped
  weighting ........ equal weights vs weights re-estimated every year from past data only (walk-forward)
  asset relevance .. forward returns by quadrant + a circular-shift test of whether quadrants separate returns
  robustness ....... hit rate and correlation by sub-period
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import indicators as I

warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Degrees of freedom", category=RuntimeWarning)

EVAL_START = pd.Timestamp("1995-01-01")


def _align(sig: pd.Series, tr: pd.Series, lag: int = 0) -> pd.DataFrame:
    """sig is indexed by month KNOWN; shift back by the publication lag to compare with the same reference month."""
    s = sig.copy()
    s.index = s.index - pd.DateOffset(months=lag)
    d = pd.concat([s.rename("s"), tr.rename("t")], axis=1).dropna()
    return d[d.index >= EVAL_START]


def sign_turns(x: pd.Series, persist: int = 3) -> pd.DatetimeIndex:
    """Dates where the sign changes and the new sign then holds for at least `persist` months."""
    sg = np.sign(x.dropna())
    out = []
    for i in range(1, len(sg) - persist + 1):
        if sg.iloc[i] != sg.iloc[i - 1] and (sg.iloc[i:i + persist] == sg.iloc[i]).all():
            out.append(sg.index[i])
    return pd.DatetimeIndex(out)


def turning_points(sig_ref: pd.Series, tr: pd.Series) -> dict:
    """sig_ref: signal indexed by REFERENCE month. Lead > 0 = indicator flipped before the truth."""
    t_turns = sign_turns(tr)
    s = sig_ref.dropna()
    s_sign = np.sign(s)
    s_flips = s_sign.index[1:][(s_sign.values[1:] != s_sign.values[:-1])]
    hits, leads = 0, []
    for d in t_turns:
        new = np.sign(tr.loc[d])
        win = [f for f in s_flips if -6 <= (d.year - f.year) * 12 + d.month - f.month <= 3 and np.sign(s.loc[f]) == new]
        if win:
            hits += 1
            best = min(win, key=lambda f: abs((d.year - f.year) * 12 + d.month - f.month))
            leads.append((d.year - best.year) * 12 + d.month - best.month)
    years = max((s.index[-1] - s.index[0]).days / 365.25, 1) if len(s) else 1
    false = sum(1 for f in s_flips if not any(abs((d.year - f.year) * 12 + d.month - f.month) <= 3 for d in t_turns))
    return {"turns": len(t_turns), "tp_hit": hits / len(t_turns) * 100 if len(t_turns) else np.nan,
            "tp_lead": np.mean(leads) if leads else np.nan, "false_per_yr": false / years}


def scorecard(reg: dict, latest: dict, first: dict, use_first: bool) -> tuple[pd.DataFrame, dict]:
    rows, sigs = [], {}
    truths = {ax: I.truth(ax, latest) for ax in ("growth", "inflation")}
    for key, (label, axis, comp, how, lag, revised, kind, in_list) in reg.items():
        tr = truths[axis]
        sig, used = I.signal(key, reg, latest, first, use_first)
        row = {"key": key, "Indicator": label, "Axis": axis, "Kind": kind, "Lag (m)": lag,
               "FRED ids used": ", ".join(used) if used else "–", "In your list": in_list}
        if sig is None or tr is None:
            row["Status"] = "unavailable"
            rows.append(row)
            continue
        sigs[key] = sig
        d0 = _align(sig, tr, 0)       # real time: value known at the end of month t vs what happened in month t
        if len(d0) < 36:
            row["Status"] = "too little history"
            rows.append(row)
            continue
        leads = {}
        for k in range(0, 7):                 # does the indicator at reference month m line up with truth at m+k?
            t_fwd = tr.shift(-k)
            dk = _align(sig, t_fwd, 0)
            leads[k] = dk["s"].corr(dk["t"]) if len(dk) > 24 else np.nan
        best_k = max(leads, key=lambda k: -np.inf if np.isnan(leads[k]) else leads[k])
        tp = turning_points(sig[sig.index >= EVAL_START], tr[tr.index >= EVAL_START])
        rev = np.nan
        if revised and first:
            s_lat, _ = I.signal(key, reg, latest, None, False)
            s_fst, _ = I.signal(key, reg, latest, first, True)
            if s_lat is not None and s_fst is not None:
                rev = pd.concat([s_lat, s_fst], axis=1).dropna().corr().iloc[0, 1]
        row.update({"Status": "ok", "Since": d0.index[0].year, "Nowcast corr": leads[0], "Best lead (m)": best_k,
                    "Corr at best lead": leads[best_k],
                    "Sign hit %": (np.sign(d0["s"]) == np.sign(d0["t"])).mean() * 100,
                    "Turns caught %": tp["tp_hit"], "Avg timing (m)": tp["tp_lead"],
                    "False flips / yr": tp["false_per_yr"], "First-vs-latest corr": rev})
        rows.append(row)
    df = pd.DataFrame(rows).set_index("key")
    ok = df["Status"] == "ok"
    if ok.any():
        # score 0-100: rank-average of the four main criteria within each axis
        parts = []
        for ax in ("growth", "inflation"):
            sub = df[ok & (df["Axis"] == ax)]
            if sub.empty:
                continue
            r = pd.concat([sub["Nowcast corr"].rank(pct=True), sub["Corr at best lead"].rank(pct=True),
                           sub["Sign hit %"].rank(pct=True), (-sub["False flips / yr"]).rank(pct=True),
                           sub["Turns caught %"].fillna(0).rank(pct=True)], axis=1).mean(axis=1)
            parts.append(r * 100)
        df["Score"] = pd.concat(parts)
    return df, sigs


def composite_stats(keys: list[str], axis: str, reg: dict, latest: dict, first: dict, use_first: bool,
                    weights: dict | None = None) -> dict:
    tr = I.truth(axis, latest)
    c = I.composite(keys, reg, latest, first, use_first, weights)
    if tr is None or c.empty:
        return {}
    lag = max(reg[k][4] for k in keys) if keys else 0
    d = _align(c, tr, 0)     # composite is indexed by month known; compare with truth for that month (nowcast)
    sref = c.copy()
    tp = turning_points(sref[sref.index >= EVAL_START], tr[tr.index >= EVAL_START])
    return {"n": len(d), "corr": d["s"].corr(d["t"]), "hit": (np.sign(d["s"]) == np.sign(d["t"])).mean() * 100,
            "tp_hit": tp["tp_hit"], "tp_lead": tp["tp_lead"], "false": tp["false_per_yr"], "series": c,
            "max_lag": lag}


def leave_one_out(keys: list[str], axis: str, reg, latest, first, use_first) -> pd.DataFrame:
    base = composite_stats(keys, axis, reg, latest, first, use_first)
    rows = []
    for k in keys:
        rest = [x for x in keys if x != k]
        if not rest:
            continue
        st = composite_stats(rest, axis, reg, latest, first, use_first)
        if not st:
            continue
        rows.append({"key": k, "Indicator": reg[k][0], "Hit % without it": st["hit"],
                     "Change in hit % if dropped": st["hit"] - base["hit"],
                     "Change in corr if dropped": st["corr"] - base["corr"],
                     "Verdict": "KEEP (adds)" if st["hit"] - base["hit"] < -0.5 or st["corr"] - base["corr"] < -0.01
                     else "DROP? (adds nothing)" if st["hit"] - base["hit"] > 0.5 else "NEUTRAL"})
    return pd.DataFrame(rows).set_index("key") if rows else pd.DataFrame()


def redundancy(keys: list[str], reg, latest, first, use_first) -> pd.DataFrame:
    z = I.signals_z(keys, reg, latest, first, use_first)
    z = z[z.index >= EVAL_START]
    z.columns = [reg[c][0][:34] for c in z.columns]
    return z.corr()


def walk_forward(keys: list[str], axis: str, reg, latest, first, use_first) -> pd.DataFrame:
    """Equal weights vs weights re-estimated each January from past data only (weight = max(0, past corr with truth))."""
    tr = I.truth(axis, latest)
    z = I.signals_z(keys, reg, latest, first, use_first)
    if tr is None or z.empty:
        return pd.DataFrame()
    eq = I.composite(keys, reg, latest, first, use_first)
    wf = pd.Series(index=z.index, dtype=float)
    for yr in range(2000, z.index[-1].year + 1):
        past = z[z.index < pd.Timestamp(yr, 1, 1) - pd.DateOffset(months=12)]   # truth needs ~a year to be known
        tpast = tr.reindex(past.index)
        w = {c: max(0.0, past[c].corr(tpast)) if past[c].notna().sum() > 36 else 0.0 for c in z.columns}
        if sum(w.values()) == 0:
            w = {c: 1.0 for c in z.columns}
        cur = z[(z.index >= pd.Timestamp(yr, 1, 1)) & (z.index < pd.Timestamp(yr + 1, 1, 1))]
        ws = pd.Series(w)
        wf.loc[cur.index] = (cur.fillna(0) * ws).sum(axis=1) / (cur.notna() * ws).sum(axis=1).replace(0, np.nan)
    out = []
    for lab, c in [("Equal weights", eq), ("Walk-forward weights", wf.dropna())]:
        d = _align(c, tr, 0)
        d = d[d.index >= pd.Timestamp(2000, 1, 1)]
        out.append({"Method": lab, "Months": len(d), "Corr with truth": d["s"].corr(d["t"]),
                    "Sign hit %": (np.sign(d["s"]) == np.sign(d["t"])).mean() * 100})
    return pd.DataFrame(out).set_index("Method")


def robustness(keys: list[str], axis: str, reg, latest, first, use_first) -> pd.DataFrame:
    st = composite_stats(keys, axis, reg, latest, first, use_first)
    if not st:
        return pd.DataFrame()
    tr = I.truth(axis, latest)
    d = _align(st["series"], tr, 0)
    periods = {"1995–2007": ("1995", "2007"), "2008–2019": ("2008", "2019"), "2020–2021 (Covid)": ("2020", "2021"),
               "2022–now": ("2022", "2100"), "All, excl. 2020–21": None}
    rows = []
    for lab, p in periods.items():
        x = d[(d.index.year < 2020) | (d.index.year > 2021)] if p is None else d.loc[p[0]:p[1]]
        if len(x) < 12:
            continue
        rows.append({"Period": lab, "Months": len(x), "Corr": x["s"].corr(x["t"]),
                     "Sign hit %": (np.sign(x["s"]) == np.sign(x["t"])).mean() * 100})
    return pd.DataFrame(rows).set_index("Period")


# ---------------------------------------------------------------------------
# asset relevance
# ---------------------------------------------------------------------------
def load_assets(demo: bool, px: pd.DataFrame | None = None) -> pd.DataFrame:
    """Monthly asset levels with long history: S&P (Yahoo ^GSPC), gold (FRED/Yahoo), dollar (Yahoo), 10Y (FRED)."""
    if demo and px is not None:
        m = px[["S&P 500", "Gold", "Dollar (DXY)", "US 10Y yield"]].resample("MS").last()
        return m.rename(columns={"US 10Y yield": "US 10Y yield"})
    out = {}
    try:
        import yfinance as yf
        raw = yf.download(["^GSPC", "GC=F", "DX-Y.NYB"], start="1990-01-01", interval="1mo", auto_adjust=True,
                          progress=False)["Close"]
        raw.index = pd.to_datetime(raw.index).tz_localize(None).to_period("M").to_timestamp()
        out["S&P 500"] = raw.get("^GSPC")
        out["Gold"] = raw.get("GC=F")
        out["Dollar (DXY)"] = raw.get("DX-Y.NYB")
    except Exception:
        pass
    try:
        import fred
        out["US 10Y yield"] = fred.series("GS10", "1990-01-01").resample("MS").last()
    except Exception:
        pass
    df = pd.DataFrame({k: v for k, v in out.items() if v is not None})
    if df.empty and px is not None:
        return load_assets(True, px)
    return df


def asset_relevance(q: pd.DataFrame, assets: pd.DataFrame, horizon: int = 3, n_shift: int = 300,
                    seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward `horizon`-month returns by quadrant known at the START of the period.
       Separation = spread between best and worst quadrant mean. p-value from circular shifts of the quadrant
       labels against returns (keeps the persistence of both series, unlike a plain shuffle)."""
    rng = np.random.default_rng(seed)
    lab = pd.Series(q["quadrant"].values, index=pd.DatetimeIndex(q["known_at"]).to_period("M").to_timestamp())
    lab = lab[~lab.index.duplicated(keep="last")]                                   # month in which it was known
    rows, tests = {}, []
    for a in assets.columns:
        s = assets[a].dropna()
        if a == "US 10Y yield":
            fwd = (s.shift(-horizon) - s) * 100          # bp
        else:
            fwd = (s.shift(-horizon) / s - 1) * 100
        fwd = fwd.shift(-1)                               # start the clock at the next month (label known at month end)
        d = pd.concat([lab.rename("q"), fwd.rename("r")], axis=1).dropna()
        d = d[d.index >= EVAL_START]
        if len(d) < 60:
            continue
        means = d.groupby("q")["r"].mean()
        rows[a] = means
        actual = means.max() - means.min()
        qs, rs = d["q"].values, d["r"].values
        cnt = 0
        for _ in range(n_shift):
            k = rng.integers(12, len(d) - 12)
            m2 = pd.Series(rs).groupby(np.roll(qs, k)).mean()
            cnt += (m2.max() - m2.min()) >= actual
        tests.append({"Asset": a, "Months": len(d), "Best quadrant": means.idxmax(), "Worst quadrant": means.idxmin(),
                      "Spread best-worst": actual, "p-value (circular shift)": cnt / n_shift})
    tab = pd.DataFrame(rows).T if rows else pd.DataFrame()
    return tab, pd.DataFrame(tests).set_index("Asset") if tests else pd.DataFrame()
