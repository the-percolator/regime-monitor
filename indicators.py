"""
Indicator registry and data pipeline for the growth x inflation quadrant (used by the Indicator lab and Macro backdrop).

Every candidate is turned into a monthly SIGNAL that answers "is this speeding up (+) or slowing down (-)?":

  transform "accel"      level series (payrolls, CPI ...): 3-month annualised growth minus 12-month growth (pp)
  transform "accel_rate" series that already are a rate (wage growth %, breakevens %): 3m average minus 12m average
  transform "diffusion"  survey diffusion indices: 3m average minus 12m average of the index
  transform "claims"     initial claims: -(log 3m average - log 12m average) x 100   (rising claims = slowing growth)

Signals are then:
  1. shifted by the indicator's PUBLICATION LAG, so month t only uses data that was out by the end of month t
  2. turned into a real-time z-score: divided by the indicator's own standard deviation over the trailing 10 years
     (only past data), so different units can be averaged.

A composite is the equal-weighted average of the z-scores of the chosen indicators.
"""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import fred
from data import CACHE_DIR

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "lab_config.json"
START = "1985-01-01"
Z_WINDOW = 120

# key: (label, axis, components, transform, publication lag in months, revised?, kind, in user's list?)
#   components: a FRED id, or ("spread", id_a, id_b) = a - b, or ("avg", [component, ...]) = average of what loads
REGISTRY = {
    # ---------------- growth ----------------
    "claims": ("Initial jobless claims, 4wk (inverted)", "growth", "ICSA", "claims", 0, False, "hard", True),
    "payrolls": ("Nonfarm payrolls", "growth", "PAYEMS", "accel", 1, True, "hard", True),
    "indpro": ("Industrial production", "growth", "INDPRO", "accel", 1, True, "hard", True),
    "retail": ("Real retail sales", "growth", "RRSFS", "accel", 1, True, "hard", True),
    "income": ("Real personal income ex transfers", "growth", "W875RX1", "accel", 1, True, "hard", False),
    "neworders_inv": ("Regional Fed new orders minus inventories (Philly, NY)", "growth",
                      ("avg", [("spread", "NOCDFSA066MSFRBPHI", "IVCDFSA066MSFRBPHI"),
                               ("spread", "NOCDISA066MSFRBNY", "IVCDISA066MSFRBNY")]), "diffusion", 0, False, "survey", True),
    "philly_future": ("Philly Fed future activity (6m ahead)", "growth", "GAFDFSA066MSFRBPHI", "diffusion", 0, False,
                      "survey", True),
    "copper": ("Copper price (IMF, monthly)", "growth", "PCOPPUSDM", "accel", 0, False, "market", True),
    # ---------------- inflation ----------------
    "core_cpi": ("Core CPI", "inflation", "CPILFESL", "accel", 1, False, "hard", True),
    "supercore": ("CPI services less rent of shelter (supercore)", "inflation", "CUSR0000SASL2RS", "accel", 1, False,
                  "hard", True),
    "ppi": ("PPI final demand", "inflation", "PPIFIS", "accel", 1, True, "hard", True),
    "wages": ("Atlanta Fed wage growth tracker", "inflation", "FRBATLWGT3MMAUMHWGO", "accel_rate", 2, False, "hard", True),
    "breakeven_5y5y": ("5y5y forward inflation expectation", "inflation", "T5YIFR", "accel_rate", 0, False, "market", True),
    "prices_paid": ("Regional Fed prices paid (Philly, NY)", "inflation",
                    ("avg", ["PPCDFSA066MSFRBPHI", "PPCDISA066MSFRBNY"]), "diffusion", 0, False, "survey", True),
}
# The dashboard's original quadrant inputs (used when no lab selection has been applied)
DEFAULT_SET = {"growth": ["payrolls", "indpro", "retail", "income"], "inflation": ["core_cpi"]}

# after-the-fact "truth" (latest revised data, not lagged): 6m annualised minus 12m growth
TRUTH = {"growth": ("Philadelphia Fed US coincident index", "USPHCI"),
         "inflation": ("Core PCE price index", "PCEPILFE")}
TRUTH_ALT = {"growth": ("Chicago Fed National Activity Index (3m avg)", "CFNAIMA3")}


# ---------------------------------------------------------------------------
# config (what the Macro backdrop uses)
# ---------------------------------------------------------------------------
def load_config() -> dict | None:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            return None
    return None


def save_config(growth: list[str], inflation: list[str], custom: dict, first_release: bool):
    CONFIG_FILE.write_text(json.dumps({"growth": growth, "inflation": inflation, "custom": custom,
                                       "first_release": first_release,
                                       "saved": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}, indent=2))


def reset_config():
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()


def registry_with_custom(custom: dict | None) -> dict:
    """custom: {key: {"label", "axis", "id", "transform", "lag"}} added by the user in the lab."""
    reg = dict(REGISTRY)
    for k, c in (custom or {}).items():
        reg[k] = (c["label"], c["axis"], c["id"], c["transform"], int(c["lag"]), False, "custom", False)
    return reg


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def fred_ids(comp) -> list[str]:
    if isinstance(comp, str):
        return [comp]
    if comp[0] == "spread":
        return [comp[1], comp[2]]
    return [i for c in comp[1] for i in fred_ids(c)]


def _monthly(s: pd.Series) -> pd.Series:
    return s.resample("MS").mean().dropna()


def load_raw(reg: dict, refresh: bool = False, demo: bool = False, first_release: bool = False) \
        -> tuple[dict, dict, dict, str]:
    """Returns (latest monthly series by id, first-release series by id, failures {id: reason}, source label)."""
    if demo:
        latest, first = make_demo_raw(reg)
        return latest, first, {}, "DEMO (synthetic indicator data)"
    ids = sorted({i for v in reg.values() for i in fred_ids(v[2])} | {v[1] for v in TRUTH.values()}
                 | {v[1] for v in TRUTH_ALT.values()})
    rev_ids = sorted({i for v in reg.values() if v[5] for i in fred_ids(v[2])})
    tag = f"{dt.date.today():%Y%m%d}"
    f_lat, f_first = CACHE_DIR / f"ind_latest_{tag}.json", CACHE_DIR / f"ind_first_{tag}.json"

    def read(f):
        d = json.loads(f.read_text())
        return {k: pd.Series(v["v"], index=pd.to_datetime(v["i"])) for k, v in d.items()}

    def write(f, d):
        CACHE_DIR.mkdir(exist_ok=True)
        f.write_text(json.dumps({k: {"i": [x.strftime("%Y-%m-%d") for x in s.index], "v": s.tolist()} for k, s in d.items()}))

    latest, first, fails = {}, {}, {}
    if f_lat.exists() and not refresh:
        latest = read(f_lat)
    missing = [i for i in ids if i not in latest]
    if missing:
        def get(i):
            try:
                return i, _monthly(fred.series(i, START)), None
            except Exception as exc:
                return i, None, exc.__class__.__name__
        with ThreadPoolExecutor(max_workers=6) as ex:
            for i, s, err in ex.map(get, missing):
                if s is not None and len(s):
                    latest[i] = s
                else:
                    fails[i] = err or "empty"
        if latest:
            for old in CACHE_DIR.glob("ind_latest_*.json"):
                if old != f_lat:
                    old.unlink()
            write(f_lat, latest)
    if first_release and fred.api_key():
        if f_first.exists() and not refresh:
            first = read(f_first)
        miss2 = [i for i in rev_ids if i not in first]
        if miss2:
            def get1(i):
                try:
                    return i, _monthly(fred.series(i, START, first_release=True))
                except Exception:
                    return i, None
            with ThreadPoolExecutor(max_workers=4) as ex:
                for i, s in ex.map(get1, miss2):
                    if s is not None and len(s):
                        first[i] = s
            if first:
                for old in CACHE_DIR.glob("ind_first_*.json"):
                    if old != f_first:
                        old.unlink()
                write(f_first, first)
    src = f"FRED ({fred.mode()})" + (", first-release values via ALFRED" if first else "")
    if not latest:
        latest, first = make_demo_raw(reg)
        return latest, first, fails, "DEMO – FRED download failed"
    return latest, first, fails, src


def make_demo_raw(reg: dict, seed: int = 21) -> tuple[dict, dict]:
    """Synthetic world: latent growth and inflation cycles; each indicator sees them with its own lead and noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, pd.Timestamp.today().normalize() - pd.offsets.MonthBegin(1), freq="MS")
    n = len(idx) + 12
    def cycle(per, ph, amp):
        t = np.arange(n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + 0.25 * rng.standard_normal()
        return amp * (np.sin(2 * np.pi * (t + ph) / per) + 0.6 * x)
    g, inf = cycle(58, 0, 1.0), cycle(70, 18, 1.0)

    def lagged(x, lead):  # lead>0: indicator moves before the cycle
        return np.r_[x[lead:], np.repeat(x[-1], lead)][:len(idx)] if lead >= 0 else np.r_[np.repeat(x[0], -lead), x[:lead]][:len(idx)]

    spec = {  # id: (axis, lead months, noise, kind, base growth/level)
        "ICSA": ("g", 2, 0.5, "claims", 330000), "PAYEMS": ("g", 0, 0.35, "lvl", 0.0012),
        "INDPRO": ("g", 0, 0.7, "lvl", 0.0015), "RRSFS": ("g", 0, 0.9, "lvl", 0.002),
        "W875RX1": ("g", -1, 0.6, "lvl", 0.002), "NOCDFSA066MSFRBPHI": ("g", 3, 0.8, "diff", 8),
        "IVCDFSA066MSFRBPHI": ("g", -2, 0.9, "diff", 2), "NOCDISA066MSFRBNY": ("g", 3, 1.0, "diff", 5),
        "IVCDISA066MSFRBNY": ("g", -2, 1.0, "diff", 3), "GAFDFSA066MSFRBPHI": ("g", 4, 1.4, "diff", 25),
        "PCOPPUSDM": ("g", 1, 1.6, "lvl", 0.003), "USPHCI": ("g", 0, 0.15, "lvl", 0.0018),
        "CFNAIMA3": ("g", 0, 0.25, "level", 0),
        "CPILFESL": ("i", 0, 0.45, "lvl", 0.0021), "CUSR0000SASL2RS": ("i", 1, 0.7, "lvl", 0.0024),
        "PPIFIS": ("i", 2, 1.0, "lvl", 0.002), "FRBATLWGT3MMAUMHWGO": ("i", -3, 0.4, "rate", 3.7),
        "T5YIFR": ("i", 1, 1.5, "rate", 2.3), "PPCDFSA066MSFRBPHI": ("i", 3, 0.9, "diff", 25),
        "PPCDISA066MSFRBNY": ("i", 3, 1.0, "diff", 28), "PCEPILFE": ("i", 0, 0.2, "lvl", 0.0019),
    }
    latest, first = {}, {}
    for sid, (ax, lead, noise, kind, base) in spec.items():
        lat = lagged(g if ax == "g" else inf, lead) + noise * rng.standard_normal(len(idx))
        if kind == "lvl":
            r = base + 0.0012 * np.cumsum(np.r_[0, np.diff(lat)]) * 0 + 0.0011 * lat
            v = 100 * np.exp(np.cumsum(r))
        elif kind == "claims":
            v = base * np.exp(-0.06 * np.cumsum(np.r_[0, np.diff(lat)]) - 0.004 * np.cumsum(lat))
            v = base * np.exp(-0.08 * pd.Series(lat).rolling(6, min_periods=1).mean().values)
        elif kind == "diff":
            v = base + 6 * pd.Series(lat).cumsum().pipe(lambda s: s - s.rolling(24, min_periods=1).mean()).values / 4
        elif kind == "rate":
            v = base + 0.25 * pd.Series(lat).cumsum().pipe(lambda s: s - s.rolling(36, min_periods=1).mean()).values / 3
        else:
            v = 0.4 * lat
        latest[sid] = pd.Series(v, index=idx)
        if sid in ("PAYEMS", "INDPRO", "RRSFS", "W875RX1", "PPIFIS"):  # first releases = latest + revision noise
            r = np.diff(np.log(v), prepend=np.log(v[0])) + 0.0012 * rng.standard_normal(len(idx))
            first[sid] = pd.Series(v[0] * np.exp(np.cumsum(r) - r[0]), index=idx)
    return latest, first


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------
def _resolve(comp, raw: dict) -> tuple[pd.Series | None, list[str]]:
    """Combine raw series per the component spec. Returns (series, list of FRED ids actually used)."""
    if isinstance(comp, str):
        return (raw[comp], [comp]) if comp in raw else (None, [])
    if comp[0] == "spread":
        a, b = raw.get(comp[1]), raw.get(comp[2])
        if a is None or b is None:
            return None, []
        return (a - b).dropna(), [comp[1], comp[2]]
    parts, used = [], []
    for c in comp[1]:
        s, u = _resolve(c, raw)
        if s is not None:
            parts.append(s); used += u
    if not parts:
        return None, []
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True), used


def transform(s: pd.Series, how: str) -> pd.Series:
    s = s.dropna()
    if how == "accel":
        s = s[s > 0]
        return ((s / s.shift(3)) ** 4 - 1) * 100 - (s / s.shift(12) - 1) * 100
    if how in ("accel_rate", "diffusion"):
        return s.rolling(3).mean() - s.rolling(12).mean()
    if how == "claims":
        return -(np.log(s.rolling(3).mean()) - np.log(s.rolling(12).mean())) * 100
    raise ValueError(how)


def signal(key: str, reg: dict, latest: dict, first: dict | None = None, use_first: bool = False) \
        -> tuple[pd.Series | None, list[str]]:
    """Real-time signal for one indicator, indexed by the month in which it was KNOWN (publication lag applied)."""
    label, axis, comp, how, lag, revised, kind, _ = reg[key]
    raw = latest
    if use_first and revised and first:
        raw = {**latest, **{k: v for k, v in first.items()}}
    s, used = _resolve(comp, raw)
    if s is None or len(s) < 30:
        return None, used
    sig = transform(s, how).dropna()
    sig.index = sig.index + pd.DateOffset(months=lag)
    return sig, used


def rt_z(sig: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    """Real-time z-score: scale by the trailing standard deviation (no look-ahead). Not demeaned: zero = no change."""
    sd = sig.rolling(window, min_periods=36).std().shift(1)
    return (sig / sd).clip(-4, 4)


def signals_z(keys: list[str], reg: dict, latest: dict, first: dict | None = None, use_first: bool = False) -> pd.DataFrame:
    out = {}
    for k in keys:
        s, _ = signal(k, reg, latest, first, use_first)
        if s is not None:
            out[k] = rt_z(s)
    return pd.DataFrame(out)


def composite(keys: list[str], reg: dict, latest: dict, first: dict | None = None, use_first: bool = False,
              weights: dict | None = None) -> pd.Series:
    z = signals_z(keys, reg, latest, first, use_first)
    if z.empty:
        return pd.Series(dtype=float)
    w = pd.Series(weights if weights else {c: 1.0 for c in z.columns}).reindex(z.columns).fillna(0)
    avail = z.notna()
    num = (z.fillna(0) * w).sum(axis=1)
    den = (avail * w).sum(axis=1)
    need = max(1, int(np.ceil(len(z.columns) / 2)))
    return (num / den).where(avail.sum(axis=1) >= need).dropna()


def truth(axis: str, latest: dict, alt: bool = False) -> pd.Series | None:
    """After-the-fact acceleration from revised data, indexed by REFERENCE month (not lagged)."""
    sid = (TRUTH_ALT if alt else TRUTH)[axis][1]
    if sid not in latest:
        return None
    s = latest[sid].dropna()
    if sid == "CFNAIMA3":                       # already a growth-vs-trend index: its change = acceleration
        return s - s.rolling(12).mean()
    return (((s / s.shift(6)) ** 2 - 1) * 100 - (s / s.shift(12) - 1) * 100).dropna()


# ---------------------------------------------------------------------------
# quadrant for the Macro backdrop from a selection
# ---------------------------------------------------------------------------
QUAD_NAMES = {(1, -1): "Goldilocks", (1, 1): "Reflation", (-1, 1): "Stagflation", (-1, -1): "Disinflationary slowdown"}


def quadrant_from_selection(sel: dict, latest: dict, first: dict | None = None) -> pd.DataFrame:
    reg = registry_with_custom(sel.get("custom"))
    use_first = bool(sel.get("first_release"))
    g = composite([k for k in sel["growth"] if k in reg], reg, latest, first, use_first)
    i = composite([k for k in sel["inflation"] if k in reg], reg, latest, first, use_first)
    q = pd.DataFrame({"growth": g, "inflation": i}).dropna()
    q["quadrant"] = [QUAD_NAMES[(1 if a >= 0 else -1, 1 if b >= 0 else -1)] for a, b in zip(q["growth"], q["inflation"])]
    q["months"] = q.groupby((q["quadrant"] != q["quadrant"].shift()).cumsum()).cumcount() + 1
    q["known_at"] = q.index + pd.offsets.MonthEnd(0)   # index is already the month the data were known
    return q
