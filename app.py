"""
Equity Regime Monitor – Streamlit app.  Start it with:  python run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find data.py / models.py from any cwd

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import crypto as K
import data as D
import fixed_income as F
import indicators as I
import lab as LAB
import macro as Q
import models as M

st.set_page_config(page_title="Regime Monitor", layout="wide")


# ---------------------------------------------------------------------------
# Compatibility with older Streamlit versions (e.g. the one bundled with Anaconda):
# newer Streamlit uses width="stretch", older versions use use_container_width=True.
# ---------------------------------------------------------------------------
def _install_width_compat():
    import inspect
    from streamlit.delta_generator import DeltaGenerator

    def supports_stretch(fn) -> bool:
        try:
            p = inspect.signature(fn).parameters.get("width")
        except (TypeError, ValueError):
            return False
        return p is not None and any(t in str(p.annotation) for t in ("Width", "stretch"))

    def make(fn):
        def wrapped(*args, width="stretch", **kwargs):
            if width == "stretch":
                kwargs.setdefault("use_container_width", True)
            elif width is not None:
                kwargs["width"] = width
            return fn(*args, **kwargs)
        return wrapped

    for name in ("button", "plotly_chart", "dataframe"):
        if not supports_stretch(getattr(DeltaGenerator, name)):
            setattr(DeltaGenerator, name, make(getattr(DeltaGenerator, name)))
        if not supports_stretch(getattr(st, name)):
            setattr(st, name, make(getattr(st, name)))


_install_width_compat()
if not hasattr(st, "segmented_control"):  # added in Streamlit 1.40
    st.segmented_control = lambda label, options, default=None, key=None, label_visibility="visible", **_: \
        st.radio(label, options, index=list(options).index(default) if default in options else 0,
                 horizontal=True, key=key, label_visibility=label_visibility)

from ui import *  # noqa: E402,F401,F403  (colours, card(), base_layout(), ...)

inject_css()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
DEMO_FLAG = "--demo" in sys.argv

SECTIONS = ["Summary", "Cross-asset", "Equities", "Fixed income", "Crypto"]
with st.sidebar:
    st.markdown("### Section")
    section = st.radio("Section", SECTIONS, key="section", label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Data")
    demo = st.toggle("Use demo data (offline)", value=DEMO_FLAG,
                     help="Synthetic prices, for trying the dashboard without internet.")
    refresh = st.button("Refresh data", width="stretch")
    if refresh:
        st.cache_data.clear()


@st.cache_data(show_spinner="Downloading ~40 daily series from Yahoo Finance (first run of the day takes ~20 s)…")
def get_data(demo: bool, refresh_token: int):
    return D.load_closes(refresh=refresh_token > 0, demo=demo)


if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = 0
if refresh:
    st.session_state.refresh_token += 1
px, source = get_data(demo, st.session_state.refresh_token)
asof = px.index[-1]


@st.cache_data(show_spinner="Downloading monthly macro data from FRED…")
def get_macro(demo: bool, refresh_token: int):
    return Q.load_macro(refresh=refresh_token > 0, demo=demo)


macro, macro_source = get_macro(demo, st.session_state.refresh_token)
quad = Q.quadrant_frame(macro)
qnow = Q.describe(quad)
MACRO_MODE = "default"


@st.cache_data(show_spinner="Downloading indicator data from FRED (first time today: ~30 series)…")
def get_ind_raw(demo: bool, refresh_token: int, first_release: bool, custom_json: str):
    import json as _json
    reg = I.registry_with_custom(_json.loads(custom_json))
    return I.load_raw(reg, refresh=refresh_token > 0, demo=demo, first_release=first_release)


LAB_CFG = I.load_config()
if LAB_CFG:
    try:
        import json as _json
        _lat, _fst, _fails, _src = get_ind_raw(demo, st.session_state.refresh_token, bool(LAB_CFG.get("first_release")),
                                               _json.dumps(LAB_CFG.get("custom", {}), sort_keys=True))
        _q = I.quadrant_from_selection(LAB_CFG, _lat, _fst)
        if len(_q) > 24:
            quad, qnow, MACRO_MODE = _q, Q.describe(_q), "lab"
            macro_source = f"Indicator lab selection (saved {LAB_CFG.get('saved', '?')}) · {_src}"
    except Exception as _exc:  # fall back to the default quadrant
        macro_source += f" · lab selection could not be built ({_exc.__class__.__name__})"


@st.cache_data(show_spinner="Downloading daily rates and credit data from FRED…")
def get_fi(px: pd.DataFrame, demo: bool, refresh_token: int):
    return F.load_fi(px, refresh=refresh_token > 0, demo=demo)


fi, fi_source = get_fi(px, demo, st.session_state.refresh_token)
fi = fi.loc[:fi["UST 10Y"].last_valid_index()]
fi_asof = fi.index[-1]

FI_NAMES = {
    "Treasury yields": ["UST 3M", "UST 2Y", "UST 5Y", "UST 10Y", "UST 30Y"],
    "Curve & Fed": ["2s10s curve", "3m10y curve", "5s30s curve", "Fed funds (EFFR)", "2Y minus Fed funds"],
    "Inflation & real yields": ["10Y breakeven inflation", "5Y breakeven inflation", "10Y real yield (TIPS)",
                                "5Y real yield (TIPS)", "10Y term premium (Kim-Wright)"],
    "Credit": ["IG credit spread (OAS)", "HY credit spread (OAS)"],
}
FI_PX = ["MOVE (rate vol)", "Short Tsy (SHY)", "Interm. Tsy (IEF)", "Long Bond (TLT)", "TIPS (TIP)",
         "IG Corp (LQD)", "High Yield (HYG)"]


@st.cache_data
def compute_fi(px: pd.DataFrame, fi: pd.DataFrame):
    fnames = [n for g in FI_NAMES.values() for n in g if n in fi]
    t1 = M.dashboard_table(fi, fnames)
    t2 = M.dashboard_table(px, [n for n in FI_PX if n in px])
    fi_table = pd.concat([t1, t2])
    crv = F.curve_regime(fi)
    split = F.fed_vs_term_premium(fi)
    rvb = F.real_vs_breakeven(fi)
    credit = F.credit_confirmation(px, fi)
    rvol = F.rate_vol(px, fi)
    mk_fi = {n: M.markov_summary(fi[n].dropna().rename(n)) for n in fnames}
    mk_fi.update({n: M.markov_summary(px[n].dropna().rename(n)) for n in FI_PX if n in px})
    mk_fi = {k: v for k, v in mk_fi.items() if v}
    return fi_table, crv, split, rvb, credit, rvol, mk_fi


fi_table, crv, split, rvb, credit, rvol, mk_fi = compute_fi(px, fi)


CR_LIST = [k for k in list(D.CRYPTO) + list(D.CRYPTO_ETF) + list(D.CRYPTO_EQ) if k in px]


@st.cache_data
def compute_crypto(px: pd.DataFrame, fi: pd.DataFrame):
    t = M.dashboard_table(px, CR_LIST)
    mk = {n: M.markov_summary(px[n].dropna().rename(n)) for n in CR_LIST}
    return t, {k: v for k, v in mk.items() if v}, K.macro_corr(px, fi, "Bitcoin")


cr_table, mk_cr, cr_corr = compute_crypto(px, fi)


@st.cache_data
def compute_core(px: pd.DataFrame):
    names = list(D.INDICES) + list(D.SECTORS) + list(D.DRIVERS) + list(D.VOL)
    table = M.dashboard_table(px, names)
    sbc = M.sbc_regime(px)
    sbcfwd = M.sbc_forward_returns(px, sbc)
    sbcorr = M.stock_bond_corr(px)
    vol = M.vol_frame(px)
    ratios = M.ratio_table(px)
    leaders = M.leaders_table(px, list(D.LEADERS))
    markov = {n: M.markov_summary(px[n].rename(n)) for n in
              list(D.INDICES) + list(D.SECTORS) + ["US 10Y yield", "Dollar (DXY)", "High Yield (HYG)", "VIX"]
              if n in px}
    breadth = M.pct_above_ma(px, list(D.SECTORS) + list(D.LEADERS), 50)
    return table, sbc, sbcfwd, sbcorr, vol, ratios, leaders, markov, breadth


table, sbc, sbcfwd, sbcorr, vol, ratios, leaders, markov, breadth = compute_core(px)

with st.sidebar:
    st.caption(f"**Markets (Yahoo):** {source} · last close {asof:%a %d %b}")
    st.caption(f"**Rates & credit (FRED):** {fi_source} · to {fi_asof:%a %d %b}")
    st.caption(f"**Macro (FRED):** {macro_source}")
    _extra = []
    for lab, src in _extra:
        st.caption(f"**{lab}:** {src}")
    bad = [lab for lab, src in [("Market prices", source), ("Rates & credit", fi_source), ("Macro", macro_source)] + _extra
           if "DEMO" in src or "STALE" in src]
    if len(bad) == 3 + len(_extra):
        st.warning("Not live: all data is demo or stale. Numbers are for illustration only.")
    elif bad:
        st.warning("Not live: " + ", ".join(bad) + ". Those numbers are demo or stale; everything else is live.")
    if "DEMO" in fi_source + macro_source and not demo:
        st.info("FRED tip: start with your API key, e.g.  FRED_API_KEY=yourkey python3 run.py")
    st.markdown("---")
    st.markdown("### Glossary")
    st.markdown("""
- **5d z**: this week's move vs. the normal size of weekly moves over 3 years. ±1.5 or more = unusual.
- **1y %ile**: where today's level sits in its 1-year range (0 = lowest, 100 = highest).
- **Trend**: 50/200-day moving-average state (slow, weeks–months).
- **Trend × volatility state**: price above/below its 21-day average × volatility above/below normal (fast, days–weeks).
- **Market-implied macro regime**: which kind of news drives markets (growth or rates), read from stocks and yields.
- **Quadrant**: is the economy's growth and inflation speeding up or slowing down (monthly)?
- **Curve regime**: bull/bear (yields down/up) × steepener/flattener over 20 days.
- **OAS**: credit spread over Treasuries, in bp.
- **bp**: basis point = 0.01%.
""")

# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------
st.markdown(f"# Regime Monitor · {section}")
st.markdown(f'<div class="sub">US markets · prices to <b>{asof:%d %b %Y}</b> · rates &amp; credit to '
            f'<b>{fi_asof:%d %b %Y}</b> · macro to <b>{qnow["as_of"]:%b %Y}</b></div>', unsafe_allow_html=True)



spx = px["S&P 500"]
tr_spx = M.trend_regime(spx)
mk_spx = markov["S&P 500"]
sbc_now = sbc.iloc[-1]
v_now = vol.iloc[-1]
row = table.loc["S&P 500"]
term = round(float(v_now.get("VIX / VIX3M", np.nan)), 2)

# ===========================================================================
# TAB 1  TODAY
# ===========================================================================
def page_eq_today():
    # ---- KPI strip ---------------------------------------------------------
    c = st.columns(5)
    c[0].metric("S&P 500", f"{spx.iloc[-1]:,.0f}", f"{row['1d']:+.2f}% today · {row['5d']:+.1f}% 5d",
                delta_color="normal" if row["1d"] >= 0 else "inverse")
    nd = table.loc["Nasdaq 100"]
    c[1].metric("Nasdaq 100", f"{px['Nasdaq 100'].iloc[-1]:,.0f}", f"{nd['1d']:+.2f}% today · {nd['5d']:+.1f}% 5d")
    vx = table.loc["VIX"]
    c[2].metric("VIX", f"{vx['Level']:.1f}", f"{vx['1d']:+.1f} pt · {M.ordinal(vx['1y %ile'])} %ile (1y)",
                delta_color="inverse")
    ty = table.loc["US 10Y yield"]
    c[3].metric("US 10Y yield", f"{ty['Level']:.2f}%", f"{ty['1d']:+.0f} bp · {ty['5d']:+.0f} bp 5d", delta_color="off")
    hy = table.loc["High Yield (HYG)"]
    c[4].metric("High yield (HYG)", f"{hy['Level']:.2f}", f"{hy['1d']:+.2f}% · {hy['5d']:+.1f}% 5d")

    st.write("")
    # ---- regime cards ------------------------------------------------------
    st.markdown("#### Where are equities? Four lenses")
    k = st.columns(4)
    br_now, br_5 = breadth.iloc[-1], breadth.iloc[-6]
    k[0].markdown(card("1 · Breadth", f"{br_now:.0f}% above 50-day",
                       f"Share of sectors and mega-caps above their 50-day average (5 days ago {br_5:.0f}%). "
                       + ("Broad weakness." if br_now < 30 else "Broad participation." if br_now > 70 else "Mixed."),
                       DOWN if br_now < 30 else UP if br_now > 70 else NEUTRAL), unsafe_allow_html=True)
    st_tr = tr_spx["state"].iloc[-1]
    k[1].markdown(card("2 · Slow trend · S&P 500", pill(st_tr),
                       f"For {int(tr_spx['days'].iloc[-1])} sessions. " + M.trend_flip_text(tr_spx),
                       STATE_COLOR[st_tr]), unsafe_allow_html=True)
    k[2].markdown(card("3 · Trend × volatility state · S&P 500", pill(mk_spx["state"]),
                       f"For {mk_spx['days']} sessions. States like this have lasted ~{mk_spx['exp_len']:.0f} days on average "
                       f"(P(stay) {mk_spx['p_stay']:.0%}) → <b>{mk_spx['style']}</b>.",
                       STATE_COLOR[mk_spx["state"]]), unsafe_allow_html=True)
    term = round(float(v_now.get("VIX / VIX3M", np.nan)), 2)
    term_state = ("INVERTED (stress)" if term > 1 else "FLAT (on the edge of inversion)" if term >= 0.97
                  else "NORMAL (upward sloping)")
    k[3].markdown(card("4 · Volatility regime", f"VIX {v_now['VIX']:.1f} · term {term_state.split()[0]}",
                       f"VIX/VIX3M = {term:.2f} ("
                       + ("above 1: near-term fear priced above longer-term" if term > 1 else
                          "close to 1: near-term vol almost as high as longer-term" if term >= 0.97 else
                          "below 1: calm, normal shape") + "). "
                       f"Implied minus realised vol = {v_now['VRP (VIX − RV)']:+.1f} pts.",
                       BAD if term > 1 else WARN if term >= 0.97 else GOOD), unsafe_allow_html=True)

    st.write("")
    # ---- the read -----------------------------------------------------------
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### The read, in plain English")
        sc = sbcorr.iloc[-1]
        mom = ratios.loc["Momentum vs market (MTUM/SPY)"] if "Momentum vs market (MTUM/SPY)" in ratios.index else None
        ew = ratios.loc["Equal-weight vs cap-weight (RSP/SPY)"] if "Equal-weight vs cap-weight (RSP/SPY)" in ratios.index else None
        lines = []
        lines.append(f"**Direction.** The S&P 500 moved {row['5d']:+.1f}% this week "
                     f"({'an unusually large' if abs(row['5d z']) >= 1.5 else 'a normal-sized'} move, z = {row['5d z']:+.1f}). "
                     f"The slow trend reads **{st_tr}**; the fast state reads **{mk_spx['state']}**.")
        lines.append(f"**Driver.** Over the last 20 sessions stocks are {'up' if sbc_now['Stocks'] > 0 else 'down'} "
                     f"({sbc_now['Stocks']:+.1f}σ) and yields are {'up' if sbc_now['Bonds (10Y yield)'] > 0 else 'down'} "
                     f"({sbc_now['Bonds (10Y yield)']:+.1f}σ): **{sbc_now['label'].lower()}**. "
                     + ("That is a growth story, so Treasuries are behaving as a hedge."
                        if sbc_now["bonds_hedge"] else
                        "Too small a move to say what is driving markets." if sbc_now["name"] == M.NO_DRIVER else
                        "That is a rates/inflation story, so Treasuries are not protecting equity holders."))
        lines.append(f"**Stock–bond link.** 60-day correlation of stocks with yields is {sc:+.2f} → "
                     + ("bonds have been rallying on down days for stocks (hedge working)." if sc > 0.1 else
                        "bonds and stocks have been falling together (hedge not working)." if sc < -0.1 else
                        "no stable relationship right now."))
        if mom is not None and ew is not None:
            lines.append(f"**Under the surface.** Momentum vs market {mom['5d %']:+.1f}% in 5 days (z {mom['5d z']:+.1f}); "
                         f"equal-weight vs cap-weight {ew['5d %']:+.1f}% (z {ew['5d z']:+.1f}). "
                         + ("Big caps / momentum names are lagging the average stock: this looks like a **leadership unwind**, not broad selling."
                            if ew["5d %"] > 0 and row["5d"] < 0 else
                            "Selling is broad: the average stock is doing no better than the index."
                            if row["5d"] < 0 else "Breadth and leadership are moving in line with the index."))
        lines.append(f"**Breadth.** {breadth.iloc[-1]:.0f}% of sectors and mega-caps are above their 50-day average "
                     f"(5 days ago: {breadth.iloc[-6]:.0f}%).")
        lines.append(f"**Volatility.** VIX at {v_now['VIX']:.1f} sits at the {M.ordinal(vx['1y %ile'])} percentile of its 1-year range; "
                     f"term structure is {term_state.lower()}.")
        st.markdown('<div class="read">' + "<br><br>".join(
            __import__("re").sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", l) for l in lines) + "</div>", unsafe_allow_html=True)
        st.caption("Generated mechanically from the models below. Thresholds, not judgment: use it as a checklist.")
    with right:
        st.markdown("#### What's stretched")
        items = M.stretched(table.drop(index=[i for i in ["VIX9D", "VIX3M"] if i in table.index]))
        rmini = ratios[ratios["5d z"].abs() >= 1.5]
        for name, r in rmini.iterrows():
            items.append(f"**{name}** – 5-day move {r['5d %']:+.1f}% (z {r['5d z']:+.1f})")
        if items:
            st.markdown("\n".join(f"- {t}" for t in items[:12]))
        else:
            st.markdown("Nothing unusual: no 5-day move beyond ±1.5σ and no level at a 1-year extreme.")
        st.caption("Flags: |5d z| ≥ 1.5, level ≤ 5th or ≥ 95th percentile of 1 year, or a trend flip in the last 5 sessions.")

    st.markdown("#### Market dashboard")
    show = table.drop(index=[i for i in ["VIX9D", "VIX3M"] if i in table.index]).copy()
    groups = {"Indices": list(D.INDICES), "Sectors": list(D.SECTORS), "Cross-asset": list(D.DRIVERS) + ["VIX", "VXN"]}
    gsel = st.segmented_control("Group", list(groups), default="Indices", key="dash_group", label_visibility="collapsed")
    show = show.loc[[n for n in groups[gsel or "Indices"] if n in show.index]]
    fmt = {"Level": "{:,.2f}", "1d": "{:+.2f}", "5d": "{:+.2f}", "20d": "{:+.2f}", "YTD": "{:+.1f}",
           "5d z": "{:+.2f}", "1y %ile": "{:.0f}", "Trend days": "{:d}"}
    sty = (show.style.format(fmt, na_rep="–")
           .map(signed_color, subset=["1d", "5d", "20d", "YTD"])
           .map(z_color, subset=["5d z"]).map(state_style, subset=["Trend", "Trend × vol"]))
    st.dataframe(sty, width="stretch", height=min(40 + 35 * len(show), 520))
    st.caption("Changes in % for prices, bp for yields, points for VIX/VXN (see Unit). Shaded z-scores are ±1.5σ or more.")

# ===========================================================================
# TAB 2  REGIMES
# ===========================================================================
def page_regimes(universe: list, frame: pd.DataFrame, markov_d: dict, prefix: str):
    st.markdown("Two engines, two horizons. They often disagree, and that is information: "
                "a market can be in a long-term uptrend (slow) while having a violent down-week (fast). "
                "The market-implied macro regime is under Cross-asset.")
    sub = st.segmented_control("Engine", ["Trend (50/200d)", "Trend × volatility state"],
                               default="Trend (50/200d)", key=prefix + "eng")
    sub = sub or "Trend (50/200d)"

    if sub == "Trend (50/200d)":
        a, b = st.columns([1, 3])
        with a:
            pick = st.selectbox("Series", [u for u in universe if u in frame], key=prefix + "tr_pick")
            yrs = st.radio("Window", [1, 2, 5], index=1, horizontal=True, format_func=lambda x: f"{x}y",
                           key=prefix + "tr_y")
        tr = M.trend_regime(frame[pick].dropna().rename(pick))
        with a:
            s_ = tr["state"].iloc[-1]
            st.markdown(card("State", pill(s_), f"{int(tr['days'].iloc[-1])} sessions in state.<br>{M.trend_flip_text(tr)}",
                             STATE_COLOR[s_]), unsafe_allow_html=True)
        t = last_n_years(tr, yrs)
        fig = go.Figure()
        # shade background by state
        blocks = (t["state"] != t["state"].shift()).cumsum()
        for _, g in t.groupby(blocks):
            stt = g["state"].iloc[0]
            if isinstance(stt, str):
                fig.add_vrect(x0=g.index[0], x1=g.index[-1], fillcolor=STATE_COLOR[stt], opacity=0.08, line_width=0)
        fig.add_scatter(x=t.index, y=t["px"], name="Level", line=dict(color="#0b0b0b", width=1.6))
        fig.add_scatter(x=t.index, y=t["ma50"], name="50-day avg", line=dict(color=SERIES[1], width=2))
        fig.add_scatter(x=t.index, y=t["ma200"], name="200-day avg", line=dict(color=SERIES[0], width=2))
        with b:
            st.plotly_chart(base_layout(fig, 420, f"{pick} with 50/200-day averages · background = state "
                                                  "(blue up, red down, gray mixed)"), width="stretch")
        explain("""
**Rule.** UPTREND when price > 50-day average > 200-day average. DOWNTREND when price < 50-day < 200-day.
Everything else is MIXED: the averages disagree with each other or with the price (typical during a pullback inside an
uptrend, or a bounce inside a downtrend).

**Why use it.** It is slow on purpose. It filters out noise and tells you which side of the market has been winning
over months. Trend-following strategies are built on exactly this kind of rule.

**Weakness.** It is late at turning points, and in sideways markets it flips a lot (whipsaw).
""")

    elif sub == "Trend × volatility state":
        a, b = st.columns([1, 2])
        with a:
            opts = list(markov_d)
            pick = st.selectbox("Series", opts, key=prefix + "mk_pick")
            ms = markov_d[pick]
            st.markdown(card("Current state", pill(ms["state"]),
                             f"{ms['days']} sessions in state · P(stay tomorrow) {ms['p_stay']:.0%} · "
                             f"typical length ~{ms['exp_len']:.1f} days → <b>{ms['style']}</b>",
                             STATE_COLOR[ms["state"]]), unsafe_allow_html=True)
            tm = ms["matrix"]
            hm = go.Figure(go.Heatmap(z=tm.values * 100, x=[f"→ {c}" for c in tm.columns], y=tm.index,
                                      colorscale=[[0, "#fcfcfb"], [1, "#184f95"]], zmin=0, zmax=100,
                                      text=np.round(tm.values * 100).astype(int), texttemplate="%{text}%",
                                      hovertemplate="today %{y} %{x}: %{z:.0f}%<extra></extra>", showscale=False))
            hm.update_yaxes(autorange="reversed")
            st.plotly_chart(base_layout(hm, 300, "Transition odds (last 3 years): today → tomorrow", False),
                            width="stretch")
        with b:
            fr = last_n_years(ms["frame"], 1)
            fig = go.Figure()
            fig.add_scatter(x=fr.index, y=fr["px"], name="Price", line=dict(color="#c3c2b7", width=1.5), showlegend=False)
            fig.add_scatter(x=fr.index, y=fr["ema"], name="21-day EMA", line=dict(color="#52514e", width=1, dash="dot"))
            for stt in M.MARKOV_STATES:
                g = fr[fr["state"] == stt]
                fig.add_scatter(x=g.index, y=g["px"], mode="markers", name=stt,
                                marker=dict(color=STATE_COLOR[stt], size=6))
            st.plotly_chart(base_layout(fig, 400, f"{pick} – last year, each day coloured by trend × volatility state"),
                            width="stretch")
        st.markdown("##### Panel: fast state across the market")
        pan = pd.DataFrame({n: {"State": v["state"], "Days in state": v["days"], "P(stay)": v["p_stay"] * 100,
                                "Typical length (d)": v["exp_len"], "Behaviour": v["style"]}
                            for n, v in markov_d.items() if v}).T
        st.dataframe(pan.style.format({"P(stay)": "{:.0f}%", "Typical length (d)": "{:.1f}"})
                     .map(state_style, subset=["State"]), width="stretch", height=min(40 + 35 * len(pan), 600))
        explain("""
**Two questions, four boxes.**
1. *Direction*: is price above (Up) or below (Down) its 21-day exponential moving average?
2. *Volatility*: is 20-day realised volatility above (High-vol) or below (Low-vol) its own 1-year median?

| | Low-vol | High-vol |
|---|---|---|
| **Up** | Steady grind higher, the friendliest state | Rally with big swings (often a squeeze or a rebound) |
| **Down** | Orderly drift lower | Sell-off with high volatility, the stress state |

**State persistence.** We count, over the last 3 years, how often each state was followed by each state the next day.
That table (the heatmap) is an empirical *transition matrix*; the diagonal is P(stay). If P(stay) = 0.80, the expected
length of a spell is 1 / (1 − 0.80) = 5 days.

**How to use it.** Long expected spells (≥ 6 days) mean states persist, so moves tend to continue (momentum).
Short spells mean the state flips often, so stretched moves tend to snap back (mean reversion).

**Naming note.** The states come from two simple rules; only the transition odds are "Markov". This is not a
Hamilton-style Markov regime-switching model, where the states are hidden and estimated statistically.
""")



# ===========================================================================
# MARKET-IMPLIED MACRO REGIME
# ===========================================================================
def regime_color(name):
    return M.REGIME_COLOR.get(name, NEUTRAL)


def regime_note(r) -> str:
    if r["name"] == M.NO_DRIVER:
        return (f"Stocks {r['Stocks']:+.2f}σ, yields {r['Bonds (10Y yield)']:+.2f}σ over 20 days: at least one is "
                f"below {M.NO_DRIVER_THRESHOLD}σ. {int(r['days'])} session(s).")
    return (f"Driver group: <b>{r['driver'].lower()}</b>. Bonds {'<b>hedge</b>' if r['bonds_hedge'] else '<b>do not hedge</b>'} "
            f"stocks in this regime. {r['dollar'].capitalize()}. {int(r['days'])} session(s) in regime; strength "
            f"{r['conviction']:.1f}σ ({'clear' if r['conviction'] >= 1 else 'moderate: can flip'}).")


def page_sbc():
    top = st.columns([1.1, 1.9])
    with top[0]:
        st.markdown(card("Market-implied macro regime", sbc_now["label"], regime_note(sbc_now),
                         regime_color(sbc_now["name"])), unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<div class="explain">{M.SBC_EXPLAIN[sbc_now["name"]]}</div>'.replace("**", ""),
                    unsafe_allow_html=True)
        st.write("")
        legs = sbc_now[["Stocks", "Bonds (10Y yield)", "Dollar"]]
        bar = go.Figure(go.Bar(x=legs.values, y=["Stocks (S&P)", "10Y yield", "Dollar (DXY)"], orientation="h",
                               marker_color=[UP if v > 0 else DOWN for v in legs.values],
                               text=[f"{v:+.2f}σ" for v in legs.values], textposition="inside",
                               insidetextanchor="middle", textfont=dict(color="#fff"),
                               hovertemplate="%{y}: %{x:+.2f}σ<extra></extra>"))
        bar.update_yaxes(autorange="reversed")
        lim = max(2.5, legs.abs().max() * 1.3)
        bar.update_xaxes(range=[-lim, lim], zeroline=True, zerolinecolor="#898781", title="σ (0 = no move)")
        for xv in (-M.NO_DRIVER_THRESHOLD, M.NO_DRIVER_THRESHOLD):
            bar.add_vline(x=xv, line_dash="dot", line_color=MUTED)
        st.plotly_chart(base_layout(bar, 220, "20-day moves, vol-scaled (dotted: too small to read)", False),
                        width="stretch")
    with top[1]:
        yrs = st.radio("Window", [1, 2, 5], index=1, horizontal=True, format_func=lambda x: f"{x}y", key="sbc_y")
        sb = last_n_years(sbc, yrs)
        fig = go.Figure()
        for i, (col, lab) in enumerate([("Stocks", "Stocks"), ("Bonds (10Y yield)", "10Y yield"), ("Dollar", "Dollar")]):
            fig.add_scatter(x=sb.index, y=sb[col], name=lab, line=dict(color=SERIES[i], width=1.6))
        fig.add_hline(y=0, line_color="#898781", line_width=1)
        st.plotly_chart(base_layout(fig, 280, "20-day vol-scaled moves over time (σ). Above 0 = up"), width="stretch")
        rib = go.Figure()
        for nm in M.REGIME_ORDER:
            g = sb[sb["name"] == nm]
            rib.add_scatter(x=g.index, y=[nm] * len(g), mode="markers", text=g["label"],
                            marker=dict(symbol="square", size=7, color=M.REGIME_COLOR[nm]), showlegend=False,
                            hovertemplate="%{text}<br>%{x|%d %b %Y}<extra></extra>")
        rib.update_yaxes(categoryorder="array", categoryarray=M.REGIME_ORDER[::-1])
        st.plotly_chart(base_layout(rib, 250, "Which regime, when", False), width="stretch")

    st.markdown("##### The regimes and what the S&P did next (20 sessions, last 10 years)")
    rows = []
    for (n, d), r in sbcfwd.iterrows():
        k = next((k for k, v in M.REGIMES.items() if v[0] == n), None)
        rows.append({"Regime": n, "Dollar": d, "Stocks": ("▲" if k[0] > 0 else "▼") if k else "small",
                     "10Y yield": ("▲" if k[1] > 0 else "▼") if k else "small",
                     "Bonds hedge?": ("yes" if M.REGIMES[k][2] else "no") if k else "–",
                     "sessions": r["sessions"], "median fwd %": r["median fwd %"], "% positive": r["% positive"],
                     "worst %": r["worst %"],
                     "now": "◀ now" if n == sbc_now["name"] and (d == sbc_now["dollar"] or n == M.NO_DRIVER) else ""})
    reg = pd.DataFrame(rows).set_index(["Regime", "Dollar"])
    st.dataframe(reg.style.format({"median fwd %": "{:+.2f}", "% positive": "{:.0f}", "worst %": "{:+.1f}",
                                   "sessions": "{:.0f}"}, na_rep="–")
                 .map(signed_color, subset=["median fwd %"])
                 .apply(lambda r: ["background-color: rgba(42,120,214,.12)" if r["now"] else "" for _ in r], axis=1),
                 width="stretch")
    st.caption("Forward returns overlap day to day, so 'sessions' overstates the number of independent episodes. "
               "Read differences between regimes as tendencies, not edges.")

    st.markdown("##### Are bonds hedging stocks? (60-day correlation of S&P returns with 10Y yield changes)")
    cc = last_n_years(sbcorr.dropna(), max(yrs, 5))
    f2 = go.Figure()
    f2.add_scatter(x=cc.index, y=cc.clip(lower=0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(42,120,214,.35)",
                   name="Hedging (positive)", hoverinfo="skip")
    f2.add_scatter(x=cc.index, y=cc.clip(upper=0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(227,73,72,.35)",
                   name="Not hedging (negative)", hoverinfo="skip")
    f2.add_scatter(x=cc.index, y=cc, line=dict(color="#0b0b0b", width=1.2), name="correlation",
                   hovertemplate="%{y:+.2f}<extra></extra>")
    st.plotly_chart(base_layout(f2, 260), width="stretch")
    explain(f"""
**The idea.** Different kinds of news move stocks and Treasury yields in characteristic directions, so the *signs*
of their joint moves tell you which kind of news is dominating (the identification logic used by Cieslak & Pang,
"Common shocks in stocks and bonds", 2021):

| Stocks | 10Y yield | Regime | Bonds hedge stocks? |
|---|---|---|---|
| ▲ | ▲ | **Growth up**: better growth lifts earnings and expected rates | yes |
| ▼ | ▼ | **Growth down / risk-off**: weaker growth *or* a flight to safety | yes |
| ▲ | ▼ | **Rates easing**: lower expected rates/inflation lift valuations | no |
| ▼ | ▲ | **Rates tightening**: higher expected rates/inflation hurt both (2022) | no |

Each 20-day move is divided by its typical 20-day size (daily volatility over 60 days × √20), so +1.0σ is a normal-sized
up move. If either the stock or the yield move is smaller than {M.NO_DRIVER_THRESHOLD}σ the regime is **No clear
driver**: too small to read, and any label would flip on one day's news.

**What the signs can't tell you.** "Growth down" and "risk-off / flight to safety" look the same (both: stocks down,
yields down). Separating them needs more information, such as how the different maturities on the curve move.

**The dollar** is shown next to the regime as a plain fact ("dollar up/down"), not used to name it. Its meaning shifts
over time: in many sell-offs the dollar rises (global de-risking), but in April 2025 US stocks fell while the dollar
*fell*, a "sell US assets" episode. So it is context, not identification.
""")


# ===========================================================================
# TAB  MACRO BACKDROP (growth x inflation quadrant)
# ===========================================================================
def page_macro():
    st.markdown("Is the US economy's **growth** speeding up or slowing down, and is **inflation** speeding up or "
                "slowing down? The answer puts the economy in one of four quadrants. This is the slow-moving "
                "backdrop: it updates monthly, while everything else in the dashboard updates daily.")
    top = st.columns([1, 1.5])
    with top[0]:
        st.markdown(card("Current quadrant", qnow["name"],
                         f"{qnow['subtitle']}.<br>In this quadrant for <b>{qnow['months']} month(s)</b>"
                         + (f" (before: {qnow['previous']})" if qnow["previous"] else "") +
                         (f".<br>Latest data month: {qnow['as_of']:%b %Y} (fully published by ~{qnow['known_at']:%d %b %Y})."
                          if MACRO_MODE == "default" else
                          f".<br>Based on data published up to {qnow['as_of']:%b %Y}. Inputs: your Indicator lab selection."),
                         Q.QUADRANT_COLOR[qnow["name"]]), unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<div class="explain">{qnow["meaning"]}</div>', unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        c1.metric("Growth acceleration", f"{qnow['growth']:+.2f} σ",
                  "accelerating" if qnow["growth"] >= 0 else "decelerating",
                  delta_color="normal" if qnow["growth"] >= 0 else "inverse")
        if MACRO_MODE == "lab":
            c2.metric("Inflation acceleration", f"{qnow['inflation']:+.2f} σ",
                      "accelerating" if qnow["inflation"] >= 0 else "decelerating", delta_color="off")
            st.caption(f"Both axes are equal-weighted composites of real-time z-scores of the indicators chosen in the "
                       f"Indicator lab ({len(LAB_CFG['growth'])} growth, {len(LAB_CFG['inflation'])} inflation).")
        else:
            c2.metric("Core CPI acceleration", f"{qnow['inflation']:+.2f} pp",
                      "accelerating" if qnow["inflation"] >= 0 else "decelerating", delta_color="off")
            st.caption("Growth is in standard-deviation units (σ) because it averages four different series; "
                       "inflation is in percentage points: 3-month annualised core CPI minus 12-month core CPI. "
                       "Choose different inputs in Cross-asset → Indicator lab.")
    with top[1]:
        trail_n = st.radio("Trail", [12, 24, 36], index=1, horizontal=True,
                           format_func=lambda x: f"last {x} months", key="q_trail")
        tr_q = quad.iloc[-trail_n:]
        gx = max(1.5, quad["growth"].iloc[-120:].abs().quantile(0.98) * 1.1, tr_q["growth"].abs().max() * 1.15)
        iy = max(1.0, quad["inflation"].iloc[-120:].abs().quantile(0.98) * 1.1, tr_q["inflation"].abs().max() * 1.15)
        f = go.Figure()
        for (gs_, is_), (nm, _, _) in Q.QUADRANTS.items():
            f.add_shape(type="rect", x0=0, x1=gs_ * gx, y0=0, y1=is_ * iy, line_width=0,
                        fillcolor=Q.QUADRANT_COLOR[nm], opacity=0.09, layer="below")
            f.add_annotation(x=gs_ * gx * 0.97, y=is_ * iy * 0.95, text=f"<b>{nm}</b>", showarrow=False,
                             xanchor="right" if gs_ > 0 else "left", yanchor="top" if is_ > 0 else "bottom",
                             font=dict(size=12, color="#52514e"))
        f.add_scatter(x=tr_q["growth"], y=tr_q["inflation"], mode="lines+markers",
                      line=dict(color="#898781", width=1.2),
                      marker=dict(size=np.linspace(5, 9, len(tr_q)), color="#898781"),
                      text=[d.strftime("%b %Y") for d in tr_q.index], name="path",
                      hovertemplate="%{text}<br>growth %{x:+.2f}σ<br>core CPI %{y:+.2f} pp<extra></extra>")
        f.add_scatter(x=[tr_q["growth"].iloc[-1]], y=[tr_q["inflation"].iloc[-1]], mode="markers+text",
                      marker=dict(size=16, color=Q.QUADRANT_COLOR[qnow["name"]], line=dict(color="#fcfcfb", width=2)),
                      text=[f"{tr_q.index[-1]:%b %Y}"], textposition="top center", name="latest",
                      hovertemplate="latest<br>growth %{x:+.2f}σ<br>core CPI %{y:+.2f} pp<extra></extra>")
        f.add_annotation(x=tr_q["growth"].iloc[0], y=tr_q["inflation"].iloc[0], text=f"{tr_q.index[0]:%b %Y}",
                         showarrow=False, yshift=-12, font=dict(size=10, color="#898781"))
        f.add_hline(y=0, line_color="#c3c2b7"); f.add_vline(x=0, line_color="#c3c2b7")
        f.update_xaxes(range=[-gx, gx], title="Growth: decelerating  ←  →  accelerating (σ)", zeroline=False)
        f.update_yaxes(range=[-iy, iy], title="Inflation: decelerating ↓ ↑ accelerating ("
                       + ("σ" if MACRO_MODE == "lab" else "pp") + ")")
        f.update_layout(hovermode="closest")
        st.plotly_chart(base_layout(f, 470, "Where the economy has been (each dot = one month)", False), width="stretch")

    yrs = st.radio("History", [5, 10, 20], index=1, horizontal=True, format_func=lambda x: f"{x}y", key="q_hist")
    qh = quad[quad.index >= quad.index[-1] - pd.DateOffset(years=yrs)]
    l, r = st.columns(2)
    iu = "σ" if MACRO_MODE == "lab" else "pp"
    ilab = "Inflation acceleration (σ)" if MACRO_MODE == "lab" else "Core CPI acceleration (pp)"
    for col, key, lab, unit_ in [(l, "growth", "Growth acceleration (σ)", "σ"), (r, "inflation", ilab, iu)]:
        f = go.Figure()
        v = qh[key]
        f.add_bar(x=v.index, y=v, marker_color=[UP if x_ >= 0 else DOWN for x_ in v], name=lab,
                  hovertemplate="%{x|%b %Y}: %{y:+.2f} " + unit_ + "<extra></extra>")
        f.add_hline(y=0, line_color="#898781", line_width=1)
        col.plotly_chart(base_layout(f, 260, lab + " · blue = accelerating", False), width="stretch")
    rib = go.Figure()
    for nm in Q.QUADRANT_COLOR:
        g = qh[qh["quadrant"] == nm]
        rib.add_bar(x=g.index, y=[1] * len(g), name=nm, marker_color=Q.QUADRANT_COLOR[nm], marker_line_width=0,
                    hovertemplate="%{x|%b %Y}: " + nm + "<extra></extra>")
    rib.update_layout(barmode="stack", bargap=0)
    rib.update_yaxes(visible=False)
    st.plotly_chart(base_layout(rib, 150, "Quadrant by month"), width="stretch")

    st.markdown("##### The inputs, latest readings")
    if MACRO_MODE == "lab":
        reg = I.registry_with_custom(LAB_CFG.get("custom"))
        rows = []
        for ax in ("growth", "inflation"):
            z = I.signals_z([k for k in LAB_CFG[ax] if k in reg], reg, _lat, _fst, bool(LAB_CFG.get("first_release")))
            for k_ in z.columns:
                zz = z[k_].dropna()
                rows.append({"Indicator": reg[k_][0], "Axis": ax, "Latest z": zz.iloc[-1], "3 months ago": zz.iloc[-4],
                             "Known as of": zz.index[-1].strftime("%b %Y"),
                             "Direction": "accelerating" if zz.iloc[-1] > 0 else "decelerating"})
        comp = pd.DataFrame(rows).set_index("Indicator")
        st.dataframe(comp.style.format({"Latest z": "{:+.2f}", "3 months ago": "{:+.2f}"})
                     .map(signed_color, subset=["Latest z", "3 months ago"]), width="stretch")
        st.caption("Each input is its real-time acceleration signal divided by its own trailing 10-year volatility. "
                   "The composite is their average (a month counts once at least half the inputs are published).")
    else:
        comp = Q.components_table(macro)
        st.dataframe(comp.style.format({"3m annualised %": "{:+.2f}", "12m %": "{:+.2f}", "Acceleration pp": "{:+.2f}"})
                     .map(signed_color, subset=["Acceleration pp"]), width="stretch")
        st.caption("Series are published with different lags, so the latest month can differ by series. "
                   "The composite needs at least 3 of the 4 growth series for a month to count.")

    st.markdown("##### How equities behaved in each quadrant")
    q_spx, q_rel = Q.quadrant_returns(px, quad, list(D.SECTORS))
    a, b = st.columns([1.25, 1])
    with a:
        q_spx = q_spx.copy()
        q_spx["now"] = ["◀ now" if i == qnow["name"] else "" for i in q_spx.index]
        st.dataframe(q_spx.style.format({"months": "{:.0f}", "avg monthly %": "{:+.2f}", "annualised %": "{:+.1f}",
                                         "% months up": "{:.0f}", "worst month %": "{:+.1f}"}, na_rep="–")
                     .map(signed_color, subset=["avg monthly %", "annualised %"]), width="stretch")
        st.caption(f"S&P 500 monthly returns, {px.index[0]:%Y}–{px.index[-1]:%Y}. Each month is assigned the quadrant "
                   "that was already published at the start of that month (no look-ahead). "
                   "Samples are small (a few dozen months each) and the data are today's revised figures, "
                   "not what was known in real time: read as tendencies.")
    with b:
        if q_rel.notna().any().any():
            rel_t = q_rel.T.dropna(how="all")
            lim = np.nanmax(np.abs(rel_t.values))
            hm = go.Figure(go.Heatmap(z=rel_t.values, x=[c.replace(" ", "<br>", 1) for c in rel_t.columns],
                                      y=rel_t.index, zmid=0, zmin=-lim, zmax=lim,
                                      colorscale=[[0, DOWN], [0.5, "#f0efec"], [1, UP]],
                                      text=np.round(rel_t.values, 2), texttemplate="%{text:+.2f}",
                                      hovertemplate="%{y}: %{z:+.2f} pp/month vs S&P<extra></extra>",
                                      colorbar=dict(title="pp", thickness=10)))
            hm.update_yaxes(autorange="reversed")
            st.plotly_chart(base_layout(hm, 430, "Sectors vs S&P by quadrant (avg pp per month)", False),
                            width="stretch")
    explain("""
**Why acceleration, not level.** Markets price the *direction of change*. 3% growth that is slowing from 5% feels
very different to investors than 3% growth rising from 1%. So each series is measured as its last-3-months pace
(annualised) minus its last-12-months pace. Positive = speeding up.

**Growth composite.** Four broad activity series (jobs, factory output, real consumer spending, real income
excluding government transfers). Each acceleration is divided by its own 10-year standard deviation so the noisy
ones (retail sales) don't swamp the smooth ones (payrolls), then averaged. **Inflation** uses core CPI (excluding
food and energy), the measure the Fed watches most closely in CPI.

**The four quadrants.**

| | Inflation decelerating | Inflation accelerating |
|---|---|---|
| **Growth accelerating** | Goldilocks | Reflation |
| **Growth decelerating** | Disinflationary slowdown | Stagflation |

**How it fits with the rest.** The quadrant is what the economy is doing; the market-implied macro regime is what the
market is *trading* right now. When they disagree (e.g. economy in Reflation but the market reads "growth down / risk-off"), the market is either anticipating
a turn in the data or overreacting to it, and the next data releases usually settle which.

**Caveats.**
* Monthly data are noisy and revised. A single month can flip the quadrant, so look at the path, not one dot.
* FRED serves today's revised history. In real time the readings were sometimes different, so the base-rate table
  is a little better than what an investor could actually have achieved.
* Released with a lag of 2–6 weeks. By the time a quadrant is confirmed, markets have often moved.
""")


# ===========================================================================
# TAB 3  DRIVERS
# ===========================================================================
def page_drivers():
    a, b, c_ = st.columns([1.2, 1, 1])
    target = a.selectbox("Index to explain", ["S&P 500", "Nasdaq 100", "Russell 2000", "S&P 500 Equal Wt (RSP)"], key="att_t")
    use_vix = b.toggle("Include VIX as a driver", value=False,
                       help="VIX is computed from S&P options, so it moves with the S&P at the same moment. "
                            "Including it 'explains' a lot but tells you little about the cause.")
    ndays = c_.radio("Show last", [20, 60], horizontal=True, format_func=lambda x: f"{x} sessions", key="att_n")

    @st.cache_data
    def _att(px, target, use_vix):
        return M.attribution(px, target, use_vix=use_vix, days=120)

    contrib, betas = _att(px, target, use_vix)
    cb = contrib.iloc[-ndays:]
    fcols = [c for c in cb.columns if c not in ("Actual",)]
    colmap = {k: SERIES[i] for i, k in enumerate(list(M.FACTORS) + ["Idiosyncratic"])}
    colmap["Idiosyncratic"] = "#c3c2b7"

    l, r = st.columns([1, 1.6])
    with l:
        span = st.segmented_control("Window to explain", ["Last day", "Last 5 days", "Last 20 days"],
                                    default="Last 5 days", key="att_span")
        nn = {"Last day": 1, "Last 5 days": 5, "Last 20 days": 20}[span or "Last 5 days"]
        d5 = contrib.iloc[-nn:].sum()
        parts = d5.drop("Actual")
        wf = go.Figure(go.Waterfall(
            orientation="v", x=[i.split(" (")[0].replace("Idiosyncratic", "Unexplained") for i in parts.index] + ["Actual"], measure=["relative"] * len(parts) + ["total"],
            y=list(parts.values) + [d5["Actual"]], text=[f"{v:+.2f}" for v in parts.values] + [f"{d5['Actual']:+.2f}"],
            textposition="outside", connector=dict(line=dict(color=MUTED, width=1)),
            increasing=dict(marker=dict(color=UP)), decreasing=dict(marker=dict(color=DOWN)),
            totals=dict(marker=dict(color="#0b0b0b")),
            hovertemplate="%{x}: %{y:+.2f} pp<extra></extra>"))
        lo = min(0, np.cumsum(parts.values).min(), d5["Actual"]); hi = max(0, np.cumsum(parts.values).max(), d5["Actual"])
        pad = (hi - lo) * 0.18 + 0.1
        wf.update_yaxes(range=[lo - pad, hi + pad], ticksuffix="")
        wf.update_xaxes(tickangle=0, tickfont=dict(size=11))
        st.plotly_chart(base_layout(wf, 360, f"{target}, {(span or '').lower()}: driver by driver (pp)",
                                    False), width="stretch")
        big = d5.drop(["Actual", "Idiosyncratic"]).abs().idxmax()
        st.markdown(f"Over this window {target} moved **{d5['Actual']:+.2f}%**. The largest identified driver was "
                    f"**{big}** ({d5[big]:+.2f} pp); **{d5['Idiosyncratic']:+.2f} pp** is left unexplained "
                    f"(equity-specific news: earnings, positioning, AI headlines…).")
    with r:
        fig = go.Figure()
        for col in fcols:
            fig.add_bar(x=cb.index, y=cb[col], name=col, marker_color=colmap.get(col, NEUTRAL))
        fig.add_scatter(x=cb.index, y=cb["Actual"], mode="markers", name="Actual return",
                        marker=dict(color="#0b0b0b", size=7, symbol="diamond"))
        fig.update_layout(barmode="relative", bargap=0.25)
        st.plotly_chart(base_layout(fig, 400, "Daily attribution (bars stack to the actual return ◆)"), width="stretch")

    l2, r2 = st.columns(2)
    with l2:
        vs = M.variance_share(contrib.iloc[-60:])
        fig = go.Figure(go.Bar(x=vs.values, y=vs.index, orientation="h",
                               marker_color=[colmap.get(i, NEUTRAL) for i in vs.index],
                               text=[f"{v:.0f}%" for v in vs.values], textposition="outside", cliponaxis=False))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[min(0, vs.min() * 1.2), vs.max() * 1.25], ticksuffix="%")
        st.plotly_chart(base_layout(fig, 280, "Share of daily variance, last 60 sessions", False), width="stretch")
        rates_share = vs.get("Rates (10Y, per bp)", 0)
        st.markdown(f"**Are rates the main equity risk right now?** "
                    + (f"**Yes** – rates explain {rates_share:.0f}% of variance, the largest identified driver."
                       if rates_share == vs.drop("Idiosyncratic").max() and rates_share > 15 else
                       f"**No** – rates explain {rates_share:.0f}%; the dominant identified driver is "
                       f"**{vs.drop('Idiosyncratic').idxmax()}**."))
    with r2:
        bt = betas.copy()
        unit_txt = {"Rates (10Y, per bp)": "% index move per +1 bp in the 10Y yield",
                    "Dollar (DXY %)": "% index move per +1% in the dollar",
                    "Credit (HYG %)": "% index move per +1% in HYG",
                    "Vol (VIX pt)": "% index move per +1 pt in VIX"}
        bt["Meaning"] = [unit_txt[i] for i in bt.index]
        _, Xf = M.factor_inputs(px, target, use_vix)
        bt["Per typical day"] = bt["beta"] * Xf.iloc[-500:].std().reindex(bt.index)
        st.markdown("##### Current sensitivities (betas)")
        st.dataframe(bt[["beta", "Per typical day", "Meaning"]].style.format({"beta": "{:+.3f}", "Per typical day": "{:+.2f} pp"}),
                     width="stretch", column_config={"Meaning": st.column_config.TextColumn(width="large")})
        st.caption("'Per typical day' = beta × a one-standard-deviation daily move in the driver: "
                   "lets you compare drivers measured in different units.")
    explain("""
**Method.** Every day we fit a regression of the index's daily % return on the same-day moves of the drivers,
using the previous 500 sessions (≈2 years) only, so there is no look-ahead. The fitted coefficients (betas)
times that day's driver moves give each driver's contribution in percentage points; whatever is left is
*idiosyncratic*. Contributions + idiosyncratic = the actual move, exactly.

**Ridge** regression is ordinary regression with a small penalty that keeps betas stable when drivers are correlated
with each other (credit and the dollar often are).

**Variance share** splits the variance of daily returns over the last 60 sessions into pieces that add up to 100%:
share_k = cov(contribution_k, return) / var(return).

**Caveats.**
* This is co-movement, not causation. A credit sell-off and an equity sell-off often have the same root cause.
* VIX is derived from S&P options, so it moves *with* the S&P at the same instant. Including it makes the model look
  better without explaining much; that's why it's off by default.
""")

# ===========================================================================
# TAB 4  BREADTH & ROTATION
# ===========================================================================
def page_breadth():
    st.markdown("Is the whole market moving, or just a few big names? Ratios answer that: when a ratio rises, "
                "the first leg is beating the second.")
    rt = ratios.copy()
    st.dataframe(rt.style.format({"5d %": "{:+.2f}", "20d %": "{:+.2f}", "5d z": "{:+.2f}", "gap to 100d %": "{:+.1f}"})
                 .map(signed_color, subset=["5d %", "20d %", "gap to 100d %"]).map(z_color, subset=["5d z"])
                 .map(state_style, subset=["vs 100d avg"]),
                 width="stretch")
    yrs = st.radio("Window", [1, 2, 5], index=0, horizontal=True, format_func=lambda x: f"{x}y", key="br_y")
    cols = st.columns(3)
    for i, (lab, (x1, x2, _)) in enumerate(M.RATIOS.items()):
        if x1 not in px or x2 not in px:
            continue
        rr = last_n_years((px[x1] / px[x2]).dropna().to_frame("r"), yrs)["r"]
        full = (px[x1] / px[x2]).dropna()
        ma = full.rolling(100).mean().reindex(rr.index)
        f = go.Figure()
        f.add_scatter(x=rr.index, y=rr / rr.iloc[0] * 100, name="ratio", line=dict(color=UP, width=1.8))
        f.add_scatter(x=rr.index, y=ma / rr.iloc[0] * 100, name="100-day avg", line=dict(color=NEUTRAL, width=1.2, dash="dot"))
        cols[i % 3].plotly_chart(base_layout(f, 230, lab, False), width="stretch")
        cols[i % 3].caption(M.RATIOS[lab][2])
    st.caption("Ratios indexed to 100 at the start of the window. Dotted = 100-day average.")

    st.markdown("##### Sector scoreboard (relative to the S&P 500, percentage points)")
    secs = [s for s in D.SECTORS if s in px]
    rel = pd.DataFrame({h: [M.change(px[s], n, s).iloc[-1] - M.change(spx, n, "S&P 500").iloc[-1] for s in secs]
                        for h, n in [("1d", 1), ("5d", 5), ("20d", 20), ("60d", 60), ("120d", 120)]}, index=secs)
    rel = rel.sort_values("20d")
    lim = np.nanmax(np.abs(rel.values))
    hm = go.Figure(go.Heatmap(z=rel.values, x=rel.columns, y=rel.index, zmid=0, zmin=-lim, zmax=lim,
                              colorscale=[[0, DOWN], [0.5, "#f0efec"], [1, UP]],
                              text=np.round(rel.values, 1), texttemplate="%{text:+.1f}",
                              hovertemplate="%{y} · %{x}: %{z:+.2f} pp vs S&P<extra></extra>",
                              colorbar=dict(title="pp", thickness=10)))
    b1, b2 = st.columns([1.3, 1])
    b1.plotly_chart(base_layout(hm, 430, "Sector return minus S&P return (sorted by 20d)", False), width="stretch")
    bb = last_n_years(breadth.to_frame("b"), 2)["b"]
    f = go.Figure()
    f.add_scatter(x=bb.index, y=bb, line=dict(color=UP, width=1.6), fill="tozeroy", fillcolor="rgba(42,120,214,.10)",
                  hovertemplate="%{y:.0f}%<extra></extra>")
    f.add_hline(y=50, line_dash="dot", line_color=NEUTRAL)
    f.update_yaxes(range=[0, 100], ticksuffix="%")
    b2.plotly_chart(base_layout(f, 430, "Breadth: % of sectors & mega-caps above their 50-day average", False),
                    width="stretch")
    explain("""
**Why ratios.** An index can fall because *everything* is falling (macro stress) or because a handful of large,
crowded winners are being sold while the rest holds up (rotation / leadership unwind). The two look the same on
the index chart but are very different situations.

* **RSP/SPY**: equal-weight S&P (every stock counts the same) vs the normal cap-weighted S&P. Rising = the
  typical stock is doing better than the giants.
* **MTUM/SPY**: momentum factor. Sharp falls = crowded winners being sold, often violently and quickly.
* **SPHB/SPLV**: the most volatile vs the least volatile S&P stocks: a clean risk-appetite gauge.
* **XLY/XLP**: consumer discretionary vs staples: cyclical vs defensive.
* **"vs 100d avg"**: above/below the ratio's own 100-day average, i.e. the slow trend of the rotation.

**Breadth line.** Share of the 12 sector ETFs and 10 mega-caps above their 50-day average. Below ~30% = broad weakness;
above ~70% = broad participation. It's a simple, robust gauge (not a full NYSE advance/decline count).
""")

# ===========================================================================
# TAB 5  VOLATILITY
# ===========================================================================
def page_vol():
    vv = vol.dropna(subset=["VIX"])
    vix_pct = (vv["VIX"].iloc[-253:-1] < vv["VIX"].iloc[-1]).mean() * 100
    vrp_pct = (vv["VRP (VIX − RV)"].iloc[-1261:-1] < vv["VRP (VIX − RV)"].iloc[-1]).mean() * 100
    k = st.columns(4)
    k[0].markdown(card("VIX (30-day implied vol)", f"{v_now['VIX']:.1f}",
                       f"{M.ordinal(vix_pct)} percentile of 1 year. Rough guide: &lt;15 calm, 15–20 normal, 20–30 stressed, &gt;30 fear.",
                       BAD if v_now["VIX"] > 25 else WARN if v_now["VIX"] > 18 else GOOD), unsafe_allow_html=True)
    k[1].markdown(card("Vol risk premium (VIX − realised)", f"{v_now['VRP (VIX − RV)']:+.1f} pts",
                       f"Realised 21d vol {v_now['RV21 S&P']:.1f}%. {M.ordinal(vrp_pct)} %ile of 5y. "
                       + ("Options priced above what the market is delivering (normal)." if v_now['VRP (VIX − RV)'] > 0
                          else "Market is moving MORE than options priced: rare, a sign of genuine stress."),
                       GOOD if v_now["VRP (VIX − RV)"] > 0 else BAD), unsafe_allow_html=True)
    t1 = round(float(v_now.get("VIX9D / VIX", np.nan)), 2)
    k[2].markdown(card("Front of curve · VIX9D / VIX", f"{t1:.2f}",
                       "&gt;1 = the next 9 days priced as riskier than the next 30 (event / panic premium).",
                       BAD if t1 > 1 else GOOD), unsafe_allow_html=True)
    k[3].markdown(card("Back of curve · VIX / VIX3M", f"{term:.2f}",
                       "&gt;1 = inverted: near-term fear above 3-month. Inversions are rare and usually short-lived.",
                       BAD if term > 1 else GOOD), unsafe_allow_html=True)
    st.write("")
    yrs = st.radio("Window", [1, 2, 5], index=1, horizontal=True, format_func=lambda x: f"{x}y", key="vol_y")
    w = last_n_years(vv, yrs)
    l, r = st.columns(2)
    f = go.Figure()
    f.add_scatter(x=w.index, y=w["VIX"], name="VIX (implied)", line=dict(color=UP, width=1.8))
    f.add_scatter(x=w.index, y=w["RV21 S&P"], name="S&P realised 21d", line=dict(color=SERIES[1], width=1.5))
    l.plotly_chart(base_layout(f, 320, "Implied vs realised volatility (%)"), width="stretch")
    f = go.Figure()
    if "VIX9D / VIX" in w:
        f.add_scatter(x=w.index, y=w["VIX9D / VIX"], name="VIX9D / VIX", line=dict(color=SERIES[1], width=1.3))
    if "VIX / VIX3M" in w:
        f.add_scatter(x=w.index, y=w["VIX / VIX3M"], name="VIX / VIX3M", line=dict(color=UP, width=1.8))
    f.add_hline(y=1, line_color=DOWN, line_dash="dot", annotation_text="above 1 = inverted", annotation_position="top right")
    r.plotly_chart(base_layout(f, 320, "Term-structure ratios"), width="stretch")

    l, r = st.columns(2)
    cone_pick = l.selectbox("Vol cone for", ["S&P 500", "Nasdaq 100", "Russell 2000"] + list(D.LEADERS), key="cone")
    cone = M.vol_cone(px[cone_pick].dropna())
    x = [f"{i}d" for i in cone.index]
    f = go.Figure()
    f.add_scatter(x=x, y=cone["max"], line=dict(color=MUTED, width=1, dash="dot"), name="max")
    f.add_scatter(x=x, y=cone["p75"], line=dict(color=UP_LIGHT, width=0), name="25–75%", showlegend=False)
    f.add_scatter(x=x, y=cone["p25"], line=dict(color=UP_LIGHT, width=0), fill="tonexty",
                  fillcolor="rgba(134,182,239,.35)", name="25th–75th pct")
    f.add_scatter(x=x, y=cone["median"], line=dict(color=UP, width=1.5, dash="dash"), name="median")
    f.add_scatter(x=x, y=cone["min"], line=dict(color=MUTED, width=1, dash="dot"), name="min")
    f.add_scatter(x=x, y=cone["current"], mode="lines+markers", line=dict(color="#0b0b0b", width=2),
                  marker=dict(size=9), name="today")
    implied = {"S&P 500": "VIX", "Nasdaq 100": "VXN"}.get(cone_pick)
    if implied and implied in px:
        f.add_scatter(x=["21d"], y=[px[implied].iloc[-1]], mode="markers", name=f"{implied} (implied 30d)",
                      marker=dict(size=13, color=SERIES[1], symbol="diamond"))
    f.update_yaxes(ticksuffix="%")
    l.plotly_chart(base_layout(f, 360, f"{cone_pick}: realised-vol cone (3-year ranges)", "right"), width="stretch")
    with r:
        st.markdown("##### Reading the cone")
        st.markdown(f"""
For each look-back window (10 days … 1 year) the shaded band shows where realised volatility has *usually* been
over the last 3 years (25th–75th percentile), the dotted lines the extremes, and the black line **today**.

* Today's short-window vol {'**above**' if cone.loc[10, 'current'] > cone.loc[10, 'p75'] else '**inside**' if cone.loc[10, 'current'] >= cone.loc[10, 'p25'] else '**below**'}
  the usual band and long-window vol {'above' if cone.loc[252, 'current'] > cone.loc[252, 'p75'] else 'inside' if cone.loc[252, 'current'] >= cone.loc[252, 'p25'] else 'below'} it.
* A **downward-sloping** black line (short > long) means a recent vol shock that historically tends to fade.
* The ◆ shows implied vol (VIX/VXN) at the 21-day point: above the black line means options are pricing
  more movement than the market is delivering.
""")
    explain("""
* **Implied volatility (VIX)**: the market's price for S&P movement over the next 30 days, backed out of options.
* **Realised volatility**: how much the S&P actually moved, annualised standard deviation of daily returns.
* **Vol risk premium = implied − realised.** Normally positive (insurance sellers want to be paid). When it turns
  negative, the market is moving more than anyone priced: real stress.
* **Term structure.** Normally longer-dated vol is higher (more time = more uncertainty): VIX < VIX3M. When short-dated
  vol is priced *above* longer-dated, traders are paying up for immediate protection. Historically VIX has tended to
  fall in the weeks after such spikes, but the spikes that don't fade (2008, 2020) are the ones that matter.
* **Not included:** option skew and dealer gamma need full option chains, which aren't reliably available for free.
""")

# ===========================================================================
# TAB 6  LEADERS
# ===========================================================================
def page_leaders():
    st.markdown("The largest stocks drive a third of the S&P. Sorted by **momentum z** "
                "(20-day return divided by what a typical 20-day move would be at current volatility).")
    lt = leaders.copy()
    st.dataframe(lt.style.format({"Price": "{:,.2f}", "1d %": "{:+.2f}", "5d %": "{:+.2f}", "20d %": "{:+.1f}",
                                  "YTD %": "{:+.1f}", "Momentum z": "{:+.2f}", "Beta (63d)": "{:.2f}",
                                  "Vol 20d %": "{:.0f}", "From 52w high %": "{:+.1f}"}, na_rep="–")
                 .map(signed_color, subset=["1d %", "5d %", "20d %", "YTD %"])
                 .map(z_color, subset=["Momentum z"]).map(state_style, subset=["Trend"]),
                 width="stretch", height=45 + 35 * len(lt))
    l, r = st.columns(2)
    f = go.Figure(go.Scatter(x=lt["Beta (63d)"], y=lt["5d %"], mode="markers+text", text=lt.index,
                             textposition="top center", marker=dict(size=12, color=[UP if v > 0 else DOWN for v in lt["5d %"]]),
                             hovertemplate="%{text}<br>beta %{x:.2f}<br>5d %{y:+.1f}%<extra></extra>"))
    bx = np.linspace(0, max(2.2, lt["Beta (63d)"].max() * 1.1), 10)
    f.add_scatter(x=bx, y=bx * row["5d"], mode="lines", line=dict(color=NEUTRAL, dash="dot"),
                  name="beta × S&P 5d", hoverinfo="skip")
    f.update_xaxes(title="Beta to S&P (63 days)")
    f.update_yaxes(title="5-day return %")
    l.plotly_chart(base_layout(f, 380, "5-day return vs beta", False), width="stretch")
    l.caption("Dotted line = beta × the S&P's 5-day move, i.e. what beta alone would predict. "
              "Below it = worse than the market explains (stock-specific selling).")
    with r:
        dd = lt["From 52w high %"].sort_values()
        f = go.Figure(go.Bar(x=dd.values, y=dd.index, orientation="h", marker_color=DOWN_LIGHT,
                             text=[f"{v:.0f}%" for v in dd.values], textposition="outside", cliponaxis=False))
        f.update_xaxes(range=[dd.min() * 1.18, 0], ticksuffix="%")
        st.plotly_chart(base_layout(f, 380, "Distance from 52-week high", False), width="stretch")
    explain("""
* **Momentum z** = 20-day return ÷ (20-day volatility × √(20/252)). It is a risk-adjusted trend score: +2 means a strong,
  clean rally; −2 a strong, clean sell-off. It lets you compare a calm stock with a volatile one.
* **Beta (63d)**: how much the stock typically moves for a 1% move in the S&P, over the last quarter.
  In the scatter, the dotted line is what each stock "should" have done given its beta and the S&P's 5-day move.
  Names far *below* it are falling for stock-specific reasons; names above it are holding up better than their beta implies.
* **From 52w high**: drawdown from the 1-year peak. Use it to see how much of a long rally has been given back.
""")

# ===========================================================================
# TAB 7  WHAT HAPPENED NEXT
# ===========================================================================
def page_next():
    st.markdown("Pick a series and a condition. We find every past session where the condition held and show what "
                "happened over the following 5 and 20 sessions. **It is a base rate, not a forecast.**")
    a, b, c_ = st.columns([1.2, 1.4, 1])
    eq_opts = ["S&P 500", "Nasdaq 100", "Russell 2000", "VIX", "High Yield (HYG)"] + list(D.SECTORS)
    fi_opts = [c for c in ["UST 2Y", "UST 10Y", "UST 30Y", "2s10s curve", "3m10y curve", "10Y real yield (TIPS)",
                           "10Y breakeven inflation", "HY credit spread (OAS)", "IG credit spread (OAS)",
                           "2Y minus Fed funds"] if c in fi] + ["MOVE (rate vol)", "Long Bond (TLT)", "Dollar (DXY)"]
    es_name = a.selectbox("Series", eq_opts + fi_opts + ["Gold"] + CR_LIST, key="es_s")
    frame = fi if es_name in fi.columns else px
    s_es = frame[es_name].dropna().rename(es_name)
    z5 = M.zscore_of_change(s_es, 5)
    tr_es = M.trend_regime(s_es)["state"]
    _mk = M.markov_states(s_es)
    mk_es = _mk["state"] if _mk is not None else pd.Series(np.nan, index=s_es.index)
    sb_codes = sbc["name"].reindex(s_es.index)
    pr = M.pct_rank(s_es)
    conds = {
        "5d z ≤ −1.5 (sharp weekly drop)": z5 <= -1.5,
        "5d z ≥ +1.5 (sharp weekly rise)": z5 >= 1.5,
        "5d z ≤ −2.0 (very sharp drop)": z5 <= -2.0,
        "Level ≥ 95th pct of 1 year": pr >= 95,
        "Level ≤ 5th pct of 1 year": pr <= 5,
        f"Same trend × volatility state as today ({mk_es.iloc[-1]})": mk_es == mk_es.iloc[-1],
        f"Same trend state as today ({tr_es.iloc[-1]})": tr_es == tr_es.iloc[-1],
        f"Same market-implied regime as today ({sbc_now['name']})": sb_codes == sbc_now["name"],
        f"Same macro quadrant as today ({qnow['name']})": Q.daily_quadrant(quad, s_es.index) == qnow["name"],
        f"Same curve regime as today ({crv.iloc[-1]['state']})": crv["state"].reindex(s_es.index) == crv.iloc[-1]["state"],
    }
    # default: first condition that is true today
    today_true = [k for k, m in conds.items()
                  if bool(m.fillna(False).iloc[-1]) and ("5d z" in k or "pct" in k)]
    default = today_true[0] if today_true else list(conds)[0]
    cond_name = b.selectbox("Condition", list(conds), index=list(conds).index(default), key="es_c")
    filt = c_.selectbox("…and only when trend was", ["any", "UPTREND", "DOWNTREND", "MIXED"], key="es_f")
    mask = conds[cond_name].fillna(False).astype(bool)
    if filt != "any":
        mask &= (tr_es == filt)
    mask.iloc[-20:] = False  # need 20 forward sessions
    res = M.event_study(s_es, mask)
    u = M.unit(es_name)
    is_true = bool(conds[cond_name].fillna(False).iloc[-1])
    st.markdown(f"Condition true **today**: {'yes' if is_true else 'no'} · "
                f"history used: {s_es.index[0]:%Y}–{s_es.index[-1]:%Y}")

    if res["n"] < 5 or res.get(20) is None:
        st.info("Fewer than 5 past sessions match. Try a looser condition or remove the trend filter.")
    else:
        k = st.columns(5)
        k[0].markdown(card("Sample", f"{res['n']} sessions",
                           f"≈ <b>{res['episodes']} independent episodes</b> (signals within 5 sessions grouped). "
                           + ("Small: treat with caution." if res["episodes"] < 15 else ""),
                           WARN if res["episodes"] < 15 else None), unsafe_allow_html=True)
        for j, h in enumerate([5, 20]):
            rr = res[h]; base = res[f"base_{h}"]
            k[1 + j].markdown(card(f"Median next {h} sessions", f"{rr['median']:+.2f} {u}",
                                   f"vs {base['median']:+.2f} {u} on any day. Up {rr['pos']:.0f}% of the time "
                                   f"(any day: {base['pos']:.0f}%).", UP if rr["median"] > 0 else DOWN),
                              unsafe_allow_html=True)
        k[3].markdown(card("Hit rate (20d)", f"{res[20]['hit']:.0f}%",
                           "Share of cases that went the same way as the median."), unsafe_allow_html=True)
        k[4].markdown(card("Worst case (20d)", f"{res[20]['worst']:+.1f} {u}",
                           f"Largest move <i>against</i> the median direction. Best: {res[20]['best']:+.1f} {u}.", BAD),
                      unsafe_allow_html=True)
        l, r = st.columns([1.4, 1])
        allf = M.change(s_es, 20, es_name).shift(-20).dropna()
        f = go.Figure()
        f.add_histogram(x=allf, histnorm="probability density", name="Any day", marker_color=MUTED, opacity=0.55, nbinsx=60)
        f.add_histogram(x=res[20]["values"], histnorm="probability density", name="After condition",
                        marker_color=UP, opacity=0.75, nbinsx=40)
        f.add_vline(x=0, line_color="#0b0b0b", line_width=1)
        f.update_layout(barmode="overlay")
        f.update_xaxes(title=f"next-20-session change ({u})")
        l.plotly_chart(base_layout(f, 340, "Distribution of outcomes: after the condition vs any day"), width="stretch")
        hits = s_es[mask]
        f = go.Figure()
        f.add_scatter(x=s_es.index, y=s_es, line=dict(color=MUTED, width=1), name=es_name)
        f.add_scatter(x=hits.index, y=hits, mode="markers", marker=dict(color=UP, size=5), name="condition true")
        f.update_yaxes(type="log" if u == "%" else "linear")
        r.plotly_chart(base_layout(f, 340, "When it happened"), width="stretch")
    explain("""
* **n sessions vs episodes.** If the S&P has a bad week, the condition is often true on 3–4 consecutive days. Each day
  counts in *n*, but they are really one event with nearly identical futures. *Episodes* groups signals within 5
  sessions of each other, and is the more honest sample size. Hit rates from 10 episodes are noisy: a 70% hit rate
  could easily be 50% in truth.
* **Compare with "any day".** The S&P rises about 55–60% of all 20-day windows. A condition is only interesting if it
  shifts that materially.
* **Worst case** is the single worst outcome against the typical direction. A small median gain can hide a tail like 2008.
* **Trend filter** asks the same question only inside one trend regime, e.g. "sharp drops *inside an uptrend*"
  (dip-buying) vs "*inside a downtrend*" (often the start of more trouble).
""")

# ===========================================================================
# TAB 8  GUIDE
# ===========================================================================
def page_guide():
    st.markdown("""
### How it's organised

Pick a **section** in the sidebar: **Cross-asset** (the big picture and tools shared by all markets), **Equities**,
**Fixed income** or **Crypto**. **Summary** is a one-page overview of everything. Each section has its own tabs. Every page has a "How this works" panel at the bottom.

### How to use it in 5 minutes

1. **Summary, top to bottom.** Key levels → the regime map (every model's current state, by market) → the
   plain-English read → what's stretched across markets.
2. **Check whether the lenses agree.** The most informative mornings are when they *disagree*: e.g. slow trend UP
   but trend × volatility state Down · High-vol = a sharp pullback inside a bull market; the question becomes whether it's a dip or a turn.
3. **Ask "what is driving it?"** The market-implied macro regime tells you whether it's a growth or a rates story; the attribution
   tab tells you whether rates, the dollar or credit explain the equity move, or whether it's equity-specific.
4. **Ask "how broad is it?"** Breadth & rotation tells you whether the whole market is selling or a crowded group.
5. **Check rates and credit.** Fixed income → Today: is the curve move about the Fed or the term premium, and is credit
   confirming the equity move?
6. **Check the stress gauges.** VIX term structure, the vol risk premium and the MOVE index tell you whether options
   markets are panicking, and whether the stress sits in equities or in rates.
7. **Look up the base rate.** What happened next gives the historical odds, including how bad the bad cases were.

### The engines at a glance

| Engine | Horizon | Inputs | Output | Typical use |
|---|---|---|---|---|
| Trend 50/200 | months | price, 50d & 200d averages | UPTREND / MIXED / DOWNTREND | which side has the long-run edge |
| Trend × volatility state | days–weeks | price vs 21d EMA; 20d vol vs 1y median | Up/Down × Low-vol/High-vol + persistence | momentum vs mean-reversion behaviour |
| Growth × inflation quadrant | months | payrolls, industrial production, real retail sales, real income; core CPI | 4 quadrants | the economic backdrop |
| Market-implied macro regime | weeks | signs of 20d vol-scaled moves in S&P and 10Y yield (dollar as context) | growth up / growth down–risk-off / rates easing / rates tightening / no clear driver | growth vs rates story; do bonds hedge? |
| Attribution | daily | S&P vs 10Y, DXY, HYG (VIX optional) | contributions + idiosyncratic | what explains today's move |
| Rotation ratios | weeks | ETF ratios vs 100d average | above/below + z-scores | breadth, leadership, risk appetite |
| Vol suite | days | VIX, VIX9D, VIX3M, realised vol | VRP, term state, cone | stress and option pricing |
| Curve regime | weeks | 20d changes in 2Y and 10Y yields | bull/bear × steepener/flattener | what the bond market is reacting to |
| 10Y splits | weeks | real yield + breakeven; beta to 2Y + residual; Kim-Wright | driver of the 10Y move | inflation vs real rates; Fed vs term premium |
| Fed pricing | months ahead | bill yields vs fed funds, forward rates | implied path | cuts or hikes priced (proxy) |
| Credit confirmation | weeks | HY spread and S&P 20d moves (z) | confirming / not / warning | is the equity move a macro move? |
| Crypto ladder & alt breadth | days–weeks | BTC/ETH key levels; altcoins vs BTC; ETF dollar volume | support/resistance, risk appetite inside crypto | where the levels are, how speculative the market is |
| Event study | 5–20d ahead | any condition above | median, hit rate, worst case | base rates |

### What is deliberately *not* here
* **Option skew and dealer gamma**: need full, paid option-chain data.
* **Futures-implied Fed path, ECB/BoE/BoJ paths, global curves, auctions, ACM term premium**: not freely available
  as daily data. The Fed pricing page uses the bill curve as a proxy, and Kim-Wright stands in for ACM.

### Data & refresh
Daily closes from Yahoo Finance (free, via `yfinance`), about 12 years of history, cached once per day in
`data_cache/`. Rates, credit and monthly macro data come from FRED (St. Louis Fed), also free; start the app with
your FRED API key for reliable downloads. FRED's daily rates usually lag by one day. Press **Refresh data** in the sidebar after the US close to pull the latest session.
Yahoo occasionally has gaps or bad prints; the models forward-fill gaps of up to 5 days.

*Educational tool: not investment advice.*
""")


# ===========================================================================
# FIXED INCOME PAGES
# ===========================================================================
CRV_COLOR = {"BULL STEEPENER": "#2a78d6", "BULL FLATTENER": "#86b6ef", "BEAR STEEPENER": "#e34948",
             "BEAR FLATTENER": "#f0a3a2", "RANGE-BOUND": "#898781"}


def bp(x):
    return x * 100


def _fi_row(name):
    return fi_table.loc[name] if name in fi_table.index else None


def page_fi_today():
    k = st.columns(6)
    for i, n in enumerate(["UST 2Y", "UST 10Y", "2s10s curve", "10Y real yield (TIPS)",
                           "HY credit spread (OAS)", "MOVE (rate vol)"]):
        r = _fi_row(n)
        if r is None:
            continue
        if n in ("2s10s curve", "HY credit spread (OAS)"):
            val = f"{bp(r['Level']):.0f} bp"
        elif n == "MOVE (rate vol)":
            val = f"{r['Level']:.0f}"
        else:
            val = f"{r['Level']:.2f}%"
        k[i].metric(n.replace(" (OAS)", "").replace(" (rate vol)", ""), val,
                    f"{r['1d']:+.0f} {r['Unit']} · {r['5d']:+.0f} {r['Unit']} 5d", delta_color="off")
    st.write("")
    st.markdown("#### Where are rates? Four lenses")
    c = st.columns(4)
    cr = crv.iloc[-1]
    c[0].markdown(card("1 · Curve regime (20 days)", cr["state"],
                       f"2Y {cr['d2Y bp']:+.0f} bp, 10Y {cr['d10Y bp']:+.0f} bp, 2s10s {cr['slope bp']:+.0f} bp. "
                       f"For {int(cr['days'])} sessions.", CRV_COLOR[cr["state"]]), unsafe_allow_html=True)
    sp = split.iloc[-1]
    c[1].markdown(card("2 · What moved the 10Y (20 days)", sp["verdict"],
                       f"10Y {sp['10Y move bp']:+.0f} bp = Fed path {sp['Fed path bp']:+.0f} + term premium & other "
                       f"{sp['Term premium & other bp']:+.0f}. Real yields {rvb['real']:+.0f} bp, breakevens "
                       f"{rvb['breakeven']:+.0f} bp.", NEUTRAL), unsafe_allow_html=True)
    fp = F.fed_path(fi)
    two = fp.loc["1-2y", "vs today bp"]
    c[2].markdown(card("3 · Fed pricing (bill-curve proxy)",
                       f"{'Cuts' if two < -5 else 'Hikes' if two > 5 else 'On hold'} priced",
                       f"Fed funds {fi['Fed funds (EFFR)'].iloc[-1]:.2f}%. Rate implied for 1–2 years out: "
                       f"{fp.loc['1-2y', 'implied rate %']:.2f}% ({two:+.0f} bp ≈ {two / 25:+.1f} × 25 bp).",
                       UP if two < -5 else DOWN if two > 5 else NEUTRAL), unsafe_allow_html=True)
    hy = _fi_row("HY credit spread (OAS)")
    c[3].markdown(card("4 · Credit", f"HY {bp(hy['Level']):.0f} bp · {M.ordinal(hy['1y %ile'])} %ile",
                       credit["verdict"].split(":")[0].capitalize() + ":" + credit["verdict"].split(":", 1)[1]
                       if ":" in credit["verdict"] else credit["verdict"],
                       DOWN if credit["verdict"].startswith(("CONFIRMING: stocks falling", "WARNING")) else
                       UP if credit["verdict"].startswith("CONFIRMING: stocks rising") else NEUTRAL),
                  unsafe_allow_html=True)
    st.write("")
    l, r = st.columns([1.35, 1])
    with l:
        st.markdown("#### The read, in plain English")
        t10, t2 = _fi_row("UST 10Y"), _fi_row("UST 2Y")
        mv = _fi_row("MOVE (rate vol)")
        lines = [
            f"**Direction.** The 10Y yield moved {t10['5d']:+.0f} bp this week (z {t10['5d z']:+.1f}) and the 2Y "
            f"{t2['5d']:+.0f} bp (z {t2['5d z']:+.1f}). Slow trend: 10Y {t10['Trend']}, 2Y {t2['Trend']}.",
            f"**Curve.** Over 20 sessions: **{cr['state'].lower()}**. {F.CURVE_REGIMES[cr['state']]}",
            f"**Why the 10Y moved.** "
            + {"FED-DRIVEN": "Mostly the expected Fed path (it tracked the 2Y).",
               "TERM-PREMIUM-DRIVEN": "Mostly term premium and other long-end factors, not the Fed path.",
               "SHARED": "Partly the Fed path, partly term premium."}[sp["verdict"]]
            + f" Split by inflation vs real rates: **{rvb['driver'].lower()}** "
              f"(real {rvb['real']:+.0f} bp, breakeven {rvb['breakeven']:+.0f} bp).",
            f"**Fed.** The bill curve implies about {fp.loc['0-3m', 'vs today bp']:+.0f} bp over the next 3 months and "
            f"{two:+.0f} bp for the 1–2 year window versus today's fed funds rate.",
            f"**Credit.** HY spreads at {bp(hy['Level']):.0f} bp ({M.ordinal(hy['1y %ile'])} percentile of 1 year), "
            f"{hy['20d']:+.0f} bp over 20 days. Versus equities: {credit['verdict'].lower()}.",
        ]
        if mv is not None:
            lines.append(f"**Rate volatility.** MOVE at {mv['Level']:.0f}, {M.ordinal(mv['1y %ile'])} percentile of 1 year.")
        st.markdown('<div class="read">' + "<br><br>".join(
            __import__("re").sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", x) for x in lines) + "</div>", unsafe_allow_html=True)
        st.caption("Generated mechanically from the models in this section. Thresholds, not judgment.")
    with r:
        st.markdown("#### What's stretched")
        items = M.stretched(fi_table)
        st.markdown("\n".join(f"- {t}" for t in items[:12]) if items else
                    "Nothing unusual: no 5-day move beyond ±1.5σ and no level at a 1-year extreme.")
        st.caption("Same rules as equities: |5d z| ≥ 1.5, ≤ 5th / ≥ 95th percentile of 1 year, or a fresh trend flip.")
    st.markdown("#### Rates dashboard")
    groups = dict(FI_NAMES)
    groups["Bond ETFs & rate vol"] = FI_PX
    g = st.segmented_control("Group", list(groups), default="Treasury yields", key="fi_group",
                             label_visibility="collapsed")
    show = fi_table.loc[[n for n in groups[g or "Treasury yields"] if n in fi_table.index]]
    sty = (show.style.format({"Level": "{:,.2f}", "1d": "{:+.1f}", "5d": "{:+.1f}", "20d": "{:+.1f}",
                              "YTD": "{:+.1f}", "5d z": "{:+.2f}", "1y %ile": "{:.0f}", "Trend days": "{:d}"},
                             na_rep="–")
           .map(signed_color, subset=["1d", "5d", "20d", "YTD"]).map(z_color, subset=["5d z"])
           .map(state_style, subset=["Trend", "Trend × vol"]))
    st.dataframe(sty, width="stretch")
    st.caption("Levels in % for yields and spreads; changes in bp. ETFs in price and %. "
               "For yields and spreads, 'UPTREND' means the yield or spread is rising.")


def page_curve():
    snap = F.curve_snapshots(fi)
    l, r = st.columns([1.4, 1])
    with l:
        f = go.Figure()
        cols = [SERIES[0], MUTED, "#c3c2b7", SERIES[1]]
        for i, c_ in enumerate(snap.columns):
            f.add_scatter(x=[n.replace("UST ", "") for n in snap.index], y=snap[c_], name=c_, mode="lines+markers",
                          line=dict(color=[SERIES[0], SERIES[2], SERIES[3], SERIES[1]][i],
                                    width=3 if i == 0 else 1.6, dash=None if i == 0 else "dot"),
                          marker=dict(size=7 if i == 0 else 5))
        f.update_yaxes(ticksuffix="%")
        f.update_layout(hovermode="x unified")
        st.plotly_chart(base_layout(f, 380, "US Treasury curve: today vs earlier"), width="stretch")
    with r:
        chg = (snap.iloc[:, 0] - snap.iloc[:, 1]) * 100
        chg3 = (snap.iloc[:, 0] - snap.iloc[:, 2]) * 100
        f = go.Figure()
        xs = [n.replace("UST ", "") for n in snap.index]
        f.add_bar(x=xs, y=chg, name="vs 1 month ago", marker_color=[UP if v < 0 else DOWN for v in chg])
        f.add_scatter(x=xs, y=chg3, name="vs 3 months ago", mode="markers",
                      marker=dict(color="#0b0b0b", size=8, symbol="diamond"))
        f.update_yaxes(ticksuffix=" bp")
        st.plotly_chart(base_layout(f, 380, "Change by maturity (bp) · blue = yields down"), width="stretch")
    cr = crv.iloc[-1]
    a, b = st.columns([1, 1.6])
    with a:
        st.markdown(card("Curve regime (last 20 sessions)", cr["state"],
                         f"2Y {cr['d2Y bp']:+.0f} bp · 10Y {cr['d10Y bp']:+.0f} bp · 2s10s {cr['slope bp']:+.0f} bp · "
                         f"for {int(cr['days'])} sessions", CRV_COLOR[cr["state"]]), unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<div class="explain">{F.CURVE_REGIMES[cr["state"]]}</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown("""
| | Short end moves more | Long end moves more |
|---|---|---|
| **Yields falling (bull)** | Bull steepener | Bull flattener |
| **Yields rising (bear)** | Bear flattener | Bear steepener |
""")
    with b:
        yrs = st.radio("Window", [1, 2, 5, 10], index=2, horizontal=True, format_func=lambda x: f"{x}y", key="crv_y")
        w = last_n_years(fi, yrs)
        f = go.Figure()
        for i, n in enumerate(["2s10s curve", "3m10y curve", "5s30s curve"]):
            f.add_scatter(x=w.index, y=w[n] * 100, name=n.replace(" curve", ""), line=dict(color=SERIES[i], width=1.6))
        f.add_hline(y=0, line_color="#898781", line_width=1)
        f.update_yaxes(ticksuffix=" bp")
        st.plotly_chart(base_layout(f, 300, "Curve spreads (bp). Below 0 = inverted"), width="stretch")
        cw = last_n_years(crv, yrs)
        rib = go.Figure()
        for nm, col in CRV_COLOR.items():
            g = cw[cw["state"] == nm]
            rib.add_scatter(x=g.index, y=[nm] * len(g), mode="markers", marker=dict(symbol="square", size=6, color=col),
                            showlegend=False, hovertemplate=nm + " · %{x|%d %b %Y}<extra></extra>")
        rib.update_yaxes(categoryorder="array", categoryarray=list(CRV_COLOR)[::-1])
        st.plotly_chart(base_layout(rib, 210, "Curve regime history", False), width="stretch")
    st.markdown("##### What equities did next, by curve regime (S&P 500, next 20 sessions)")
    spx_f = (px["S&P 500"].shift(-20) / px["S&P 500"] - 1) * 100
    d = pd.DataFrame({"state": crv["state"], "fwd": spx_f.reindex(crv.index)}).dropna()
    g = d.groupby("state")["fwd"]
    tab = pd.DataFrame({"sessions": g.size(), "median fwd %": g.median(),
                        "% positive": g.apply(lambda x: (x > 0).mean() * 100), "worst %": g.min()}).reindex(list(CRV_COLOR))
    tab["now"] = ["◀ now" if i == cr["state"] else "" for i in tab.index]
    st.dataframe(tab.style.format({"sessions": "{:.0f}", "median fwd %": "{:+.2f}", "% positive": "{:.0f}",
                                   "worst %": "{:+.1f}"}, na_rep="–").map(signed_color, subset=["median fwd %"]),
                 width="stretch")
    st.caption("Overlapping 20-day windows: 'sessions' overstates independent episodes. Read as tendencies.")
    explain("""
**Curve regimes** classify the last 20 sessions by two questions:
* **Bull or bear?** Did yields fall (bull, bond prices up) or rise (bear)? Measured as the average 20-day change of the
  2Y and 10Y yields.
* **Steepener or flattener?** Did the gap between the 10Y and 2Y yield (2s10s) widen or narrow?

If yields moved less than 5 bp and the curve less than 3 bp, the state is RANGE-BOUND.

The **front end** (2Y and shorter) mostly reflects where the Fed is expected to set rates. The **long end** (10Y, 30Y)
adds growth and inflation expectations and a *term premium*: the extra yield investors demand for locking money up
for a long time. So who leads tells you what the market is reacting to.

**Spreads.** 2s10s is the most watched; 3m10y is the one with the best recession track record (inversions preceded
most US recessions, but with long and variable lags); 5s30s isolates the long end.
""")


def page_10y():
    st.markdown("Two ways to split a move in the 10-year yield. They answer different questions, so read them together.")
    a, b = st.columns(2)
    with a:
        st.markdown("##### 1 · Inflation expectations vs real yields")
        st.markdown(card("Last 20 sessions", rvb["driver"],
                         f"10Y {rvb['nominal']:+.0f} bp = real yield {rvb['real']:+.0f} bp + breakeven inflation "
                         f"{rvb['breakeven']:+.0f} bp", NEUTRAL), unsafe_allow_html=True)
        yrs = st.radio("Window", [1, 2, 5, 10], index=1, horizontal=True, format_func=lambda x: f"{x}y", key="rb_y")
        w = last_n_years(fi, yrs)
        f = go.Figure()
        f.add_scatter(x=w.index, y=w["UST 10Y"], name="10Y nominal", line=dict(color="#0b0b0b", width=1.6))
        f.add_scatter(x=w.index, y=w["10Y real yield (TIPS)"], name="10Y real (TIPS)", line=dict(color=SERIES[0], width=1.6))
        f.add_scatter(x=w.index, y=w["10Y breakeven inflation"], name="10Y breakeven", line=dict(color=SERIES[1], width=1.6))
        f.update_yaxes(ticksuffix="%")
        st.plotly_chart(base_layout(f, 330, "Nominal = real + breakeven (%)"), width="stretch")
    with b:
        st.markdown("##### 2 · Fed path vs term premium")
        sp = split.iloc[-1]
        st.markdown(card("Last 20 sessions", sp["verdict"],
                         f"10Y {sp['10Y move bp']:+.0f} bp = Fed path {sp['Fed path bp']:+.0f} bp "
                         f"(β {sp['beta to 2Y']:.2f} × 2Y move) + term premium & other {sp['Term premium & other bp']:+.0f} bp. "
                         f"Cross-check: Kim-Wright term premium {sp['Kim-Wright TP move bp']:+.0f} bp.", NEUTRAL),
                    unsafe_allow_html=True)
        yrs2 = st.radio("Window", [1, 2, 5], index=0, horizontal=True, format_func=lambda x: f"{x}y", key="tp_y")
        w = last_n_years(split, yrs2)
        f = go.Figure()
        f.add_scatter(x=w.index, y=w["Fed path bp"], name="Fed path", line=dict(color=SERIES[0], width=1.4),
                      stackgroup="one")
        f.add_scatter(x=w.index, y=w["Term premium & other bp"], name="Term premium & other",
                      line=dict(color=SERIES[1], width=1.4), stackgroup="two")
        f.add_scatter(x=w.index, y=w["10Y move bp"], name="10Y 20-day move", line=dict(color="#0b0b0b", width=1.4))
        f.update_yaxes(ticksuffix=" bp")
        st.plotly_chart(base_layout(f, 330, "Rolling 20-day 10Y move, split (bp)"), width="stretch")
    yrs3 = st.radio("Term premium history", [5, 10], index=1, horizontal=True, format_func=lambda x: f"{x}y", key="kw_y")
    w = last_n_years(fi, yrs3)
    f = go.Figure()
    f.add_scatter(x=w.index, y=w["10Y term premium (Kim-Wright)"] * 100, line=dict(color=SERIES[1], width=1.6),
                  fill="tozeroy", fillcolor="rgba(235,104,52,.12)", name="Kim-Wright 10Y term premium")
    f.add_hline(y=0, line_color="#898781", line_width=1)
    f.update_yaxes(ticksuffix=" bp")
    st.plotly_chart(base_layout(f, 260, "10Y term premium, Kim-Wright model (bp)", False), width="stretch")
    explain("""
**Split 1: real vs breakeven.** A nominal Treasury yield = real yield (what TIPS pay above inflation) + breakeven
inflation (the inflation rate the market is pricing). Rising real yields = tighter financial conditions, the harder
hit for equity valuations. Rising breakevens = inflation worries.

**Split 2: Fed path vs term premium.** The 2Y yield is mostly the expected path of Fed policy. We estimate, over the
last 250 sessions, how many bp the 10Y typically moves per bp of 2Y move (beta). Over the last 20 days:
*Fed path part* = beta × 2Y move; the rest is *term premium & other* (supply, inflation uncertainty, global yields).
Verdict: FED-DRIVEN if the Fed part is at least 2/3 of the total size, TERM-PREMIUM-DRIVEN if at most 1/3, else SHARED.

**Cross-check.** The Kim-Wright model (Federal Reserve Board, on FRED) estimates the term premium with a full term-structure
model. If our residual and Kim-Wright move in the same direction, the split is more trustworthy. The pack uses the
NY Fed's ACM model instead; the two usually agree on direction.
""")


def page_fed():
    fp = F.fed_path(fi)
    fp_1m = F.fed_path(fi, -22) if len(fi) > 22 else None
    l, r = st.columns([1.3, 1])
    with l:
        f = go.Figure()
        f.add_scatter(x=fp.index, y=fp["implied rate %"], name="today", mode="lines+markers",
                      line=dict(color=SERIES[0], width=2.5), marker=dict(size=9))
        if fp_1m is not None:
            f.add_scatter(x=fp_1m.index, y=fp_1m["implied rate %"], name="1 month ago", mode="lines+markers",
                          line=dict(color=MUTED, width=1.5, dash="dot"))
        f.update_yaxes(ticksuffix="%")
        f.update_layout(hovermode="x unified")
        st.plotly_chart(base_layout(f, 360, "Policy rate implied by the bill curve, by future window"), width="stretch")
    with r:
        ff = fi["Fed funds (EFFR)"].iloc[-1]
        st.markdown(card("Fed funds (effective)", f"{ff:.2f}%",
                         f"Implied 1–2 years out: {fp.loc['1-2y', 'implied rate %']:.2f}% "
                         f"({fp.loc['1-2y', 'vs today bp']:+.0f} bp ≈ {fp.loc['1-2y', 'vs today bp'] / 25:+.1f} × 25 bp). "
                         + (f"A month ago: {fp_1m.loc['1-2y', 'vs today bp']:+.0f} bp." if fp_1m is not None else ""),
                         NEUTRAL), unsafe_allow_html=True)
        st.write("")
        tab = F.fed_pricing(fi)
        st.dataframe(tab.style.format({"Yield %": "{:.2f}", "vs Fed funds bp": "{:+.0f}",
                                       "≈ 25bp moves": "{:+.1f}"}), width="stretch")
        st.caption("Yield minus fed funds = average change priced over the bill's life. "
                   "The chart's forward windows show *when* the moves are priced.")
    yrs = st.radio("Window", [2, 5, 10], index=1, horizontal=True, format_func=lambda x: f"{x}y", key="fed_y")
    w = last_n_years(fi, yrs)
    f = go.Figure()
    f.add_scatter(x=w.index, y=w["Fed funds (EFFR)"], name="Fed funds", line=dict(color="#0b0b0b", width=2, shape="hv"))
    f.add_scatter(x=w.index, y=w["UST 2Y"], name="2Y yield", line=dict(color=SERIES[0], width=1.5))
    f.add_scatter(x=w.index, y=w["UST 3M"], name="3M bill", line=dict(color=SERIES[2], width=1.2))
    f.update_yaxes(ticksuffix="%")
    st.plotly_chart(base_layout(f, 320, "Fed funds vs 3M and 2Y yields: 2Y below fed funds = cuts priced"), width="stretch")
    explain("""
**Idea.** A Treasury bill maturing in 6 months yields roughly the average overnight rate investors expect over those
6 months. So the gap between bill yields and today's fed funds rate shows how much easing (negative) or tightening
(positive) is priced. **Forward rates** between maturities, e.g. the rate implied for months 6–12 from the 6M and
1Y yields, show *when*: f = (y₁ × 1 − y₀.₅ × 0.5) / 0.5.

**Why this is only a proxy.** The pack's path comes from fed funds / SOFR futures, meeting by meeting. Those
contracts aren't freely downloadable. Bills also carry their own supply/demand effects (bills can trade a few bp
below or above expected policy), and the 2Y includes a small term premium. Treat the levels as ±10 bp and focus on
direction and changes.
""")


def page_credit():
    ig, hy = _fi_row("IG credit spread (OAS)"), _fi_row("HY credit spread (OAS)")
    k = st.columns(3)
    for col, r_, lab in [(k[0], ig, "Investment grade (IG) spread"), (k[1], hy, "High yield (HY) spread")]:
        if r_ is not None:
            col.markdown(card(lab, f"{bp(r_['Level']):.0f} bp",
                              f"{M.ordinal(r_['1y %ile'])} percentile of 1 year · 5d {r_['5d']:+.0f} bp · 20d {r_['20d']:+.0f} bp · "
                              f"trend {r_['Trend']}", DOWN if r_["1y %ile"] >= 80 else UP if r_["1y %ile"] <= 20 else NEUTRAL),
                         unsafe_allow_html=True)
    k[2].markdown(card("Is credit confirming equities? (20 days)", credit["verdict"].split(":")[0],
                       f"S&P {credit['eq_move']:+.1f}% (z {credit['eq_z']:+.1f}) · HY spread {credit['hy_move_bp']:+.0f} bp "
                       f"(z {credit['hy_z']:+.1f}). " + (credit["verdict"].split(":", 1)[1] if ":" in credit["verdict"] else ""),
                       NEUTRAL), unsafe_allow_html=True)
    st.write("")
    w_all = fi[["IG credit spread (OAS)", "HY credit spread (OAS)"]].dropna(how="all")
    span = (w_all.index[-1] - w_all.index[0]).days / 365
    l, r = st.columns(2)
    for col, n, c_ in [(l, "HY credit spread (OAS)", SERIES[1]), (r, "IG credit spread (OAS)", SERIES[0])]:
        s_ = fi[n].dropna() * 100
        f = go.Figure()
        f.add_scatter(x=s_.index, y=s_, line=dict(color=c_, width=1.6), name=n,
                      hovertemplate="%{x|%d %b %Y}: %{y:.0f} bp<extra></extra>")
        f.add_hline(y=s_.iloc[-252:].median(), line_dash="dot", line_color=NEUTRAL,
                    annotation_text="1y median", annotation_position="top left")
        f.update_yaxes(ticksuffix=" bp")
        col.plotly_chart(base_layout(f, 300, n, False), width="stretch")
    if span < 5:
        st.caption(f"Spread history available: about {span:.0f} years (FRED only serves recent history for the ICE BofA "
                   "indices). Percentiles and z-scores use what is available; long event studies on spreads are not possible.")
    yrs = st.radio("Window", [1, 2, 5], index=0, horizontal=True, format_func=lambda x: f"{x}y", key="cr_y")
    l, r = st.columns(2)
    rr = last_n_years((px["High Yield (HYG)"] / px["IG Corp (LQD)"]).dropna().to_frame("r"), yrs)["r"] \
        if "IG Corp (LQD)" in px else None
    if rr is not None:
        f = go.Figure()
        f.add_scatter(x=rr.index, y=rr / rr.iloc[0] * 100, line=dict(color=UP, width=1.6), name="HYG/LQD")
        l.plotly_chart(base_layout(f, 280, "HYG / LQD (indexed): falling = junk lagging quality", False), width="stretch")
        l.caption("A price-based credit risk-appetite gauge with long history. It moves with HY spreads but is available daily from Yahoo.")
    hyw = last_n_years((fi["HY credit spread (OAS)"].dropna() * 100).to_frame("hy"), yrs)["hy"]
    spw = px["S&P 500"].reindex(hyw.index).ffill()
    f = go.Figure()
    f.add_scatter(x=hyw.index, y=(hyw - hyw.iloc[0]), name="HY spread change (bp)", line=dict(color=SERIES[1], width=1.6))
    f.add_scatter(x=spw.index, y=-(spw / spw.iloc[0] - 1) * 100 * 10, name="S&P fall × 10 (inverted, %)",
                  line=dict(color=SERIES[0], width=1.2, dash="dot"))
    f.add_hline(y=0, line_color="#898781", line_width=1)
    r.plotly_chart(base_layout(f, 280, "Spread widening vs S&P falling (same direction = confirming)"), width="stretch")
    r.caption("S&P is inverted and scaled ×10 so both lines rise in stress. Shape matters, not the scale.")
    explain("""
**OAS (option-adjusted spread)** is the extra yield a corporate bond pays over a Treasury of the same maturity,
adjusted for embedded options. It is the market's price of default and liquidity risk.
* **IG**: investment-grade companies (BBB and above). Moves slowly; widening signals broad stress.
* **HY**: high yield (junk). Much more sensitive to growth and risk appetite, often moves *before* equities.

**Confirmation.** Equity sell-offs that credit does *not* confirm (spreads barely widen) have historically been more
likely to be rotation or positioning events than the start of a recession trade. When stocks fall and HY spreads
widen sharply at the same time, the move is broad-based risk aversion. When stocks *rise* but spreads widen, credit
may be leading equities lower.

The verdict uses 20-day moves, each scaled by its own history (z-score): z below −1 / above +1 counts as a strong move.
""")


def page_ratevol():
    rv = rvol.dropna(subset=["10Y realised vol (bp/yr)"])
    has_move = "MOVE" in rv and rv["MOVE"].notna().any()
    k = st.columns(3)
    if has_move:
        mv = rv["MOVE"].dropna()
        pct = (mv.iloc[-253:-1] < mv.iloc[-1]).mean() * 100
        k[0].markdown(card("MOVE index (implied Treasury vol)", f"{mv.iloc[-1]:.0f}",
                           f"{M.ordinal(pct)} percentile of 1 year. Rough guide: below 80 calm, 80–120 normal, above 120 stressed.",
                           BAD if mv.iloc[-1] > 120 else WARN if mv.iloc[-1] > 95 else GOOD), unsafe_allow_html=True)
        prem = rv["MOVE minus realised"].dropna().iloc[-1]
        k[2].markdown(card("Implied minus realised", f"{prem:+.0f} bp/yr",
                           "Positive = options price more rate movement than is happening (normal).",
                           GOOD if prem > 0 else BAD), unsafe_allow_html=True)
    k[1].markdown(card("10Y realised vol (21 days)", f"{rv['10Y realised vol (bp/yr)'].iloc[-1]:.0f} bp/yr",
                       f"≈ {rv['10Y realised vol (bp/yr)'].iloc[-1] / np.sqrt(252):.1f} bp per day typical move.",
                       NEUTRAL), unsafe_allow_html=True)
    yrs = st.radio("Window", [1, 2, 5, 10], index=1, horizontal=True, format_func=lambda x: f"{x}y", key="rv_y")
    w = last_n_years(rv, yrs)
    l, r = st.columns(2)
    f = go.Figure()
    if has_move:
        f.add_scatter(x=w.index, y=w["MOVE"], name="MOVE (implied)", line=dict(color=UP, width=1.8))
    f.add_scatter(x=w.index, y=w["10Y realised vol (bp/yr)"], name="10Y realised", line=dict(color=SERIES[1], width=1.5))
    l.plotly_chart(base_layout(f, 320, "Implied vs realised rate volatility (bp per year)"), width="stretch")
    if "MOVE / VIX" in w:
        f = go.Figure()
        f.add_scatter(x=w.index, y=w["MOVE / VIX"], line=dict(color=SERIES[2], width=1.6), name="MOVE / VIX")
        f.add_hline(y=w["MOVE / VIX"].median(), line_dash="dot", line_color=NEUTRAL,
                    annotation_text="median of window", annotation_position="top left")
        r.plotly_chart(base_layout(f, 320, "MOVE / VIX: is the stress in rates or in equities?", False), width="stretch")
    explain("""
* **MOVE** is to Treasuries what VIX is to the S&P: implied volatility from Treasury options, expressed in basis points
  per year. MOVE 100 ≈ the market expects 10Y-ish yields to move about 100 bp over a year (≈ 6 bp a day).
* **Realised** = standard deviation of daily 10Y changes over 21 days × √252, in the same units.
* **MOVE / VIX** rising = stress concentrated in rates (policy or inflation uncertainty); falling = stress concentrated
  in equities. High rate vol also tends to weigh on equity valuations and on liquidity across markets.
""")


# ===========================================================================
# CROSS-ASSET OVERVIEW
# ===========================================================================
def mini_card(label: str, value: str, note: str, color: str | None = None) -> str:
    bar = f"border-left:4px solid {color};" if color else ""
    return (f'<div class="card" style="{bar}padding:10px 12px"><div class="lbl">{label}</div>'
            f'<div class="val" style="font-size:1.0rem">{value}</div><div class="note" style="font-size:.8rem">{note}</div></div>')


def regime_row(title: str, horizon: str, cards: list, n_cols: int = 4):
    st.markdown(f"<div style='margin:14px 0 6px 0'><b>{title}</b> <span style='color:#52514e;font-size:.85rem'>"
                f"· {horizon}</span></div>", unsafe_allow_html=True)
    cols = st.columns(max(n_cols, len(cards)))
    for c_, html in zip(cols, cards):
        c_.markdown(html, unsafe_allow_html=True)


def page_summary():
    k = st.columns(8)
    t10, s210, hy, mv = (_fi_row(n) for n in ["UST 10Y", "2s10s curve", "HY credit spread (OAS)", "MOVE (rate vol)"])
    b = cr_table.loc["Bitcoin"] if "Bitcoin" in cr_table.index else None
    items = [
        ("S&P 500", f"{spx.iloc[-1]:,.0f}", f"{row['5d']:+.1f}% 5d"),
        ("UST 10Y", f"{fi['UST 10Y'].iloc[-1]:.2f}%", f"{t10['5d']:+.0f} bp 5d"),
        ("2s10s", f"{bp(fi['2s10s curve'].iloc[-1]):.0f} bp", f"{s210['5d']:+.0f} bp 5d"),
        ("HY spread", f"{bp(hy['Level']):.0f} bp", f"{hy['5d']:+.0f} bp 5d"),
        ("VIX", f"{table.loc['VIX', 'Level']:.1f}", f"{table.loc['VIX', '5d']:+.1f} 5d"),
        ("MOVE", f"{mv['Level']:.0f}" if mv is not None else "–", f"{mv['5d']:+.0f} 5d" if mv is not None else ""),
        ("Dollar (DXY)", f"{table.loc['Dollar (DXY)', 'Level']:.1f}", f"{table.loc['Dollar (DXY)', '5d']:+.1f}% 5d"),
        ("Bitcoin", f"{b['Level']:,.0f}" if b is not None else "–", f"{b['5d']:+.1f}% 5d" if b is not None else ""),
    ]
    for c_, (lab, v, d) in zip(k, items):
        c_.metric(lab, v, d, delta_color="off")

    st.markdown("#### Regime map: the current state of every model")
    st.caption("Grouped by market, slowest models first within each group. The coloured edge shows the direction "
               "(blue supportive / rising, red adverse / falling, grey neutral). Details in each section.")
    # ---- macro ----
    sc = sbcorr.iloc[-1]
    regime_row("Macro", "monthly to weekly", [
        mini_card("Economy · growth × inflation", qnow["name"],
                  f"{qnow['subtitle']}. {qnow['months']} month(s). Data to {qnow['as_of']:%b %Y}.",
                  Q.QUADRANT_COLOR[qnow["name"]]),
        mini_card("Market-implied macro regime", sbc_now["label"],
                  f"{int(sbc_now['days'])} session(s). Stocks {sbc_now['Stocks']:+.1f}σ, yields "
                  f"{sbc_now['Bonds (10Y yield)']:+.1f}σ over 20 days.", regime_color(sbc_now["name"])),
        mini_card("Stock–bond hedge (60d)", "Bonds hedging" if sc > 0.1 else "Not hedging" if sc < -0.1 else "No stable link",
                  f"Correlation of S&P returns with 10Y yield changes {sc:+.2f}.",
                  UP if sc > 0.1 else DOWN if sc < -0.1 else NEUTRAL),
    ])
    # ---- equities ----
    st_tr = tr_spx["state"].iloc[-1]
    br_now = breadth.iloc[-1]
    regime_row("Equities", "S&P 500 unless stated", [
        mini_card("Trend (50/200-day)", pill(st_tr), f"{int(tr_spx['days'].iloc[-1])} sessions. {M.trend_flip_text(tr_spx)}",
                  STATE_COLOR[st_tr]),
        mini_card("Trend × volatility state", pill(mk_spx["state"]),
                  f"{mk_spx['days']} sessions; typically lasts ~{mk_spx['exp_len']:.0f} days ({mk_spx['style']}).",
                  STATE_COLOR[mk_spx["state"]]),
        mini_card("Breadth", f"{br_now:.0f}% above 50-day",
                  f"Sectors and mega-caps above their 50-day average (5 days ago {breadth.iloc[-6]:.0f}%).",
                  DOWN if br_now < 30 else UP if br_now > 70 else NEUTRAL),
        mini_card("Equity volatility", f"VIX {v_now['VIX']:.1f} · term {'inverted' if term > 1 else 'flat' if term >= 0.97 else 'normal'}",
                  f"VIX/VIX3M {term:.2f}; implied minus realised {v_now['VRP (VIX − RV)']:+.1f} pts.",
                  BAD if term > 1 else WARN if term >= 0.97 else GOOD),
    ])
    # ---- fixed income ----
    cr_ = crv.iloc[-1]
    sp_ = split.iloc[-1]
    fp = F.fed_path(fi)
    two = fp.loc["1-2y", "vs today bp"]
    t10_state = mk_fi.get("UST 10Y")
    hy_state = mk_fi.get("HY credit spread (OAS)")
    regime_row("Fixed income · rates", "for yields, Up = yields rising", [
        mini_card("10Y yield trend & state", pill(t10["Trend"]) + (pill(t10_state["state"]) if t10_state else ""),
                  f"Trend for {int(t10['Trend days'])} sessions; 10Y {t10['20d']:+.0f} bp over 20 days.",
                  STATE_COLOR.get(t10["Trend"], NEUTRAL)),
        mini_card("Curve regime (20d)", cr_["state"],
                  f"2Y {cr_['d2Y bp']:+.0f} bp, 10Y {cr_['d10Y bp']:+.0f} bp, 2s10s {cr_['slope bp']:+.0f} bp. "
                  f"{int(cr_['days'])} sessions.", CRV_COLOR[cr_["state"]]),
        mini_card("What moved the 10Y (20d)", sp_["verdict"],
                  f"Fed path {sp_['Fed path bp']:+.0f} bp, term premium & other {sp_['Term premium & other bp']:+.0f} bp; "
                  f"real yields {rvb['real']:+.0f}, breakevens {rvb['breakeven']:+.0f} bp.", NEUTRAL),
        mini_card("Fed pricing (bill curve)", f"{'Cuts' if two < -5 else 'Hikes' if two > 5 else 'On hold'} priced",
                  f"1–2 years out {two:+.0f} bp vs fed funds ({two / 25:+.1f} × 25 bp).",
                  UP if two < -5 else DOWN if two > 5 else NEUTRAL),
    ])
    regime_row("Fixed income · credit & rate volatility", "spreads: Up = spreads widening", [
        mini_card("Credit", f"HY {bp(hy['Level']):.0f} bp · {M.ordinal(hy['1y %ile'])} pct",
                  f"Spread trend {hy['Trend'].lower()}" + (f", state {hy_state['state']}" if hy_state else "")
                  + f". vs equities: {credit['verdict'].split(':')[0].lower()}.",
                  DOWN if hy["1y %ile"] >= 80 else UP if hy["1y %ile"] <= 20 else NEUTRAL),
        mini_card("Rate volatility", f"MOVE {mv['Level']:.0f} · {M.ordinal(mv['1y %ile'])} pct" if mv is not None else "–",
                  "Implied Treasury volatility vs its 1-year range.",
                  (BAD if mv["1y %ile"] >= 85 else GOOD if mv["1y %ile"] <= 25 else NEUTRAL) if mv is not None else NEUTRAL),
    ])
    # ---- crypto ----
    if b is not None:
        mk_b = mk_cr.get("Bitcoin")
        alts, share = K.alt_breadth(px)
        regime_row("Crypto", "Bitcoin unless stated", [
            mini_card("Trend (50/200-day)", pill(b["Trend"]), f"{int(b['Trend days'])} sessions; "
                      f"{K.drawdown(px['Bitcoin']).iloc[-1]:+.0f}% from its high.", STATE_COLOR.get(b["Trend"], NEUTRAL)),
            mini_card("Trend × volatility state", pill(mk_b["state"]) if mk_b else "–",
                      f"{mk_b['days']} sessions ({mk_b['style']})." if mk_b else "", STATE_COLOR.get(mk_b["state"], NEUTRAL) if mk_b else None),
            mini_card("Link to equities (60d)", f"Nasdaq corr {cr_corr.iloc[0, 0]:+.2f}",
                      "Above 0.5 = trading as a high-beta tech asset.", NEUTRAL),
            mini_card("Risk appetite inside crypto", f"{share:.0f}% of alts beat BTC",
                      "Share of large altcoins outperforming Bitcoin over 20 days.",
                      UP if share >= 60 else DOWN if share <= 30 else NEUTRAL),
        ])

    st.write("")
    l, r = st.columns([1.35, 1])
    with l:
        st.markdown("#### The read, in plain English")
        ew = ratios.loc["Equal-weight vs cap-weight (RSP/SPY)"] if "Equal-weight vs cap-weight (RSP/SPY)" in ratios.index else None
        lines = [
            f"**Economy.** On data to {qnow['as_of']:%B %Y}, the US is in **{qnow['name']}** ({qnow['subtitle'].lower()}).",
            f"**What markets trade.** The market-implied macro regime is **{sbc_now['label'].lower()}**. "
            f"60-day stock/yield correlation {sc:+.2f}: "
            + ("bonds are hedging stocks." if sc > 0.1 else "bonds are not hedging stocks." if sc < -0.1 else "no stable link."),
            f"**Equities.** S&P {row['5d']:+.1f}% this week (z {row['5d z']:+.1f}); trend {st_tr.lower()}, "
            f"trend × volatility state {mk_spx['state']}."
            + (f" Equal-weight vs cap-weight {ew['5d %']:+.1f}% in 5 days"
               + (": the average stock is holding up better than the giants (leadership unwind, not broad selling)."
                  if ew["5d %"] > 0 and row["5d"] < 0 else ".") if ew is not None else ""),
            f"**Rates.** 10Y trend {t10['Trend'].lower()}; curve {cr_['state'].lower()} over 20 sessions; the 10Y move was "
            f"{sp_['verdict'].lower()}. The bill curve prices {two:+.0f} bp vs fed funds for the 1–2 year window.",
            f"**Credit.** HY spread {bp(hy['Level']):.0f} bp ({M.ordinal(hy['1y %ile'])} percentile of 1 year). "
            f"{credit['verdict'].capitalize()}.",
            f"**Volatility.** VIX {table.loc['VIX', 'Level']:.1f} ({M.ordinal(table.loc['VIX', '1y %ile'])} percentile), "
            f"term structure {'inverted' if term > 1 else 'flat' if term >= 0.97 else 'normal'}"
            + (f"; MOVE {mv['Level']:.0f} ({M.ordinal(mv['1y %ile'])} percentile)." if mv is not None else "."),
        ]
        if b is not None:
            lines.append(f"**Crypto.** Bitcoin {b['5d']:+.1f}% this week, {K.drawdown(px['Bitcoin']).iloc[-1]:+.0f}% from its "
                         f"high; trend {b['Trend'].lower()}; 60-day correlation with the Nasdaq {cr_corr.iloc[0, 0]:+.2f}.")
        read_box(lines)
    with r:
        st.markdown("#### What's stretched, across markets")
        key_eq = table.loc[[n for n in list(D.INDICES) + ["VIX", "Dollar (DXY)"] if n in table.index]]
        key_fi = fi_table.loc[[n for n in ["UST 2Y", "UST 10Y", "2s10s curve", "10Y real yield (TIPS)",
                                           "10Y breakeven inflation", "HY credit spread (OAS)",
                                           "IG credit spread (OAS)", "MOVE (rate vol)"] if n in fi_table.index]]
        key_cr = cr_table.loc[[n for n in ["Bitcoin", "Ether"] if n in cr_table.index]]
        items = M.stretched(pd.concat([key_eq, key_fi, key_cr]))
        st.markdown("\n".join(f"- {t}" for t in items[:14]) if items else "Nothing unusual across the key series.")
        st.caption("Key series only. Each section has its own fuller list.")


# ===========================================================================
# SHARED SMALL HELPERS
# ===========================================================================
def read_box(lines):
    st.markdown('<div class="read">' + "<br><br>".join(
        __import__("re").sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", x) for x in lines) + "</div>", unsafe_allow_html=True)
    st.caption("Generated mechanically from the models in this section. Thresholds, not judgment.")


def dash_table(tab: pd.DataFrame, decimals: str = "{:,.4f}"):
    sty = (tab.style.format({"Level": decimals, "1d": "{:+.2f}", "5d": "{:+.2f}", "20d": "{:+.2f}", "YTD": "{:+.1f}",
                             "5d z": "{:+.2f}", "1y %ile": "{:.0f}", "Trend days": "{:d}"}, na_rep="–")
           .map(signed_color, subset=["1d", "5d", "20d", "YTD"]).map(z_color, subset=["5d z"])
           .map(state_style, subset=["Trend", "Trend × vol"]))
    st.dataframe(sty, width="stretch", height=min(45 + 35 * len(tab), 600))


# ===========================================================================
# CRYPTO PAGES
# ===========================================================================
def page_cr_today():
    k = st.columns(6)
    for i, n in enumerate(["Bitcoin", "Ether", "Solana", "XRP", "Bitcoin ETF (IBIT)", "Strategy (MSTR)"]):
        if n in cr_table.index:
            r_ = cr_table.loc[n]
            k[i].metric(n, f"{r_['Level']:,.0f}" if r_["Level"] > 100 else f"{r_['Level']:,.2f}",
                        f"{r_['1d']:+.1f}% · {r_['5d']:+.1f}% 5d", delta_color="off")
    st.write("")
    st.markdown("#### Where is crypto? Four lenses")
    b = cr_table.loc["Bitcoin"]
    mk_b = mk_cr.get("Bitcoin")
    dd = K.drawdown(px["Bitcoin"]).iloc[-1]
    vs = K.vol_stats(px["Bitcoin"])
    c = st.columns(4)
    c[0].markdown(card("1 · Bitcoin trend", pill(b["Trend"]) + (pill(mk_b["state"]) if mk_b else ""),
                       f"Slow trend for {int(b['Trend days'])} sessions. {dd:+.0f}% from its all-time high. "
                       + (f"Fast state typically lasts ~{mk_b['exp_len']:.0f} days ({mk_b['style']})." if mk_b else ""),
                       STATE_COLOR.get(b["Trend"], NEUTRAL)), unsafe_allow_html=True)
    v30 = vs[30]
    c[1].markdown(card("2 · Bitcoin volatility (30d)", f"{v30['now']:.0f}% · {M.ordinal(v30['pct'])} pct",
                       f"Realised volatility, annualised, vs its last 3 years. 90-day: {vs[90]['now']:.0f}%.",
                       BAD if v30["pct"] >= 85 else GOOD if v30["pct"] <= 25 else NEUTRAL), unsafe_allow_html=True)
    nq = cr_corr.loc["Nasdaq 100"] if "Nasdaq 100" in cr_corr.index else None
    c[2].markdown(card("3 · Trading like…", ("Tech stocks" if nq is not None and nq.iloc[0] > 0.5 else
                                             "Its own asset" if nq is not None and nq.iloc[0] < 0.2 else "Partly risk asset"),
                       (f"60-day correlation with the Nasdaq {nq.iloc[0]:+.2f}, beta {nq.iloc[2]:.1f} "
                        f"(a 1% Nasdaq move ≈ {nq.iloc[2]:.1f}% in Bitcoin)." if nq is not None else ""), NEUTRAL),
                  unsafe_allow_html=True)
    alts, share = K.alt_breadth(px)
    ethbtc = (px["Ether"] / px["Bitcoin"]).dropna()
    eb20 = (ethbtc.iloc[-1] / ethbtc.iloc[-21] - 1) * 100
    c[3].markdown(card("4 · Risk appetite inside crypto", f"{share:.0f}% of alts beat BTC",
                       f"Share of {len(alts)} large altcoins outperforming Bitcoin over 20 days. ETH/BTC {eb20:+.1f}%. "
                       + ("Speculation broadening." if share >= 60 else "Flight to Bitcoin (quality)." if share <= 30 else ""),
                       UP if share >= 60 else DOWN if share <= 30 else NEUTRAL), unsafe_allow_html=True)
    st.write("")
    l, r = st.columns([1.35, 1])
    with l:
        st.markdown("#### The read, in plain English")
        lad = K.ladder(px["Bitcoin"])
        above = lad[lad["Level"] > b["Level"]].iloc[-1] if (lad["Level"] > b["Level"]).any() else None
        below = lad[lad["Level"] < b["Level"]].iloc[0] if (lad["Level"] < b["Level"]).any() else None
        etf = K.etf_activity(px)
        lines = [
            f"**Bitcoin.** {b['5d']:+.1f}% this week (z {b['5d z']:+.1f}), {b['20d']:+.1f}% over 20 days, "
            f"{dd:+.0f}% from its high. Slow trend {b['Trend'].lower()}"
            + (f", fast state {mk_b['state']}." if mk_b else "."),
            "**Nearest levels.** "
            + (f"Above: {above.name.lower()} at {above['Level']:,.0f} ({above['Distance %']:+.1f}%). " if above is not None else "")
            + (f"Below: {below.name.lower()} at {below['Level']:,.0f} ({below['Distance %']:+.1f}%)." if below is not None else ""),
            f"**Macro link.** " + (f"60-day correlation with the Nasdaq {nq.iloc[0]:+.2f}: "
                                   + ("Bitcoin is trading as a high-beta tech asset, so equity risk drives it."
                                      if nq.iloc[0] > 0.5 else "the link to equities is loose right now.") if nq is not None else ""),
            f"**Inside crypto.** {share:.0f}% of large altcoins beat Bitcoin over 20 days; ETH/BTC {eb20:+.1f}%.",
        ]
        if not etf.empty:
            lines.append("**ETF activity.** " + "; ".join(f"{i.split('(')[1][:-1]} trading {r_['5-day vs 50-day']:.1f}× its 50-day "
                                                          f"average dollar volume, price {r_['Price 5d %']:+.1f}% in 5 days"
                                                          for i, r_ in etf.iterrows()) + ".")
        read_box(lines)
    with r:
        st.markdown("#### What's stretched")
        items = M.stretched(cr_table)
        st.markdown("\n".join(f"- {t}" for t in items[:12]) if items else "Nothing unusual in crypto.")
        st.caption("Crypto trades 7 days a week; this dashboard uses US trading days, so Monday includes the weekend.")
    st.markdown("#### Crypto dashboard")
    dash_table(cr_table, "{:,.2f}")


def page_cr_levels():
    coin = st.segmented_control("Coin", [c_ for c_ in ["Bitcoin", "Ether", "Solana"] if c_ in px], default="Bitcoin",
                                key="cr_coin") or "Bitcoin"
    s_ = px[coin].dropna()
    lad = K.ladder(s_)
    l, r = st.columns([1, 1.5])
    with l:
        st.markdown(f"##### {coin} level ladder (price {s_.iloc[-1]:,.2f})")
        st.dataframe(lad.style.format({"Level": "{:,.2f}", "Distance %": "{:+.1f}"})
                     .map(lambda v: f"color:{DOWN if 'resistance' in str(v) else UP}", subset=["Role"]),
                     width="stretch", height=45 + 35 * len(lad))
    with r:
        yrs = st.radio("Window", [1, 2, 5], index=0, horizontal=True, format_func=lambda x: f"{x}y", key="cr_lv_y")
        w = last_n_years(s_.to_frame("p"), yrs)["p"]
        f = go.Figure()
        f.add_scatter(x=w.index, y=w, line=dict(color="#0b0b0b", width=1.5), name=coin)
        for nm, col in [("50-day average", SERIES[1]), ("200-day average", SERIES[0])]:
            win = 50 if "50" in nm else 200
            f.add_scatter(x=w.index, y=s_.rolling(win).mean().reindex(w.index), name=nm, line=dict(color=col, width=1.4))
        for nm in ["52-week high", "52-week low", "Round number above", "Round number below"]:
            if nm in lad.index:
                f.add_hline(y=lad.loc[nm, "Level"], line_dash="dot", line_color=MUTED,
                            annotation_text=nm, annotation_position="top left", annotation_font_size=10)
        f.update_yaxes(type="log")
        st.plotly_chart(base_layout(f, 420, f"{coin} with key levels (log scale)"), width="stretch")
    l, r = st.columns(2)
    ddw = last_n_years(K.drawdown(s_).to_frame("d"), 5)["d"]
    f = go.Figure()
    f.add_scatter(x=ddw.index, y=ddw, fill="tozeroy", line=dict(color=DOWN, width=1), fillcolor="rgba(227,73,72,.18)",
                  name="drawdown", hovertemplate="%{x|%d %b %Y}: %{y:.0f}%<extra></extra>")
    f.update_yaxes(ticksuffix="%")
    l.plotly_chart(base_layout(f, 300, f"{coin}: distance from all-time high", False), width="stretch")
    vs = K.vol_stats(s_)
    f = go.Figure()
    for i, w_ in enumerate((30, 90)):
        vv = last_n_years(vs[w_]["series"].dropna().to_frame("v"), 5)["v"]
        f.add_scatter(x=vv.index, y=vv, name=f"{w_}-day realised vol", line=dict(color=SERIES[i], width=1.5))
    f.update_yaxes(ticksuffix="%")
    r.plotly_chart(base_layout(f, 300, f"{coin}: realised volatility (annualised)"), width="stretch")
    explain("""
**Level ladder.** The levels traders commonly watch, sorted around today's price: moving averages, recent and 52-week
highs/lows, the all-time high in the data, the year's opening price, and the nearest round numbers. They matter
because many participants react at the same places, not because of any fundamental value. The nearest level above
is the first likely resistance, the nearest below the first likely support.

**Drawdown.** How far the price is below its previous peak. Bitcoin has had several 70–80% drawdowns; 20–30% pullbacks
inside bull markets are common.

**Realised volatility** is annualised from daily moves. Bitcoin typically runs at 40–80%, three to four times the S&P.
""")


def page_cr_macro():
    coin = st.segmented_control("Coin", [c_ for c_ in ["Bitcoin", "Ether"] if c_ in px], default="Bitcoin",
                                key="cr_mcoin") or "Bitcoin"
    tab = K.macro_corr(px, fi, coin)
    l, r = st.columns([1, 1.4])
    with l:
        st.dataframe(tab.style.format("{:+.2f}"), width="stretch")
        st.caption("Correlation of daily moves. Beta = % move in the coin for a 1% move in the driver "
                   "(per 1 percentage point for the real yield).")
    with r:
        yrs = st.radio("Window", [1, 2, 5], index=1, horizontal=True, format_func=lambda x: f"{x}y", key="cr_m_y")
        f = go.Figure()
        for i, d in enumerate(["Nasdaq 100", "Gold", "Dollar (DXY)"]):
            rc = last_n_years(K.rolling_corr(px, coin, d).dropna().to_frame("c"), yrs)["c"]
            f.add_scatter(x=rc.index, y=rc, name=d, line=dict(color=SERIES[i], width=1.5))
        f.add_hline(y=0, line_color=MUTED)
        f.update_yaxes(range=[-1, 1])
        st.plotly_chart(base_layout(f, 340, f"{coin}: 60-day correlation with macro assets"), width="stretch")
    explain("""
Bitcoin's story ("digital gold", "tech risk asset", "dollar hedge") changes over time, and correlations show which one
the market is trading *now*.
* **High correlation with the Nasdaq** (above ~0.5): Bitcoin behaves like a leveraged tech stock. Equity risk, liquidity
  and real yields drive it; it won't diversify an equity portfolio in a sell-off.
* **Correlation with gold rising**: the debasement / store-of-value narrative is in charge.
* **Negative with the dollar and real yields**: the usual pattern for a liquidity-sensitive asset; tighter financial
  conditions hurt it.
""")


def page_cr_alts():
    alts, share = K.alt_breadth(px)
    l, r = st.columns([1, 1.3])
    with l:
        st.markdown(f"##### Altcoins vs Bitcoin, last 20 days ({share:.0f}% outperforming)")
        f = go.Figure(go.Bar(x=alts["vs Bitcoin pp"], y=alts.index, orientation="h",
                             marker_color=[UP if v > 0 else DOWN for v in alts["vs Bitcoin pp"]],
                             text=[f"{v:+.1f}" for v in alts["vs Bitcoin pp"]], textposition="outside", cliponaxis=False))
        f.update_yaxes(autorange="reversed")
        lim = alts["vs Bitcoin pp"].abs().max() * 1.35 + 2
        f.update_xaxes(range=[-lim, lim], title="20-day return minus Bitcoin's (pp)")
        st.plotly_chart(base_layout(f, 340, None, False), width="stretch")
    with r:
        yrs = st.radio("Window", [1, 2, 5], index=1, horizontal=True, format_func=lambda x: f"{x}y", key="cr_a_y")
        eb = last_n_years((px["Ether"] / px["Bitcoin"]).dropna().to_frame("r"), yrs)["r"]
        f = go.Figure()
        f.add_scatter(x=eb.index, y=eb, line=dict(color=SERIES[0], width=1.6), name="ETH/BTC")
        st.plotly_chart(base_layout(f, 200, "ETH/BTC: rising = appetite for riskier crypto", False), width="stretch")
        ab = last_n_years(K.alt_breadth_series(px).dropna().to_frame("b"), yrs)["b"]
        f = go.Figure()
        f.add_scatter(x=ab.index, y=ab.rolling(5).mean(), line=dict(color=SERIES[2], width=1.6), fill="tozeroy",
                      fillcolor="rgba(27,175,122,.12)", name="% alts beating BTC (5d avg)")
        f.add_hline(y=50, line_dash="dot", line_color=MUTED)
        f.update_yaxes(range=[0, 100], ticksuffix="%")
        st.plotly_chart(base_layout(f, 200, "% of altcoins beating Bitcoin over 20 days", False), width="stretch")
    st.markdown("##### Spot ETF activity")
    etf = K.etf_activity(px)
    if etf.empty:
        st.info("ETF volume data not available.")
    else:
        st.dataframe(etf.style.format({"Dollar volume today $m": "{:,.0f}", "5-day avg $m": "{:,.0f}",
                                       "50-day avg $m": "{:,.0f}", "5-day vs 50-day": "{:.2f}×", "Price 5d %": "{:+.1f}"})
                     .map(signed_color, subset=["Price 5d %"]), width="stretch")
        pick = st.selectbox("Volume history", list(etf.index), key="cr_etf")
        dv = (px[pick] * px[f"{pick} volume"]).dropna() / 1e6
        dv = last_n_years(dv.to_frame("v"), 2)["v"]
        f = go.Figure()
        f.add_bar(x=dv.index, y=dv, marker_color=MUTED, name="daily $ volume")
        f.add_scatter(x=dv.index, y=dv.rolling(50).mean(), line=dict(color=SERIES[0], width=2), name="50-day average")
        f.update_yaxes(ticksuffix="m")
        st.plotly_chart(base_layout(f, 280, f"{pick}: dollar trading volume ($m)"), width="stretch")
    explain("""
**Alt breadth.** When most large altcoins beat Bitcoin, speculative appetite is broadening (late-cycle froth in past
cycles). When Bitcoin beats nearly everything, money is retreating to the most liquid coin, a defensive sign
*inside* crypto. ETH/BTC is the single most watched version of this.

**ETF activity.** Daily *net flows* into spot Bitcoin and Ether ETFs are published by the issuers and data vendors,
not available free in a clean daily feed. As a proxy we use **dollar trading volume** (price × shares traded) against
its 50-day average. Heavy volume with rising prices suggests buying; heavy volume with falling prices suggests selling.
It is a proxy: volume also includes market makers and arbitrage, not only investor flows.
""")


# ===========================================================================
# INDICATOR LAB
# ===========================================================================
@st.cache_data(show_spinner="Scoring indicators…")
def _lab_score(custom_json: str, use_first: bool, demo: bool, refresh_token: int):
    import json as _json
    reg = I.registry_with_custom(_json.loads(custom_json))
    lat, fst, fails, src = get_ind_raw(demo, refresh_token, use_first, custom_json)
    sc, _ = LAB.scorecard(reg, lat, fst, use_first)
    return sc, fails, src, bool(fst)


@st.cache_data(show_spinner="Downloading long monthly asset history…")
def _lab_assets(demo: bool, px: pd.DataFrame):
    return LAB.load_assets(demo, px)


SCORE_FMT = {"Nowcast corr": "{:+.2f}", "Corr at best lead": "{:+.2f}", "Sign hit %": "{:.0f}", "Turns caught %": "{:.0f}",
             "Avg timing (m)": "{:+.1f}", "False flips / yr": "{:.2f}", "First-vs-latest corr": "{:.2f}",
             "Score": "{:.0f}", "Since": "{:.0f}", "Best lead (m)": "{:.0f}", "Lag (m)": "{:.0f}"}


def page_lab():
    import json as _json
    st.markdown("Test which indicators belong in the **growth** and **inflation** composites behind the Macro backdrop. "
                "Every test is in **real time**: an indicator only counts from the month it was actually published. "
                "It is scored against what really happened, measured afterwards from fully revised data "
                f"(growth: {I.TRUTH['growth'][0]}; inflation: {I.TRUTH['inflation'][0]}).")
    if "lab_custom" not in st.session_state:
        st.session_state.lab_custom = (LAB_CFG or {}).get("custom", {})
    has_key = __import__("fred").api_key() is not None
    c1, c2 = st.columns([1.3, 1])
    use_first = c1.toggle("Use first-release data (ALFRED) for revised series", value=has_key and not demo,
                          disabled=not has_key or demo, key="lab_first",
                          help="Payrolls, IP, retail sales, income and PPI get revised a lot. With this on they are "
                               "scored on the numbers as first published. Needs the FRED API key.")
    with c2.expander("Add your own FRED series"):
        with st.form("lab_add", clear_on_submit=True):
            fid = st.text_input("FRED series ID (e.g. UMCSENT)")
            lab_ = st.text_input("Label")
            ax = st.radio("Axis", ["growth", "inflation"], horizontal=True)
            how = st.selectbox("Type of series", ["accel", "diffusion", "accel_rate", "claims"],
                               format_func=lambda x: {"accel": "Level (index, count, $) → growth acceleration",
                                                      "diffusion": "Survey diffusion index",
                                                      "accel_rate": "Already a rate (%, yield)",
                                                      "claims": "Claims-like (rising = weaker)"}[x])
            lag_ = st.number_input("Publication lag (months)", 0, 3, 1)
            if st.form_submit_button("Add") and fid.strip():
                key = "custom_" + fid.strip().upper()
                st.session_state.lab_custom[key] = {"label": lab_.strip() or fid.strip().upper(), "axis": ax,
                                                    "id": fid.strip().upper(), "transform": how, "lag": int(lag_)}
        if st.session_state.lab_custom:
            st.caption("Added: " + ", ".join(v["label"] for v in st.session_state.lab_custom.values()))
            if st.button("Remove all added series", key="lab_clear"):
                st.session_state.lab_custom = {}
    custom_json = _json.dumps(st.session_state.lab_custom, sort_keys=True)
    reg = I.registry_with_custom(st.session_state.lab_custom)
    sc, fails, src, got_first = _lab_score(custom_json, use_first, demo, st.session_state.refresh_token)
    lat, fst, _, _ = get_ind_raw(demo, st.session_state.refresh_token, use_first, custom_json)
    st.caption(f"Data: {src}." + (" Revised series scored on first-release values." if got_first and use_first else
                                  " Scored on today's revised data (no first-release values)." if not demo else ""))
    if fails:
        st.warning("Not found on FRED (shown as unavailable): " + ", ".join(sorted(fails)) +
                   ". A combined indicator still works with the components that did load.")

    # ---------------- 1. scorecard ----------------
    st.markdown("### 1 · Scorecard: each indicator on its own")
    cols = ["Indicator", "Kind", "Lag (m)", "Since", "Score", "Nowcast corr", "Best lead (m)", "Corr at best lead",
            "Sign hit %", "Turns caught %", "Avg timing (m)", "False flips / yr", "First-vs-latest corr", "Status"]
    for ax in ("growth", "inflation"):
        t = sc[sc["Axis"] == ax].copy()
        t = t[[c for c in cols if c in t]].sort_values("Score", ascending=False, na_position="last") if "Score" in t else t
        st.markdown(f"**{ax.capitalize()}**")
        st.dataframe(t.set_index("Indicator").style.format({k: v for k, v in SCORE_FMT.items() if k in t}, na_rep="–")
                     .map(lambda v: "font-weight:700" if isinstance(v, float) and v >= 70 else "", subset=["Score"] if "Score" in t else []),
                     width="stretch")
    st.caption("Score = average rank (0–100, within each axis) of nowcast corr, corr at best lead, sign hit rate, "
               "turns caught and (inverted) false flips. A ranking aid, not a verdict: read the columns.")

    # ---------------- 2. choose sets ----------------
    st.markdown("### 2 · Build a composite and compare")
    ok = sc[sc["Status"] == "ok"]
    cur = LAB_CFG if LAB_CFG else I.DEFAULT_SET
    start = LAB_CFG if LAB_CFG else {ax: [k for k, v in reg.items() if v[1] == ax and v[7]] for ax in ("growth", "inflation")}
    sel = {}
    a, b = st.columns(2)
    for col, ax in [(a, "growth"), (b, "inflation")]:
        opts = list(ok[ok["Axis"] == ax].index)
        default = [k for k in start.get(ax, []) if k in opts] or [k for k in opts if reg[k][7]]
        sel[ax] = col.multiselect(f"{ax.capitalize()} composite", opts, default=default,
                                  format_func=lambda k: reg[k][0], key=f"lab_sel_{ax}")
    if not sel["growth"] or not sel["inflation"]:
        st.info("Pick at least one indicator for each axis.")
        return
    cur_reg = I.registry_with_custom((LAB_CFG or {}).get("custom"))
    rows = []
    for ax in ("growth", "inflation"):
        for lab_name, keys, rg in [("Current (Macro backdrop)", [k for k in cur.get(ax, []) if k in cur_reg], cur_reg),
                                   ("Your selection", sel[ax], reg)]:
            stt = LAB.composite_stats(keys, ax, rg, lat, fst, use_first)
            if stt:
                rows.append({"Axis": ax, "Composite": lab_name, "Indicators": len(keys), "Corr with truth": stt["corr"],
                             "Sign hit %": stt["hit"], "Turns caught %": stt["tp_hit"], "Avg timing (m)": stt["tp_lead"],
                             "False flips / yr": stt["false"]})
    cmp_ = pd.DataFrame(rows).set_index(["Axis", "Composite"])
    st.dataframe(cmp_.style.format({"Corr with truth": "{:+.2f}", "Sign hit %": "{:.1f}", "Turns caught %": "{:.0f}",
                                    "Avg timing (m)": "{:+.1f}", "False flips / yr": "{:.2f}"}, na_rep="–"), width="stretch")
    yrs = st.radio("Chart window", [10, 20, 30], index=1, horizontal=True, format_func=lambda x: f"{x}y", key="lab_y")
    l, r = st.columns(2)
    for col, ax in [(l, "growth"), (r, "inflation")]:
        tr = I.truth(ax, lat)
        f = go.Figure()
        cs = LAB.composite_stats(sel[ax], ax, reg, lat, fst, use_first)["series"]
        cutoff = cs.index[-1] - pd.DateOffset(years=yrs)
        cs = cs[cs.index >= cutoff]
        f.add_bar(x=cs.index, y=cs, name="your composite (σ, real time)",
                  marker_color=[UP if v >= 0 else DOWN for v in cs], opacity=0.55)
        if tr is not None:
            t_ = tr[tr.index >= cutoff]
            f.add_scatter(x=t_.index, y=t_ / t_.std(), name="truth (revised, scaled)", line=dict(color="#0b0b0b", width=1.6))
        f.add_hline(y=0, line_color=MUTED)
        col.plotly_chart(base_layout(f, 300, f"{ax.capitalize()}: composite vs what actually happened"), width="stretch")

    # ---------------- 3. what each indicator adds ----------------
    st.markdown("### 3 · What each indicator adds")
    l, r = st.columns(2)
    for col, ax in [(l, "growth"), (r, "inflation")]:
        loo = LAB.leave_one_out(sel[ax], ax, reg, lat, fst, use_first)
        col.markdown(f"**{ax.capitalize()}: leave-one-out**")
        if loo.empty:
            col.caption("Needs at least two indicators.")
        else:
            loo = loo.drop(columns=["Hit % without it"]).rename(columns={"Change in hit % if dropped": "Δ hit % if dropped",
                                                                          "Change in corr if dropped": "Δ corr if dropped"})
            col.dataframe(loo.set_index("Indicator").style.format({"Δ hit % if dropped": "{:+.1f}",
                                                                   "Δ corr if dropped": "{:+.3f}"})
                          .map(lambda v: f"color:{UP if 'KEEP' in str(v) else DOWN if 'DROP' in str(v) else NEUTRAL};"
                                         "font-weight:600", subset=["Verdict"]), width="stretch")
    red = LAB.redundancy(sel["growth"] + sel["inflation"], reg, lat, fst, use_first)
    if len(red) > 1:
        hm = go.Figure(go.Heatmap(z=red.values, x=red.columns, y=red.index, zmin=-1, zmax=1, zmid=0,
                                  colorscale=[[0, DOWN], [0.5, "#f0efec"], [1, UP]], text=np.round(red.values, 2),
                                  texttemplate="%{text:.2f}", colorbar=dict(thickness=10)))
        hm.update_yaxes(autorange="reversed")
        hm.update_xaxes(tickangle=-35)
        st.plotly_chart(base_layout(hm, 120 + 34 * len(red), "Overlap between selected indicators (correlation of signals)",
                                    False), width="stretch")
        st.caption("Pairs above ~0.7 largely tell the same story: keeping both double-weights that story. "
               "Leave-one-out: Δ = composite with the indicator removed minus composite with it; negative = it was helping.")

    # ---------------- 4. weighting and robustness ----------------
    st.markdown("### 4 · Weighting and robustness")
    l, r = st.columns(2)
    for col, ax in [(l, "growth"), (r, "inflation")]:
        col.markdown(f"**{ax.capitalize()}**")
        wf = LAB.walk_forward(sel[ax], ax, reg, lat, fst, use_first)
        if not wf.empty:
            col.dataframe(wf.style.format({"Corr with truth": "{:+.3f}", "Sign hit %": "{:.1f}"}), width="stretch")
        rb = LAB.robustness(sel[ax], ax, reg, lat, fst, use_first)
        if not rb.empty:
            col.dataframe(rb.style.format({"Corr": "{:+.2f}", "Sign hit %": "{:.1f}"}), width="stretch")
    st.caption("Walk-forward weights are re-estimated each January using only data that was available then "
               "(and truth that was already known, about a year back). Only switch away from equal weights if the "
               "gain is clear and stable across periods.")

    # ---------------- 5. asset relevance ----------------
    st.markdown("### 5 · Do the quadrants separate asset returns?")
    assets = _lab_assets(demo, px)
    h = st.radio("Forward horizon", [1, 3, 6], index=1, horizontal=True, format_func=lambda x: f"{x} months", key="lab_h")
    qa = I.quadrant_from_selection({**sel, "custom": st.session_state.lab_custom, "first_release": use_first}, lat, fst)
    qb = quad   # whatever the Macro backdrop currently shows (default model or an applied selection)
    l, r = st.columns(2)
    for col, lab_name, qq in [(l, "Your selection", qa), (r, "Current (Macro backdrop)", qb)]:
        tab, tests = LAB.asset_relevance(qq, assets, h)
        col.markdown(f"**{lab_name}**: average forward {h}-month return by quadrant (bp for the 10Y)")
        if tab.empty:
            col.caption("Not enough overlapping history.")
            continue
        col.dataframe(tab.style.format("{:+.2f}").map(signed_color), width="stretch")
        col.dataframe(tests[["Months", "Best quadrant", "Worst quadrant", "Spread best-worst", "p-value (circular shift)"]]
                      .style.format({"Spread best-worst": "{:.2f}", "p-value (circular shift)": "{:.2f}"}), width="stretch")
    st.caption("p-value: how often randomly shifted quadrant labels separate returns at least as much. Below ~0.10 "
               "suggests the quadrants carry real information for that asset; with a few hundred overlapping months "
               "and a handful of cycles, treat everything above that as noise. Use this as a check, never to pick "
               "indicators (that would fit the noise).")

    # ---------------- 6. apply ----------------
    st.markdown("### 6 · Use this selection in the Macro backdrop")
    a, b, _ = st.columns([1, 1, 2])
    if a.button("Apply to Macro backdrop", type="primary", key="lab_apply"):
        I.save_config(sel["growth"], sel["inflation"],
                      {k: v for k, v in st.session_state.lab_custom.items() if k in sel["growth"] + sel["inflation"]},
                      use_first)
        st.success("Saved. The Macro backdrop now uses this selection.")
        st.rerun()
    if b.button("Reset to default inputs", key="lab_reset"):
        I.reset_config()
        st.rerun()
    st.caption("Current Macro backdrop inputs: " + (
        f"your selection saved {LAB_CFG.get('saved')} ({len(LAB_CFG['growth'])} growth, {len(LAB_CFG['inflation'])} inflation)."
        if LAB_CFG else "default (payrolls, industrial production, real retail sales, real income; core CPI)."))
    explain("""
**Why real time.** An indicator that is only published six weeks later, and then revised, looks great in hindsight
and is much less useful on the day. So each indicator is lagged by its publication delay, and (with the API key)
revised series are scored on their first-published numbers.

**The truth.** What actually happened, measured with hindsight: the 6-month annualised pace minus the 12-month pace of
the Philadelphia Fed coincident index (growth) and of core PCE prices (inflation), both from today's revised data.

**Columns.**
* *Nowcast corr*: does the value you know at month end line up with what was really happening that month?
* *Best lead*: how many months ahead the indicator lines up best. Above 0 = it leads.
* *Sign hit %*: how often it gets the direction (accelerating vs decelerating) right. 50% = a coin flip.
* *Turns caught / timing*: of the truth's direction changes, how many the indicator caught within 6 months before to
  3 months after, and how early on average (positive = early).
* *False flips / yr*: direction changes with no real turn nearby. Noise.
* *First-vs-latest corr*: how similar the first-published and fully revised signals are (1 = no revision problem).

**Avoiding overfitting.** With ~30 years of monthly data and only a handful of cycles, it is easy to find a combination
that looked perfect in the past. Rules of thumb: prefer 4–7 indicators per axis from *different* sources (hard data,
surveys, labour market), keep equal weights unless walk-forward weights win clearly and in every sub-period, and do
not choose indicators by their asset-return results.
""")


# ===========================================================================
# NAVIGATION
# ===========================================================================
PAGES = {
    "Summary": [("Summary", page_summary)],
    "Cross-asset": [("Macro backdrop", page_macro), ("Indicator lab", page_lab),
                    ("Market-implied macro regime", page_sbc), ("What happened next", page_next), ("Guide", page_guide)],
    "Equities": [("Today", page_eq_today),
                 ("Regimes", lambda: page_regimes(list(D.INDICES) + list(D.SECTORS) + list(D.LEADERS), px, markov, "eq_")),
                 ("What's driving it", page_drivers), ("Breadth & rotation", page_breadth),
                 ("Volatility", page_vol), ("Mega-cap leaders", page_leaders)],
    "Crypto": [("Today", page_cr_today), ("Levels & drawdown", page_cr_levels), ("Macro links", page_cr_macro),
               ("Alts & ETFs", page_cr_alts), ("Regimes", lambda: page_regimes(CR_LIST, px, mk_cr, "cr_"))],
    "Fixed income": [("Today", page_fi_today), ("Curve", page_curve), ("What's moving the 10Y", page_10y),
                     ("Fed pricing", page_fed), ("Credit", page_credit), ("Rate vol", page_ratevol),
                     ("Regimes", lambda: page_regimes(["UST 10Y"] + [n for g in FI_NAMES.values() for n in g if n != "UST 10Y"] + FI_PX,
                                                      fi.join(px[[c for c in FI_PX if c in px]], how="outer"),
                                                      mk_fi, "fi_"))],
}
if len(PAGES[section]) == 1:
    PAGES[section][0][1]()
else:
    _tabs = st.tabs([name for name, _ in PAGES[section]])
    for _tab, (_name, _fn) in zip(_tabs, PAGES[section]):
        with _tab:
            _fn()
