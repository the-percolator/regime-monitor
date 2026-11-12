"""
Small FRED client (Federal Reserve Bank of St. Louis data).

* With an API key  -> official API (reliable). The key is read from the environment variable FRED_API_KEY,
  or from a file fred_api_key.txt next to this script.
* Without a key    -> FRED's public CSV download (works, but often slow or timing out).

Free key: https://fredaccount.stlouisfed.org/apikeys
"""
from __future__ import annotations

import io
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
API_URL = "https://api.stlouisfed.org/fred/series/observations?"
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?"
TIMEOUT = 30
ATTEMPTS = 2


def api_key() -> str | None:
    k = os.environ.get("FRED_API_KEY", "").strip()
    if k:
        return k
    f = HERE / "fred_api_key.txt"
    if f.exists():
        k = f.read_text().strip()
        return k or None
    return None


def mode() -> str:
    return "FRED API (key found)" if api_key() else "FRED public CSV (no API key)"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (equity-dashboard)"})
    last = None
    for i in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except Exception as exc:  # timeout, HTTP error, DNS ...
            last = exc
            time.sleep(1.5 * (i + 1))
    raise last


def parse_api_json(raw: bytes, series_id: str) -> pd.Series:
    obs = json.loads(raw)["observations"]
    s = pd.Series({pd.Timestamp(o["date"]): o["value"] for o in obs}, name=series_id)
    return pd.to_numeric(s, errors="coerce").dropna()


def parse_csv(text: str) -> pd.Series:
    """FRED CSV: first column date ('DATE' or 'observation_date'), second the values ('.' = missing)."""
    df = pd.read_csv(io.StringIO(text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    s.name = df.columns[1]
    return s.dropna()


def series(series_id: str, start: str = "1990-01-01", first_release: bool = False) -> pd.Series:
    """first_release=True: each observation's value as FIRST published (ALFRED, API key required)."""
    key = api_key()
    if first_release and not key:
        raise RuntimeError("first-release (ALFRED) data needs a FRED API key")
    if key:
        params = {"series_id": series_id, "api_key": key, "file_type": "json", "observation_start": start}
        if first_release:
            params.update({"output_type": 4, "realtime_start": "1776-07-04", "realtime_end": "9999-12-31"})
        q = urllib.parse.urlencode(params)
        s = parse_api_json(_get(API_URL + q), series_id)
    else:
        q = urllib.parse.urlencode({"id": series_id, "cosd": start})
        s = parse_csv(_get(CSV_URL + q).decode("utf-8"))
    s.name = series_id
    return s


def many(ids: list[str], start: str = "1990-01-01") -> pd.DataFrame:
    """Download several series in parallel. Raises if any one fails."""
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = dict(zip(ids, ex.map(lambda i: series(i, start), ids)))
    return pd.concat(res, axis=1)
