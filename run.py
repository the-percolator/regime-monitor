"""
One-command launcher for the Equity Regime Monitor.

    python run.py           # live data from Yahoo Finance
    python run.py --demo    # synthetic offline data
    options: --port 8502, --no-browser

On first run it installs any missing packages (streamlit, plotly, pandas, numpy, yfinance),
then opens the dashboard in your browser at http://localhost:8501. Stop it with Ctrl+C.
"""
import importlib.util
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEEDED = {"streamlit": "streamlit>=1.50", "plotly": "plotly>=5.20", "pandas": "pandas>=2.0",
          "numpy": "numpy>=1.26", "yfinance": "yfinance>=0.2.50"}


def ensure_packages():
    missing = [spec for mod, spec in NEEDED.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        import streamlit
        major, minor = (int(x) for x in streamlit.__version__.split(".")[:2])
        if (major, minor) < (1, 50):
            missing.append(NEEDED["streamlit"])
    if missing:
        print(f"Installing: {', '.join(missing)} …")
        cmd = [sys.executable, "-m", "pip", "install", "--quiet", *missing]
        if subprocess.call(cmd) != 0:
            # some system Pythons (Homebrew, Debian) need this flag; a virtualenv is the clean alternative
            subprocess.check_call(cmd + ["--break-system-packages"])


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or newer is required.")
    ensure_packages()
    extra = ["--", "--demo"] if "--demo" in sys.argv else []
    port = 8501
    for i, a in enumerate(sys.argv):
        if a == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    # headless=true skips Streamlit's first-run e-mail prompt; we open the browser ourselves
    cmd = [sys.executable, "-m", "streamlit", "run", str(HERE / "app.py"), "--server.port", str(port),
           "--server.headless", "true", "--browser.gatherUsageStats", "false", *extra]
    url = f"http://localhost:{port}"
    print(f"\nEquity Regime Monitor starting at {url}  (Ctrl+C to stop)\n")
    if "--no-browser" not in sys.argv:
        threading.Timer(3.0, lambda: webbrowser.open(url)).start()
    try:
        subprocess.call(cmd, cwd=HERE)
    except KeyboardInterrupt:
        pass
