#!/usr/bin/env python3
"""
Euro area vs US rates / inflation dashboard (ECB Data Portal + FRED, no API keys needed).

RUN:   pip install pandas matplotlib requests
       python macro_dashboard.py            # real data (needs internet)
       python macro_dashboard.py --demo     # fake data, only to test the code runs

OUTPUT: macro_dashboard.png (4 panels), macro_data.csv, and a printed fact sheet.
The script gives you FACTS. The one-page VIEW (what it implies for rates) must be yours.

SERIES USED
  ECB Data Portal (data-api.ecb.europa.eu):
    YC  B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y / SR_10Y   euro area AAA govt spot yield curve (2y, 10y), daily
    FM  B.U2.EUR.4F.KR.DFR.LEV                      ECB deposit facility rate, daily
    HICP M.U2.N.000000.4D0.ANR                      euro area HICP headline inflation, % y/y, monthly
    HICP M.U2.N.XEF000.4D0.ANR                      euro area HICP excl. energy, food, alcohol & tobacco, % y/y
  FRED (fred.stlouisfed.org, public CSV endpoint):
    DGS2, DGS10      US Treasury constant maturity 2y, 10y (daily)
    DFEDTARU         Fed funds target range upper bound (daily)
    CPIAUCSL         US CPI all items (monthly index; y/y computed here)
    CPILFESL         US core CPI (monthly index; y/y computed here)
If an ECB series code is rejected (they occasionally change), the script says which one and carries on.
"""
import argparse, io, sys
import numpy as np
import pandas as pd

START = "2015-01-01"
ECB_BASE = "https://data-api.ecb.europa.eu/service/data"
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

ECB_SERIES = {
    "EA_2Y":     "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y",
    "EA_10Y":    "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
    "ECB_DFR":   "FM/B.U2.EUR.4F.KR.DFR.LEV",
    # ECB discontinued the older ICP dataset after December 2025.
    "EA_HICP":   "HICP/M.U2.N.000000.4D0.ANR",
    "EA_CORE":   "HICP/M.U2.N.XEF000.4D0.ANR",
}
FRED_SERIES = {"US_2Y": "DGS2", "US_10Y": "DGS10", "FED_UB": "DFEDTARU",
               "US_CPI_IDX": "CPIAUCSL", "US_CORE_IDX": "CPILFESL"}

def get_ecb(key):
    import requests
    url = f"{ECB_BASE}/{key}"
    r = requests.get(url, params={"format": "csvdata", "startPeriod": START}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.Series(df["OBS_VALUE"].values, index=pd.to_datetime(df["TIME_PERIOD"]), dtype=float)
    return s.sort_index()

def get_fred(code):
    import requests
    r = requests.get(FRED_BASE, params={"id": code, "cosd": START}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values, index=pd.to_datetime(df["date"]))
    return s.dropna().sort_index()

def load_real():
    out, failed = {}, []
    for name, key in ECB_SERIES.items():
        try: out[name] = get_ecb(key)
        except Exception as e: failed.append((name, f"ECB {key}: {e}"))
    for name, code in FRED_SERIES.items():
        try: out[name] = get_fred(code)
        except Exception as e: failed.append((name, f"FRED {code}: {e}"))
    for n, msg in failed: print(f"  [warn] could not load {n}: {msg}", file=sys.stderr)
    return out

def load_demo():
    rng = np.random.default_rng(1)
    d = pd.bdate_range(START, "2026-11-06"); m = pd.date_range(START, "2026-10-31", freq="ME")
    def walk(n, start, vol, lo=-0.7): return np.maximum(start + np.cumsum(rng.normal(0, vol, n)), lo)
    return {
        "EA_2Y": pd.Series(walk(len(d), 0.0, 0.03), d), "EA_10Y": pd.Series(walk(len(d), 0.4, 0.03), d),
        "ECB_DFR": pd.Series(np.round(np.clip(walk(len(d), -0.5, 0.01), -0.5, 4) * 4) / 4, d),
        "EA_HICP": pd.Series(np.clip(walk(len(m), 1.0, 0.25), -0.5, 10), m), "EA_CORE": pd.Series(np.clip(walk(len(m), 1.0, 0.2), 0, 8), m),
        "US_2Y": pd.Series(walk(len(d), 1.5, 0.04), d), "US_10Y": pd.Series(walk(len(d), 2.0, 0.04), d),
        "FED_UB": pd.Series(np.round(np.clip(walk(len(d), 0.25, 0.02), 0, 6) * 4) / 4, d),
        "US_CPI_IDX": pd.Series(250 * np.exp(np.cumsum(rng.normal(0.002, 0.002, len(m)))), m),
        "US_CORE_IDX": pd.Series(250 * np.exp(np.cumsum(rng.normal(0.002, 0.001, len(m)))), m),
    }

def last(s):
    s = s.dropna(); return s.index[-1], s.iloc[-1]

def at_or_before(s, date):
    s = s.dropna(); s = s.loc[:date]; return s.iloc[-1] if len(s) else np.nan

def build(raw):
    df = {}
    for k, s in raw.items(): df[k] = s
    daily = pd.concat({k: v for k, v in df.items() if k in ("EA_2Y", "EA_10Y", "ECB_DFR", "US_2Y", "US_10Y", "FED_UB")}, axis=1, sort=True).ffill().dropna(how="all")
    monthly = {}
    if "EA_HICP" in df: monthly["EA_HICP"] = df["EA_HICP"].resample("ME").last()
    if "EA_CORE" in df: monthly["EA_CORE"] = df["EA_CORE"].resample("ME").last()
    if "US_CPI_IDX" in df: monthly["US_CPI"] = (df["US_CPI_IDX"].resample("ME").last().pct_change(12) * 100)
    if "US_CORE_IDX" in df: monthly["US_CORE"] = (df["US_CORE_IDX"].resample("ME").last().pct_change(12) * 100)
    mdf = pd.DataFrame(monthly)
    if {"EA_10Y", "EA_2Y"} <= set(daily.columns): daily["EA_2s10s"] = daily["EA_10Y"] - daily["EA_2Y"]
    if {"US_10Y", "US_2Y"} <= set(daily.columns): daily["US_2s10s"] = daily["US_10Y"] - daily["US_2Y"]
    if {"EA_10Y", "US_10Y"} <= set(daily.columns): daily["US_minus_EA_10Y"] = daily["US_10Y"] - daily["EA_10Y"]
    return daily, mdf

def fact_sheet(daily, mdf):
    print("\n" + "=" * 70 + "\nFACT SHEET (copy these into your one-page view; the interpretation is yours)\n" + "=" * 70)
    def line(label, s, unit="%", yr_ago=True):
        s = s.dropna()
        if s.empty: print(f"  {label}: n/a"); return
        d, v = s.index[-1], s.iloc[-1]
        extra = ""
        if yr_ago:
            v0 = at_or_before(s, d - pd.DateOffset(years=1))
            if pd.notna(v0): extra = f"  (1y ago {v0:.2f}, change {v - v0:+.2f})"
        print(f"  {label:<34}{v:6.2f}{unit}  as of {d.date()}{extra}")
    for lab, col in [("ECB deposit rate", "ECB_DFR"), ("Euro area 2y yield", "EA_2Y"), ("Euro area 10y yield", "EA_10Y"),
                     ("Euro area 2s10s slope", "EA_2s10s"), ("Fed funds upper bound", "FED_UB"),
                     ("US 2y yield", "US_2Y"), ("US 10y yield", "US_10Y"), ("US 2s10s slope", "US_2s10s"),
                     ("US 10y minus euro area 10y", "US_minus_EA_10Y")]:
        if col in daily: line(lab, daily[col])
    for lab, col in [("Euro area HICP y/y", "EA_HICP"), ("Euro area HICP ex energy/food y/y", "EA_CORE"),
                     ("US CPI y/y", "US_CPI"), ("US core CPI y/y", "US_CORE")]:
        if col in mdf: line(lab, mdf[col])
    latest_daily = daily.index.max() if not daily.empty else pd.Timestamp.min
    for col in ("EA_HICP", "EA_CORE", "US_CPI", "US_CORE"):
        if col in mdf and not mdf[col].dropna().empty:
            latest = mdf[col].dropna().index.max()
            if latest < latest_daily - pd.Timedelta(days=45):
                print(f"  [warn] {col} is stale versus daily data: latest observation {latest.date()} (daily data through {latest_daily.date()})", file=sys.stderr)
    if "ECB_DFR" in daily and "EA_HICP" in mdf:
        rr = daily["ECB_DFR"].dropna().iloc[-1] - mdf["EA_HICP"].dropna().iloc[-1]
        print(f"  {'Euro area real policy rate (DFR - HICP)':<34}{rr:6.2f}%  (negative = policy still loose vs inflation)")
    if "EA_2Y" in daily and "ECB_DFR" in daily:
        gap = daily["EA_2Y"].dropna().iloc[-1] - daily["ECB_DFR"].dropna().iloc[-1]
        print(f"  {'2y yield minus deposit rate':<34}{gap:6.2f}%  (positive = market pricing hikes; negative = cuts)")
    inv = (daily.get("EA_2s10s", pd.Series(dtype=float)).dropna() < 0).mean() * 100
    if inv == inv: print(f"  Share of days since {START[:4]} with an inverted euro area 2s10s: {inv:.0f}%")
    print("\nQuestions your view must answer: (1) Is the curve pricing hikes or cuts, and is inflation justifying that?"
          "\n(2) Core vs headline: is the pressure broad? (3) Euro vs US: why the 10y gap, and what would close it?"
          "\n(4) What single data release would change your mind?")

def plot(daily, mdf, path="macro_dashboard.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    a = ax[0, 0]
    for c, lab in [("ECB_DFR", "ECB deposit rate"), ("EA_2Y", "EA 2y"), ("EA_10Y", "EA 10y")]:
        if c in daily: a.plot(daily[c], label=lab)
    a.set_title("Euro area: policy rate and yields (%)"); a.legend(); a.grid(alpha=.3)
    a = ax[0, 1]
    for c, lab in [("EA_2s10s", "EA 2s10s"), ("US_2s10s", "US 2s10s")]:
        if c in daily: a.plot(daily[c], label=lab)
    a.axhline(0, color="k", lw=.7); a.set_title("Curve slope, 10y minus 2y (pp)"); a.legend(); a.grid(alpha=.3)
    a = ax[1, 0]
    for c, lab in [("EA_HICP", "EA HICP"), ("EA_CORE", "EA HICP ex energy/food"), ("US_CPI", "US CPI"), ("US_CORE", "US core CPI")]:
        if c in mdf: a.plot(mdf[c], label=lab)
    a.axhline(2, color="grey", ls="--", lw=.8); a.set_title("Inflation, % y/y (dashed = 2%)"); a.legend(); a.grid(alpha=.3)
    a = ax[1, 1]
    if "ECB_DFR" in daily and "EA_HICP" in mdf:
        dm = daily["ECB_DFR"].resample("ME").last(); rr = (dm - mdf["EA_HICP"]).dropna()
        a.plot(rr, label="Euro area real policy rate (DFR - HICP)")
    if "FED_UB" in daily and "US_CPI" in mdf:
        dm = daily["FED_UB"].resample("ME").last(); rr = (dm - mdf["US_CPI"]).dropna()
        a.plot(rr, label="US real policy rate (Fed upper - CPI)")
    a.axhline(0, color="k", lw=.7); a.set_title("Real policy rates (pp)"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); print(f"Saved {path}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--demo", action="store_true"); a = ap.parse_args()
    raw = load_demo() if a.demo else load_real()
    if not raw: sys.exit("No series loaded. Check your internet connection.")
    daily, mdf = build(raw)
    if a.demo: print("DEMO DATA: numbers below are random and meaningless.")
    fact_sheet(daily, mdf)
    daily.join(mdf, how="outer").to_csv("macro_data.csv"); print("\nSaved macro_data.csv")
    plot(daily, mdf)

if __name__ == "__main__":
    main()
