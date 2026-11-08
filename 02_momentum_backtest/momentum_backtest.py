#!/usr/bin/env python3
"""
Momentum backtest on liquid ETFs, with transaction costs and honest out-of-sample testing.

RUN:   pip install yfinance pandas numpy matplotlib
       python momentum_backtest.py                 # real data (needs internet)
       python momentum_backtest.py --synthetic     # fake data, only to test the code runs

WHAT IT DOES (read this, you must be able to explain every line in an interview)
1. Universe: liquid ETFs across equities, bonds, gold, commodities, real estate.
2. Signal: every month-end, rank assets by total return over the past L months, skipping the most
   recent month (the classic "12-1" momentum; skipping avoids short-term reversal).
3. Portfolio: hold the top K assets equally weighted for the next month. Absolute-momentum filter:
   an asset is only held if its own momentum is > 0, otherwise that slot goes to cash (SHY).
4. Costs: COST_BPS per unit of one-way turnover, charged at each rebalance.
5. No look-ahead: the signal at month-end t is used for returns in month t+1.
6. Out-of-sample: (a) a fixed train/test split; (b) a walk-forward where each year the parameters are
   re-chosen using ONLY past data, then applied to the next year.
7. Benchmarks: SPY buy-and-hold, and equal-weight buy-and-hold of the universe.
8. Reports: CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, turnover, and a bootstrap
   confidence interval for the Sharpe difference vs the benchmark.
"""
import argparse, itertools, sys
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ CONFIG (every number is a choice you must defend)
UNIVERSE = ["SPY", "EFA", "EEM", "IWM", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC", "VNQ"]
CASH = "SHY"                 # short-term Treasuries as the cash proxy
BENCHMARK = "SPY"
START = "2007-06-01"         # HYG/DBC start 2007; earlier start would shrink the universe
COST_BPS = 10                # one-way cost per unit of turnover (spread + commission + slippage)
LOOKBACKS = [3, 6, 9, 12]    # months: parameter grid
TOP_KS = [2, 3, 4, 5]        # number of assets held
SKIP = 1                     # months skipped (recent-month reversal)
TRAIN_END = "2016-12-31"     # fixed split: everything up to here is in-sample
MIN_TRAIN_MONTHS = 60        # walk-forward needs at least 5 years before the first out-of-sample year
RF_ANNUAL = 0.0              # set to e.g. 0.02 if you want excess-return Sharpe; 0 keeps it simple
N_BOOT = 2000
SEED = 42

# ------------------------------------------------------------------ DATA
def load_prices(synthetic=False):
    tickers = UNIVERSE + [CASH]
    if synthetic:
        rng = np.random.default_rng(SEED)
        idx = pd.bdate_range("2007-06-01", "2025-12-31")
        n = len(tickers)
        A = rng.normal(size=(n, n)); cov = A @ A.T / n
        d = np.sqrt(np.diag(cov)); corr = cov / np.outer(d, d)
        vol = np.array([0.16, 0.17, 0.22, 0.20, 0.14, 0.07, 0.07, 0.10, 0.15, 0.17, 0.20, 0.01])
        mu = np.array([0.07, 0.05, 0.05, 0.06, 0.03, 0.025, 0.035, 0.05, 0.04, 0.01, 0.06, 0.015])
        L = np.linalg.cholesky(corr * 0.5 + np.eye(n) * 0.5)
        z = rng.normal(size=(len(idx), n)) @ L.T
        daily = mu / 252 + z * vol / np.sqrt(252)
        return pd.DataFrame(100 * np.exp(np.cumsum(daily, axis=0)), index=idx, columns=tickers)
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Install yfinance first: pip install yfinance")
    as_of = pd.Timestamp.today().normalize()
    px = yf.download(tickers, start=START, end=as_of + pd.Timedelta(days=1), auto_adjust=True, progress=False)["Close"]
    # Some data providers can return forward-dated rows in a sandbox or during
    # timezone transitions; never let those rows enter a historical backtest.
    px = px.loc[px.index <= as_of]
    px = px[tickers].dropna(how="any")
    if px.empty:
        sys.exit("No data returned. Check your internet connection / tickers.")
    return px

def monthly(px):
    m = px.resample("ME").last() if hasattr(px, "resample") else px
    return m

# ------------------------------------------------------------------ STRATEGY
def momentum_weights(prices_m, lookback, top_k, skip=SKIP):
    """Weights decided at each month-end using only data up to that month-end."""
    assets = UNIVERSE
    mom = prices_m[assets].shift(skip) / prices_m[assets].shift(lookback) - 1
    w = pd.DataFrame(0.0, index=prices_m.index, columns=assets + [CASH])
    for t in prices_m.index:
        s = mom.loc[t].dropna()
        if len(s) < top_k:
            continue
        top = s.sort_values(ascending=False).head(top_k)
        for a, val in top.items():
            if val > 0:
                w.loc[t, a] = 1.0 / top_k
            else:
                w.loc[t, CASH] += 1.0 / top_k
    return w

def run_strategy(prices_m, lookback, top_k, cost_bps=COST_BPS):
    """Return a DataFrame with net monthly returns and turnover. Signal at t earns returns in t+1."""
    rets = prices_m.pct_change()
    w = momentum_weights(prices_m, lookback, top_k)
    w_held = w.shift(1)                       # NO LOOK-AHEAD: trade on the weights decided last month-end
    gross = (w_held * rets[w.columns]).sum(axis=1, min_count=1)
    # The signal/portfolio change occurs at month-end t and earns returns in t+1.
    # Align the cost with the first return period affected by that trade. The old
    # code charged the previous trade one month late and omitted the initial trade.
    turnover = w.diff().abs().sum(axis=1).shift(1)
    cost = turnover * cost_bps / 1e4
    net = gross - cost
    out = pd.DataFrame({"net": net, "gross": gross, "turnover": turnover})
    first_valid = w_held.dropna().index[0] if not w_held.dropna().empty else None
    return out.loc[first_valid:] if first_valid is not None else out.dropna()

# ------------------------------------------------------------------ METRICS
def metrics(r, rf=RF_ANNUAL):
    r = r.dropna()
    if len(r) < 12:
        return dict(CAGR=np.nan, Vol=np.nan, Sharpe=np.nan, Sortino=np.nan, MaxDD=np.nan, Calmar=np.nan, Months=len(r))
    yrs = len(r) / 12
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(12)
    ex = r - rf / 12
    sharpe = ex.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan
    down = ex[ex < 0].std() * np.sqrt(12)
    sortino = ex.mean() * 12 / down if down and down > 0 else np.nan
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    return dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, Sortino=sortino, MaxDD=maxdd, Calmar=calmar, Months=len(r))

def bootstrap_sharpe_diff(a, b, n=N_BOOT, block=6, seed=SEED):
    """Block-bootstrap the Sharpe difference (strategy minus benchmark). Returns (point, 5th pct, 95th pct)."""
    df = pd.concat([a, b], axis=1).dropna()
    x = df.values; T = len(x)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = []
        while len(idx) < T:
            s = rng.integers(0, T)
            idx.extend(range(s, min(s + block, T)))
        samp = x[idx[:T]]
        s1 = samp[:, 0].mean() / samp[:, 0].std() * np.sqrt(12)
        s2 = samp[:, 1].mean() / samp[:, 1].std() * np.sqrt(12)
        diffs.append(s1 - s2)
    pt = (x[:, 0].mean() / x[:, 0].std() - x[:, 1].mean() / x[:, 1].std()) * np.sqrt(12)
    return pt, np.percentile(diffs, 5), np.percentile(diffs, 95)

# ------------------------------------------------------------------ VALIDATION
def grid(prices_m, end=None, start=None):
    """Sharpe of every parameter pair on a date window."""
    res = {}
    for L, K in itertools.product(LOOKBACKS, TOP_KS):
        r = run_strategy(prices_m, L, K)["net"]
        if start: r = r.loc[start:]
        if end: r = r.loc[:end]
        res[(L, K)] = metrics(r)["Sharpe"]
    return res

def walk_forward(prices_m):
    """Each calendar year: pick the (lookback, top_k) with best Sharpe over ALL PRIOR data, apply to that year."""
    all_runs = {(L, K): run_strategy(prices_m, L, K) for L, K in itertools.product(LOOKBACKS, TOP_KS)}
    first = prices_m.index[0] + pd.DateOffset(months=MIN_TRAIN_MONTHS)
    years = sorted(set(prices_m.loc[first:].index.year))
    pieces, choices = [], []
    for y in years:
        cutoff = pd.Timestamp(f"{y}-01-01")
        best, best_s = None, -np.inf
        for key, df in all_runs.items():
            s = metrics(df["net"].loc[:cutoff - pd.Timedelta(days=1)])["Sharpe"]
            if pd.notna(s) and s > best_s:
                best, best_s = key, s
        if best is None:
            continue
        seg = all_runs[best].loc[str(y)]
        pieces.append(seg); choices.append((y, best, best_s))
    return pd.concat(pieces), choices

# ------------------------------------------------------------------ MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="use fake data to test that the code runs")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    px = load_prices(args.synthetic)
    pm = px.resample("ME").last()
    bench = pm[BENCHMARK].pct_change()
    ew = pm[UNIVERSE].pct_change().mean(axis=1)     # monthly-rebalanced equal weight (simple, mildly flattering)
    n_trials = len(LOOKBACKS) * len(TOP_KS)

    print("=" * 78)
    print(f"Data: {pm.index[0].date()} to {pm.index[-1].date()} (raw prices through {px.index[-1].date()})   |   {'SYNTHETIC - results meaningless' if args.synthetic else 'real data'}")
    print(f"Universe: {', '.join(UNIVERSE)}   Cash: {CASH}   Cost: {COST_BPS} bps one-way")
    print(f"Parameter combinations tried (this is your 'number of trials'): {n_trials}")
    print("=" * 78)

    # 1) In-sample grid and fixed split
    g_in = grid(pm, end=TRAIN_END)
    best = max(g_in, key=lambda k: (g_in[k] if pd.notna(g_in[k]) else -9))
    print(f"\n[1] Fixed split. In-sample (to {TRAIN_END}) best params: lookback={best[0]}m, top_k={best[1]} (Sharpe {g_in[best]:.2f})")
    full = run_strategy(pm, *best)
    is_r = full["net"].loc[:TRAIN_END]; oos_r = full["net"].loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1):]
    print("    The honest number is the out-of-sample one. Expect it to be worse than in-sample.")

    # 2) Walk-forward
    wf, choices = walk_forward(pm)
    print("\n[2] Walk-forward parameter choices (year: lookback, top_k):")
    print("    " + ", ".join(f"{y}: {k[0]}m/{k[1]}" for y, k, _ in choices))
    wf_net = wf["net"]; wf_to = wf["turnover"].mean() * 12

    # 3) Table
    start_oos = wf_net.index[0]
    b_r = bench.loc[start_oos:]; e_r = ew.loc[start_oos:]
    rows = {
        "Strategy: in-sample (fixed split)": metrics(is_r),
        "Strategy: out-of-sample (fixed split)": metrics(oos_r),
        "Strategy: walk-forward (all OOS)": metrics(wf_net),
        f"{BENCHMARK} buy&hold (same window)": metrics(b_r),
        "Equal-weight universe (same window)": metrics(e_r),
    }
    tbl = pd.DataFrame(rows).T
    fmt = tbl.copy()
    for c in ["CAGR", "Vol", "MaxDD"]: fmt[c] = tbl[c].map(lambda x: f"{x:6.1%}" if pd.notna(x) else "n/a")
    for c in ["Sharpe", "Sortino", "Calmar"]: fmt[c] = tbl[c].map(lambda x: f"{x:5.2f}" if pd.notna(x) else "n/a")
    fmt["Months"] = tbl["Months"].astype(int)
    print("\n[3] Performance (net of costs)\n")
    print(fmt.to_string())
    print(f"\n    Walk-forward annual turnover: {wf_to:.1f}x   Cost drag: ~{(wf['gross'].mean()-wf['net'].mean())*12:.2%} a year")

    # 4) Is the edge distinguishable from luck?
    pt, lo, hi = bootstrap_sharpe_diff(wf_net, b_r)
    print(f"\n[4] Walk-forward Sharpe minus {BENCHMARK} Sharpe: {pt:+.2f}  (90% bootstrap interval {lo:+.2f} to {hi:+.2f})")
    verdict = "interval includes zero: NO statistically convincing edge vs the benchmark." if lo <= 0 <= hi else ("interval above zero: some evidence of an edge (still check regimes)." if lo > 0 else "interval below zero: the strategy underperformed the benchmark.")
    print("    " + verdict)
    print(f"    You tried {n_trials} parameter sets, so even a 'good' in-sample Sharpe is inflated by selection.")

    # 5) Parameter stability
    print("\n[5] Full-sample Sharpe by parameter pair (look for a smooth plateau, not one lucky spike):")
    g_all = grid(pm)
    heat = pd.DataFrame([[g_all[(L, K)] for K in TOP_KS] for L in LOOKBACKS], index=[f"{L}m" for L in LOOKBACKS], columns=[f"top{K}" for K in TOP_KS])
    print(heat.round(2).to_string())

    # 6) Cost sensitivity on chosen params
    print("\n[6] Cost sensitivity (in-sample-chosen params, whole sample):")
    for c in [0, 5, 10, 25, 50]:
        r = run_strategy(pm, best[0], best[1], cost_bps=c)["net"]
        m = metrics(r)
        print(f"    {c:>3} bps: CAGR {m['CAGR']:.1%}, Sharpe {m['Sharpe']:.2f}")

    pd.DataFrame(rows).T.to_csv("backtest_summary.csv")
    pd.concat({"walk_forward": wf_net, BENCHMARK: b_r, "equal_weight": e_r}, axis=1).to_csv("backtest_monthly_returns.csv")

    if not args.no_plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(10, 11), sharex=False)
        for name, r in [("Walk-forward strategy", wf_net), (BENCHMARK, b_r), ("Equal-weight", e_r)]:
            ax[0].plot((1 + r.dropna()).cumprod(), label=name)
        ax[0].set_yscale("log"); ax[0].set_title("Growth of 1 (net of costs, out-of-sample window)"); ax[0].legend(); ax[0].grid(alpha=.3)
        for name, r in [("Walk-forward strategy", wf_net), (BENCHMARK, b_r)]:
            eq = (1 + r.dropna()).cumprod(); ax[1].plot(eq / eq.cummax() - 1, label=name)
        ax[1].set_title("Drawdown"); ax[1].legend(); ax[1].grid(alpha=.3)
        rs = wf_net.rolling(36).apply(lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan)
        rb = b_r.rolling(36).apply(lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan)
        ax[2].plot(rs, label="Strategy"); ax[2].plot(rb, label=BENCHMARK); ax[2].axhline(0, color="k", lw=.5)
        ax[2].set_title("Rolling 36-month Sharpe"); ax[2].legend(); ax[2].grid(alpha=.3)
        plt.tight_layout(); plt.savefig("backtest_charts.png", dpi=130)
        print("\nSaved: backtest_summary.csv, backtest_monthly_returns.csv, backtest_charts.png")

if __name__ == "__main__":
    main()
