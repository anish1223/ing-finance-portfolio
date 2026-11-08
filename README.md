# ING valuation, momentum backtest and euro-area rates dashboard

Three small finance projects built as interview preparation (equity valuation, systematic trading, macro/rates).
Each one reports its limitations openly; none of this is investment advice.

| Folder | What it is |
|---|---|
| `01_ing_valuation/` | Excel valuation model for ING Groep N.V. (inputs, forecast, valuation, comps) |
| `02_momentum_backtest/` | Python backtest of ETF momentum with costs and walk-forward validation |
| `03_macro_dashboard/` | Python dashboard of euro-area vs US rates and inflation (ECB + FRED) |

## 1. ING valuation (Excel)
A bank is valued on its equity, not its enterprise value, so a standard DCF/WACC does not apply. The model discounts
the cash ING can distribute to shareholders (net result minus capital retained to support RWA growth at a 13% CET1 target)
at the cost of equity, with a terminal value from the justified P/TBV formula (ROTE - g) / (ke - g).

- Anchored on ING's 3Q2026 press release (29 Oct 2026) and management guidance for 2026-27.
- Bear / Base / Bull switch (`Inputs!B4`); sensitivity tables for cost of equity vs terminal growth and vs terminal ROTE.
- Base case: EUR 28.96 per share vs EUR 31.49 reference price (-8%); bear EUR 21.75; bull EUR 33.59.
- Reverse DCF: the reference price implies roughly 14.6% terminal ROTE.
- **Limitations:** beta (1.10) and ERP (5.5%) are my assumptions, not sourced; peer comps are intentionally left blank
  (no reliable peer data was available); terminal value is ~59% of equity value; AT1 coupon treatment and the ongoing
  buyback are simplified; the Russia exit is not modelled.

## 2. Momentum backtest (Python)
Monthly 12-1 style momentum across 11 liquid ETFs with an absolute-momentum filter (cash = SHY), 10 bps one-way costs,
no look-ahead, a fixed train/test split and an annual walk-forward parameter re-selection.

Result (2012-2026 walk-forward window, net of costs):

| | CAGR | Sharpe | Max drawdown |
|---|---|---|---|
| Walk-forward strategy | 6.8% | 0.70 | -15.6% |
| SPY buy & hold | 14.5% | 1.05 | -23.9% |
| Equal-weight universe | 6.3% | 0.72 | -19.5% |

The strategy did **not** beat SPY (Sharpe difference -0.35, 90% block-bootstrap interval -0.67 to -0.02). It is
statistically similar to an equal-weight basket with a shallower drawdown than SPY.
- **Limitations:** Yahoo adjusted-price data; survivorship/availability bias in the ETF universe; 16 parameter
  combinations tried; the final month is partial; the fixed-split out-of-sample Sharpe (0.82) exceeds in-sample (0.46),
  which reflects the regime of the split date rather than skill.

```bash
pip install -r requirements.txt
python 02_momentum_backtest/momentum_backtest.py            # real data
python 02_momentum_backtest/momentum_backtest.py --synthetic # code test on fake data
```

## 3. Euro-area rates dashboard (Python)
Pulls ECB (yield curve, deposit rate, HICP) and FRED (US yields, Fed funds, CPI) data through public endpoints, builds a
four-panel chart and prints a fact sheet. Snapshot from the included run (8 Nov 2026): ECB deposit rate 2.50%,
euro-area 2y yield 46 bp above it (market pricing hikes), headline HICP 4.0% vs core 2.5%, real policy rate -1.5%,
US-minus-euro-area 10y gap 1.55 pp. The script supplies facts; the interpretation is left to the analyst.

```bash
python 03_macro_dashboard/macro_dashboard.py          # real data
python 03_macro_dashboard/macro_dashboard.py --demo   # code test on fake data
```

## Reproducing
Outputs in each `results/` folder are the saved runs referred to above. Re-running will give different numbers as new
data arrives.
