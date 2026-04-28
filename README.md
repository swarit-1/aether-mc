# AETHER_CRYSTAL Monte Carlo

Single-page Monte Carlo simulator for the AETHER_CRYSTAL options challenge.

- GBM with zero drift, σ = 251% (annual), 4 steps/day, 252 trading days/year
- Simulate 1 / 100 / 1000 / 2000 paths
- Visualizes sample paths and the terminal price distribution
- Trade entry for the underlying, 2- and 3-week vanilla calls/puts, plus three exotics:
  - Chooser (decide call vs put at T+14)
  - Binary put (configurable payout)
  - Knock-out put (configurable barrier)
- Marks each position to the simulated fair value and reports expected PnL × contract size (3000), with std-dev and 5/95th percentile bands

## Run locally

Just open `index.html` in a browser. No build step.
