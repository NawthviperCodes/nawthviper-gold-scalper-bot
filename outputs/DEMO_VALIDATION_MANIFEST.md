# Demo Validation Manifest

## Deployment seal

- Validation start boundary (broker tick time): **2026-06-30T08:46:26.171Z**
- MT5 broker epoch (milliseconds): **1782809186171**
- Boundary source: latest `EURUSDz` tick returned by the connected Exness terminal during the final read-only preflight.
- The bot was not started when this manifest was sealed. Only trades opened after this boundary and after the operator's explicit launch authorization are eligible for validation.

## Account and instrument lock

- Broker/account type: **Exness demo** (`trade_mode=0`)
- Server: **Exness-MT5Trial10**
- Account login: **redacted from the public repository**
- Starting balance/equity: **$2,000.00 / $2,000.00**
- Open positions at seal: **0**
- Instrument pool: **XAUUSDz, EURUSDz only**
- Portfolio concurrency: **one open position globally across the account**
- Risk model: **fixed 1% of account balance per trade**
- Nominal target: **2R**

## Four authorized safeguards

1. **Instrument pool:** verified that `config.json` contains exactly `XAUUSDz` and `EURUSDz`.
2. **Global position limit:** account-wide position inspection is serialized with order submission. The first completed signal to acquire the execution lock wins; every later simultaneous signal is rejected while any account position exists.
3. **Frozen adaptive behavior:** performance-based risk scaling, fixed-lot override, and the adaptive performance strategy gate are disabled. Fixed 1% risk remains. Hard safety circuit breakers that only stop entries remain active.
4. **Complete trade log:** entry and closure records include UTC timestamps, instrument, side, signal/fill prices, adverse slippage, stop, target, confidence, zone quality, HTF bias, close reason, realized P&L, and realized R.

Verification evidence:

- Full installed unit suite: **53 tests passed** while execution remained safely re-blocked with `DRY_RUN=true`; after verification, the authorized deployment value `DRY_RUN=false` was restored and checked separately without starting the process.
- Concurrent two-signal executor simulation: **one order accepted, one rejected**.
- Logger round trip: **25 fields persisted; TP closure and +2.0R calculated correctly**.
- Final MT5 preflight: terminal initialized and connected, expert trading permitted, demo account confirmed, and zero positions open.
- Locked decision, zone, confirmation, ATR/departure/width thresholds, structural-stop logic, and 2R strategy rules were not changed.

## Locked development reference

T10 pooled two-instrument development result:

- Expectancy: **+0.2184R**
- Profit factor: **1.371**
- Trades: **209**
- Expectancy confidence-interval lower bound: **+0.0257R**
- Development win rate: **41.15%**
- Development average win/loss: **+1.961R / -1.000R**

## Demo pass gate

Assess only the post-boundary, explicitly authorized demo trades. Passing requires every condition below:

- At least **100 closed trades**
- Expectancy **>= +0.10R**
- Profit factor **>= 1.15**
- Expectancy confidence-interval lower bound **> 0R**
- Win rate within 10 percentage points of 41.15%: **31.15% to 51.15%**
- Average win within 15% of +1.961R: **+1.6669R to +2.2552R**
- Average loss magnitude within 15% of 1.000R: **0.8500R to 1.1500R**

## Governance

- This manifest freezes the demo-validation configuration and eligibility boundary.
- No pre-boundary observation or trade may be included in the validation sample.
- No strategy, threshold, instrument, sizing, concurrency, or account change is permitted during the sample without invalidating the run and issuing a new manifest.
- The first order requires explicit operator go-ahead after readiness confirmation.
