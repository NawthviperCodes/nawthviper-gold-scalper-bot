# NawthViper Gold Scalper Bot

NawthViper Gold Scalper Bot is a MetaTrader 5 trading bot with a Python/Flask
backend and a React dashboard. It is built around a research-driven zone
reversal strategy, currently locked to a pooled **XAUUSDz + EURUSDz**
configuration validated through a structured, preregistered backtest research
program (see [Research Status](#research-status) below).

This project is for education, research, and demo-account testing. Trading is
risky, and no strategy can guarantee profit. **The current configuration has
passed development backtest gates but has not yet completed out-of-sample
validation — see Research Status before drawing any conclusions about live
viability.**

## Research Status

This strategy did not start in its current form. It went through two full
research cycles and ten preregistered trials (T1–T10) before reaching the
configuration currently live in `config.json`. Briefly:

- **T1–T7** (Cycle 1 and 2): tested H4 regime alignment, zone-quality
  filtering, session timing, failed-auction confirmation, pure momentum, a
  trade-management protection rule, and a VIX75 instrument transfer.
  All falsified or statistically inconclusive.
- **Five-change rebuild** (between Cycle 2 and 3): hard-rejected B-quality
  zones, removed candlestick pattern bonuses, made the H4 EMA trend bias a
  hard directional gate, moved stops to the displacement-origin structural
  level, and rebased take-profit to the actual fill price instead of the
  pre-execution signal price.
- **T8–T9**: pooled XAUUSDz with GBPUSDz and USDJPYz. Both real, liquid
  instruments — but both diluted the pool's profit factor below gate.
  GBPUSDz and USDJPYz are not part of the current configuration.
- **T10 (current locked development reference)**: pooled XAUUSDz + EURUSDz
  only, on causal grounds that GBP and JPY underwent unusually sustained
  directional regimes in the development window that conflict with this
  strategy's mean-reversion entry logic. **Passed all four preregistered
  development gates:** 209 trades, +0.2184R expectancy, profit factor 1.371,
  95% CI lower bound +0.0257R.
- **Backtest validation status:** blocked. A contamination audit found that
  prior full-backtest artifacts in this repository had already exposed the
  intended validation/holdout window (2024-02-19 to 2026-03-20) before the
  T1–T10 trial sequence began. No clean, unexposed historical window remains
  with current data.
- **Current phase: demo forward-validation.** Since historical out-of-sample
  data is exhausted, validation is being conducted by running the locked T10
  configuration on an Exness demo account and treating every trade from the
  deployment boundary forward as genuine out-of-sample evidence. See
  `outputs/DEMO_VALIDATION_MANIFEST.md` for the sealed deployment boundary,
  account details, and pass gate. Target: 100 closed trades before any
  pass/fail determination, expected to take approximately 6–8 months given
  historical trade frequency.

**No live capital has been deployed.** No live-trading authorization has been
given pending the demo validation outcome.

## What It Does

- Connects to MetaTrader 5 through the `MetaTrader5` Python API.
- Runs the trading engine in a background thread from the Flask API.
- Monitors configured symbols (currently `XAUUSDz`, `EURUSDz`) in parallel
  using a thread pool.
- Detects higher-timeframe trend bias (H4 EMA50/EMA200), supply/demand
  zones (H1), and recent zone touches (M5).
- Generates buy/sell signals from a research-validated zone reversal model.
- Places MT5 market orders when `DRY_RUN` is disabled.
- Trails stop losses, checks partial TP logic, and logs closed trade outcomes
  with extended fields (fill price, slippage, confidence, zone quality, HTF
  bias, close reason, realized R) for ongoing validation analysis.
- Blocks trading during risk events such as daily loss/drawdown limits,
  consecutive-loss circuit breaker, and high-impact news.
- Enforces a **global one-position limit across all configured instruments**
  (not per-symbol) for the duration of the demo validation period.
- Sends Telegram alerts when `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are
  configured.
- Provides a React dashboard for MT5 login, starting/stopping the bot,
  positions, account stats, and recent account history.

## How It Trades

The live engine is centered on `backend/scalper_strategy_engine.py` and
`backend/trade_decision_engine.py`.

1. The bot reads settings from `backend/config.json`, including symbols,
   timeframes, TP ratio, confidence threshold, and dry-run mode.
2. For each symbol, it pulls market data from MT5:
   - H4 for higher-timeframe bias using EMA 50 vs EMA 200.
   - H1 for supply/demand zone detection.
   - M5 for confirmation and recent touch context.
3. It only continues when the higher-timeframe bias is not neutral and
   enough H1/M5 candles are available.
4. It detects demand and supply zones, then looks for recent tap or deep
   touches into zones.
5. **Only A-quality zones are tradeable.** B-quality zones are hard-rejected
   — research found they carried negative expectancy (-$7.88/trade average
   in development testing) and were diluting the A-zone signal
   (+$6.03/trade average).
6. A confirmed opposite H4 trend bias is a **hard reject**, not a soft
   confidence adjustment. A buy is blocked if HTF bias is DOWN; a sell is
   blocked if HTF bias is UP. NEUTRAL bias passes both directions.
7. It rejects overused zones, full touches, missing reclaim confirmation,
   low RR, inactive strategies, and conflicting live trades.
8. A valid buy setup comes from demand-zone rejection. A valid sell setup
   comes from supply-zone rejection.
9. Stop loss is built around the **displacement-origin structural level**
   (the candle that created the zone) plus a 0.15 ATR buffer, falling back
   to the touch-candle wick if the origin level isn't available.
10. Take profit is fixed by the configured R multiple (`TP_RATIO: 2.0`),
    **rebased to the actual signal-side fill price** (bid/ask plus spread)
    rather than the pre-execution mid-price, to avoid systematically
    undershooting the nominal target.
11. Confidence is adjusted by zone quality, freshness, touch depth, and M5
    trend. **Candlestick pattern bonuses (doji, pin bar) have been removed**
    — research found every pattern-augmented trade underperformed the plain
    reclaim signal.
12. If confidence meets the configured threshold, the bot sizes the trade
    (fixed 1% account risk) and sends the order through MT5, respecting the
    global one-position limit.

By default, `DRY_RUN` is set to `true` for local testing. The current demo
validation deployment runs with `DRY_RUN: false` against the Exness demo
account specified in `outputs/DEMO_VALIDATION_MANIFEST.md` only.

## Risk Controls

- `DRY_RUN` mode for testing without execution.
- Daily loss and max drawdown limits through `emergency_control.py`.
- Consecutive-loss circuit breaker per symbol.
- High-impact news filter.
- **Global one-position limit** across all configured instruments (added for
  demo validation — previously per-symbol only).
- Fixed 1% account risk per trade. **Dynamic/adaptive risk scaling and
  fixed-lot override are frozen** for the duration of the demo validation
  period so that every trade reflects the exact locked T10 configuration.
- Broker-aware lot sizing and filling-mode detection.
- Optional Telegram status and trade notifications.

## Tech Stack

| Layer | Tools |
| --- | --- |
| Backend | Python, Flask, Flask-CORS |
| Trading | MetaTrader 5 Python API |
| Analysis | pandas, numpy, ta |
| News/HTTP | requests, beautifulsoup4 |
| Frontend | React, Vite, Tailwind CSS, Chart.js |

## Project Structure

```text
Currency_v2/
|-- backend/
|   |-- api_server.py              # Flask API and dashboard endpoints
|   |-- main.py                    # Real-time bot loop
|   |-- scalper_strategy_engine.py # Symbol cycle, filters, execution trigger
|   |-- trade_decision_engine.py   # Zone reversal decision model (locked
|   |                              # five-change rebuild: A-zone only, hard
|   |                              # HTF gate, structural stops, rebased TP,
|   |                              # no pattern bonuses)
|   |-- trade_executor.py          # MT5 order placement, SL/TP management,
|   |                              # global one-position limit
|   |-- zone_detector.py           # Supply/demand zone detection
|   |-- zone_reversal.py           # Zone quality classification, HTF bias
|   |-- trade_logger.py            # Extended trade logging (25 fields)
|   |-- emergency_control.py       # Daily loss and drawdown controls
|   |-- performance_tracker.py     # Trade logging and summaries
|   |-- telegram_notifier.py       # Telegram alerts via environment variables
|   |-- config.json                # Bot settings; locked to XAUUSDz, EURUSDz
|   `-- requirements.txt
|-- outputs/
|   |-- DEMO_VALIDATION_MANIFEST.md   # Sealed demo deployment boundary,
|   |                                  # account, and pass gate
|   `-- (research artifacts: T1-T10 trial reports, backtest results)
|-- frontend/
|   |-- src/
|   |-- package.json
|   `-- vite.config.js
`-- README.md
```

## Setup

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python api_server.py
```

The backend runs on `http://127.0.0.1:5000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on the Vite URL shown in the terminal, usually
`http://127.0.0.1:5173`.

## Configuration

Edit `backend/config.json` to change:

- `BotSettings.SYMBOLS` — **currently locked to `XAUUSDz` and `EURUSDz`.**
  This is not an arbitrary default: the strategy's passing development
  result (T10) was a pooled two-instrument result, not validated for either
  instrument alone or for any other instrument. Adding or removing symbols
  invalidates the current demo validation run and requires a new
  preregistered trial before any conclusions can be drawn from it.
- `BotSettings.DRY_RUN`
- `BotSettings.MAGIC`
- Strategy timeframes
- TP ratio
- confidence thresholds
- risk/research settings

**Do not change strategy thresholds (zone quality ATR bounds, departure
strength minimums, confidence threshold, stop buffer, TP ratio) without
running a new preregistered backtest trial first.** Every parameter in the
current configuration was arrived at through a structured falsification
process; changing them silently breaks the chain of evidence behind the
T10 result.

Telegram alerts use environment variables:

```text
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Keep `.env` files private. They are ignored by Git.

## Important Notes

- MetaTrader 5 must be installed and logged in on the machine running the
  backend.
- The bot is currently configured for `XAUUSDz` and `EURUSDz` only. This
  pairing was specifically validated together — see Research Status.
- Large local datasets, generated CSVs, virtual environments, and
  `node_modules` are intentionally ignored.
- **The bot is currently running a live demo forward-validation deployment.**
  Do not modify `trade_decision_engine.py`, `zone_reversal.py`, or any
  threshold in `config.json` while this validation is in progress — doing so
  invalidates the sample and requires starting a new sealed manifest from
  zero trades.
- Test on a demo account before considering any live use. **As of this
  writing, this strategy has not completed out-of-sample validation and
  should not be deployed with live capital.**

## Disclaimer

This software is provided for educational and research purposes only.
Financial markets are risky, automated trading can lose money, and past
results do not guarantee future performance. The current strategy
configuration has passed a structured development backtest but has not yet
completed forward out-of-sample validation. Use at your own risk.
