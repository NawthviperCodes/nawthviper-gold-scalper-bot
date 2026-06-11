# NawthViper Gold Scalper Bot

NawthViper Gold Scalper Bot is a MetaTrader 5 trading bot with a Python/Flask backend and a React dashboard. It is built around a research-driven zone reversal strategy for gold, with `XAUUSDz` enabled by default in `backend/config.json`.

This project is for education, research, and demo-account testing. Trading is risky, and no strategy can guarantee profit.

## What It Does

- Connects to MetaTrader 5 through the `MetaTrader5` Python API.
- Runs the trading engine in a background thread from the Flask API.
- Monitors configured gold symbols in parallel using a thread pool.
- Detects higher-timeframe trend bias, supply/demand zones, and recent zone touches.
- Generates buy/sell signals from a simplified v2 zone reversal model.
- Places MT5 market orders when `DRY_RUN` is disabled.
- Trails stop losses, checks partial TP logic, and logs closed trade outcomes.
- Blocks trading during risk events such as daily loss/drawdown limits, consecutive-loss circuit breaker, and high-impact news.
- Sends Telegram alerts when `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are configured.
- Provides a React dashboard for MT5 login, starting/stopping the bot, positions, account stats, and recent account history.

## How It Trades

The live engine is centered on `backend/scalper_strategy_engine.py` and `backend/trade_decision_engine.py`.

1. The bot reads settings from `backend/config.json`, including symbols, timeframes, TP ratio, confidence threshold, and dry-run mode.
2. For each symbol, it pulls market data from MT5:
   - H4 for higher-timeframe bias using EMA 50 vs EMA 200.
   - H1 for supply/demand zone detection.
   - M5 for confirmation and recent touch context.
3. It only continues when the higher-timeframe bias is not neutral and enough H1/M5 candles are available.
4. It detects demand and supply zones, then looks for recent tap or deep touches into A/B quality zones.
5. It rejects weak setups, overused zones, full touches, missing reclaim confirmation, low RR, inactive strategies, and conflicting live trades.
6. A valid buy setup comes from demand-zone rejection. A valid sell setup comes from supply-zone rejection.
7. Stop loss is built around the sweep wick plus a 0.15 ATR buffer.
8. Take profit is fixed by the configured R multiple, currently `TP_RATIO: 2.0`.
9. Confidence is adjusted by zone quality, freshness, touch depth, M5 trend, H4 bias, and optional doji/pin-bar bonuses.
10. If confidence meets the configured threshold, the bot sizes the trade and sends the order through MT5.

By default, `DRY_RUN` is set to `true`, so the bot can generate signals without placing live orders.

## Risk Controls

- `DRY_RUN` mode for testing without execution.
- Daily loss and max drawdown limits through `emergency_control.py`.
- Consecutive-loss circuit breaker per symbol.
- High-impact news filter.
- One active same-symbol conflict check before placing trades.
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
|   |-- trade_decision_engine.py   # Zone reversal decision model
|   |-- trade_executor.py          # MT5 order placement and SL/TP management
|   |-- zone_detector.py           # Supply/demand zone detection
|   |-- emergency_control.py       # Daily loss and drawdown controls
|   |-- performance_tracker.py     # Trade logging and summaries
|   |-- telegram_notifier.py       # Telegram alerts via environment variables
|   |-- config.json                # Bot settings and strategy parameters
|   `-- requirements.txt
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

The frontend runs on the Vite URL shown in the terminal, usually `http://127.0.0.1:5173`.

## Configuration

Edit `backend/config.json` to change:

- `BotSettings.SYMBOLS` for gold symbols such as `XAUUSDz`
- `BotSettings.DRY_RUN`
- `BotSettings.MAGIC`
- Strategy timeframes
- TP ratio
- confidence thresholds
- risk/research settings

Telegram alerts use environment variables:

```text
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Keep `.env` files private. They are ignored by Git.

## Important Notes

- MetaTrader 5 must be installed and logged in on the machine running the backend.
- The bot is currently configured for gold trading through broker symbols like `XAUUSDz`.
- Large local datasets, generated CSVs, virtual environments, and `node_modules` are intentionally ignored.
- Test on a demo account before considering any live use.

## Disclaimer

This software is provided for educational and research purposes only. Financial markets are risky, automated trading can lose money, and past results do not guarantee future performance. Use at your own risk.

