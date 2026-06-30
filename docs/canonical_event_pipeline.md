# Canonical Event Pipeline

The production target uses one causal domain and one strategy interface. Live,
paper, and replay environments may differ only in their data and execution
adapters.

```mermaid
flowchart LR
    A["Market data adapter"] --> B["Closed MarketBar snapshot"]
    B --> C["Feature and zone events"]
    C --> D["TradingStrategy.evaluate(snapshot)"]
    D --> E["SignalEvent"]
    E --> F["Portfolio risk adapter"]
    F --> G["RiskDecisionEvent"]
    G --> H["OrderIntentEvent"]
    H --> I["Execution adapter"]
    I --> J["Orders, fills, positions"]
    J --> K["Event store and attribution"]
```

## Causal contract

- Every timestamp is timezone-aware UTC.
- `occurred_at` describes market occurrence; `known_at` describes when the
  system could first use it.
- `known_at` may never precede `occurred_at`.
- Snapshots reject bars and zones whose `known_at` exceeds `as_of`.
- Strategies receive no broker, filesystem, network, pandas, or MT5 object.
- A signal must be created exactly at `snapshot.as_of`.
- Stable IDs make replay/live processing idempotent and auditable.
- Domain objects are frozen and slot-based; downstream code cannot mutate the
  historical facts used to make a decision.

## Target structure

```text
backend/
  trading_core/
    events.py       # immutable domain facts and intents
    strategy.py     # canonical snapshot, Protocol, and runner
  adapters/
    market_data/    # MT5 live and CSV replay implementations
    execution/      # MT5, paper, and simulated broker implementations
    persistence/    # SQLite/PostgreSQL event stores
  strategies/
    zone_reversal.py
  risk/
    portfolio.py
```

The next integration step is an adapter around the existing zone-reversal
decision logic. Both live and replay paths will construct the same
`StrategySnapshot` and call the same `StrategyRunner`.

## Implemented compatibility boundary

`strategies/zone_reversal.py` translates the immutable snapshot into the
existing v2 decision function and converts its output into deterministic
`SignalEvent` objects. Both `scalper_strategy_engine.py` and `backtester.py`
now reach it through the same feature-gated pipeline boundary.

## Runtime migration modes

`BotSettings.STRATEGY_PIPELINE_MODE` controls both live and replay entry points:

- `legacy` (default): invokes the existing decision function directly and does
  not construct canonical state.
- `shadow`: returns the legacy decision, evaluates the canonical path with
  telemetry disabled, and records whether their execution-relevant outputs
  match. Canonical failures cannot affect the returned production decision.
- `canonical`: returns the canonical adapter's decision through the existing
  legacy dictionary contract. Snapshot or causality failures propagate to the
  caller's fail-closed guard.

Canonical snapshots admit only candles whose calculated close timestamp is not
later than `as_of`. Replay uses the end of the decision candle as its clock;
live uses the MT5 quote timestamp. A deployment should run in `shadow` and
explain every material mismatch before enabling `canonical`.

Replay lower- and higher-timeframe windows now use that same close-time rule:
M1/M5 observations close no later than the H1 decision clock, and H4 data is
unavailable until four hours after its open. Replay stop construction and the
canonical adapter both consume the same 80-bar closed-M5 window and the same
Wilder ATR implementation. This removes the former one-hour M5 lag, M1/M5 ATR
divergence, and historical H4 forming-candle leakage.

The MT5 live engine applies the identical rule at the broker quote timestamp.
It requests extra MT5 rows, removes any H4/H1/M5 candle whose calculated close
is later than the quote, and then keeps the configured history length. Live,
replay, and canonical paths also share the same Wilder ATR and current completed
H1 reference. The real-time quote remains the entry reference; only incomplete
candle-derived features are excluded.

## Live fail-closed boundary

`DRY_RUN` now guards every broker-mutating path in the strategy cycle: initial
trailing-stop management, partial-close management, strong-signal position
closure, and order placement. A dry run may read broker state and write local
diagnostics, but it cannot change orders or positions.

Before candle retrieval, the live engine also rejects missing, zero, inverted,
non-finite, stale, or materially future-dated quotes. The maximum accepted age
is configured as `BotSettings.MAX_QUOTE_AGE_SECONDS` (120 seconds by default),
with a five-second future-clock tolerance. Rejection counters distinguish
invalid, stale, and future quotes for operations monitoring.

## Replay cadence

The production backtester now uses closed M5 candles as its decision,
execution, trade-management, cooldown, and equity clock. H1 zones and trend are
refreshed only when a new H1 candle becomes available; H4 bias is refreshed only
after a new H4 close. Signals execute at the next M5 open. The former H1-clock
implementation remains named `_run_h1_simulation_legacy` for audit comparison
but is not used by full, walk-forward, or stress runs.

Backtest decisions explicitly force the strategy active and disable live
rejection telemetry, preventing local performance-memory files from changing a
historical experiment. Sorted timestamp indices provide logarithmic bar-window
selection without changing the causal close-time rule.
