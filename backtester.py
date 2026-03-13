"""
Crypto Backtesting Engine
Loads CSV data, runs EMA/RSI/MACD strategy combinations across all coins and timeframes,
evaluates performance, and saves the best strategy + PineScript output.
"""

import os
import re
import glob
import json
import itertools
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Data Loading ──────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STRATEGIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategies")


def parse_filename(filepath: str) -> tuple[str, str]:
    """Extract symbol and timeframe from filename like 'BINANCE_BTCUSDT 2h ...'."""
    basename = os.path.basename(filepath)
    # Handle comma in filename (BTCUSDT, 1h)
    basename = basename.replace(",", "")
    parts = basename.split()
    symbol = parts[0].replace("BINANCE_", "")
    timeframe_raw = parts[1]
    # Normalize 'u' to 'h' (NEAR files use 'u')
    timeframe = timeframe_raw.replace("u", "h")
    return symbol, timeframe


def load_all_datasets() -> list[dict]:
    """Load all CSV files and return list of {symbol, timeframe, df}."""
    datasets = []
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    for f in sorted(csv_files):
        symbol, timeframe = parse_filename(f)
        df = pd.read_csv(f)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.sort_values("time").reset_index(drop=True)
        datasets.append({"symbol": symbol, "timeframe": timeframe, "df": df})
    print(f"Loaded {len(datasets)} datasets")
    return datasets


# ── Indicator Calculations ────────────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── Strategy Definitions ──────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    name: str
    # EMA params
    use_ema: bool = False
    ema_fast: int = 9
    ema_slow: int = 21
    # RSI params
    use_rsi: bool = False
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    # MACD params
    use_macd: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    def describe(self) -> str:
        parts = []
        if self.use_ema:
            parts.append(f"EMA({self.ema_fast}/{self.ema_slow})")
        if self.use_rsi:
            parts.append(f"RSI({self.rsi_period}, {self.rsi_oversold}/{self.rsi_overbought})")
        if self.use_macd:
            parts.append(f"MACD({self.macd_fast}/{self.macd_slow}/{self.macd_signal})")
        return " + ".join(parts)


def generate_strategies() -> list[StrategyConfig]:
    """Generate strategy combinations using EMA, RSI, MACD."""
    strategies = []
    ema_params = [(9, 21), (12, 26), (20, 50), (8, 34)]
    rsi_params = [(14, 30, 70), (14, 25, 75), (21, 30, 70), (7, 20, 80)]
    macd_params = [(12, 26, 9), (8, 21, 5), (5, 13, 8)]

    # Single indicator strategies
    for fast, slow in ema_params:
        strategies.append(StrategyConfig(
            name=f"EMA_{fast}_{slow}",
            use_ema=True, ema_fast=fast, ema_slow=slow
        ))
    for period, oversold, overbought in rsi_params:
        strategies.append(StrategyConfig(
            name=f"RSI_{period}_{oversold}_{overbought}",
            use_rsi=True, rsi_period=period, rsi_oversold=oversold, rsi_overbought=overbought
        ))
    for fast, slow, sig in macd_params:
        strategies.append(StrategyConfig(
            name=f"MACD_{fast}_{slow}_{sig}",
            use_macd=True, macd_fast=fast, macd_slow=slow, macd_signal=sig
        ))

    # Two-indicator combos
    for (ef, es), (rp, ro, rb) in itertools.product(ema_params, rsi_params):
        strategies.append(StrategyConfig(
            name=f"EMA_{ef}_{es}_RSI_{rp}_{ro}_{rb}",
            use_ema=True, ema_fast=ef, ema_slow=es,
            use_rsi=True, rsi_period=rp, rsi_oversold=ro, rsi_overbought=rb
        ))
    for (ef, es), (mf, ms, msig) in itertools.product(ema_params, macd_params):
        strategies.append(StrategyConfig(
            name=f"EMA_{ef}_{es}_MACD_{mf}_{ms}_{msig}",
            use_ema=True, ema_fast=ef, ema_slow=es,
            use_macd=True, macd_fast=mf, macd_slow=ms, macd_signal=msig
        ))
    for (rp, ro, rb), (mf, ms, msig) in itertools.product(rsi_params, macd_params):
        strategies.append(StrategyConfig(
            name=f"RSI_{rp}_{ro}_{rb}_MACD_{mf}_{ms}_{msig}",
            use_rsi=True, rsi_period=rp, rsi_oversold=ro, rsi_overbought=rb,
            use_macd=True, macd_fast=mf, macd_slow=ms, macd_signal=msig
        ))

    # Triple combos
    for (ef, es), (rp, ro, rb), (mf, ms, msig) in itertools.product(
        ema_params, rsi_params, macd_params
    ):
        strategies.append(StrategyConfig(
            name=f"EMA_{ef}_{es}_RSI_{rp}_{ro}_{rb}_MACD_{mf}_{ms}_{msig}",
            use_ema=True, ema_fast=ef, ema_slow=es,
            use_rsi=True, rsi_period=rp, rsi_oversold=ro, rsi_overbought=rb,
            use_macd=True, macd_fast=mf, macd_slow=ms, macd_signal=msig
        ))

    print(f"Generated {len(strategies)} strategy combinations")
    return strategies


# ── Signal Generation ─────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    """
    Generate buy/sell signals. Returns a Series of 1 (buy), -1 (sell), 0 (hold).
    For multi-indicator strategies, ALL indicators must agree for a signal.
    """
    n = len(df)
    buy_conditions = pd.Series(True, index=df.index)
    sell_conditions = pd.Series(True, index=df.index)

    if config.use_ema:
        ema_fast = calc_ema(df["close"], config.ema_fast)
        ema_slow = calc_ema(df["close"], config.ema_slow)
        # Buy: fast crosses above slow, Sell: fast crosses below slow
        ema_above = ema_fast > ema_slow
        buy_conditions &= ema_above & ~ema_above.shift(1).fillna(False)
        sell_conditions &= ~ema_above & ema_above.shift(1).fillna(True)

    if config.use_rsi:
        rsi = calc_rsi(df["close"], config.rsi_period)
        # Buy: RSI crosses above oversold, Sell: RSI crosses below overbought
        rsi_prev = rsi.shift(1)
        buy_conditions &= (rsi > config.rsi_oversold) & (rsi_prev <= config.rsi_oversold)
        sell_conditions &= (rsi < config.rsi_overbought) & (rsi_prev >= config.rsi_overbought)

    if config.use_macd:
        macd_line, signal_line, histogram = calc_macd(
            df["close"], config.macd_fast, config.macd_slow, config.macd_signal
        )
        # Buy: MACD crosses above signal, Sell: MACD crosses below signal
        macd_above = macd_line > signal_line
        buy_conditions &= macd_above & ~macd_above.shift(1).fillna(False)
        sell_conditions &= ~macd_above & macd_above.shift(1).fillna(True)

    signals = pd.Series(0, index=df.index)
    signals[buy_conditions] = 1
    signals[sell_conditions] = -1
    return signals


# ── Backtesting Engine ────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    timeframe: str
    total_return_pct: float
    max_drawdown_pct: float
    num_trades: int
    win_rate: float
    sharpe_ratio: float = 0.0


def backtest(df: pd.DataFrame, config: StrategyConfig, symbol: str, timeframe: str) -> BacktestResult:
    """Run backtest on a single dataset with a single strategy."""
    signals = generate_signals(df, config)

    initial_capital = 10000.0
    capital = initial_capital
    position = 0.0  # number of units held
    entry_price = 0.0
    peak_capital = initial_capital
    max_drawdown = 0.0
    trades = []

    for i in range(len(df)):
        price = df["close"].iloc[i]

        if signals.iloc[i] == 1 and position == 0:
            # Buy
            position = capital / price
            entry_price = price
            capital = 0.0
        elif signals.iloc[i] == -1 and position > 0:
            # Sell
            capital = position * price
            pnl = (price - entry_price) / entry_price
            trades.append(pnl)
            position = 0.0
            entry_price = 0.0

        # Track equity for drawdown
        equity = capital + position * price
        if equity > peak_capital:
            peak_capital = equity
        dd = (peak_capital - equity) / peak_capital
        if dd > max_drawdown:
            max_drawdown = dd

    # Close any open position at the end
    if position > 0:
        final_price = df["close"].iloc[-1]
        capital = position * final_price
        pnl = (final_price - entry_price) / entry_price
        trades.append(pnl)
        position = 0.0

    final_equity = capital if capital > 0 else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital * 100
    win_rate = (sum(1 for t in trades if t > 0) / len(trades) * 100) if trades else 0.0

    return BacktestResult(
        strategy_name=config.name,
        symbol=symbol,
        timeframe=timeframe,
        total_return_pct=round(total_return, 2),
        max_drawdown_pct=round(max_drawdown * 100, 2),
        num_trades=len(trades),
        win_rate=round(win_rate, 2),
    )


# ── Aggregate Scoring ─────────────────────────────────────────────────────────

def score_strategy(results: list[BacktestResult]) -> float:
    """
    Score a strategy across all datasets.
    Balances return, drawdown, win rate, and trade frequency.
    """
    if not results:
        return -999
    avg_return = np.mean([r.total_return_pct for r in results])
    avg_drawdown = np.mean([r.max_drawdown_pct for r in results])
    avg_winrate = np.mean([r.win_rate for r in results])
    avg_trades = np.mean([r.num_trades for r in results])

    # Penalize strategies with too few trades
    trade_penalty = 0 if avg_trades >= 10 else -50

    # Score: high return, low drawdown, high win rate
    score = avg_return - (avg_drawdown * 0.5) + (avg_winrate * 0.3) + trade_penalty
    return round(score, 2)


# ── PineScript Generator ─────────────────────────────────────────────────────

def generate_pinescript(config: StrategyConfig) -> str:
    """Generate TradingView PineScript v5 for the strategy."""
    lines = [
        '//@version=5',
        f'strategy("{config.describe()}", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100)',
        '',
    ]

    conditions_buy = []
    conditions_sell = []

    if config.use_ema:
        lines.append(f'emaFast = ta.ema(close, {config.ema_fast})')
        lines.append(f'emaSlow = ta.ema(close, {config.ema_slow})')
        lines.append('plot(emaFast, "EMA Fast", color=color.blue)')
        lines.append('plot(emaSlow, "EMA Slow", color=color.red)')
        lines.append('')
        conditions_buy.append('ta.crossover(emaFast, emaSlow)')
        conditions_sell.append('ta.crossunder(emaFast, emaSlow)')

    if config.use_rsi:
        lines.append(f'rsiVal = ta.rsi(close, {config.rsi_period})')
        lines.append(f'rsiOversold = {config.rsi_oversold}')
        lines.append(f'rsiOverbought = {config.rsi_overbought}')
        lines.append('')
        conditions_buy.append('ta.crossover(rsiVal, rsiOversold)')
        conditions_sell.append('ta.crossunder(rsiVal, rsiOverbought)')

    if config.use_macd:
        lines.append(f'[macdLine, signalLine, histLine] = ta.macd(close, {config.macd_fast}, {config.macd_slow}, {config.macd_signal})')
        lines.append('')
        conditions_buy.append('ta.crossover(macdLine, signalLine)')
        conditions_sell.append('ta.crossunder(macdLine, signalLine)')

    buy_cond = ' and '.join(conditions_buy)
    sell_cond = ' and '.join(conditions_sell)

    lines.append(f'buySignal = {buy_cond}')
    lines.append(f'sellSignal = {sell_cond}')
    lines.append('')
    lines.append('if buySignal')
    lines.append('    strategy.entry("Long", strategy.long)')
    lines.append('if sellSignal')
    lines.append('    strategy.close("Long")')
    lines.append('')
    lines.append('plotshape(buySignal, title="Buy", location=location.belowbar, color=color.green, style=shape.triangleup, size=size.small)')
    lines.append('plotshape(sellSignal, title="Sell", location=location.abovebar, color=color.red, style=shape.triangledown, size=size.small)')

    return '\n'.join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    datasets = load_all_datasets()
    strategies = generate_strategies()

    # Run all backtests
    strategy_results: dict[str, list[BacktestResult]] = {}
    total_tests = len(strategies) * len(datasets)
    print(f"Running {total_tests} backtests...")

    for i, config in enumerate(strategies):
        results = []
        for ds in datasets:
            result = backtest(ds["df"], config, ds["symbol"], ds["timeframe"])
            results.append(result)
        strategy_results[config.name] = results
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(strategies)} strategies completed")

    print(f"All backtests complete.\n")

    # Score and rank strategies
    scored = []
    for config in strategies:
        results = strategy_results[config.name]
        s = score_strategy(results)
        avg_ret = np.mean([r.total_return_pct for r in results])
        avg_dd = np.mean([r.max_drawdown_pct for r in results])
        avg_wr = np.mean([r.win_rate for r in results])
        avg_tr = np.mean([r.num_trades for r in results])
        scored.append({
            "config": config,
            "score": s,
            "avg_return": round(avg_ret, 2),
            "avg_drawdown": round(avg_dd, 2),
            "avg_winrate": round(avg_wr, 2),
            "avg_trades": round(avg_tr, 1),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Print top 10
    print("=" * 90)
    print(f"{'Rank':<5} {'Strategy':<45} {'Return%':>8} {'MaxDD%':>8} {'WinR%':>7} {'Trades':>7} {'Score':>7}")
    print("=" * 90)
    for i, s in enumerate(scored[:10]):
        print(f"{i+1:<5} {s['config'].describe():<45} {s['avg_return']:>8.1f} {s['avg_drawdown']:>8.1f} {s['avg_winrate']:>7.1f} {s['avg_trades']:>7.1f} {s['score']:>7.1f}")
    print("=" * 90)

    # Best strategy
    best = scored[0]
    best_config = best["config"]
    best_results = strategy_results[best_config.name]

    print(f"\nBest Strategy: {best_config.describe()}")
    print(f"  Avg Return: {best['avg_return']:.2f}%")
    print(f"  Avg Max Drawdown: {best['avg_drawdown']:.2f}%")
    print(f"  Avg Win Rate: {best['avg_winrate']:.2f}%")
    print(f"  Avg Trades: {best['avg_trades']:.1f}")
    print(f"  Score: {best['score']:.2f}")

    # Per-dataset breakdown for best strategy
    print(f"\n{'Symbol':<12} {'TF':<5} {'Return%':>9} {'MaxDD%':>9} {'WinR%':>8} {'Trades':>7}")
    print("-" * 55)
    for r in sorted(best_results, key=lambda x: x.total_return_pct, reverse=True):
        print(f"{r.symbol:<12} {r.timeframe:<5} {r.total_return_pct:>9.2f} {r.max_drawdown_pct:>9.2f} {r.win_rate:>8.2f} {r.num_trades:>7}")

    # Save best strategy
    os.makedirs(STRATEGIES_DIR, exist_ok=True)

    # Save strategy config as JSON
    strategy_data = {
        "name": best_config.name,
        "description": best_config.describe(),
        "parameters": {
            "use_ema": best_config.use_ema,
            "ema_fast": best_config.ema_fast,
            "ema_slow": best_config.ema_slow,
            "use_rsi": best_config.use_rsi,
            "rsi_period": best_config.rsi_period,
            "rsi_oversold": best_config.rsi_oversold,
            "rsi_overbought": best_config.rsi_overbought,
            "use_macd": best_config.use_macd,
            "macd_fast": best_config.macd_fast,
            "macd_slow": best_config.macd_slow,
            "macd_signal": best_config.macd_signal,
        },
        "performance": {
            "avg_return_pct": best["avg_return"],
            "avg_max_drawdown_pct": best["avg_drawdown"],
            "avg_win_rate_pct": best["avg_winrate"],
            "avg_trades": best["avg_trades"],
            "score": best["score"],
        },
        "per_dataset_results": [
            {
                "symbol": r.symbol,
                "timeframe": r.timeframe,
                "total_return_pct": r.total_return_pct,
                "max_drawdown_pct": r.max_drawdown_pct,
                "num_trades": r.num_trades,
                "win_rate": r.win_rate,
            }
            for r in best_results
        ],
    }

    config_path = os.path.join(STRATEGIES_DIR, "best_strategy.json")
    with open(config_path, "w") as f:
        json.dump(strategy_data, f, indent=2)
    print(f"\nStrategy config saved to: {config_path}")

    # Save PineScript
    pinescript = generate_pinescript(best_config)
    pine_path = os.path.join(STRATEGIES_DIR, "best_strategy.pine")
    with open(pine_path, "w") as f:
        f.write(pinescript)
    print(f"PineScript saved to: {pine_path}")

    # Save full leaderboard
    leaderboard = []
    for s in scored:
        leaderboard.append({
            "rank": scored.index(s) + 1,
            "name": s["config"].name,
            "description": s["config"].describe(),
            "avg_return_pct": s["avg_return"],
            "avg_max_drawdown_pct": s["avg_drawdown"],
            "avg_win_rate_pct": s["avg_winrate"],
            "avg_trades": s["avg_trades"],
            "score": s["score"],
        })
    lb_path = os.path.join(STRATEGIES_DIR, "leaderboard.json")
    with open(lb_path, "w") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"Full leaderboard saved to: {lb_path}")


if __name__ == "__main__":
    main()
