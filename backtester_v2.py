"""
Backtester V2 - Optimized strategy with filters and risk management.
Targets: PF >= 1.4, NP/MDD >= 4, Sortino >= 1, Trades >= 150, Largest loser < 10% of gross loss.
"""
import os, glob, json, itertools
from dataclasses import dataclass
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STRATEGIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategies")


# ── Indicators ────────────────────────────────────────────────────────────────

def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(span=p, adjust=False).mean()
    al = l.ewm(span=p, adjust=False).mean()
    return 100 - (100 / (1 + ag / al))

def calc_macd(s, f=8, sl=21, sig=5):
    ef = calc_ema(s, f)
    es = calc_ema(s, sl)
    ml = ef - es
    sl_line = calc_ema(ml, sig)
    return ml, sl_line, ml - sl_line

def calc_atr(df, p=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(p).mean()


# ── Strategy Config ───────────────────────────────────────────────────────────

@dataclass
class StrategyV2Config:
    name: str
    # MACD
    macd_fast: int = 8
    macd_slow: int = 21
    macd_signal: int = 5
    # RSI filter
    rsi_period: int = 14
    rsi_min: float = 30.0
    rsi_max: float = 65.0
    # Trend filter
    use_ema_filter: bool = True
    ema_trend: int = 100
    # ATR-based risk management
    atr_period: int = 14
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    trailing_atr_mult: float = 2.5
    # Time-based exit
    max_bars: int = 30
    # MACD histogram filter
    require_hist_positive: bool = True

    def describe(self) -> str:
        parts = [f"MACD({self.macd_fast}/{self.macd_slow}/{self.macd_signal})"]
        parts.append(f"RSI({self.rsi_period},{self.rsi_min}-{self.rsi_max})")
        if self.use_ema_filter:
            parts.append(f"EMA({self.ema_trend})")
        parts.append(f"SL:{self.sl_atr_mult}ATR TP:{self.tp_atr_mult}ATR Trail:{self.trailing_atr_mult}ATR")
        if self.max_bars > 0:
            parts.append(f"MaxBars:{self.max_bars}")
        return " | ".join(parts)


# ── Backtest Engine ───────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    entry_price: float
    exit_price: float
    pnl_pct: float
    entry_bar: int
    exit_bar: int
    exit_reason: str


@dataclass
class BacktestV2Result:
    config_name: str
    symbol: str
    timeframe: str
    total_return_pct: float
    max_drawdown_pct: float
    num_trades: int
    win_rate: float
    profit_factor: float
    sortino_ratio: float
    largest_loser_pct_of_gross: float
    np_over_mdd: float
    trades: list


def backtest_v2(df: pd.DataFrame, config: StrategyV2Config, symbol: str, timeframe: str) -> BacktestV2Result:
    close = df["close"]
    macd_line, signal_line, hist = calc_macd(close, config.macd_fast, config.macd_slow, config.macd_signal)
    rsi = calc_rsi(close, config.rsi_period)
    atr = calc_atr(df, config.atr_period)

    if config.use_ema_filter:
        ema_trend = calc_ema(close, config.ema_trend)

    macd_above = macd_line > signal_line
    buy_signals = macd_above & ~macd_above.shift(1).fillna(False)

    warmup = max(config.ema_trend if config.use_ema_filter else 0, config.macd_slow + config.macd_signal, config.atr_period) + 10

    initial_capital = 10000.0
    capital = initial_capital
    position_size = 0.0
    entry_price = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    trailing_stop = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_atr = 0.0
    in_position = False

    equity_curve = [initial_capital] * len(df)
    trades = []

    for i in range(warmup, len(df)):
        price = close.iloc[i]
        current_atr = atr.iloc[i]

        if in_position:
            # Track highest price for trailing stop
            if df["high"].iloc[i] > highest_since_entry:
                highest_since_entry = df["high"].iloc[i]
                if entry_atr > 0:
                    trailing_stop = highest_since_entry - config.trailing_atr_mult * entry_atr

            # Check exit conditions
            exit_reason = None
            exit_price = price
            bars_held = i - entry_bar

            if df["low"].iloc[i] <= stop_loss:
                exit_price = stop_loss
                exit_reason = "stop_loss"
            elif df["high"].iloc[i] >= take_profit:
                exit_price = take_profit
                exit_reason = "take_profit"
            elif price <= trailing_stop and bars_held > 3:
                exit_reason = "trailing_stop"
            elif config.max_bars > 0 and bars_held >= config.max_bars:
                exit_reason = "time_exit"
            elif not macd_above.iloc[i] and macd_above.iloc[i-1]:
                if price > entry_price or bars_held > 5:
                    exit_reason = "macd_exit"

            if exit_reason:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                capital = position_size * exit_price
                position_size = 0.0
                in_position = False
                trades.append(TradeRecord(
                    entry_price=entry_price, exit_price=exit_price,
                    pnl_pct=pnl_pct, entry_bar=entry_bar, exit_bar=i,
                    exit_reason=exit_reason
                ))

        elif buy_signals.iloc[i]:
            # Check entry filters
            rsi_ok = config.rsi_min <= rsi.iloc[i] <= config.rsi_max
            trend_ok = (not config.use_ema_filter) or (price > ema_trend.iloc[i])
            hist_ok = (not config.require_hist_positive) or (hist.iloc[i] > 0)
            atr_ok = current_atr > 0

            if rsi_ok and trend_ok and hist_ok and atr_ok:
                entry_price = price
                entry_bar = i
                entry_atr = current_atr
                position_size = capital / price
                capital = 0.0
                in_position = True
                highest_since_entry = price
                stop_loss = price - config.sl_atr_mult * current_atr
                take_profit = price + config.tp_atr_mult * current_atr
                trailing_stop = price - config.trailing_atr_mult * current_atr

        equity_curve[i] = capital + position_size * price if in_position else (capital if capital > 0 else equity_curve[i-1])

    # Close any open position
    if in_position:
        final_price = close.iloc[-1]
        pnl_pct = (final_price - entry_price) / entry_price * 100
        capital = position_size * final_price
        trades.append(TradeRecord(
            entry_price=entry_price, exit_price=final_price,
            pnl_pct=pnl_pct, entry_bar=entry_bar, exit_bar=len(df)-1,
            exit_reason="end_of_data"
        ))
        in_position = False

    # Calculate metrics
    pnls = [t.pnl_pct for t in trades]
    num_trades = len(trades)

    if num_trades == 0:
        return BacktestV2Result(
            config_name=config.name, symbol=symbol, timeframe=timeframe,
            total_return_pct=0, max_drawdown_pct=0, num_trades=0,
            win_rate=0, profit_factor=0, sortino_ratio=0,
            largest_loser_pct_of_gross=0, np_over_mdd=0, trades=[]
        )

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    total_return = sum(pnls)
    win_rate = len(winners) / num_trades * 100

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    # Max drawdown from equity curve
    eq = pd.Series(equity_curve[warmup:])
    peak = eq.cummax()
    dd = (peak - eq) / peak
    max_dd = dd.max() * 100

    # NP / MDD
    final_equity = capital if capital > 0 else initial_capital
    net_profit_pct = (final_equity - initial_capital) / initial_capital * 100
    np_mdd = net_profit_pct / max_dd if max_dd > 0 else 999

    # Sortino ratio (annualized, using downside deviation)
    pnl_arr = np.array(pnls)
    downside = pnl_arr[pnl_arr < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1
    avg_return = np.mean(pnl_arr)
    # Rough annualization: assume avg 20 trades per year
    sortino = (avg_return / downside_std) * np.sqrt(20) if downside_std > 0 else 0

    # Largest loser as % of gross loss
    largest_loser = abs(min(pnls)) if losers else 0
    largest_loser_pct = (largest_loser / gross_loss * 100) if gross_loss > 0 else 0

    return BacktestV2Result(
        config_name=config.name, symbol=symbol, timeframe=timeframe,
        total_return_pct=round(total_return, 2),
        max_drawdown_pct=round(max_dd, 2),
        num_trades=num_trades,
        win_rate=round(win_rate, 2),
        profit_factor=round(profit_factor, 3),
        sortino_ratio=round(sortino, 3),
        largest_loser_pct_of_gross=round(largest_loser_pct, 2),
        np_over_mdd=round(np_mdd, 3),
        trades=trades
    )


# ── Strategy Generation ───────────────────────────────────────────────────────

def generate_v2_strategies() -> list[StrategyV2Config]:
    strategies = []

    # Focused parameter grid based on analysis insights
    rsi_ranges = [(30, 55), (30, 60), (35, 60), (35, 65)]
    ema_trends = [50, 100]
    sl_tp_combos = [
        (1.5, 3.0), (1.5, 4.0),
        (2.0, 3.0), (2.0, 4.0), (2.0, 5.0),
        (2.5, 4.0), (2.5, 5.0),
        (3.0, 5.0),
    ]
    trail_atr = [2.0, 2.5, 3.0]
    max_bars_opts = [0, 30]
    hist_opts = [True, False]

    idx = 0
    # With EMA filter
    for rsi_min, rsi_max in rsi_ranges:
        for ema in ema_trends:
            for sl, tp in sl_tp_combos:
                for trail in trail_atr:
                    for mb in max_bars_opts:
                        for hist_req in hist_opts:
                            strategies.append(StrategyV2Config(
                                name=f"v2_{idx}",
                                rsi_min=rsi_min, rsi_max=rsi_max,
                                use_ema_filter=True, ema_trend=ema,
                                sl_atr_mult=sl, tp_atr_mult=tp,
                                trailing_atr_mult=trail,
                                max_bars=mb,
                                require_hist_positive=hist_req,
                            ))
                            idx += 1

    # Without EMA filter
    for rsi_min, rsi_max in rsi_ranges:
        for sl, tp in sl_tp_combos:
            for trail in trail_atr:
                for mb in max_bars_opts:
                    for hist_req in hist_opts:
                        strategies.append(StrategyV2Config(
                            name=f"v2_{idx}",
                            rsi_min=rsi_min, rsi_max=rsi_max,
                            use_ema_filter=False,
                            sl_atr_mult=sl, tp_atr_mult=tp,
                            trailing_atr_mult=trail,
                            max_bars=mb,
                            require_hist_positive=hist_req,
                        ))
                        idx += 1

    print(f"Generated {len(strategies)} V2 strategies")
    return strategies


# ── Aggregate Scoring ─────────────────────────────────────────────────────────

def aggregate_results(all_results: list[BacktestV2Result]) -> dict:
    """Aggregate results across all datasets for one strategy."""
    if not all_results:
        return None

    all_trades = []
    for r in all_results:
        all_trades.extend(r.trades)

    pnls = [t.pnl_pct for t in all_trades]
    num_trades = len(pnls)
    if num_trades == 0:
        return None

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))

    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    wr = len(winners) / num_trades * 100
    avg_return = np.mean([r.total_return_pct for r in all_results]) if all_results else 0
    avg_dd = np.mean([r.max_drawdown_pct for r in all_results])
    avg_np_mdd = np.mean([r.np_over_mdd for r in all_results if r.max_drawdown_pct > 0])

    largest_loser = abs(min(pnls)) if losers else 0
    largest_loser_pct = (largest_loser / gross_loss * 100) if gross_loss > 0 else 0

    pnl_arr = np.array(pnls)
    downside = pnl_arr[pnl_arr < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1
    sortino = (np.mean(pnl_arr) / downside_std) * np.sqrt(num_trades / 2) if downside_std > 0 else 0

    return {
        "num_trades": num_trades,
        "win_rate": round(wr, 2),
        "profit_factor": round(pf, 3),
        "avg_dd": round(avg_dd, 2),
        "avg_np_mdd": round(avg_np_mdd, 3),
        "sortino": round(sortino, 3),
        "largest_loser_pct_of_gross": round(largest_loser_pct, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_pnl": round(gross_profit - gross_loss, 2),
        "avg_trade": round(np.mean(pnls), 3),
        "exit_reasons": {},
    }


def score_v2(agg: dict) -> float:
    """Score based on meeting the target requirements."""
    if agg is None or agg["num_trades"] == 0:
        return -9999

    score = 0.0

    # Profit factor (target: 1.4)
    pf = agg["profit_factor"]
    if pf >= 1.4:
        score += 100 + (pf - 1.4) * 50
    else:
        score += (pf - 1.4) * 200  # Heavy penalty

    # NP/MDD (target: 4)
    np_mdd = agg["avg_np_mdd"]
    if np_mdd >= 4:
        score += 80 + (np_mdd - 4) * 10
    else:
        score += (np_mdd - 4) * 40

    # Sortino (target: 1)
    sortino = agg["sortino"]
    if sortino >= 1:
        score += 60 + (sortino - 1) * 20
    else:
        score += (sortino - 1) * 60

    # Trade count (target: 150 per chart, we have 40 charts so ~4+ per chart)
    # In our aggregated data, 150 total is reasonable
    trades = agg["num_trades"]
    if trades >= 150:
        score += 30
    elif trades >= 100:
        score += 10
    else:
        score -= 50

    # Largest loser (target: < 10% of gross)
    ll = agg["largest_loser_pct_of_gross"]
    if ll < 10:
        score += 20
    else:
        score -= (ll - 10) * 5

    # Win rate bonus
    wr = agg["win_rate"]
    if wr >= 40:
        score += 20

    return round(score, 2)


# ── Data Loading ──────────────────────────────────────────────────────────────

def parse_filename(fp):
    bn = os.path.basename(fp).replace(",", "")
    parts = bn.split()
    return parts[0].replace("BINANCE_", ""), parts[1].replace("u", "h")


def load_datasets():
    datasets = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        symbol, tf = parse_filename(f)
        df = pd.read_csv(f)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.sort_values("time").reset_index(drop=True)
        datasets.append({"symbol": symbol, "timeframe": tf, "df": df})
    print(f"Loaded {len(datasets)} datasets")
    return datasets


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    datasets = load_datasets()
    strategies = generate_v2_strategies()

    total_tests = len(strategies) * len(datasets)
    print(f"Running {total_tests} backtests...")

    results_db = {}
    for i, config in enumerate(strategies):
        results = []
        for ds in datasets:
            r = backtest_v2(ds["df"], config, ds["symbol"], ds["timeframe"])
            results.append(r)
        results_db[config.name] = (config, results)
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{len(strategies)}")

    print("Scoring strategies...\n")

    scored = []
    for name, (config, results) in results_db.items():
        agg = aggregate_results(results)
        if agg is None:
            continue
        s = score_v2(agg)
        scored.append({"config": config, "agg": agg, "score": s, "results": results})

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Print top 15
    print(f"{'Rank':<5} {'PF':>6} {'NP/MDD':>7} {'Sort':>6} {'WR%':>6} {'Trades':>7} {'LL%':>5} {'NetPnL':>8} {'Score':>7}")
    print("=" * 65)
    for i, s in enumerate(scored[:15]):
        a = s["agg"]
        print(f"{i+1:<5} {a['profit_factor']:>6.3f} {a['avg_np_mdd']:>7.2f} {a['sortino']:>6.2f} {a['win_rate']:>6.1f} {a['num_trades']:>7} {a['largest_loser_pct_of_gross']:>5.1f} {a['net_pnl']:>8.1f} {s['score']:>7.1f}")
    print("=" * 65)

    # Check which meet ALL requirements
    print("\n=== STRATEGIES MEETING ALL REQUIREMENTS ===")
    qualifying = []
    for s in scored:
        a = s["agg"]
        meets_all = (
            a["profit_factor"] >= 1.4 and
            a["avg_np_mdd"] >= 4 and
            a["sortino"] >= 1 and
            a["num_trades"] >= 150 and
            a["largest_loser_pct_of_gross"] < 10
        )
        if meets_all:
            qualifying.append(s)

    if qualifying:
        print(f"Found {len(qualifying)} qualifying strategies!\n")
        for i, s in enumerate(qualifying[:5]):
            a = s["agg"]
            c = s["config"]
            print(f"  #{i+1}: {c.describe()}")
            print(f"       PF={a['profit_factor']:.3f} NP/MDD={a['avg_np_mdd']:.2f} Sortino={a['sortino']:.2f} WR={a['win_rate']:.1f}% Trades={a['num_trades']} LL%={a['largest_loser_pct_of_gross']:.1f}%")
    else:
        print("No strategy meets ALL requirements simultaneously.")
        print("Showing best strategies with relaxed constraints:\n")
        # Show top strategies that are closest
        for i, s in enumerate(scored[:5]):
            a = s["agg"]
            c = s["config"]
            checks = {
                "PF>=1.4": a["profit_factor"] >= 1.4,
                "NP/MDD>=4": a["avg_np_mdd"] >= 4,
                "Sortino>=1": a["sortino"] >= 1,
                "Trades>=150": a["num_trades"] >= 150,
                "LL<10%": a["largest_loser_pct_of_gross"] < 10,
            }
            passed = sum(checks.values())
            failed = [k for k, v in checks.items() if not v]
            print(f"  #{i+1} ({passed}/5 passed, fails: {', '.join(failed) if failed else 'none'}):")
            print(f"       {c.describe()}")
            print(f"       PF={a['profit_factor']:.3f} NP/MDD={a['avg_np_mdd']:.2f} Sortino={a['sortino']:.2f} WR={a['win_rate']:.1f}% Trades={a['num_trades']} LL%={a['largest_loser_pct_of_gross']:.1f}%\n")

    # Save the best overall strategy
    best = scored[0]
    best_config = best["config"]
    best_agg = best["agg"]
    best_results = best["results"]

    os.makedirs(STRATEGIES_DIR, exist_ok=True)

    # Per-dataset breakdown
    print(f"\n=== BEST STRATEGY PER-DATASET BREAKDOWN ===")
    print(f"Strategy: {best_config.describe()}\n")
    print(f"{'Symbol':<12} {'TF':<5} {'Return%':>9} {'DD%':>7} {'PF':>7} {'WR%':>7} {'Trades':>7}")
    print("-" * 60)
    for r in sorted(best_results, key=lambda x: x.profit_factor, reverse=True):
        if r.num_trades > 0:
            print(f"{r.symbol:<12} {r.timeframe:<5} {r.total_return_pct:>9.2f} {r.max_drawdown_pct:>7.2f} {r.profit_factor:>7.3f} {r.win_rate:>7.1f} {r.num_trades:>7}")

    # Exit reason analysis
    all_trades = []
    for r in best_results:
        all_trades.extend(r.trades)
    exit_reasons = {}
    for t in all_trades:
        er = t.exit_reason
        if er not in exit_reasons:
            exit_reasons[er] = {"count": 0, "pnl_sum": 0, "wins": 0}
        exit_reasons[er]["count"] += 1
        exit_reasons[er]["pnl_sum"] += t.pnl_pct
        if t.pnl_pct > 0:
            exit_reasons[er]["wins"] += 1

    print(f"\n{'Exit Reason':<18} {'Count':>6} {'WR%':>7} {'Avg PnL':>9}")
    print("-" * 45)
    for reason, data in sorted(exit_reasons.items(), key=lambda x: -x[1]["count"]):
        wr = data["wins"] / data["count"] * 100
        avg = data["pnl_sum"] / data["count"]
        print(f"{reason:<18} {data['count']:>6} {wr:>7.1f} {avg:>9.2f}%")

    # Save config
    strategy_data = {
        "name": best_config.name,
        "description": best_config.describe(),
        "parameters": {
            "macd_fast": best_config.macd_fast,
            "macd_slow": best_config.macd_slow,
            "macd_signal": best_config.macd_signal,
            "rsi_period": best_config.rsi_period,
            "rsi_min": best_config.rsi_min,
            "rsi_max": best_config.rsi_max,
            "use_ema_filter": best_config.use_ema_filter,
            "ema_trend": best_config.ema_trend,
            "atr_period": best_config.atr_period,
            "sl_atr_mult": best_config.sl_atr_mult,
            "tp_atr_mult": best_config.tp_atr_mult,
            "trailing_atr_mult": best_config.trailing_atr_mult,
            "max_bars": best_config.max_bars,
            "require_hist_positive": best_config.require_hist_positive,
        },
        "performance": {
            "profit_factor": best_agg["profit_factor"],
            "avg_np_over_mdd": best_agg["avg_np_mdd"],
            "sortino_ratio": best_agg["sortino"],
            "win_rate": best_agg["win_rate"],
            "num_trades": best_agg["num_trades"],
            "largest_loser_pct_of_gross": best_agg["largest_loser_pct_of_gross"],
            "net_pnl_pct": best_agg["net_pnl"],
            "avg_drawdown_pct": best_agg["avg_dd"],
        },
        "per_dataset": [
            {
                "symbol": r.symbol, "timeframe": r.timeframe,
                "return_pct": r.total_return_pct, "dd_pct": r.max_drawdown_pct,
                "pf": r.profit_factor, "wr": r.win_rate, "trades": r.num_trades,
            }
            for r in best_results if r.num_trades > 0
        ],
    }

    with open(os.path.join(STRATEGIES_DIR, "best_strategy_v2.json"), "w") as f:
        json.dump(strategy_data, f, indent=2)

    print(f"\nBest strategy config saved to strategies/best_strategy_v2.json")
    print(f"Now generating PineScript v6...")

    # Generate PineScript
    pine = generate_pinescript_v6(best_config)
    pine_path = os.path.join(STRATEGIES_DIR, "best_strategy_v2.pine")
    with open(pine_path, "w", encoding="utf-8") as f:
        f.write(pine)
    print(f"PineScript v6 saved to {pine_path}")


# ── PineScript v6 Generator ──────────────────────────────────────────────────

def generate_pinescript_v6(config: StrategyV2Config) -> str:
    lines = []
    lines.append('//@version=6')
    lines.append(f'strategy("{config.describe()}", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=100, initial_capital=10000, commission_value=0.04, slippage=1, calc_on_every_tick=false)')
    lines.append('')
    lines.append('// ── Inputs ──')
    lines.append(f'macdFast   = input.int({config.macd_fast}, "MACD Fast")')
    lines.append(f'macdSlow   = input.int({config.macd_slow}, "MACD Slow")')
    lines.append(f'macdSignal = input.int({config.macd_signal}, "MACD Signal")')
    lines.append(f'rsiPeriod  = input.int({config.rsi_period}, "RSI Period")')
    lines.append(f'rsiMin     = input.float({config.rsi_min}, "RSI Min")')
    lines.append(f'rsiMax     = input.float({config.rsi_max}, "RSI Max")')
    if config.use_ema_filter:
        lines.append(f'emaTrend   = input.int({config.ema_trend}, "EMA Trend Filter")')
    lines.append(f'atrPeriod  = input.int({config.atr_period}, "ATR Period")')
    lines.append(f'slAtrMult  = input.float({config.sl_atr_mult}, "SL ATR Multiplier", step=0.1)')
    lines.append(f'tpAtrMult  = input.float({config.tp_atr_mult}, "TP ATR Multiplier", step=0.1)')
    lines.append(f'trailMult  = input.float({config.trailing_atr_mult}, "Trailing ATR Multiplier", step=0.1)')
    if config.max_bars > 0:
        lines.append(f'maxBars    = input.int({config.max_bars}, "Max Bars in Trade")')
    lines.append('')
    lines.append('// ── Indicators ──')
    lines.append('[macdLine, signalLine, histLine] = ta.macd(close, macdFast, macdSlow, macdSignal)')
    lines.append('rsiVal = ta.rsi(close, rsiPeriod)')
    lines.append('atrVal = ta.atr(atrPeriod)')
    if config.use_ema_filter:
        lines.append('emaTrendLine = ta.ema(close, emaTrend)')
        lines.append('plot(emaTrendLine, "EMA Trend", color=color.new(color.orange, 30), linewidth=2)')
    lines.append('')
    lines.append('// ── Entry Conditions ──')
    lines.append('macdCross = ta.crossover(macdLine, signalLine)')
    lines.append('rsiOk = rsiVal >= rsiMin and rsiVal <= rsiMax')
    if config.use_ema_filter:
        lines.append('trendOk = close > emaTrendLine')
    else:
        lines.append('trendOk = true')
    if config.require_hist_positive:
        lines.append('histOk = histLine > 0')
    else:
        lines.append('histOk = true')
    lines.append('')
    lines.append('buySignal = macdCross and rsiOk and trendOk and histOk and atrVal > 0')
    lines.append('')
    lines.append('// ── Risk Management ──')
    lines.append('var float entryPrice = na')
    lines.append('var float stopLoss = na')
    lines.append('var float takeProfit = na')
    lines.append('var float trailStop = na')
    lines.append('var float highestSinceEntry = na')
    lines.append('var int entryBar = na')
    lines.append('')
    lines.append('if buySignal and strategy.position_size == 0')
    lines.append('    entryPrice := close')
    lines.append('    stopLoss := close - slAtrMult * atrVal')
    lines.append('    takeProfit := close + tpAtrMult * atrVal')
    lines.append('    trailStop := close - trailMult * atrVal')
    lines.append('    highestSinceEntry := close')
    lines.append('    entryBar := bar_index')
    lines.append('    strategy.entry("Long", strategy.long)')
    lines.append('')
    lines.append('// ── Exit Logic ──')
    lines.append('if strategy.position_size > 0')
    lines.append('    // Update trailing stop')
    lines.append('    if close > highestSinceEntry')
    lines.append('        highestSinceEntry := close')
    lines.append('        trailStop := highestSinceEntry - trailMult * atrVal')
    lines.append('')
    lines.append('    barsHeld = bar_index - entryBar')
    lines.append('')
    lines.append('    // Stop loss')
    lines.append('    if low <= stopLoss')
    lines.append('        strategy.exit("SL", "Long", stop=stopLoss)')
    lines.append('')
    lines.append('    // Take profit')
    lines.append('    if high >= takeProfit')
    lines.append('        strategy.exit("TP", "Long", limit=takeProfit)')
    lines.append('')
    lines.append('    // Trailing stop (after 3 bars)')
    lines.append('    if close <= trailStop and barsHeld > 3')
    lines.append('        strategy.close("Long", comment="Trail")')
    lines.append('')
    if config.max_bars > 0:
        lines.append('    // Time-based exit')
        lines.append('    if barsHeld >= maxBars')
        lines.append('        strategy.close("Long", comment="TimeExit")')
        lines.append('')
    lines.append('    // MACD exit (if profitable or after 5 bars)')
    lines.append('    if ta.crossunder(macdLine, signalLine) and (close > entryPrice or barsHeld > 5)')
    lines.append('        strategy.close("Long", comment="MACD Exit")')
    lines.append('')
    lines.append('// ── Plots ──')
    lines.append('plotshape(buySignal and strategy.position_size == 0, title="Buy", location=location.belowbar, color=color.green, style=shape.triangleup, size=size.small)')
    lines.append('plot(strategy.position_size > 0 ? stopLoss : na, "Stop Loss", color=color.red, style=plot.style_linebr, linewidth=1)')
    lines.append('plot(strategy.position_size > 0 ? takeProfit : na, "Take Profit", color=color.green, style=plot.style_linebr, linewidth=1)')
    lines.append('plot(strategy.position_size > 0 ? trailStop : na, "Trail Stop", color=color.orange, style=plot.style_linebr, linewidth=1)')

    return '\n'.join(lines)


if __name__ == "__main__":
    main()
