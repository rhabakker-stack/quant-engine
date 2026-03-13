"""
Analyze losing trades from the MACD(8/21/5) strategy to find patterns we can filter.
"""
import os, glob
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


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
    ml = calc_ema(s, f) - calc_ema(s, sl)
    sl_line = calc_ema(ml, sig)
    return ml, sl_line, ml - sl_line

def calc_atr(df, p=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(p).mean()


def parse_filename(fp):
    bn = os.path.basename(fp).replace(",", "")
    parts = bn.split()
    symbol = parts[0].replace("BINANCE_", "")
    tf = parts[1].replace("u", "h")
    return symbol, tf


def analyze():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    all_trades = []

    for f in sorted(csv_files):
        symbol, tf = parse_filename(f)
        df = pd.read_csv(f)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.sort_values("time").reset_index(drop=True)

        # Compute indicators
        macd_line, signal_line, hist = calc_macd(df["close"], 8, 21, 5)
        ema200 = calc_ema(df["close"], 200)
        ema50 = calc_ema(df["close"], 50)
        rsi = calc_rsi(df["close"], 14)
        atr = calc_atr(df, 14)

        macd_above = macd_line > signal_line
        buy_signals = macd_above & ~macd_above.shift(1).fillna(False)
        sell_signals = ~macd_above & macd_above.shift(1).fillna(True)

        position = False
        entry_idx = None

        for i in range(200, len(df)):
            if buy_signals.iloc[i] and not position:
                position = True
                entry_idx = i
            elif sell_signals.iloc[i] and position:
                entry_price = df["close"].iloc[entry_idx]
                exit_price = df["close"].iloc[i]
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                duration = i - entry_idx

                # Context at entry
                trade = {
                    "symbol": symbol, "tf": tf,
                    "entry_date": df["time"].iloc[entry_idx],
                    "exit_date": df["time"].iloc[i],
                    "pnl_pct": pnl_pct,
                    "duration_bars": duration,
                    "price_vs_ema200": (df["close"].iloc[entry_idx] / ema200.iloc[entry_idx] - 1) * 100,
                    "price_vs_ema50": (df["close"].iloc[entry_idx] / ema50.iloc[entry_idx] - 1) * 100,
                    "ema50_vs_ema200": (ema50.iloc[entry_idx] / ema200.iloc[entry_idx] - 1) * 100,
                    "rsi_at_entry": rsi.iloc[entry_idx],
                    "hist_at_entry": hist.iloc[entry_idx],
                    "atr_pct": (atr.iloc[entry_idx] / df["close"].iloc[entry_idx]) * 100,
                    "macd_slope": macd_line.iloc[entry_idx] - macd_line.iloc[max(0, entry_idx-3)],
                    # Max adverse excursion
                    "max_adverse_pct": ((df["low"].iloc[entry_idx:i+1].min() - entry_price) / entry_price) * 100,
                    # Max favorable excursion
                    "max_favorable_pct": ((df["high"].iloc[entry_idx:i+1].max() - entry_price) / entry_price) * 100,
                }
                all_trades.append(trade)
                position = False

    trades_df = pd.DataFrame(all_trades)
    winners = trades_df[trades_df["pnl_pct"] > 0]
    losers = trades_df[trades_df["pnl_pct"] <= 0]

    print(f"Total trades: {len(trades_df)}")
    print(f"Winners: {len(winners)} ({len(winners)/len(trades_df)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(trades_df)*100:.1f}%)")
    print(f"Avg win: {winners['pnl_pct'].mean():.2f}%  Avg loss: {losers['pnl_pct'].mean():.2f}%")
    print(f"Median win: {winners['pnl_pct'].median():.2f}%  Median loss: {losers['pnl_pct'].median():.2f}%")
    print(f"Largest winner: {winners['pnl_pct'].max():.2f}%  Largest loser: {losers['pnl_pct'].min():.2f}%")

    print("\n=== PATTERN ANALYSIS: Winners vs Losers ===\n")

    for col in ["price_vs_ema200", "price_vs_ema50", "ema50_vs_ema200", "rsi_at_entry",
                 "atr_pct", "duration_bars", "max_adverse_pct", "max_favorable_pct"]:
        w_mean = winners[col].mean()
        l_mean = losers[col].mean()
        print(f"{col:>22s}:  Winners={w_mean:>8.2f}  Losers={l_mean:>8.2f}  Diff={w_mean-l_mean:>8.2f}")

    print("\n=== TREND FILTER ANALYSIS (Price vs EMA200) ===\n")
    above_200 = trades_df[trades_df["price_vs_ema200"] > 0]
    below_200 = trades_df[trades_df["price_vs_ema200"] <= 0]
    if len(above_200) > 0:
        wr_above = (above_200["pnl_pct"] > 0).sum() / len(above_200) * 100
        print(f"Above EMA200: {len(above_200)} trades, WR={wr_above:.1f}%, Avg PnL={above_200['pnl_pct'].mean():.2f}%")
    if len(below_200) > 0:
        wr_below = (below_200["pnl_pct"] > 0).sum() / len(below_200) * 100
        print(f"Below EMA200: {len(below_200)} trades, WR={wr_below:.1f}%, Avg PnL={below_200['pnl_pct'].mean():.2f}%")

    print("\n=== EMA50 vs EMA200 (Trend Alignment) ===\n")
    trend_up = trades_df[trades_df["ema50_vs_ema200"] > 0]
    trend_down = trades_df[trades_df["ema50_vs_ema200"] <= 0]
    if len(trend_up) > 0:
        wr = (trend_up["pnl_pct"] > 0).sum() / len(trend_up) * 100
        print(f"EMA50 > EMA200 (uptrend): {len(trend_up)} trades, WR={wr:.1f}%, Avg={trend_up['pnl_pct'].mean():.2f}%")
    if len(trend_down) > 0:
        wr = (trend_down["pnl_pct"] > 0).sum() / len(trend_down) * 100
        print(f"EMA50 < EMA200 (downtrend): {len(trend_down)} trades, WR={wr:.1f}%, Avg={trend_down['pnl_pct'].mean():.2f}%")

    print("\n=== RSI FILTER ANALYSIS ===\n")
    for lo, hi, label in [(0,30,"RSI<30"), (30,50,"RSI 30-50"), (50,70,"RSI 50-70"), (70,100,"RSI>70")]:
        subset = trades_df[(trades_df["rsi_at_entry"] >= lo) & (trades_df["rsi_at_entry"] < hi)]
        if len(subset) > 0:
            wr = (subset["pnl_pct"] > 0).sum() / len(subset) * 100
            print(f"{label}: {len(subset)} trades, WR={wr:.1f}%, Avg PnL={subset['pnl_pct'].mean():.2f}%")

    print("\n=== VOLATILITY (ATR%) ANALYSIS ===\n")
    for lo, hi, label in [(0,1,"Low vol <1%"), (1,2,"Med vol 1-2%"), (2,4,"High vol 2-4%"), (4,100,"Very high >4%")]:
        subset = trades_df[(trades_df["atr_pct"] >= lo) & (trades_df["atr_pct"] < hi)]
        if len(subset) > 0:
            wr = (subset["pnl_pct"] > 0).sum() / len(subset) * 100
            print(f"{label}: {len(subset)} trades, WR={wr:.1f}%, Avg PnL={subset['pnl_pct'].mean():.2f}%")

    print("\n=== STOP LOSS ANALYSIS (Max Adverse Excursion) ===\n")
    for sl_pct in [1, 2, 3, 4, 5, 7, 10]:
        # Trades that would have been stopped out
        stopped = trades_df[trades_df["max_adverse_pct"] <= -sl_pct]
        not_stopped = trades_df[trades_df["max_adverse_pct"] > -sl_pct]
        # Of stopped trades, how many would have ended as winners?
        stopped_winners = (stopped["pnl_pct"] > 0).sum()
        # Simulated PnL with stop
        sim_pnl = []
        for _, t in trades_df.iterrows():
            if t["max_adverse_pct"] <= -sl_pct:
                sim_pnl.append(-sl_pct)
            else:
                sim_pnl.append(t["pnl_pct"])
        avg_pnl = np.mean(sim_pnl)
        total_pnl = np.sum(sim_pnl)
        print(f"SL={sl_pct}%: Stopped={len(stopped)}, Saved_winners={stopped_winners}, Avg PnL={avg_pnl:.2f}%, Total={total_pnl:.1f}%")

    print("\n=== TAKE PROFIT ANALYSIS (Max Favorable Excursion) ===\n")
    for tp_pct in [2, 3, 5, 7, 10, 15]:
        sim_pnl = []
        tp_hit = 0
        for _, t in trades_df.iterrows():
            if t["max_favorable_pct"] >= tp_pct:
                sim_pnl.append(tp_pct)
                tp_hit += 1
            else:
                sim_pnl.append(t["pnl_pct"])
        avg_pnl = np.mean(sim_pnl)
        total_pnl = np.sum(sim_pnl)
        print(f"TP={tp_pct}%: Hit={tp_hit}/{len(trades_df)}, Avg PnL={avg_pnl:.2f}%, Total={total_pnl:.1f}%")

    print("\n=== COMBINED SL + TP SIMULATION ===\n")
    best_combo = None
    best_pf = 0
    for sl in [2, 3, 4, 5]:
        for tp in [3, 5, 7, 10]:
            gross_profit = 0
            gross_loss = 0
            wins = 0
            trades_count = 0
            for _, t in trades_df.iterrows():
                trades_count += 1
                # Check SL first (conservative)
                if t["max_adverse_pct"] <= -sl:
                    gross_loss += sl
                elif t["max_favorable_pct"] >= tp:
                    gross_profit += tp
                    wins += 1
                else:
                    if t["pnl_pct"] > 0:
                        gross_profit += t["pnl_pct"]
                        wins += 1
                    else:
                        gross_loss += abs(t["pnl_pct"])
            pf = gross_profit / gross_loss if gross_loss > 0 else 0
            wr = wins / trades_count * 100 if trades_count > 0 else 0
            np_mdd_approx = (gross_profit - gross_loss) / (sl * 5)  # rough estimate
            if pf > best_pf:
                best_pf = pf
                best_combo = (sl, tp)
            print(f"SL={sl}% TP={tp}%: PF={pf:.3f}, WR={wr:.1f}%, Net={gross_profit-gross_loss:.1f}%")

    print(f"\nBest combo: SL={best_combo[0]}% TP={best_combo[1]}% with PF={best_pf:.3f}")


if __name__ == "__main__":
    analyze()
