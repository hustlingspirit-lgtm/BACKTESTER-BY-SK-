"""
backtest_engine.py

Pure-Python/pandas trading logic for the Streamlit backtester app.
Deliberately kept free of any Streamlit imports so it can be tested
and reused on its own.

Sections:
    1. Indicators
    2. Condition builder / evaluator
    3. Position sizing
    4. Backtest simulation loop
    5. Performance metrics
"""

from __future__ import annotations
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. INDICATORS
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


INDICATOR_CHOICES = ["SMA", "EMA", "RSI", "ATR", "MACD_LINE", "MACD_SIGNAL",
                      "BB_UPPER", "BB_MID", "BB_LOWER"]
PRICE_CHOICES = ["Open", "High", "Low", "Close"]


def build_indicator_column(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    config example:
      {"name": "SMA_20", "type": "SMA", "period": 20}
      {"name": "BB_UPPER_20", "type": "BB_UPPER", "period": 20, "std": 2.0}
      {"name": "MACD_LINE_12_26_9", "type": "MACD_LINE", "fast":12,"slow":26,"signal":9}
    Returns a Series aligned to df.index.
    """
    t = config["type"]
    if t == "SMA":
        return sma(df["Close"], config["period"])
    if t == "EMA":
        return ema(df["Close"], config["period"])
    if t == "RSI":
        return rsi(df["Close"], config["period"])
    if t == "ATR":
        return atr(df, config["period"])
    if t in ("MACD_LINE", "MACD_SIGNAL"):
        line, sig = macd(df["Close"], config.get("fast", 12),
                          config.get("slow", 26), config.get("signal", 9))
        return line if t == "MACD_LINE" else sig
    if t in ("BB_UPPER", "BB_MID", "BB_LOWER"):
        upper, mid, lower = bollinger_bands(df["Close"], config.get("period", 20),
                                             config.get("std", 2.0))
        return {"BB_UPPER": upper, "BB_MID": mid, "BB_LOWER": lower}[t]
    raise ValueError(f"Unknown indicator type: {t}")


def add_indicators(df: pd.DataFrame, indicator_configs: list[dict]) -> pd.DataFrame:
    """Adds one column per indicator config, named by config['name']."""
    out = df.copy()
    for cfg in indicator_configs:
        out[cfg["name"]] = build_indicator_column(out, cfg)
    return out


# ---------------------------------------------------------------------------
# 2. CONDITION BUILDER
# ---------------------------------------------------------------------------

OPERATORS = ["crosses above", "crosses below", ">", "<", ">=", "<="]


def _series_or_constant(df: pd.DataFrame, ref: str | float):
    """ref is either a column name (str present in df.columns) or a number."""
    if isinstance(ref, (int, float)):
        return pd.Series(ref, index=df.index)
    if ref in df.columns:
        return df[ref]
    raise ValueError(f"Unknown series/column reference: {ref}")


def evaluate_condition(df: pd.DataFrame, left: str, operator: str, right) -> pd.Series:
    """
    left: column name (indicator or price)
    operator: one of OPERATORS
    right: column name OR a numeric constant
    Returns boolean Series aligned to df.index.
    """
    a = _series_or_constant(df, left)
    b = _series_or_constant(df, right)

    if operator == ">":
        return a > b
    if operator == "<":
        return a < b
    if operator == ">=":
        return a >= b
    if operator == "<=":
        return a <= b
    if operator == "crosses above":
        return (a > b) & (a.shift(1) <= b.shift(1))
    if operator == "crosses below":
        return (a < b) & (a.shift(1) >= b.shift(1))
    raise ValueError(f"Unknown operator: {operator}")


def evaluate_condition_group(df: pd.DataFrame, conditions: list[dict]) -> pd.Series:
    """
    conditions: list of {"left": ..., "operator": ..., "right": ...}
    All conditions are AND-ed together.
    Returns a boolean Series (False where any indicator is still NaN/warming up).
    """
    if not conditions:
        return pd.Series(False, index=df.index)
    result = pd.Series(True, index=df.index)
    for c in conditions:
        result &= evaluate_condition(df, c["left"], c["operator"], c["right"]).fillna(False)
    return result


# ---------------------------------------------------------------------------
# 3. POSITION SIZING
# ---------------------------------------------------------------------------

class PositionSizer:
    """
    Tracks consecutive losses and applies a temporary quantity reduction
    after `loss_streak_trigger` consecutive losing trades, for
    `reduction_duration_trades` trades, cutting size by `reduction_pct`.
    """

    def __init__(self, base_lot_size: float, loss_streak_trigger: int,
                 reduction_pct: float, reduction_duration_trades: int):
        self.base_lot_size = base_lot_size
        self.loss_streak_trigger = loss_streak_trigger
        self.reduction_pct = reduction_pct
        self.reduction_duration_trades = reduction_duration_trades

        self.consecutive_losses = 0
        self.reduction_trades_remaining = 0

    def current_qty(self) -> float:
        if self.reduction_trades_remaining > 0:
            return self.base_lot_size * (1 - self.reduction_pct / 100.0)
        return self.base_lot_size

    def register_trade_result(self, pnl: float):
        if self.reduction_trades_remaining > 0:
            self.reduction_trades_remaining -= 1

        if pnl < 0:
            self.consecutive_losses += 1
            if (self.loss_streak_trigger > 0 and
                    self.consecutive_losses >= self.loss_streak_trigger):
                self.reduction_trades_remaining = self.reduction_duration_trades
        else:
            self.consecutive_losses = 0


# ---------------------------------------------------------------------------
# 4. BACKTEST SIMULATION
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame,
                  entry_conditions: list[dict],
                  exit_conditions: list[dict],
                  atr_period: int,
                  sl_multiplier: float,
                  risk_reward: float,
                  lot_size: float,
                  max_trades_per_day: int,
                  max_daily_loss: float,
                  loss_streak_trigger: int,
                  reduction_pct: float,
                  reduction_duration_trades: int) -> pd.DataFrame:
    """
    Long-only, one-position-at-a-time bar-by-bar simulation.
    df must contain: Date, Open, High, Low, Close, plus any indicator columns
    already added (see add_indicators) and referenced by entry/exit conditions.
    Fill price = Close of the signal bar. Stops/targets checked intrabar
    using High/Low of the following bars.

    Returns a DataFrame trade log with one row per closed trade.
    """
    df = df.reset_index(drop=True).copy()
    df["ATR_calc"] = atr(df, atr_period)

    entry_signal = evaluate_condition_group(df, entry_conditions)
    exit_signal = evaluate_condition_group(df, exit_conditions) if exit_conditions else \
        pd.Series(False, index=df.index)

    sizer = PositionSizer(lot_size, loss_streak_trigger, reduction_pct,
                           reduction_duration_trades)

    trades = []
    in_position = False
    entry_price = entry_date = stop_price = target_price = qty = None

    current_date = None
    trades_today = 0
    daily_pnl = 0.0
    daily_loss_hit = False

    for i in range(len(df)):
        row = df.iloc[i]
        row_date = pd.to_datetime(row["Date"]).date()

        if row_date != current_date:
            current_date = row_date
            trades_today = 0
            daily_pnl = 0.0
            daily_loss_hit = False

        if in_position:
            exit_price = None
            exit_reason = None

            if row["Low"] <= stop_price:
                exit_price = stop_price
                exit_reason = "Stop Loss"
            elif row["High"] >= target_price:
                exit_price = target_price
                exit_reason = "Take Profit"
            elif bool(exit_signal.iloc[i]):
                exit_price = row["Close"]
                exit_reason = "Signal Exit"

            if exit_price is not None:
                pnl = (exit_price - entry_price) * qty
                trades.append({
                    "Entry Date": entry_date,
                    "Exit Date": row["Date"],
                    "Entry Price": round(entry_price, 4),
                    "Exit Price": round(exit_price, 4),
                    "Qty": round(qty, 4),
                    "PnL": round(pnl, 2),
                    "Exit Reason": exit_reason,
                })
                daily_pnl += pnl
                sizer.register_trade_result(pnl)
                in_position = False
                if max_daily_loss > 0 and daily_pnl <= -abs(max_daily_loss):
                    daily_loss_hit = True

        if (not in_position and not daily_loss_hit
                and trades_today < max_trades_per_day
                and bool(entry_signal.iloc[i])
                and not np.isnan(row["ATR_calc"])):
            entry_price = row["Close"]
            entry_date = row["Date"]
            risk_per_unit = row["ATR_calc"] * sl_multiplier
            if risk_per_unit <= 0 or np.isnan(risk_per_unit):
                continue
            stop_price = entry_price - risk_per_unit
            target_price = entry_price + risk_per_unit * risk_reward
            qty = sizer.current_qty()
            in_position = True
            trades_today += 1

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE METRICS
# ---------------------------------------------------------------------------

def compute_metrics(trade_log: pd.DataFrame) -> dict:
    if trade_log.empty:
        return {
            "Total Net Profit": 0.0, "Total Trades": 0, "Wins": 0, "Losses": 0,
            "Win Rate %": 0.0, "Profit Factor": np.nan, "Max Drawdown": 0.0,
        }

    pnl = trade_log["PnL"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    equity_curve = pnl.cumsum()
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_drawdown = drawdown.min()

    return {
        "Total Net Profit": round(pnl.sum(), 2),
        "Total Trades": len(trade_log),
        "Wins": int((pnl > 0).sum()),
        "Losses": int((pnl < 0).sum()),
        "Win Rate %": round(100 * (pnl > 0).mean(), 2),
        "Profit Factor": round(profit_factor, 2) if not np.isnan(profit_factor) else np.nan,
        "Max Drawdown": round(max_drawdown, 2),
}
  
