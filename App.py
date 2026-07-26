"""
Flexible Trading Strategy Backtester
-------------------------------------
Run with:
    streamlit run app.py

Requires: streamlit, pandas, numpy, plotly  (see requirements.txt)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from backtest_engine import (
    add_indicators, evaluate_condition_group, run_backtest, compute_metrics,
    INDICATOR_CHOICES, PRICE_CHOICES, OPERATORS,
)

st.set_page_config(page_title="Trading Backtester", layout="wide")
st.title("📈 Flexible Trading Strategy Backtester")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "indicators" not in st.session_state:
    st.session_state.indicators = []       # list of dicts: name/type/period/...
if "entry_conditions" not in st.session_state:
    st.session_state.entry_conditions = []
if "exit_conditions" not in st.session_state:
    st.session_state.exit_conditions = []
if "trade_log" not in st.session_state:
    st.session_state.trade_log = None


def available_refs():
    """Column names usable on either side of a condition: price cols + indicators."""
    return PRICE_CHOICES + [ind["name"] for ind in st.session_state.indicators]


# ---------------------------------------------------------------------------
# SIDEBAR — 1. Data upload
# ---------------------------------------------------------------------------
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Historical price CSV", type=["csv"])

df_raw = None
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

df_mapped = None
if df_raw is not None:
    st.sidebar.subheader("Column Mapping")
    cols = list(df_raw.columns)

    def guess(colnames, keywords):
        for c in colnames:
            if any(k in c.lower() for k in keywords):
                return c
        return colnames[0]

    date_col = st.sidebar.selectbox("Date/Time column", cols,
                                     index=cols.index(guess(cols, ["date", "time"])))
    open_col = st.sidebar.selectbox("Open column", cols,
                                     index=cols.index(guess(cols, ["open"])))
    high_col = st.sidebar.selectbox("High column", cols,
                                     index=cols.index(guess(cols, ["high"])))
    low_col = st.sidebar.selectbox("Low column", cols,
                                    index=cols.index(guess(cols, ["low"])))
    close_col = st.sidebar.selectbox("Close column", cols,
                                      index=cols.index(guess(cols, ["close", "adj"])))

    df_mapped = df_raw.rename(columns={
        date_col: "Date", open_col: "Open", high_col: "High",
        low_col: "Low", close_col: "Close",
    })[["Date", "Open", "High", "Low", "Close"]].copy()

    df_mapped["Date"] = pd.to_datetime(df_mapped["Date"], errors="coerce")
    df_mapped = df_mapped.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for c in ["Open", "High", "Low", "Close"]:
        df_mapped[c] = pd.to_numeric(df_mapped[c], errors="coerce")
    df_mapped = df_mapped.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# SIDEBAR — 2. Strategy Builder: Indicators
# ---------------------------------------------------------------------------
if df_mapped is not None:
    st.sidebar.header("2. Strategy Builder")
    st.sidebar.subheader("Indicators")

    with st.sidebar.expander("➕ Add indicator"):
        ind_type = st.selectbox("Indicator type", INDICATOR_CHOICES, key="new_ind_type")
        needs_period = ind_type not in ("MACD_LINE", "MACD_SIGNAL")
        period = st.number_input("Lookback period", min_value=1, value=14, step=1,
                                  key="new_ind_period") if needs_period else None
        extra = {}
        if ind_type in ("MACD_LINE", "MACD_SIGNAL"):
            extra["fast"] = st.number_input("MACD fast", min_value=1, value=12, key="macd_fast")
            extra["slow"] = st.number_input("MACD slow", min_value=1, value=26, key="macd_slow")
            extra["signal"] = st.number_input("MACD signal", min_value=1, value=9, key="macd_signal")
        if ind_type.startswith("BB_"):
            extra["std"] = st.number_input("Std dev multiplier", min_value=0.1, value=2.0,
                                            step=0.1, key="bb_std")

        default_name = f"{ind_type}_{period}" if period else f"{ind_type}"
        ind_name = st.text_input("Name (used in conditions)", value=default_name, key="new_ind_name")

        if st.button("Add indicator"):
            cfg = {"name": ind_name, "type": ind_type}
            if period:
                cfg["period"] = int(period)
            cfg.update(extra)
            st.session_state.indicators.append(cfg)
            st.rerun()

    if st.session_state.indicators:
        for i, ind in enumerate(st.session_state.indicators):
            c1, c2 = st.sidebar.columns([4, 1])
            c1.write(f"`{ind['name']}` ({ind['type']})")
            if c2.button("✕", key=f"del_ind_{i}"):
                st.session_state.indicators.pop(i)
                st.rerun()
    else:
        st.sidebar.caption("No indicators added yet.")

    # -----------------------------------------------------------------------
    # SIDEBAR — Entry / Exit conditions
    # -----------------------------------------------------------------------
    def condition_builder(label, state_key):
        st.sidebar.subheader(label)
        refs = available_refs()
        with st.sidebar.expander(f"➕ Add {label.lower()}"):
            left = st.selectbox("Left", refs, key=f"{state_key}_left")
            op = st.selectbox("Operator", OPERATORS, key=f"{state_key}_op")
            right_mode = st.radio("Compare to", ["Series/Indicator", "Fixed number"],
                                   key=f"{state_key}_rmode", horizontal=True)
            if right_mode == "Series/Indicator":
                right = st.selectbox("Right", refs, key=f"{state_key}_right_series")
            else:
                right = st.number_input("Value", value=0.0, key=f"{state_key}_right_const")
            if st.button(f"Add to {label}", key=f"{state_key}_addbtn"):
                st.session_state[state_key].append({"left": left, "operator": op, "right": right})
                st.rerun()

        conditions = st.session_state[state_key]
        if conditions:
            for i, c in enumerate(conditions):
                c1, c2 = st.sidebar.columns([4, 1])
                c1.write(f"`{c['left']}` {c['operator']} `{c['right']}`")
                if c2.button("✕", key=f"del_{state_key}_{i}"):
                    st.session_state[state_key].pop(i)
                    st.rerun()
        else:
            st.sidebar.caption(f"No {label.lower()} defined yet. (All added conditions are AND-ed together.)")

    condition_builder("Entry Conditions", "entry_conditions")
    condition_builder("Exit Conditions", "exit_conditions")

    # -----------------------------------------------------------------------
    # SIDEBAR — 3. Risk management
    # -----------------------------------------------------------------------
    st.sidebar.header("3. Risk Management")
    atr_period = st.sidebar.number_input("ATR period", min_value=1, value=14)
    sl_multiplier = st.sidebar.number_input("Stop-loss ATR multiplier", min_value=0.1, value=1.5, step=0.1)
    risk_reward = st.sidebar.number_input("Risk-to-reward ratio", min_value=0.1, value=2.0, step=0.1)
    lot_size = st.sidebar.number_input("Lot size (base quantity)", min_value=0.01, value=1.0, step=0.01)
    max_trades_per_day = st.sidebar.number_input("Max trades per day", min_value=1, value=5, step=1)
    max_daily_loss = st.sidebar.number_input("Max daily loss ($, 0 = no limit)", min_value=0.0, value=0.0, step=10.0)

    st.sidebar.header("4. Dynamic Position Sizing")
    loss_streak_trigger = st.sidebar.number_input(
        "Consecutive losses to trigger reduction", min_value=0, value=3, step=1)
    reduction_pct = st.sidebar.number_input(
        "Size reduction (%)", min_value=0.0, max_value=100.0, value=50.0, step=5.0)
    reduction_duration_trades = st.sidebar.number_input(
        "Reduction duration (number of trades)", min_value=0, value=5, step=1)

    run_clicked = st.sidebar.button("🚀 Run Backtest", type="primary")
else:
    run_clicked = False
    st.info("👈 Upload a CSV file to get started.")

# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------
if df_mapped is not None:
    with st.expander("Preview mapped data", expanded=False):
        st.dataframe(df_mapped.head(20), use_container_width=True)

if run_clicked:
    if not st.session_state.entry_conditions:
        st.error("Add at least one entry condition before running the backtest.")
    else:
        df_ind = add_indicators(df_mapped, st.session_state.indicators)
        trade_log = run_backtest(
            df_ind,
            entry_conditions=st.session_state.entry_conditions,
            exit_conditions=st.session_state.exit_conditions,
            atr_period=atr_period,
            sl_multiplier=sl_multiplier,
            risk_reward=risk_reward,
            lot_size=lot_size,
            max_trades_per_day=max_trades_per_day,
            max_daily_loss=max_daily_loss,
            loss_streak_trigger=loss_streak_trigger,
            reduction_pct=reduction_pct,
            reduction_duration_trades=reduction_duration_trades,
        )
        st.session_state.trade_log = trade_log

trade_log = st.session_state.trade_log

if trade_log is not None:
    st.header("Results")

    if trade_log.empty:
        st.warning("No trades were generated with this configuration. Try loosening your "
                    "entry conditions or check your indicator lookback periods against your data length.")
    else:
        metrics = compute_metrics(trade_log)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Net Profit", f"${metrics['Total Net Profit']:,.2f}")
        c2.metric("Wins / Losses", f"{metrics['Wins']} / {metrics['Losses']}")
        c3.metric("Win Rate", f"{metrics['Win Rate %']:.1f}%")
        c4.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}"
                  if not np.isnan(metrics["Profit Factor"]) else "—")
        c5.metric("Max Drawdown", f"${metrics['Max Drawdown']:,.2f}")

        st.subheader("Charts")
        col1, col2 = st.columns(2)

        with col1:
            fig_dist = px.histogram(trade_log, x="PnL", nbins=30, title="PnL Distribution")
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            tl = trade_log.copy()
            tl["Exit Date"] = pd.to_datetime(tl["Exit Date"])
            tl["Month"] = tl["Exit Date"].dt.to_period("M").astype(str)
            monthly = tl.groupby("Month")["PnL"].sum().reset_index()
            fig_month = px.bar(monthly, x="Month", y="PnL", title="Monthly PnL")
            st.plotly_chart(fig_month, use_container_width=True)

        has_time = pd.to_datetime(tl["Entry Date"]).dt.time.astype(str).ne("00:00:00").any()
        if has_time:
            tl["EntryHour"] = pd.to_datetime(tl["Entry Date"]).dt.hour
            hourly = tl.groupby("EntryHour")["PnL"].mean().reset_index()
            fig_tod = px.bar(hourly, x="EntryHour", y="PnL",
                              title="Avg PnL by Entry Hour (Time-of-Day)")
            st.plotly_chart(fig_tod, use_container_width=True)
        else:
            st.caption("Time-of-day analysis needs a Date/Time column with actual "
                       "intraday timestamps — your data appears to be daily bars only.")

        st.subheader("Trade Log")
        st.dataframe(trade_log, use_container_width=True)
        st.download_button("Download trade log as CSV",
                            trade_log.to_csv(index=False).encode("utf-8"),
                            "trade_log.csv", "text/csv")
  
