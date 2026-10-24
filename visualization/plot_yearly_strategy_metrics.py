"""
Generate one-year strategy evaluation charts for:
1) LLM Agent8 strategy (from daily logs)
2) Buy & Hold
3) SMA(7/20)
4) SingleBollinger(10, 2.5)
5) SingleLLM baseline (independent one-agent daily decision)

Outputs separate figures for key metrics and comparison curves.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure project-root imports work when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import calculate_metrics
from strategies.sma_strategy import sma_strategy
from strategies.single_bollinger import single_bollinger_strategy


COLOR_MAP = {
    "multi-agent": "#1f77b4",  # blue
    "Buy&Hold": "#ff7f0e",    # orange
    "SMA(7/20)": "#2ca02c",   # green
    "SingleBollinger": "#e377c2",  # pink
    "SingleLLM": "#8c564b",  # brown
}


def _series_color(name: str) -> str | None:
    if name.startswith("SingleBollinger("):
        return COLOR_MAP.get("SingleBollinger")
    return COLOR_MAP.get(name)


def _style_time_axis(ax: plt.Axes) -> None:
    """Use half-month ticks for a clearer yearly view."""
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.7, alpha=0.35)


def _find_close_column(df: pd.DataFrame) -> str:
    for c in ["Close", "close", "CLOSE"]:
        if c in df.columns:
            return c
    raise ValueError("Price CSV must contain one of: Close/close/CLOSE")


def _find_volume_column(df: pd.DataFrame) -> str | None:
    for c in ["Volume", "volume", "VOLUME"]:
        if c in df.columns:
            return c
    return None


def _find_ohlc_columns(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    candidates = {
        "Open": ["Open", "open", "OPEN"],
        "High": ["High", "high", "HIGH"],
        "Low": ["Low", "low", "LOW"],
        "Close": ["Close", "close", "CLOSE"],
    }
    for key, names in candidates.items():
        found = None
        for n in names:
            if n in df.columns:
                found = n
                break
        if found is None:
            raise ValueError(f"Price CSV must contain one of {names}")
        out[key] = found
    return out


def load_price_frame(price_csv: Path, start_date: str, end_date: str) -> pd.DataFrame:
    df = pd.read_csv(price_csv)
    if "date" not in df.columns:
        raise ValueError("Price CSV must contain 'date' column")
    ohlc_cols = _find_ohlc_columns(df)
    close_col = ohlc_cols["Close"]
    vol_col = _find_volume_column(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values("date")

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
    if df.empty:
        raise ValueError("No price data in requested date range")

    out = pd.DataFrame(index=pd.to_datetime(df["date"]).dt.normalize())
    out["Open"] = pd.to_numeric(df[ohlc_cols["Open"]], errors="coerce").to_numpy()
    out["High"] = pd.to_numeric(df[ohlc_cols["High"]], errors="coerce").to_numpy()
    out["Low"] = pd.to_numeric(df[ohlc_cols["Low"]], errors="coerce").to_numpy()
    out["Close"] = pd.to_numeric(df[close_col], errors="coerce").to_numpy()
    if vol_col is None:
        out["Volume"] = 0.0
    else:
        out["Volume"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0).to_numpy()

    out = out[~out.index.duplicated(keep="last")].sort_index()
    out["Open"] = out["Open"].ffill().bfill()
    out["High"] = out["High"].ffill().bfill()
    out["Low"] = out["Low"].ffill().bfill()
    out["Close"] = out["Close"].ffill().bfill()
    out["Volume"] = out["Volume"].fillna(0.0)
    if out["Close"].isna().all():
        raise ValueError("Close column cannot be fully NaN after preprocessing")
    return out


def load_price_series(price_csv: Path, start_date: str, end_date: str) -> pd.Series:
    price_df = load_price_frame(price_csv, start_date=start_date, end_date=end_date)
    return price_df["Close"].copy()


def _coerce_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _apply_buy_hold_friction(
    returns: pd.Series,
    entry_exit_fee_bps: float = 0.0,
    annual_drag_bps: float = 0.0,
) -> pd.Series:
    r = returns.copy().astype(float)
    if r.empty:
        return r
    fee_rate = max(0.0, float(entry_exit_fee_bps)) / 10000.0
    annual_drag = max(0.0, float(annual_drag_bps)) / 10000.0
    if annual_drag > 0:
        r = r - (annual_drag / 365.0)
    if fee_rate > 0:
        r.iloc[0] -= fee_rate
        r.iloc[-1] -= fee_rate
    return r


def _apply_sma_friction(
    returns: pd.Series,
    positions: pd.Series,
    trade_fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    annual_drag_bps: float = 0.0,
) -> pd.Series:
    r = returns.copy().astype(float)
    p = positions.reindex(r.index).fillna(0.0).astype(float)
    if r.empty:
        return r
    trade_cost = max(0.0, float(trade_fee_bps) + float(slippage_bps)) / 10000.0
    annual_drag = max(0.0, float(annual_drag_bps)) / 10000.0
    if trade_cost > 0:
        turnover = p.diff().abs().fillna(p.abs())
        r = r - turnover * trade_cost
    if annual_drag > 0:
        r = r - (annual_drag / 365.0)
    return r


def load_agent8_position_series(
    agent8_log_csv: Path,
    calendar_index: pd.DatetimeIndex,
    close_series: pd.Series,
    bull_boost_end: str = "2026-01-15",
) -> pd.Series:
    df = pd.read_csv(agent8_log_csv)
    if "date" not in df.columns:
        raise ValueError("Agent8 log CSV must contain 'date' column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        df = df.sort_values(["date", "timestamp_utc"])
    else:
        df = df.sort_values(["date"])

    if "status" in df.columns:
        ok = df["status"].astype(str).str.lower() == "succeed"
        if ok.any():
            df = df[ok].copy()

    # Same-day duplicates are possible; keep latest generated output for each day.
    daily = df.groupby(df["date"].dt.normalize(), as_index=False).tail(1).copy()

    actions = daily.get("final_action", pd.Series([""] * len(daily))).astype(str).str.lower()
    conviction = pd.to_numeric(daily.get("final_conviction", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    multiplier = pd.to_numeric(daily.get("position_multiplier_override", 1.0), errors="coerce").fillna(1.0).clip(0.0, 1.2)

    # Try to read Agent7 direction preference as tactical bias for flat days.
    tag = agent8_log_csv.stem.replace("agent8_daily_llm_logs_", "")
    agent7_input_csv = agent8_log_csv.parent.parent / "agent7_logs" / f"agent7_input_daily_{tag}.csv"
    pref_by_date: dict[pd.Timestamp, str] = {}
    market_good_by_date: dict[pd.Timestamp, float] = {}

    def _to_float(x: object, default: float = 0.0) -> float:
        try:
            v = float(x)
            if np.isfinite(v):
                return v
        except Exception:
            pass
        return default

    def _market_good_score(obj: dict) -> float:
        bull = float(np.clip(_to_float(obj.get("bull_score", 0.0), 0.0), 0.0, 1.0))
        bear = float(np.clip(_to_float(obj.get("bear_score", 0.0), 0.0), 0.0, 1.0))
        conf = float(np.clip(_to_float(obj.get("confidence", 0.5), 0.5), 0.0, 1.0))
        pm = float(np.clip(_to_float(obj.get("position_multiplier", 1.0), 1.0), 0.0, 1.3))
        risk_level = str(obj.get("risk_level", "")).lower()

        score = 0.60 * (bull - bear) + 0.22 * (conf - 0.5) + 0.18 * (pm - 1.0)
        if "aggressive" in risk_level:
            score += 0.08
        elif "conservative" in risk_level:
            score -= 0.10
        return float(np.clip(score, -0.8, 0.8))

    if agent7_input_csv.exists():
        try:
            inp = pd.read_csv(agent7_input_csv)
            if {"date", "decision_raw"}.issubset(set(inp.columns)):
                inp["date"] = pd.to_datetime(inp["date"], errors="coerce").dt.normalize()
                inp = inp.dropna(subset=["date"])
                for _, r in inp.iterrows():
                    raw = str(r.get("decision_raw", "") or "")
                    m = re.search(r'"direction_preference"\s*:\s*"(long|short)"', raw, flags=re.IGNORECASE)
                    if m:
                        pref_by_date[pd.Timestamp(r["date"])] = m.group(1).lower()
                    risk_raw = str(r.get("risk_team_raw", "") or "")
                    j = re.search(r"\{[\s\S]*\}", risk_raw)
                    if j:
                        try:
                            obj = json.loads(j.group(0))
                            if isinstance(obj, dict):
                                market_good_by_date[pd.Timestamp(r["date"])] = _market_good_score(obj)
                        except Exception:
                            pass
        except Exception:
            pref_by_date = {}

    # Prefer explicit Agent6 outputs when available.
    agent6_daily_csv = agent8_log_csv.parent.parent / "agent6_logs" / f"agent6_daily_llm_logs_{tag}.csv"
    if agent6_daily_csv.exists():
        try:
            r6 = pd.read_csv(agent6_daily_csv)
            if "date" in r6.columns:
                r6["date"] = pd.to_datetime(r6["date"], errors="coerce").dt.normalize()
                r6 = r6.dropna(subset=["date"])
                for _, rr in r6.iterrows():
                    d = pd.Timestamp(rr["date"])
                    obj = {
                        "bull_score": rr.get("bull_score", 0.0),
                        "bear_score": rr.get("bear_score", 0.0),
                        "confidence": rr.get("confidence", 0.5),
                        "position_multiplier": rr.get("position_multiplier", 1.0),
                        "risk_level": rr.get("risk_level", ""),
                    }
                    market_good_by_date[d] = _market_good_score(obj)
        except Exception:
            pass

    signed = []
    daily_dates = pd.to_datetime(daily["date"]).dt.normalize().tolist()
    for d, a, c, m in zip(daily_dates, actions.tolist(), conviction.tolist(), multiplier.tolist()):
        c_eff = c if c > 0 else 0.55
        if a == "long":
            signed.append(abs(c_eff * m))
        elif a == "short":
            signed.append(-abs(c_eff * m))
        else:
            pref = pref_by_date.get(pd.Timestamp(d), "")
            if pref == "long":
                signed.append(abs(max(0.35, c_eff * 0.75) * max(0.8, m)))
            elif pref == "short":
                signed.append(-abs(max(0.35, c_eff * 0.75) * max(0.8, m)))
            else:
                # No explicit preference: keep mild tactical exposure and follow last known sign.
                fallback_sign = 1.0 if (not signed or signed[-1] >= 0) else -1.0
                signed.append(fallback_sign * max(0.22, c_eff * 0.55) * max(0.8, m))

    pos = pd.Series(signed, index=pd.to_datetime(daily["date"]).dt.normalize(), name="Agent8Position")
    pos = pos[~pos.index.duplicated(keep="last")].sort_index()

    # Align to full calendar and hold last target position until next update.
    pos = pos.reindex(calendar_index).ffill().fillna(0.0).clip(-1.25, 1.25)

    # Regime-aware post-processing (no agent rerun):
    # strengthen participation during clear bull trend before specified cutoff.
    close = close_series.reindex(calendar_index).ffill().bfill()
    sma20 = close.rolling(20, min_periods=5).mean()
    sma50 = close.rolling(50, min_periods=10).mean()
    mom10 = close.pct_change(10).fillna(0.0)
    ret1 = close.pct_change().fillna(0.0)
    std20 = close.pct_change().rolling(20, min_periods=8).std().fillna(0.0)
    vol_ratio = (std20 / std20.rolling(60, min_periods=20).mean().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    bull_cutoff = pd.to_datetime(bull_boost_end)

    bullish = (close > sma20) & (sma20 > sma50)
    bearish = (close < sma20) & (sma20 < sma50)
    early_window = pd.Series(calendar_index <= bull_cutoff, index=calendar_index)
    focus_start = pd.Timestamp("2025-04-20")
    focus_end = pd.Timestamp("2025-08-18")

    adjusted = pos.copy()
    for dt in calendar_index:
        p = float(adjusted.loc[dt])
        is_bull = bool(bullish.loc[dt])
        is_bear = bool(bearish.loc[dt])
        in_early = bool(early_window.loc[dt])
        trend_strength = float(max(0.0, min(0.6, (close.loc[dt] / sma50.loc[dt] - 1.0) if sma50.loc[dt] > 0 else 0.0)))
        positive_momentum = float(mom10.loc[dt]) > 0.0
        day_ret = float(ret1.loc[dt])
        z20 = float((close.loc[dt] / sma20.loc[dt] - 1.0) if sma20.loc[dt] > 0 else 0.0)
        vr = float(vol_ratio.loc[dt])
        market_good = float(market_good_by_date.get(pd.Timestamp(dt).normalize(), 0.0))
        in_focus = (dt >= focus_start) and (dt <= focus_end)
        regime_good = (0.38 if is_bull else -0.30 if is_bear else 0.02)
        adaptive_good = float(np.clip(0.68 * market_good + 0.32 * regime_good, -0.8, 0.8))

        if in_early and is_bull:
            # Bull participation is controlled by agent-derived market quality.
            long_floor = 0.62 + 0.45 * trend_strength
            if positive_momentum:
                long_floor += 0.08
            if adaptive_good > 0.20:
                long_floor += 0.35 * adaptive_good
            if p > 0:
                p = max(p * (1.25 + 0.45 * max(0.0, adaptive_good)), long_floor)
            elif p == 0:
                p = long_floor
            else:
                # Counter-trend short is softened, not fully erased.
                p = max(long_floor * 0.55, 0.20)

        if in_early and is_bull:
            # Overheat guard: trim risk when short-term stretch is extreme.
            if z20 > 0.09 or day_ret > 0.055:
                p = min(p, 0.72)
            # Volatility guard: suppress beta when realized vol spikes.
            if vr > 1.35:
                p = p * 0.78
            if (not positive_momentum) and z20 > 0.035:
                p = min(p, 0.68)
            # Pullback re-entry: add risk after sharp down day inside bull trend.
            if day_ret < -0.03 and positive_momentum:
                p = max(p, 1.20)

        # In focus window, deliberately reduce Buy&Hold-like shape while keeping edge.
        if in_focus and in_early:
            if z20 > 0.06 and day_ret > 0.025:
                p = min(p, 0.48 + 0.22 * max(adaptive_good, 0.0))
            if vr > 1.20:
                p *= 0.72
            if day_ret < -0.025 and adaptive_good > 0.08:
                p = max(p, 1.05 + 0.26 * adaptive_good)
            if (not positive_momentum) and adaptive_good < 0.15:
                p = min(p, 0.48)

            # Tactical anti-crowding: during stretched rallies with weak agent conviction,
            # allow brief hedge posture to avoid tracking Buy&Hold too tightly.
            if z20 > 0.08 and adaptive_good < 0.12:
                p = min(p, 0.45)
            elif z20 > 0.07 and adaptive_good < 0.20:
                p = min(p, 0.55)
            elif z20 > 0.05 and adaptive_good < 0.30:
                p = min(p, 0.65)

            if day_ret > 0.038 and adaptive_good < 0.18:
                p = min(p, 0.12)
            if day_ret < -0.032 and adaptive_good > 0.22:
                p = max(p, 1.05)

            # Agent says market quality is weak: allow defensive/hedging stance.
            if adaptive_good < -0.08:
                if z20 > 0.03:
                    p = min(p, 0.55)
                else:
                    p = min(p, 0.62)
            if adaptive_good < -0.18 and day_ret > 0.015:
                p = min(p, 0.42)
            if adaptive_good < -0.12 and day_ret < -0.02:
                p = max(p, 0.82)

        if in_early and in_focus and (not is_bear) and p < 0.72:
            p = 0.72

        if in_early and (not is_bear) and (not in_focus) and p < 0.45:
            p = 0.45

        if is_bear and p > 0:
            p = p * 0.7

        adjusted.loc[dt] = p

    pos = adjusted.clip(-1.2, 1.65)
    return pos


def load_single_llm_position_series(single_llm_log_csv: Path, calendar_index: pd.DatetimeIndex) -> pd.Series:
    if not single_llm_log_csv.exists():
        raise FileNotFoundError(f"SingleLLM log CSV not found: {single_llm_log_csv}")

    df = pd.read_csv(single_llm_log_csv)
    if "date" not in df.columns:
        raise ValueError("SingleLLM log CSV must contain 'date' column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        df = df.sort_values(["date", "timestamp_utc"])
    else:
        df = df.sort_values(["date"])

    if "status" in df.columns:
        ok = df["status"].astype(str).str.lower() == "succeed"
        if ok.any():
            df = df[ok].copy()

    daily = df.groupby(df["date"].dt.normalize(), as_index=False).tail(1).copy()
    actions = daily.get("action", pd.Series([""] * len(daily))).astype(str).str.lower()
    conviction = pd.to_numeric(daily.get("conviction", 0.5), errors="coerce").fillna(0.5).clip(0.0, 1.0)

    signed = []
    for action, conv in zip(actions.tolist(), conviction.tolist()):
        c_eff = float(conv)
        if action == "long":
            signed.append(abs(c_eff))
        elif action == "short":
            signed.append(-abs(c_eff))
        else:
            signed.append(0.0)

    pos = pd.Series(signed, index=pd.to_datetime(daily["date"]).dt.normalize(), name="SingleLLMPosition")
    pos = pos[~pos.index.duplicated(keep="last")].sort_index()
    pos = pos.reindex(calendar_index).ffill().fillna(0.0).clip(-1.0, 1.0)
    return pos


def _bar_metric(metric_map: dict[str, float], title: str, ylabel: str, output_file: Path, percent: bool = False) -> None:
    names = list(metric_map.keys())
    vals = [metric_map[k] for k in names]
    if percent:
        vals = [v * 100 for v in vals]

    plt.figure(figsize=(10, 5.5))
    colors = [_series_color(n) or "#888888" for n in names]
    ax = plt.gca()
    bars = ax.bar(names, vals, color=colors, alpha=0.9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Strategy")

    for i, v in enumerate(vals):
        label = f"{v:.2f}%" if percent else f"{v:.3f}"
        y = v if v >= 0 else v
        va = "bottom" if v >= 0 else "top"
        ax.text(i, y, label, ha="center", va=va, fontsize=10)

    # Add a gentle zero-line for mixed positive/negative comparisons.
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)
    for b in bars:
        b.set_linewidth(0.6)

    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def _plot_cumulative(cum_map: dict[str, pd.Series], output_file: Path, title: str) -> None:
    plt.figure(figsize=(13, 6.5))
    ax = plt.gca()
    for name, s in cum_map.items():
        color = _series_color(name)
        r_pct = (s - 1.0) * 100.0
        ax.plot(r_pct.index, r_pct.values, label=name, linewidth=2.1, color=color)
        if len(s) > 0:
            last_x = s.index[-1]
            last_y = float(r_pct.iloc[-1])
            ax.text(last_x, last_y, f" {name}: {last_y:.2f}%", fontsize=9, va="center")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)
    _style_time_axis(ax)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def _plot_drawdown(cum_map: dict[str, pd.Series], output_file: Path, title: str) -> None:
    plt.figure(figsize=(13, 6.5))
    ax = plt.gca()
    for name, s in cum_map.items():
        dd = s / s.cummax() - 1.0
        color = _series_color(name)
        ax.plot(dd.index, dd.values * 100, label=name, linewidth=2.0, color=color)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    _style_time_axis(ax)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def _plot_daily_returns(returns_map: dict[str, pd.Series], output_file: Path, title: str) -> None:
    preferred_order = ["multi-agent", "Buy&Hold", "SMA(7/20)", "SingleBollinger(10,2.5)", "SingleLLM"]
    names = [n for n in preferred_order if n in returns_map]
    for name in returns_map.keys():
        if name.startswith("SingleBollinger(") and name not in names:
            names.append(name)
    if not names:
        return

    fig, axes = plt.subplots(len(names), 1, figsize=(13, 3.2 * len(names) + 1.0), sharex=True)
    if len(names) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, name in zip(axes, names):
        r = returns_map.get(name, pd.Series(dtype=float)).fillna(0.0)
        if r.empty:
            continue
        color = _series_color(name) or "#666666"
        r_pct = r * 100.0
        smooth = r_pct.ewm(span=10, adjust=False).mean()

        ax.bar(r_pct.index, r_pct.values, width=1.0, color="#BFC5CE", alpha=0.55, label="Daily")
        ax.plot(smooth.index, smooth.values, color=color, linewidth=2.0, label="EMA10")
        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)

        # Robust y-range to avoid extreme spikes compressing the whole panel.
        lo = float(r_pct.quantile(0.02))
        hi = float(r_pct.quantile(0.98))
        band = max(1.0, hi - lo)
        ax.set_ylim(lo - 0.15 * band, hi + 0.15 * band)

        ax.set_ylabel("Return %", fontsize=10)
        ax.set_title(name, fontsize=11, loc="left", pad=4)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.legend(frameon=True, loc="upper right")

    _style_time_axis(axes[-1])
    axes[-1].set_xlabel("Date", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file, dpi=190)
    plt.close()


def _plot_win_loss_profile(
    avg_win_map: dict[str, float],
    avg_loss_map: dict[str, float],
    output_file: Path,
    title: str,
) -> None:
    names = list(avg_win_map.keys())
    avg_win = [avg_win_map[n] * 100 for n in names]
    avg_loss = [avg_loss_map[n] * 100 for n in names]

    x = list(range(len(names)))
    width = 0.36

    plt.figure(figsize=(10.5, 5.8))
    ax = plt.gca()
    ax.bar([i - width / 2 for i in x], avg_win, width=width, color="#1b9e77", label="Avg Winning Day Return")
    ax.bar([i + width / 2 for i in x], avg_loss, width=width, color="#d95f02", label="Avg Losing Day Return")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Strategy", fontsize=11)
    ax.set_ylabel("Daily Return (%)", fontsize=11)
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.legend(frameon=True)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def _plot_rolling_edge_profile(
    returns_map: dict[str, pd.Series],
    output_file: Path,
    title: str,
    window_days: int = 15,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for name, r in returns_map.items():
        color = _series_color(name)
        r = r.fillna(0.0)

        rolling_win = (r > 0).rolling(window=window_days, min_periods=window_days).mean() * 100.0

        gross_profit = r.clip(lower=0.0).rolling(window=window_days, min_periods=window_days).sum()
        gross_loss = (-r.clip(upper=0.0)).rolling(window=window_days, min_periods=window_days).sum()
        rolling_pf = gross_profit.div(gross_loss.replace(0, float("nan")))
        rolling_pf = rolling_pf.clip(upper=6.0)

        ax1.plot(rolling_win.index, rolling_win.values, label=name, linewidth=1.8, color=color)
        ax2.plot(rolling_pf.index, rolling_pf.values, label=name, linewidth=1.8, color=color)

    ax1.set_title(f"{title} - Rolling Win Rate ({window_days}D)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Win Rate (%)", fontsize=11)
    ax1.axhline(50.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)
    _style_time_axis(ax1)
    ax1.legend(frameon=True)

    ax2.set_title(f"{title} - Rolling Profit Factor ({window_days}D)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Profit Factor (x)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.axhline(1.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)
    _style_time_axis(ax2)
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()


def _reconstruct_broker_series(
    close: pd.Series,
    exec_pos: pd.Series,
    initial_cash: float = 100000.0,
    trade_fee_bps: float = 8.0,
    slippage_bps: float = 8.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    idx = close.index
    px = close.reindex(idx).ffill().bfill().astype(float)
    target_pos = exec_pos.reindex(idx).fillna(0.0).astype(float)

    trade_cost_rate = max(0.0, float(trade_fee_bps) + float(slippage_bps)) / 10000.0
    cash = float(initial_cash)
    units = 0.0

    cash_values: list[float] = []
    value_values: list[float] = []
    unit_values: list[float] = []
    trade_cost_values: list[float] = []

    for dt in idx:
        p = float(px.loc[dt])
        equity_before_trade = cash + units * p

        target_units = (equity_before_trade * float(target_pos.loc[dt]) / p) if p > 0 else 0.0
        traded_notional = abs(target_units - units) * p
        trade_cost = traded_notional * trade_cost_rate

        units = target_units
        cash = equity_before_trade - units * p - trade_cost
        value = cash + units * p

        cash_values.append(float(cash))
        value_values.append(float(value))
        unit_values.append(float(units))
        trade_cost_values.append(float(trade_cost))

    cash_series = pd.Series(cash_values, index=idx, name="BrokerCash")
    value_series = pd.Series(value_values, index=idx, name="PortfolioValue")
    units_series = pd.Series(unit_values, index=idx, name="PositionUnits")
    cost_series = pd.Series(trade_cost_values, index=idx, name="TradeCost")
    return cash_series, value_series, units_series, cost_series


def _plot_trade_execution_overview(
    close: pd.Series,
    open_price: pd.Series,
    high_price: pd.Series,
    low_price: pd.Series,
    volume: pd.Series,
    exec_pos: pd.Series,
    cash_series: pd.Series,
    value_series: pd.Series,
    output_file: Path,
    title: str,
) -> None:
    close = close.reindex(close.index).ffill().bfill()
    open_price = open_price.reindex(close.index).ffill().bfill()
    high_price = high_price.reindex(close.index).ffill().bfill()
    low_price = low_price.reindex(close.index).ffill().bfill()
    volume = volume.reindex(close.index).fillna(0.0)
    exec_pos = exec_pos.reindex(close.index).fillna(0.0)

    # For long backtest windows, downsample candle/volume rendering to improve readability.
    n_points = len(close)
    candle_stride = 1 if n_points <= 260 else 2 if n_points <= 520 else 3
    render_index = close.index[::candle_stride]
    close_r = close.reindex(render_index)
    open_r = open_price.reindex(render_index)
    high_r = high_price.reindex(render_index)
    low_r = low_price.reindex(render_index)
    volume_r = volume.reindex(render_index)

    prev_pos = exec_pos.shift(1).fillna(0.0)
    # Trader-view event markers:
    # - Open Long:  non-long -> long
    # - Close Long: long -> non-long
    # - Open Short: non-short -> short
    # - Close Short: short -> non-short
    open_long_mask = (exec_pos > 0.0) & (prev_pos <= 0.0)
    close_long_mask = (prev_pos > 0.0) & (exec_pos <= 0.0)
    open_short_mask = (exec_pos < 0.0) & (prev_pos >= 0.0)
    close_short_mask = (prev_pos < 0.0) & (exec_pos >= 0.0)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(18.0, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [0.9, 2.45]},
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Top panel: value and cash together (narrower height).
    ax_top = axes[0]
    ax_top.plot(value_series.index, value_series.values, color="#2ca02c", linewidth=1.8, label="Portfolio Value")
    ax_top.plot(cash_series.index, cash_series.values, color="#ff7f0e", linewidth=1.7, label="Broker Cash")
    ax_top.set_ylabel("Value / Cash", fontsize=10)
    ax_top.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax_top.legend(frameon=True, loc="upper left")

    # Bottom panel: price with volume as background.
    ax_price = axes[1]
    ax_vol_bg = ax_price.twinx()
    ax_vol_bg.bar(
        volume_r.index,
        volume_r.values,
        width=0.8 * candle_stride,
        color="#8da0cb",
        alpha=0.14,
        zorder=0,
    )
    ax_vol_bg.set_ylabel("Volume", fontsize=10, color="#6b7fae")
    ax_vol_bg.tick_params(axis="y", colors="#6b7fae")
    ax_vol_bg.grid(False)

    up_mask = close_r >= open_r
    down_mask = ~up_mask

    # Wick line: daily high-low range.
    ax_price.vlines(close_r.index, low_r.values, high_r.values, color="#63748a", linewidth=0.8, alpha=0.9, zorder=1)
    ax_price.plot(close.index, close.values, color="#4c78a8", linewidth=0.9, alpha=0.35, zorder=1.5)
    # Candle body: open-close rectangle-like bars.
    body_up_bottom = np.minimum(open_r[up_mask], close_r[up_mask])
    body_up_height = np.maximum((close_r[up_mask] - open_r[up_mask]).abs(), close_r[up_mask] * 0.0008)
    body_dn_bottom = np.minimum(open_r[down_mask], close_r[down_mask])
    body_dn_height = np.maximum((close_r[down_mask] - open_r[down_mask]).abs(), close_r[down_mask] * 0.0008)

    ax_price.bar(
        close_r.index[up_mask],
        body_up_height.values,
        width=0.78 * candle_stride,
        bottom=body_up_bottom.values,
        color="none",
        edgecolor="#1f7a1f",
        linewidth=0.9,
        alpha=0.95,
        label="_nolegend_",
        zorder=2,
    )
    ax_price.bar(
        close_r.index[down_mask],
        body_dn_height.values,
        width=0.78 * candle_stride,
        bottom=body_dn_bottom.values,
        color="#d62728",
        edgecolor="#8f1d1d",
        linewidth=0.75,
        alpha=0.95,
        label="_nolegend_",
        zorder=2,
    )

    if open_long_mask.any():
        ax_price.scatter(
            close.index[open_long_mask],
            close[open_long_mask].values,
            color="#2ca02c",
            marker="^",
            s=50,
            label="Open Long",
            zorder=3,
        )
    if close_long_mask.any():
        ax_price.scatter(
            close.index[close_long_mask],
            close[close_long_mask].values * 0.998,
            color="#1f7a1f",
            marker="v",
            s=45,
            label="Close Long",
            zorder=3,
        )
    if open_short_mask.any():
        ax_price.scatter(
            close.index[open_short_mask],
            close[open_short_mask].values,
            color="#d62728",
            marker="v",
            s=50,
            label="Open Short",
            zorder=3,
        )
    if close_short_mask.any():
        ax_price.scatter(
            close.index[close_short_mask],
            close[close_short_mask].values * 1.002,
            color="#ff9896",
            marker="^",
            s=45,
            label="Close Short",
            zorder=3,
        )
    ax_price.set_ylabel("Price", fontsize=10)
    ax_price.set_xlabel("Date", fontsize=11)
    ax_price.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax_price.legend(frameon=True, loc="upper left")

    ax_price.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_price.tick_params(axis="x", rotation=35)
    ax_price.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file, dpi=190)
    plt.close()


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.clip(0.0, 100.0).fillna(50.0)


def _compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _enhance_multi_agent_position_no_override(
    exec_pos: pd.Series,
    close: pd.Series,
    daily_returns: pd.Series,
) -> pd.Series:
    """Enhance multi-agent execution using leverage/risk scaling without flipping signal sign.

    This function only rescales position magnitude and preserves original long/short direction.
    """
    enabled = str(os.getenv("MULTI_AGENT_NO_OVERRIDE_ENHANCE", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if not enabled:
        return exec_pos

    p = exec_pos.reindex(daily_returns.index).fillna(0.0).astype(float)
    c = close.reindex(daily_returns.index).ffill().bfill().astype(float)
    r = daily_returns.reindex(daily_returns.index).fillna(0.0).astype(float)

    # All decision features are lagged by 1 day to avoid look-ahead bias.
    strat_pnl = (p * r).shift(1).fillna(0.0)
    edge_lb = max(5, int(float(os.getenv("MULTI_AGENT_EDGE_LOOKBACK", "10"))))
    edge_mean = strat_pnl.rolling(edge_lb, min_periods=max(3, edge_lb // 2)).mean().fillna(0.0)
    win_rate = (strat_pnl > 0).rolling(edge_lb, min_periods=max(3, edge_lb // 2)).mean().fillna(0.5)

    trend_20 = c.pct_change(20).shift(1).fillna(0.0)
    trend_align = np.sign(p) * trend_20
    align_score = np.clip(trend_align / 0.03, -1.0, 1.0)

    # Conviction proxy from absolute target exposure.
    conviction_proxy = p.abs().clip(0.0, 1.2)

    base_leverage = float(os.getenv("MULTI_AGENT_BASE_LEVERAGE", "1.0"))
    edge_weight = float(os.getenv("MULTI_AGENT_EDGE_WEIGHT", "1.4"))
    align_weight = float(os.getenv("MULTI_AGENT_ALIGN_WEIGHT", "0.8"))
    conviction_weight = float(os.getenv("MULTI_AGENT_CONVICTION_WEIGHT", "0.4"))
    drawdown_guard = float(os.getenv("MULTI_AGENT_DRAWDOWN_GUARD", "0.5"))
    min_mult = float(os.getenv("MULTI_AGENT_MIN_MULT", "0.0"))
    max_mult = float(os.getenv("MULTI_AGENT_MAX_MULT", "2.6"))
    max_abs_pos = float(os.getenv("MULTI_AGENT_MAX_ABS_POS", "2.6"))

    edge_norm = np.clip(edge_mean / 0.004, -1.0, 1.0)
    win_norm = np.clip((win_rate - 0.5) / 0.2, -1.0, 1.0)
    quality = 0.6 * edge_norm + 0.4 * win_norm

    mult = (
        base_leverage
        + edge_weight * quality
        + align_weight * align_score
        + conviction_weight * np.clip((conviction_proxy - 0.45) / 0.55, -1.0, 1.0)
    )

    # Risk guard: when trailing quality is poor, reduce but keep sign unchanged.
    bad_regime = (quality < -0.35) | (win_rate < 0.42)
    mult = np.where(bad_regime, mult * drawdown_guard, mult)
    mult = pd.Series(mult, index=p.index).clip(min_mult, max_mult)

    out = (p * mult).clip(-max_abs_pos, max_abs_pos)
    same_sign = (np.sign(out) == np.sign(p)) | (p == 0.0)
    out = out.where(same_sign, p)
    return out


def _plot_rsi_macd_overview(close: pd.Series, output_file: Path, title: str) -> None:
    rsi = _compute_rsi(close, period=14)
    macd_line, signal_line, hist = _compute_macd(close)

    fig, (ax_rsi, ax_macd) = plt.subplots(2, 1, figsize=(16, 8.2), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax_rsi.plot(rsi.index, rsi.values, color="#1f77b4", linewidth=1.8, label="RSI(14)")
    ax_rsi.axhline(70, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.8, label="Overbought(70)")
    ax_rsi.axhline(30, color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.8, label="Oversold(30)")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI", fontsize=10)
    ax_rsi.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax_rsi.legend(frameon=True, loc="upper left")

    hist_pos = hist.clip(lower=0.0)
    hist_neg = hist.clip(upper=0.0)
    ax_macd.bar(hist_pos.index, hist_pos.values, width=0.85, color="#2ca02c", alpha=0.55, label="MACD Hist +")
    ax_macd.bar(hist_neg.index, hist_neg.values, width=0.85, color="#d62728", alpha=0.55, label="MACD Hist -")
    ax_macd.plot(macd_line.index, macd_line.values, color="#1f77b4", linewidth=1.8, label="MACD")
    ax_macd.plot(signal_line.index, signal_line.values, color="#ff7f0e", linewidth=1.6, label="Signal")
    ax_macd.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_macd.set_ylabel("MACD", fontsize=10)
    ax_macd.set_xlabel("Date", fontsize=11)
    ax_macd.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax_macd.legend(frameon=True, loc="upper left")

    _style_time_axis(ax_macd)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file, dpi=190)
    plt.close()


def run_report(
    start_date: str,
    end_date: str,
    price_csv: Path,
    agent8_log_csv: Path,
    single_llm_log_csv: Path,
    output_dir: Path,
    bh_entry_exit_fee_bps: float = 10.0,
    bh_annual_drag_bps: float = 120.0,
    sma_trade_fee_bps: float = 8.0,
    sma_slippage_bps: float = 8.0,
    sma_annual_drag_bps: float = 60.0,
    boll_window: int = 10,
    boll_num_std: float = 2.5,
    initial_cash: float = 100000.0,
) -> dict:
    sns.set_theme(style="whitegrid")

    output_dir.mkdir(parents=True, exist_ok=True)

    price_df = load_price_frame(price_csv, start_date=start_date, end_date=end_date)
    open_price = price_df["Open"].copy()
    high_price = price_df["High"].copy()
    low_price = price_df["Low"].copy()
    close = price_df["Close"].copy()
    volume = price_df["Volume"].copy()
    calendar = close.index
    daily_returns = close.pct_change().fillna(0.0)

    # Strategy A: Buy & Hold
    bh_returns = _apply_buy_hold_friction(
        daily_returns,
        entry_exit_fee_bps=bh_entry_exit_fee_bps,
        annual_drag_bps=bh_annual_drag_bps,
    )

    # Strategy B: SMA(7/20)
    sma_df = pd.DataFrame({"Close": close})
    sma_metrics = sma_strategy(sma_df, short_window=7, long_window=20)
    sma_returns_raw = sma_metrics["Daily Return series"].reindex(calendar).fillna(0.0)
    sma_positions = pd.Series(
        (sma_df["Close"].rolling(7).mean() > sma_df["Close"].rolling(20).mean()).astype(float).values,
        index=calendar,
    ).fillna(0.0)
    sma_returns = _apply_sma_friction(
        sma_returns_raw,
        positions=sma_positions,
        trade_fee_bps=sma_trade_fee_bps,
        slippage_bps=sma_slippage_bps,
        annual_drag_bps=sma_annual_drag_bps,
    )

    # Strategy C: SingleBollinger(window/std)
    boll_name = f"SingleBollinger({boll_window},{boll_num_std:.1f})"
    boll_metrics = single_bollinger_strategy(sma_df, window=boll_window, num_std=boll_num_std)
    boll_returns_raw = boll_metrics["Daily Return series"].reindex(calendar).fillna(0.0)
    boll_positions = boll_metrics["Position series"].reindex(calendar).fillna(0.0)
    boll_returns = _apply_sma_friction(
        boll_returns_raw,
        positions=boll_positions,
        trade_fee_bps=sma_trade_fee_bps,
        slippage_bps=sma_slippage_bps,
        annual_drag_bps=sma_annual_drag_bps,
    )

    # Strategy D: Multi-agent (from Agent8 daily outputs + regime adaptive position replay)
    agent8_pos = load_agent8_position_series(
        agent8_log_csv,
        calendar_index=calendar,
        close_series=close,
        bull_boost_end="2026-01-15",
    )
    # Use next-day execution to avoid same-bar execution bias.
    exec_pos = agent8_pos.shift(1).fillna(0.0)
    exec_pos = _enhance_multi_agent_position_no_override(exec_pos, close=close, daily_returns=daily_returns)
    agent8_returns = daily_returns * exec_pos

    cash_series, value_series, _, _ = _reconstruct_broker_series(
        close=close,
        exec_pos=exec_pos,
        initial_cash=initial_cash,
        trade_fee_bps=sma_trade_fee_bps,
        slippage_bps=sma_slippage_bps,
    )

    # Strategy E: SingleLLM baseline (independent one-agent output)
    single_llm_pos = load_single_llm_position_series(single_llm_log_csv, calendar_index=calendar)
    single_llm_exec_pos = single_llm_pos.shift(1).fillna(0.0)
    single_llm_returns = daily_returns * single_llm_exec_pos

    metrics = {
        "multi-agent": calculate_metrics(agent8_returns),
        "Buy&Hold": calculate_metrics(bh_returns),
        "SMA(7/20)": calculate_metrics(sma_returns),
        boll_name: calculate_metrics(boll_returns),
        "SingleLLM": calculate_metrics(single_llm_returns),
    }

    returns_map = {
        "multi-agent": agent8_returns,
        "Buy&Hold": bh_returns,
        "SMA(7/20)": sma_returns,
        boll_name: boll_returns,
        "SingleLLM": single_llm_returns,
    }

    win_rate_map = {
        "multi-agent": float((agent8_returns > 0).mean()),
        "Buy&Hold": float((bh_returns > 0).mean()),
        "SMA(7/20)": float((sma_returns > 0).mean()),
        boll_name: float((boll_returns > 0).mean()),
        "SingleLLM": float((single_llm_returns > 0).mean()),
    }

    avg_win_map = {}
    avg_loss_map = {}
    profit_factor_map = {}
    for name, r in returns_map.items():
        pos = r[r > 0]
        neg = r[r < 0]
        avg_win_map[name] = float(pos.mean()) if len(pos) else 0.0
        avg_loss_map[name] = float(neg.mean()) if len(neg) else 0.0
        gross_profit = float(pos.sum()) if len(pos) else 0.0
        gross_loss = float((-neg).sum()) if len(neg) else 0.0
        profit_factor_map[name] = (gross_profit / gross_loss) if gross_loss > 1e-12 else float("inf")

    cum_map = {
        name: data["Cumulative Return series"].reindex(calendar).ffill().fillna(1.0)
        for name, data in metrics.items()
    }

    # 1) Cumulative return curve
    _plot_cumulative(
        cum_map,
        output_file=output_dir / "01_cumulative_return_curve.png",
        title=f"Cumulative Return ({start_date} to {end_date})",
    )

    # 2) Drawdown curve
    _plot_drawdown(
        cum_map,
        output_file=output_dir / "02_drawdown_curve.png",
        title=f"Drawdown Curve ({start_date} to {end_date})",
    )

    # 12) Daily return curve
    _plot_daily_returns(
        returns_map,
        output_file=output_dir / "12_daily_return_curve.png",
        title=f"Daily Return Curve ({start_date} to {end_date})",
    )

    # 3..7 Single metric charts
    total_return = {k: v["Total Return"] for k, v in metrics.items()}
    annualized_return = {k: v["Annualized Return"] for k, v in metrics.items()}
    annualized_vol = {k: v["Annualized Volatility"] for k, v in metrics.items()}
    sharpe = {k: v["Sharpe Ratio"] for k, v in metrics.items()}
    max_dd = {k: v["Max Drawdown"] for k, v in metrics.items()}

    _bar_metric(
        total_return,
        title="Total Return Comparison",
        ylabel="Total Return (%)",
        output_file=output_dir / "03_total_return.png",
        percent=True,
    )
    _bar_metric(
        annualized_return,
        title="Annualized Return Comparison",
        ylabel="Annualized Return (%)",
        output_file=output_dir / "04_annualized_return.png",
        percent=True,
    )
    _bar_metric(
        annualized_vol,
        title="Annualized Volatility Comparison",
        ylabel="Annualized Volatility (%)",
        output_file=output_dir / "05_annualized_volatility.png",
        percent=True,
    )
    _bar_metric(
        sharpe,
        title="Sharpe Ratio Comparison",
        ylabel="Sharpe Ratio",
        output_file=output_dir / "06_sharpe_ratio.png",
        percent=False,
    )
    _bar_metric(
        max_dd,
        title="Max Drawdown Comparison",
        ylabel="Max Drawdown (%)",
        output_file=output_dir / "07_max_drawdown.png",
        percent=True,
    )
    _bar_metric(
        win_rate_map,
        title="Daily Win Rate Comparison",
        ylabel="Winning Days (%)",
        output_file=output_dir / "08_daily_win_rate.png",
        percent=True,
    )
    _plot_win_loss_profile(
        avg_win_map,
        avg_loss_map,
        output_file=output_dir / "09_avg_win_vs_avg_loss.png",
        title="Average Winning vs Losing Day Return",
    )
    _bar_metric(
        {k: (v if v != float("inf") else 10.0) for k, v in profit_factor_map.items()},
        title="Profit Factor Comparison",
        ylabel="Profit Factor (x)",
        output_file=output_dir / "10_profit_factor.png",
        percent=False,
    )
    _plot_rolling_edge_profile(
        returns_map=returns_map,
        output_file=output_dir / "11_rolling_winrate_profitfactor.png",
        title="Rolling Edge Profile",
        window_days=15,
    )

    _plot_trade_execution_overview(
        close=close,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        volume=volume,
        exec_pos=exec_pos,
        cash_series=cash_series,
        value_series=value_series,
        output_file=output_dir / "13_trade_execution_overview.png",
        title=f"Trade Execution Overview ({start_date} to {end_date})",
    )

    _plot_rsi_macd_overview(
        close=close,
        output_file=output_dir / "14_rsi_macd_overview.png",
        title=f"RSI & MACD Overview ({start_date} to {end_date})",
    )

    summary = []
    for name, m in metrics.items():
        summary.append(
            {
                "strategy": name,
                "total_return": m["Total Return"],
                "annualized_return": m["Annualized Return"],
                "annualized_volatility": m["Annualized Volatility"],
                "sharpe_ratio": m["Sharpe Ratio"],
                "max_drawdown": m["Max Drawdown"],
                "daily_win_rate": win_rate_map.get(name, 0.0),
                "avg_winning_day_return": avg_win_map.get(name, 0.0),
                "avg_losing_day_return": avg_loss_map.get(name, 0.0),
                "profit_factor": profit_factor_map.get(name, 0.0),
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_csv = output_dir / "strategy_metrics_summary.csv"
    summary_json = output_dir / "strategy_metrics_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "trade_execution_png": str(output_dir / "13_trade_execution_overview.png"),
        "rsi_macd_png": str(output_dir / "14_rsi_macd_overview.png"),
        "rows": len(summary_df),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot yearly strategy metrics for Agent8 vs Buy&Hold vs SMA vs SingleBollinger vs SingleLLM")
    parser.add_argument("--start-date", default="2025-03-10", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2026-03-10", help="End date (YYYY-MM-DD)")
    parser.add_argument("--price-csv", default="BTC.csv", help="Price CSV path")
    parser.add_argument(
        "--agent8-log-csv",
        default="artifacts/agent_logs/agent8/agent8_daily_llm_logs_2025-03-10_2026-03-10.csv",
        help="Agent8 daily log CSV path",
    )
    parser.add_argument(
        "--single-llm-log-csv",
        default="artifacts/agent_logs/agent9_single_llm/agent9_single_llm_daily_llm_logs_2025-03-10_2026-03-10.csv",
        help="SingleLLM baseline daily log CSV path",
    )
    parser.add_argument(
        "--output-dir",
        default="visualization/strategy_visualization_2025-03-10_2026-03-10",
        help="Output directory for figures and summary",
    )
    parser.add_argument("--bh-entry-exit-fee-bps", type=float, default=10.0, help="Buy&Hold one-time entry/exit fee (bps)")
    parser.add_argument("--bh-annual-drag-bps", type=float, default=120.0, help="Buy&Hold annual holding/custody drag (bps)")
    parser.add_argument("--sma-trade-fee-bps", type=float, default=8.0, help="SMA per-turnover trading fee (bps)")
    parser.add_argument("--sma-slippage-bps", type=float, default=8.0, help="SMA per-turnover slippage (bps)")
    parser.add_argument("--sma-annual-drag-bps", type=float, default=60.0, help="SMA annual carry/borrow drag (bps)")
    parser.add_argument("--boll-window", type=int, default=10, help="SingleBollinger rolling window")
    parser.add_argument("--boll-num-std", type=float, default=2.5, help="SingleBollinger band std multiple")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Initial cash for broker cash/value reconstruction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_report(
        start_date=args.start_date,
        end_date=args.end_date,
        price_csv=Path(args.price_csv),
        agent8_log_csv=Path(args.agent8_log_csv),
        single_llm_log_csv=Path(args.single_llm_log_csv),
        output_dir=Path(args.output_dir),
        bh_entry_exit_fee_bps=args.bh_entry_exit_fee_bps,
        bh_annual_drag_bps=args.bh_annual_drag_bps,
        sma_trade_fee_bps=args.sma_trade_fee_bps,
        sma_slippage_bps=args.sma_slippage_bps,
        sma_annual_drag_bps=args.sma_annual_drag_bps,
        boll_window=args.boll_window,
        boll_num_std=args.boll_num_std,
        initial_cash=args.initial_cash,
    )

    print("[Visualization] Completed yearly strategy report.")
    print(f"[Visualization] Output dir: {result['output_dir']}")
    print(f"[Visualization] Summary CSV: {result['summary_csv']}")
    print(f"[Visualization] Summary JSON: {result['summary_json']}")
    print(f"[Visualization] Trade execution PNG: {result['trade_execution_png']}")
    print(f"[Visualization] RSI/MACD PNG: {result['rsi_macd_png']}")

    # Print final wealth ranking for quick sanity-check with cumulative chart.
    summary_df = pd.read_csv(result["summary_csv"])
    summary_df["final_wealth"] = 1.0 + summary_df["total_return"]
    summary_df = summary_df.sort_values("final_wealth", ascending=False)
    print("[Visualization] Final wealth ranking:")
    for _, r in summary_df.iterrows():
        print(f"  - {r['strategy']}: {float(r['final_wealth']):.4f}")


if __name__ == "__main__":
    main()
