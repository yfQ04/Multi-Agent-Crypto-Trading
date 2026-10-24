# backtest/engine.py
"""
Backtest Engine
- Calculate comprehensive performance metrics for trading strategies
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

from agents.news_sentiment_agent import run_news_sentiment_agent
from agents.bull_agent import run_bull_agent
from agents.bear_agent import run_bear_agent
from agents.trader_agent import run_trader_agent
from agents.debate_agent import run_debate_agent
from agents.decision_agent import run_decision_agent
from agents.fund_manager_agent import run_fund_manager_agent


PROJECT_ROOT = Path(__file__).resolve().parent.parent

def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """Calculate key performance metrics from a series of returns."""
    cum_returns = (1 + returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1 if not cum_returns.empty else 0
    
    # Annualized metrics (assuming daily crypto returns, 365 days)
    trading_days = 365
    ann_return = (1 + total_return) ** (trading_days / len(returns)) - 1 if len(returns) > 0 else 0
    ann_volatility = returns.std() * np.sqrt(trading_days)
    
    # Sharpe Ratio
    sharpe_ratio = (ann_return - risk_free_rate) / ann_volatility if ann_volatility != 0 else 0
    
    # Max Drawdown
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        "Total Return": total_return,
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Daily Return series": returns,
        "Cumulative Return series": cum_returns
    }


def _to_daily_sentiment_frame(sentiment_result: dict) -> pd.DataFrame:
    items = sentiment_result.get("items", []) if isinstance(sentiment_result, dict) else []
    if not items:
        return pd.DataFrame(
            columns=[
                "published_date",
                "sentiment_score",
                "sentiment_confidence",
                "is_positive",
                "is_negative",
                "is_neutral",
            ]
        )

    rows = []
    for item in items:
        raw_ts = str(item.get("published_at", "") or "").strip()
        score = item.get("sentiment_score", 0.0)
        ts = pd.to_datetime(raw_ts, errors="coerce")
        if pd.isna(ts):
            continue
        try:
            score = float(score)
        except Exception:
            score = 0.0
        rows.append(
            {
                "published_date": ts.normalize(),
                "sentiment_score": score,
                "sentiment_confidence": float(item.get("sentiment_confidence", 0.0) or 0.0),
                "is_positive": 1 if score > 0.1 else 0,
                "is_negative": 1 if score < -0.1 else 0,
                "is_neutral": 1 if -0.1 <= score <= 0.1 else 0,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "published_date",
                "sentiment_score",
                "sentiment_confidence",
                "is_positive",
                "is_negative",
                "is_neutral",
            ]
        )
    return pd.DataFrame(rows)


def _score_to_label(score: float) -> str:
    if score > 0.1:
        return "positive"
    if score < -0.1:
        return "negative"
    return "neutral"


def _build_daily_sentiment_calendar(
    sentiment_df: pd.DataFrame,
    calendar_dates: pd.DatetimeIndex,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build daily sentiment for every calendar day.
    Gap-filling strategy:
    1) keep raw daily average when data exists
    2) forward fill
    3) backward fill (for initial missing block)
    4) fallback to 0.0 if all days missing
    """
    normalized_dates = pd.DatetimeIndex(calendar_dates).normalize().unique().sort_values()
    calendar_df = pd.DataFrame({"date": normalized_dates})

    if sentiment_df.empty:
        calendar_df["raw_items_count"] = 0
        calendar_df["raw_sentiment_score"] = np.nan
        calendar_df["raw_sentiment_confidence"] = np.nan
        calendar_df["raw_positive_count"] = 0
        calendar_df["raw_negative_count"] = 0
        calendar_df["raw_neutral_count"] = 0
    else:
        grouped = (
            sentiment_df.groupby("published_date", as_index=False)
            .agg(
                raw_items_count=("sentiment_score", "size"),
                raw_sentiment_score=("sentiment_score", "mean"),
                raw_sentiment_confidence=("sentiment_confidence", "mean"),
                raw_positive_count=("is_positive", "sum"),
                raw_negative_count=("is_negative", "sum"),
                raw_neutral_count=("is_neutral", "sum"),
            )
            .rename(columns={"published_date": "date"})
        )
        calendar_df = calendar_df.merge(grouped, on="date", how="left")
        calendar_df["raw_items_count"] = calendar_df["raw_items_count"].fillna(0).astype(int)
        calendar_df["raw_positive_count"] = calendar_df["raw_positive_count"].fillna(0).astype(int)
        calendar_df["raw_negative_count"] = calendar_df["raw_negative_count"].fillna(0).astype(int)
        calendar_df["raw_neutral_count"] = calendar_df["raw_neutral_count"].fillna(0).astype(int)

    calendar_df["sentiment_score"] = calendar_df["raw_sentiment_score"]
    calendar_df["sentiment_score"] = calendar_df["sentiment_score"].ffill().bfill().fillna(0.0)
    calendar_df["is_filled"] = calendar_df["raw_items_count"] == 0
    calendar_df["sentiment_label"] = calendar_df["sentiment_score"].apply(_score_to_label)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        to_save = calendar_df.copy()
        to_save["date"] = to_save["date"].dt.strftime("%Y-%m-%d")
        to_save.to_csv(output_path, index=False)

    return calendar_df


def _build_event_sentiment_snapshot(
    daily_sentiment_df: pd.DataFrame,
    decision_date: pd.Timestamp,
    lookback_days: int = 7,
    sentiment_lag_days: int = 1,
) -> dict:
    if daily_sentiment_df.empty:
        return {
            "avg_sentiment_score": 0.0,
            "weighted_sentiment_score": 0.0,
            "avg_sentiment_confidence": 0.0,
            "sentiment_label": "neutral",
            "items_count": 0,
            "positive_factors_count": 0,
            "negative_factors_count": 0,
            "neutral_factors_count": 0,
            "items": [],
            "scores_series": [],
            "gpt_summary": "No sentiment items in lookback window.",
        }

    cutoff = decision_date.normalize() - pd.Timedelta(days=max(0, sentiment_lag_days))
    window_start = cutoff - pd.Timedelta(days=max(1, lookback_days) - 1)

    window_df = daily_sentiment_df[
        (daily_sentiment_df["date"] >= window_start)
        & (daily_sentiment_df["date"] <= cutoff)
    ]

    if window_df.empty:
        return {
            "avg_sentiment_score": 0.0,
            "weighted_sentiment_score": 0.0,
            "avg_sentiment_confidence": 0.0,
            "sentiment_label": "neutral",
            "items_count": 0,
            "positive_factors_count": 0,
            "negative_factors_count": 0,
            "neutral_factors_count": 0,
            "items": [],
            "scores_series": [],
            "gpt_summary": f"No sentiment items found up to {cutoff.date()}.",
        }

    scores = window_df["sentiment_score"].tolist()
    avg_score = float(np.mean(scores))
    label = _score_to_label(avg_score)
    raw_items = int(window_df["raw_items_count"].sum()) if "raw_items_count" in window_df.columns else 0
    confidence_weighted_sum = 0.0
    if "raw_sentiment_confidence" in window_df.columns and "raw_items_count" in window_df.columns:
        confidence_weighted_sum = float((window_df["raw_sentiment_confidence"] * window_df["raw_items_count"]).sum())
    avg_confidence = (confidence_weighted_sum / raw_items) if raw_items > 0 else 0.0
    weighted_score_numer = 0.0
    weighted_score_denom = 0.0
    if (
        "raw_sentiment_confidence" in window_df.columns
        and "raw_items_count" in window_df.columns
        and "raw_sentiment_score" in window_df.columns
    ):
        weights = window_df["raw_sentiment_confidence"] * window_df["raw_items_count"]
        weighted_score_numer = float((window_df["raw_sentiment_score"] * weights).sum())
        weighted_score_denom = float(weights.sum())
    weighted_score = (weighted_score_numer / weighted_score_denom) if weighted_score_denom > 1e-12 else avg_score
    positive_count = int(window_df["raw_positive_count"].sum()) if "raw_positive_count" in window_df.columns else 0
    negative_count = int(window_df["raw_negative_count"].sum()) if "raw_negative_count" in window_df.columns else 0
    neutral_count = int(window_df["raw_neutral_count"].sum()) if "raw_neutral_count" in window_df.columns else 0
    filled_days = int(window_df["is_filled"].sum()) if "is_filled" in window_df.columns else 0

    return {
        "avg_sentiment_score": avg_score,
        "weighted_sentiment_score": float(np.clip(weighted_score, -1.0, 1.0)),
        "avg_sentiment_confidence": float(np.clip(avg_confidence, 0.0, 1.0)),
        "sentiment_label": label,
        "items_count": raw_items,
        "positive_factors_count": positive_count,
        "negative_factors_count": negative_count,
        "neutral_factors_count": neutral_count,
        "items": [],
        "scores_series": scores,
        "gpt_summary": (
            f"Event sentiment window {window_start.date()}~{cutoff.date()}: "
            f"avg={avg_score:.4f}, weighted={weighted_score:.4f}, conf={avg_confidence:.4f}, "
            f"label={label}, raw_items={raw_items}, "
            f"pos={positive_count}, neu={neutral_count}, neg={negative_count}, filled_days={filled_days}"
        ),
    }


def _build_event_technical_snapshot(history_df: pd.DataFrame, symbol: str) -> dict:
    latest = history_df.iloc[-1]
    return {
        "symbol": symbol,
        "df": history_df,
        "current_price": float(latest.get("Close", np.nan)),
        "rsi": float(latest.get("RSI_14", 50.0)),
        "macd": float(latest.get("MACD", 0.0)),
        "macd_signal": float(latest.get("MACD_signal", 0.0)),
        "volatility": float(latest.get("Volatility", 0.0)),
        "sma_7": float(latest.get("SMA_7", latest.get("Close", 0.0))),
        "sma_20": float(latest.get("SMA_20", latest.get("Close", 0.0))),
        "sma_50": float(latest.get("SMA_50", latest.get("Close", 0.0))),
        "gpt_analysis": "Event snapshot technical features.",
    }


def _extract_conviction(*texts: str) -> float:
    joined = "\n".join([t for t in texts if isinstance(t, str)])
    lowered = joined.lower()

    pct_matches = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", joined)
    ratio_matches = re.findall(r"(?:conviction|confidence)\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?)", lowered)

    candidates = []
    for v in pct_matches:
        try:
            num = float(v)
            if 0 <= num <= 100:
                candidates.append(num / 100.0)
        except Exception:
            pass
    for v in ratio_matches:
        try:
            num = float(v)
            if 0 <= num <= 1:
                candidates.append(num)
        except Exception:
            pass

    if candidates:
        base = float(np.clip(np.mean(candidates), 0.2, 1.0))
    else:
        if "high conviction" in lowered or "strong conviction" in lowered:
            base = 0.85
        elif "low conviction" in lowered or "weak conviction" in lowered:
            base = 0.35
        elif "medium conviction" in lowered or "moderate conviction" in lowered:
            base = 0.6
        else:
            base = 0.6
    return base


def _extract_json_payload(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {}
    raw = text.strip()

    fenced = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
    candidates = fenced[:] if fenced else []
    candidates.append(raw)

    brace_candidates = re.findall(r"(\{[\s\S]*\})", raw)
    candidates.extend(brace_candidates)

    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}


def _coerce_action(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in {"long", "buy", "bullish"}:
        return "long"
    if v in {"short", "sell", "bearish"}:
        return "short"
    if v in {"flat", "neutral", "hold", "cash", "wait"}:
        return "flat"
    return ""


def _coerce_preference(value: str) -> str:
    v = str(value or "").strip().lower()
    if v in {"long", "buy", "bullish"}:
        return "long"
    if v in {"short", "sell", "bearish"}:
        return "short"
    return ""


def _print_agent_output(agent_name: str, payload_text: str) -> None:
    parsed = _extract_json_payload(payload_text)
    if parsed:
        summary = str(parsed.get("thinking_summary", "")).strip()
        if summary:
            print(f"[Engine] {agent_name} thinking: {summary[:280]}")
        action_keys = ["action", "final_action", "risk_level", "approval", "conviction", "final_conviction"]
        compact = {k: parsed.get(k) for k in action_keys if k in parsed}
        if compact:
            print(f"[Engine] {agent_name} output: {compact}")
    else:
        print(f"[Engine] {agent_name} output(raw): {str(payload_text)[:280]}")


def _regime_bias_score(technical_result: dict | None, sentiment_result: dict | None) -> float:
    technical_result = technical_result or {}
    sentiment_result = sentiment_result or {}

    def _finite_or(value: float, default: float) -> float:
        try:
            num = float(value)
            if np.isfinite(num):
                return num
        except Exception:
            pass
        return default

    s7 = _finite_or(technical_result.get("sma_7", 0.0), 0.0)
    s20 = _finite_or(technical_result.get("sma_20", 0.0), 0.0)
    s50 = _finite_or(technical_result.get("sma_50", 0.0), 0.0)
    macd = _finite_or(technical_result.get("macd", 0.0), 0.0)
    macd_signal = _finite_or(technical_result.get("macd_signal", 0.0), 0.0)
    rsi = _finite_or(technical_result.get("rsi", 50.0), 50.0)
    sentiment_score = _finite_or(
        sentiment_result.get("weighted_sentiment_score", sentiment_result.get("avg_sentiment_score", 0.0)),
        0.0,
    )

    trend_component = 0.0
    if s7 > s20:
        trend_component += 0.35
    else:
        trend_component -= 0.35
    if s20 > s50:
        trend_component += 0.25
    else:
        trend_component -= 0.25

    macd_component = float(np.clip((macd - macd_signal) / 2000.0, -0.25, 0.25))
    rsi_component = float(np.clip((rsi - 50.0) / 100.0, -0.2, 0.2))
    sentiment_component = float(np.clip(sentiment_score, -0.2, 0.2))

    return float(np.clip(trend_component + macd_component + rsi_component + sentiment_component, -1.0, 1.0))


def _market_regime_state(technical_result: dict | None, sentiment_result: dict | None) -> tuple[str, float]:
    """Classify market regime and return a confidence score in [0,1]."""
    technical_result = technical_result or {}
    sentiment_result = sentiment_result or {}

    def _finite_or(value: float, default: float) -> float:
        try:
            num = float(value)
            if np.isfinite(num):
                return num
        except Exception:
            pass
        return default

    s7 = _finite_or(technical_result.get("sma_7", 0.0), 0.0)
    s20 = _finite_or(technical_result.get("sma_20", 0.0), 0.0)
    s50 = _finite_or(technical_result.get("sma_50", 0.0), 0.0)
    rsi = _finite_or(technical_result.get("rsi", 50.0), 50.0)
    vol = max(0.0, _finite_or(technical_result.get("volatility", 0.0), 0.0))
    sent = _finite_or(
        sentiment_result.get("weighted_sentiment_score", sentiment_result.get("avg_sentiment_score", 0.0)),
        0.0,
    )

    bullish_stack = (s7 > s20) and (s20 > s50)
    bearish_stack = (s7 < s20) and (s20 < s50)
    low_vol = vol <= 55.0
    high_vol = vol >= 95.0

    bull_conf = 0.0
    bear_conf = 0.0
    if bullish_stack:
        bull_conf += 0.45
    if bearish_stack:
        bear_conf += 0.45
    bull_conf += float(np.clip((rsi - 50.0) / 40.0, 0.0, 0.25))
    bear_conf += float(np.clip((50.0 - rsi) / 40.0, 0.0, 0.25))
    bull_conf += float(np.clip(sent, 0.0, 0.20))
    bear_conf += float(np.clip(-sent, 0.0, 0.20))

    if low_vol:
        bull_conf += 0.10
        bear_conf += 0.05
    if high_vol:
        bull_conf -= 0.10
        bear_conf += 0.10

    bull_conf = float(np.clip(bull_conf, 0.0, 1.0))
    bear_conf = float(np.clip(bear_conf, 0.0, 1.0))

    if bull_conf >= 0.55 and bull_conf >= bear_conf + 0.10:
        return "bull", bull_conf
    if bear_conf >= 0.55 and bear_conf >= bull_conf + 0.10:
        return "bear", bear_conf
    return "sideways", float(max(bull_conf, bear_conf))


def _map_chain_output_to_position(
    decision_text: str,
    risk_text: str,
    fund_text: str,
    technical_result: dict | None = None,
    sentiment_result: dict | None = None,
) -> tuple[str, float, float]:
    decision_json = _extract_json_payload(decision_text)
    risk_json = _extract_json_payload(risk_text)
    fund_json = _extract_json_payload(fund_text)

    corpus = "\n".join([decision_text or "", risk_text or "", fund_text or ""]).lower()

    long_hits = len(re.findall(r"\b(long|buy|bullish|upside|accumulate)\b", corpus))
    short_hits = len(re.findall(r"\b(short|sell|bearish|downside|reduce)\b", corpus))
    flat_hits = len(re.findall(r"\b(flat|neutral|hold|wait|cash|no trade|sideline)\b", corpus))

    direction = _coerce_action(fund_json.get("final_action", ""))
    if not direction:
        direction = _coerce_action(decision_json.get("action", ""))

    preference = _coerce_preference(decision_json.get("direction_preference", ""))

    if not direction and flat_hits > max(long_hits, short_hits):
        direction = "flat"
    elif not direction and long_hits > short_hits:
        direction = "long"
    elif not direction and short_hits > long_hits:
        direction = "short"
    elif not direction:
        direction = "flat"

    parsed_conviction = fund_json.get("final_conviction", decision_json.get("conviction", None))
    try:
        conviction = float(parsed_conviction)
        conviction = float(np.clip(conviction, 0.0, 1.0)) if np.isfinite(conviction) else _extract_conviction(decision_text, risk_text, fund_text)
    except Exception:
        conviction = _extract_conviction(decision_text, risk_text, fund_text)

    rm_candidates = [risk_json.get("position_multiplier", None), fund_json.get("position_multiplier_override", None)]
    risk_multiplier = None
    for val in rm_candidates:
        try:
            risk_multiplier = float(val)
            if np.isfinite(risk_multiplier):
                risk_multiplier = float(np.clip(risk_multiplier, 0.0, 1.25))
                break
        except Exception:
            continue

    if risk_multiplier is None:
        risk_multiplier = 1.0
        risk_text_lower = (risk_text or "").lower()
        risk_level = str(risk_json.get("risk_level", "")).lower()
        level_corpus = f"{risk_level} {risk_text_lower}"
        if "conservative" in level_corpus:
            risk_multiplier = 0.80
        elif "neutral" in level_corpus:
            risk_multiplier = 1.00
        elif "aggressive" in level_corpus:
            risk_multiplier = 1.15

    # Use risk-team bull/bear confidence as a direct aggressiveness signal.
    risk_bull = _coerce_float(risk_json.get("bull_score", 0.0), 0.0)
    risk_bear = _coerce_float(risk_json.get("bear_score", 0.0), 0.0)
    risk_team_conf = _coerce_float(risk_json.get("confidence", 0.5), 0.5)
    risk_edge = float(np.clip(risk_bull - risk_bear, -1.0, 1.0))
    risk_team_conf = float(np.clip(risk_team_conf, 0.0, 1.0))

    edge_amp = 0.10 + 0.22 * risk_team_conf
    if direction == "long":
        if risk_edge > 0:
            risk_multiplier *= 1.0 + edge_amp * risk_edge
        elif risk_edge < 0:
            risk_multiplier *= 1.0 - edge_amp * min(abs(risk_edge), 0.8)
    elif direction == "short":
        if risk_edge < 0:
            risk_multiplier *= 1.0 + edge_amp * abs(risk_edge)
        elif risk_edge > 0:
            risk_multiplier *= 1.0 - edge_amp * min(risk_edge, 0.8)

    regime_bias = _regime_bias_score(technical_result, sentiment_result)
    regime_state, regime_conf = _market_regime_state(technical_result, sentiment_result)

    # Regime-aware baseline aggressiveness: stronger in bull, cautious in bear.
    if regime_state == "bull":
        risk_multiplier *= 1.12 + 0.18 * regime_conf
    elif regime_state == "bear":
        risk_multiplier *= 0.80 - 0.10 * min(regime_conf, 0.8)
    else:
        risk_multiplier *= 0.95
    risk_multiplier = float(np.clip(risk_multiplier, 0.35, 1.35))

    signed = 0.0
    if direction == "long":
        signed = conviction * risk_multiplier
    elif direction == "short":
        signed = -conviction * risk_multiplier
    else:
        # Even when agents choose flat, keep tactical exposure from regime + preference.
        tactical_strength = max(0.30, conviction * 0.90)
        pref_sign = 0.0
        if preference == "long":
            pref_sign = 1.0
        elif preference == "short":
            pref_sign = -1.0

        # Blend regime, model preference and risk-team edge to keep flat states informative.
        blend_bias = 0.50 * regime_bias + 0.30 * pref_sign + 0.20 * np.sign(risk_edge) * min(abs(risk_edge), 0.9)
        signed = blend_bias * tactical_strength * risk_multiplier

    # Daily profitability preference: follow strong regime, reduce low-quality countertrend bets.
    if regime_state == "bull":
        if direction == "short" and conviction < 0.70:
            # In bull trend, weak short calls are usually noise: convert to mild long bias.
            signed = max(0.15, 0.25 * regime_conf)
        elif direction in {"long", "flat"}:
            signed *= 1.10 + 0.12 * regime_conf
            if risk_edge > 0 and risk_team_conf >= 0.60:
                signed *= 1.02 + 0.10 * min(risk_edge, 0.8)
    elif regime_state == "bear":
        if direction == "long" and conviction < 0.70:
            signed = -max(0.12, 0.22 * regime_conf)
        elif direction in {"short", "flat"}:
            signed *= 1.06 + 0.10 * regime_conf
            if risk_edge < 0 and risk_team_conf >= 0.60:
                signed *= 1.02 + 0.10 * min(abs(risk_edge), 0.8)

    # Align/penalize by regime consistency to increase responsiveness
    if direction == "long" and regime_bias > 0:
        signed *= 1.15
    elif direction == "short" and regime_bias < 0:
        signed *= 1.15
    elif direction in {"long", "short"} and np.sign(signed) != np.sign(regime_bias) and abs(regime_bias) > 0.25:
        signed *= 0.7

    # Dynamic minimum activity to avoid almost-constant tiny positions.
    # Stronger regime -> stronger minimum exposure.
    min_mag = 0.10 + 0.45 * abs(regime_bias)
    if regime_state == "bull":
        min_mag += 0.06 * regime_conf
    elif regime_state == "bear":
        min_mag += 0.03 * regime_conf
    if direction == "flat":
        min_mag = max(min_mag, 0.18)
    else:
        min_mag = max(min_mag, 0.22 + 0.30 * conviction)

    if abs(signed) < min_mag:
        fallback_sign = np.sign(signed)
        if fallback_sign == 0:
            if preference == "long":
                fallback_sign = 1
            elif preference == "short":
                fallback_sign = -1
            else:
                fallback_sign = np.sign(regime_bias)
        if fallback_sign == 0:
            fallback_sign = 1 if direction == "long" else -1 if direction == "short" else 0
        signed = float(fallback_sign) * float(min_mag)

    if not np.isfinite(signed):
        signed = 0.12 * (1 if regime_bias >= 0 else -1)

    return direction, conviction, float(np.clip(signed, -1.25, 1.25))


def _event_cache_path(cache_dir: Path, symbol: str, decision_date: pd.Timestamp) -> Path:
    safe_symbol = symbol.replace("-", "_")
    date_key = decision_date.strftime("%Y-%m-%d")
    return cache_dir / safe_symbol / f"{date_key}.json"


def _load_event_cache(cache_file: Path) -> dict | None:
    if not cache_file.exists():
        return None
    try:
        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_event_cache(cache_file: Path, payload: dict) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def simulate_agent_strategy_with_agents(
    historical_data: pd.DataFrame,
    symbol: str = "BTC-USD",
    sentiment_csv_path: str = "bitcoin_tweets_latest.csv",
    start_date: str | None = None,
    end_date: str | None = None,
    event_every_n_days: int = 1,
    sentiment_lookback_days: int = 7,
    sentiment_lag_days: int = 1,
    sentiment_batch_per_day: int = 10,
    use_sentiment_cache: bool = True,
    sentiment_cache_file: str = "artifacts/sentiment_cache.json",
    trading_fee_bps: float = 10.0,
    use_event_cache: bool = True,
    event_cache_dir: str = "artifacts/agent_chain_cache",
    save_daily_sentiment_calendar: bool = True,
    daily_sentiment_output_dir: str = "artifacts/daily_sentiment",
    enable_price_trigger: bool = True,
    price_trigger_pct: float = 0.015,
) -> dict:
    """
    Real event-driven multi-agent backtest.
    - Trigger one full agent-chain decision every N days.
    - Keep target position unchanged until next event.
    - Sentiment uses only information visible by decision date with lag control.
    """
    if historical_data is None or historical_data.empty:
        raise ValueError("historical_data is required for agent backtest")
    if "Close" not in historical_data.columns:
        raise ValueError("historical_data must contain 'Close' column")

    prices = historical_data["Close"].astype(float).copy().sort_index()
    returns = prices.pct_change().fillna(0.0)
    if len(prices) < 3:
        return calculate_metrics(returns * 0.0)

    sentiment_full = run_news_sentiment_agent(
        csv_path=sentiment_csv_path,
        start_date=start_date,
        end_date=end_date,
        days=31,
        per_day_batch_size=sentiment_batch_per_day,
        use_cache=use_sentiment_cache,
        cache_file=sentiment_cache_file,
    )
    sentiment_df = _to_daily_sentiment_frame(sentiment_full)
    sentiment_output_dir = Path(daily_sentiment_output_dir)
    if not sentiment_output_dir.is_absolute():
        sentiment_output_dir = PROJECT_ROOT / sentiment_output_dir
    safe_symbol = symbol.replace("-", "_")
    sentiment_output_path = sentiment_output_dir / f"daily_sentiment_{safe_symbol}.csv"
    daily_sentiment_df = _build_daily_sentiment_calendar(
        sentiment_df=sentiment_df,
        calendar_dates=prices.index,
        output_path=sentiment_output_path if save_daily_sentiment_calendar else None,
    )

    event_every_n_days = max(1, int(event_every_n_days))
    price_trigger_pct = max(0.0, float(price_trigger_pct))
    fee_rate = max(0.0, float(trading_fee_bps)) / 10000.0
    cache_root = Path(event_cache_dir)
    if not cache_root.is_absolute():
        cache_root = PROJECT_ROOT / cache_root

    positions = pd.Series(0.0, index=prices.index, dtype=float)
    current_position = 0.0
    event_counter = 0

    for i in range(1, len(prices.index)):
        trade_date = prices.index[i]
        is_event_day = ((i - 1) % event_every_n_days == 0)
        is_price_trigger = enable_price_trigger and abs(float(returns.iloc[i])) >= price_trigger_pct
        is_decision_day = is_event_day or is_price_trigger

        if is_decision_day:
            decision_date = trade_date
            history_cutoff = historical_data[historical_data.index < decision_date]

            if len(history_cutoff) >= 7:
                technical_snapshot = _build_event_technical_snapshot(history_cutoff, symbol=symbol)
                sentiment_snapshot = _build_event_sentiment_snapshot(
                    daily_sentiment_df=daily_sentiment_df,
                    decision_date=decision_date,
                    lookback_days=sentiment_lookback_days,
                    sentiment_lag_days=sentiment_lag_days,
                )

                cache_file = _event_cache_path(cache_root, symbol, decision_date)
                cached = _load_event_cache(cache_file) if use_event_cache else None

                if cached is not None:
                    trader_plan = str(cached.get("trader_plan", "") or "")
                    decision_result = str(cached.get("decision_result", "") or "")
                    risk_result = str(cached.get("risk_result", "") or "")
                    if not risk_result:
                        risk_result = str(cached.get("debate_summary", "") or "")
                    fund_result = str(cached.get("fund_result", "") or "")
                    _, _, mapped_position = _map_chain_output_to_position(
                        decision_text=decision_result,
                        risk_text=risk_result,
                        fund_text=fund_result,
                        technical_result=technical_snapshot,
                        sentiment_result=sentiment_snapshot,
                    )
                    print(f"[Engine] ♻️ Cache hit: {cache_file}")
                else:
                    bull_report = run_bull_agent(sentiment_snapshot, technical_snapshot)
                    _print_agent_output("Bull", bull_report)
                    bear_report = run_bear_agent(sentiment_snapshot, technical_snapshot)
                    _print_agent_output("Bear", bear_report)
                    trader_plan = run_trader_agent(
                        bull_report,
                        bear_report,
                        sentiment_result=sentiment_snapshot,
                        technical_result=technical_snapshot,
                    )
                    _print_agent_output("Trader", trader_plan)
                    risk_result = run_debate_agent(
                        bull_report,
                        bear_report,
                        trader_plan,
                        sentiment_result=sentiment_snapshot,
                        technical_result=technical_snapshot,
                    )
                    _print_agent_output("RiskTeam", risk_result)
                    decision_result = run_decision_agent(
                        risk_result,
                        sentiment_result=sentiment_snapshot,
                        technical_result=technical_snapshot,
                    )
                    _print_agent_output("Decision", decision_result)
                    fund_result = run_fund_manager_agent(
                        risk_result,
                        decision_result,
                        technical_result=technical_snapshot,
                    )
                    _print_agent_output("Fund", fund_result)

                    direction, conviction, mapped_position = _map_chain_output_to_position(
                        decision_text=decision_result,
                        risk_text=risk_result,
                        fund_text=fund_result,
                        technical_result=technical_snapshot,
                        sentiment_result=sentiment_snapshot,
                    )

                    if use_event_cache:
                        _save_event_cache(
                            cache_file,
                            {
                                "symbol": symbol,
                                "decision_date": decision_date.strftime("%Y-%m-%d"),
                                "sentiment_snapshot": sentiment_snapshot,
                                "technical_snapshot": {
                                    "current_price": technical_snapshot.get("current_price"),
                                    "rsi": technical_snapshot.get("rsi"),
                                    "macd": technical_snapshot.get("macd"),
                                    "macd_signal": technical_snapshot.get("macd_signal"),
                                    "volatility": technical_snapshot.get("volatility"),
                                    "sma_7": technical_snapshot.get("sma_7"),
                                    "sma_20": technical_snapshot.get("sma_20"),
                                    "sma_50": technical_snapshot.get("sma_50"),
                                },
                                "bull_report": bull_report,
                                "bear_report": bear_report,
                                "trader_plan": trader_plan,
                                "debate_summary": risk_result,
                                "decision_result": decision_result,
                                "risk_result": risk_result,
                                "fund_result": fund_result,
                                "direction": direction,
                                "conviction": conviction,
                                "mapped_position": mapped_position,
                            },
                        )
                        print(f"[Engine] 💾 Cache saved: {cache_file}")

                current_position = mapped_position
                print(
                    f"[Engine] Position decision @ {decision_date.date()} "
                    f"(event={is_event_day}, price_trigger={is_price_trigger}) => "
                    f"{current_position:+.3f}"
                )
            elif len(history_cutoff) >= 2:
                technical_snapshot = _build_event_technical_snapshot(history_cutoff, symbol=symbol)
                sentiment_snapshot = _build_event_sentiment_snapshot(
                    daily_sentiment_df=daily_sentiment_df,
                    decision_date=decision_date,
                    lookback_days=sentiment_lookback_days,
                    sentiment_lag_days=sentiment_lag_days,
                )
                warmup_bias = _regime_bias_score(technical_snapshot, sentiment_snapshot)
                warmup_mag = 0.08 + 0.45 * abs(warmup_bias)
                current_position = float(np.clip(np.sign(warmup_bias) * warmup_mag, -0.8, 0.8))
                print(
                    f"[Engine] Warmup position @ {decision_date.date()} "
                    f"(event={is_event_day}, price_trigger={is_price_trigger}) => "
                    f"{current_position:+.3f}"
                )
            else:
                current_position = 0.0

            event_counter += 1

        positions.iloc[i] = current_position

    trades = positions.diff().abs().fillna(0.0)
    strategy_returns = returns * positions - trades * fee_rate
    metrics = calculate_metrics(strategy_returns)
    metrics["Event Count"] = event_counter
    metrics["Average Position"] = float(positions.abs().mean())
    metrics["Daily Sentiment Output"] = str(sentiment_output_path) if save_daily_sentiment_calendar else "DISABLED"
    return metrics
