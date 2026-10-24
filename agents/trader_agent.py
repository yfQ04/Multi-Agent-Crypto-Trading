# agents/trader_agent.py
"""
Agent 5: Trader Strategy Proposal
- Receive Bull/Bear outputs and market context
- Propose an actionable trading strategy with risk boundaries
"""

from agents.llm_utils import ask_llm


def run_trader_agent(
    bull_report: str,
    bear_report: str,
    sentiment_result: dict | None = None,
    technical_result: dict | None = None,
) -> str:
    """
    Trader Agent: Convert Bull/Bear analysis into an executable strategy proposal
    """
    print("\n" + "=" * 60)
    print("📈 Agent 5: Trader Strategy Proposal Starting...")
    print("=" * 60)

    sentiment_result = sentiment_result or {}
    technical_result = technical_result or {}

    prompt = f"""
You are a professional BTC derivatives trader. You must synthesize conflicting viewpoints into one executable trading plan.
Goal: maximize risk-adjusted return with disciplined entries, exits, and position sizing.

Bull Report:
{bull_report}

Bear Report:
{bear_report}

Sentiment Context:
- avg_sentiment_score: {float(sentiment_result.get('avg_sentiment_score', 0.0)):.4f}
- sentiment_label: {sentiment_result.get('sentiment_label', 'neutral')}
- items_count: {int(sentiment_result.get('items_count', 0))}
- summary: {str(sentiment_result.get('gpt_summary', 'N/A'))[:500]}

Technical Context:
- current_price: {float(technical_result.get('current_price', 0.0)):.4f}
- RSI(14): {float(technical_result.get('rsi', 50.0)):.2f}
- MACD / MACD_signal: {float(technical_result.get('macd', 0.0)):.4f} / {float(technical_result.get('macd_signal', 0.0)):.4f}
- SMA_7 / SMA_20 / SMA_50: {float(technical_result.get('sma_7', 0.0)):.4f} / {float(technical_result.get('sma_20', 0.0)):.4f} / {float(technical_result.get('sma_50', 0.0)):.4f}
- annualized_volatility_pct: {float(technical_result.get('volatility', 0.0)):.2f}

Decision policy:
1) Weigh Bull/Bear evidence quality, not just conclusion count.
2) If signals conflict heavily, prefer smaller size or flat.
3) Must define invalidation and risk controls.
4) Avoid overconfidence under high volatility.

Return ONLY one valid JSON object:
{{
  "thinking_summary": "3-6 concise points in one paragraph",
  "strategy_type": "trend_following|mean_reversion|breakout|range|defensive",
  "action": "long|short|flat",
  "conviction": 0.0,
  "position_size": 0.0,
  "entry_plan": "specific entry logic",
  "stop_loss_rule": "clear stop logic",
  "take_profit_rule": "clear TP logic",
  "time_horizon_days": 1,
  "key_risks": ["risk1", "risk2", "risk3"],
  "monitoring_triggers": ["trigger1", "trigger2"]
}}

Rules:
- conviction and position_size must be in [0,1]
- when action is flat, position_size should be <= 0.2
- if confidence is low or volatility is high, reduce position_size
- no markdown and no extra text outside JSON
"""

    return ask_llm(prompt)
