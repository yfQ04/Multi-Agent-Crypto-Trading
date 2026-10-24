# agents/decision_agent.py
"""
Agent 7: Final Decision Proposal
- Generate final decision based on debate results
"""

from agents.llm_utils import ask_llm

def run_decision_agent(
    debate_summary: str,
    sentiment_result: dict | None = None,
    technical_result: dict | None = None,
) -> str:
    """
    Decision Agent: Generate final trading proposal based on debate assessment
    """
    print("\n" + "="*60)
    print("🧐 Agent 7: Final Decision Proposal Starting...")
    print("="*60)

    sentiment_result = sentiment_result or {}
    technical_result = technical_result or {}

    prompt = f"""
  You are a trading strategist for BTC. Your objective is high risk-adjusted return and high daily positive-PnL probability, not blind optimism.
You MUST jointly consider sentiment + technical indicators.

  Priority:
  1) Improve the chance of making money day-by-day.
  2) In strong bull regime, be more willing to hold directional long exposure.
  3) In weak/uncertain regime, reduce exposure or use flat.

Sentiment Inputs:
- avg_sentiment_score: {float(sentiment_result.get('avg_sentiment_score', 0.0)):.4f}
- sentiment_label: {sentiment_result.get('sentiment_label', 'neutral')}
- items_count: {int(sentiment_result.get('items_count', 0))}
- summary: {str(sentiment_result.get('gpt_summary', 'N/A'))[:500]}

Technical Inputs:
- current_price: {float(technical_result.get('current_price', 0.0)):.4f}
- RSI(14): {float(technical_result.get('rsi', 50.0)):.2f}
- MACD: {float(technical_result.get('macd', 0.0)):.4f}
- MACD_signal: {float(technical_result.get('macd_signal', 0.0)):.4f}
- SMA_7: {float(technical_result.get('sma_7', 0.0)):.4f}
- SMA_20: {float(technical_result.get('sma_20', 0.0)):.4f}
- SMA_50: {float(technical_result.get('sma_50', 0.0)):.4f}
- volatility: {float(technical_result.get('volatility', 0.0)):.2f}%

Agent 6 Risk Management Team Output:
{debate_summary}

Return ONLY one valid JSON object with this schema:
{{
  "thinking_summary": "3-6 concise bullets in one paragraph, evidence-based",
  "action": "long|short|flat",
    "direction_preference": "long|short",
  "conviction": 0.0,
  "time_horizon_days": 1,
  "key_drivers": ["driver1", "driver2", "driver3"],
  "invalidations": ["condition1", "condition2"],
  "execution_plan": "how to implement"
}}

Rules:
- conviction must be in [0,1]
- direction_preference is mandatory and must be long or short (never flat)
- choose flat ONLY when BOTH conditions hold:
    1) conviction < 0.35
    2) risk team output indicates high uncertainty/conflict (e.g., tie winner, high uncertainty, or conservative low-confidence stance)
- if market regime appears bullish (SMA stack up, RSI supportive, positive sentiment), avoid unnecessary flat and prefer long with actionable conviction
- do not include markdown, explanations, or extra text outside JSON
- if action is flat, still provide direction_preference for tactical bias mapping
"""

    return ask_llm(prompt)
