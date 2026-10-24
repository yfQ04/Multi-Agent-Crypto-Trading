# agents/debate_agent.py
"""
Agent 6: Risk Management Team
- Merge debate + risk assessment into one team output
- Use three internal debators: Aggressive, Neutral, Conservative
"""

from agents.llm_utils import ask_llm


def run_debate_agent(
    bull_report: str,
    bear_report: str,
    trader_plan: str | None = None,
    sentiment_result: dict | None = None,
    technical_result: dict | None = None,
) -> str:
    """
    Agent 6 Risk Management Team:
    Run three internal debators (Aggressive/Neutral/Conservative)
    and output one consolidated team conclusion.
    """
    print("\n" + "=" * 60)
    print("🛡️ Agent 6: Risk Management Team Starting...")
    print("=" * 60)

    trader_plan = trader_plan or "N/A"
    sentiment_result = sentiment_result or {}
    technical_result = technical_result or {}

    prompt = f"""
You are Agent 6: a BTC Risk Management Team composed of three internal debators.

Team Structure and Goals:
1) Aggressive Debator
Goal: Advocate for high-reward, high-risk investment strategies.

2) Neutral Debator
Goal: Provide a balanced perspective on Apple investment.

3) Conservative Debator
Goal: Emphasize a conservative investment strategy with risk mitigation.

Even though one goal mentions Apple investment, you are operating on BTC market inputs here.
Treat it as a "balanced perspective" role and apply it to BTC.

You must run an internal debate and then publish one final team output.
The final output must include both debate synthesis and actionable risk controls.

🐂 Bullish Argument:
{bull_report}

🐻 Bearish Argument:
{bear_report}

📈 Trader Proposed Strategy (Agent 5):
{trader_plan}

Sentiment Context:
- avg_sentiment_score: {float(sentiment_result.get('avg_sentiment_score', 0.0)):.4f}
- weighted_sentiment_score: {float(sentiment_result.get('weighted_sentiment_score', sentiment_result.get('avg_sentiment_score', 0.0))):.4f}
- sentiment_label: {sentiment_result.get('sentiment_label', 'neutral')}
- items_count: {int(sentiment_result.get('items_count', 0))}
- avg_sentiment_confidence: {float(sentiment_result.get('avg_sentiment_confidence', 0.0)):.4f}
- positive_factors_count: {int(sentiment_result.get('positive_factors_count', 0))}
- negative_factors_count: {int(sentiment_result.get('negative_factors_count', 0))}
- neutral_factors_count: {int(sentiment_result.get('neutral_factors_count', 0))}

Technical Context:
- current_price: {float(technical_result.get('current_price', 0.0)):.4f}
- RSI(14): {float(technical_result.get('rsi', 50.0)):.2f}
- MACD / MACD_signal: {float(technical_result.get('macd', 0.0)):.4f} / {float(technical_result.get('macd_signal', 0.0)):.4f}
- SMA_7 / SMA_20 / SMA_50: {float(technical_result.get('sma_7', 0.0)):.4f} / {float(technical_result.get('sma_20', 0.0)):.4f} / {float(technical_result.get('sma_50', 0.0)):.4f}
- annualized_volatility_pct: {float(technical_result.get('volatility', 0.0)):.2f}

Return ONLY one valid JSON object:
{{
    "thinking_summary": "team synthesis with key conflicts and rationale",
    "aggressive_view": "high-reward high-risk argument",
    "neutral_view": "balanced argument",
    "conservative_view": "risk-first argument",
    "bull_score": 0.0,
    "bear_score": 0.0,
    "winner": "bull|bear|tie",
    "confidence": 0.0,
    "risk_level": "Aggressive|Neutral|Conservative",
    "position_multiplier": 0.0,
    "max_drawdown_alert": "low|medium|high",
    "risk_controls": ["control1", "control2", "control3"],
    "key_uncertainties": ["u1", "u2"]
}}

Rules:
- bull_score, bear_score, confidence, position_multiplier in [0,1]
- output must be internally consistent:
  - if risk_level is Conservative, usually lower position_multiplier
  - if volatility is high or uncertainty is high, lower position_multiplier
- bull regime handling:
    - if SMA_7 > SMA_20 > SMA_50 and RSI >= 52 and weighted_sentiment_score > 0,
        avoid overly defensive outputs
    - in the above bullish setup, if winner is bull and confidence >= 0.55,
        position_multiplier should usually be >= 0.60 unless max_drawdown_alert is high
- no markdown and no extra text outside JSON
"""
    return ask_llm(prompt)
