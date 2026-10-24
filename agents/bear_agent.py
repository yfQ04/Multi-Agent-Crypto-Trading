# agents/bear_agent.py
"""
Agent 4: Bearish Analyst
- Receive sentiment data + technical data
- Build arguments from a bearish perspective
- Output bearish report
"""

from agents.llm_utils import ask_llm

def run_bear_agent(sentiment_result: dict, technical_result: dict) -> str:
    """
    Bear Agent: Analyze all data from a negative perspective and build the strongest bearish argument
    """
    print("\n" + "="*60)
    print("🐻 Agent 4: Bearish Analyst Starting...")
    print("="*60)

    prompt = f"""
You are a bearish BTC analyst. Build the strongest SHORT/RISK-OFF case using BOTH sentiment and technical indicators.
Focus on evidence quality and practical trading signal.

Provided Data:
- Average Sentiment Score: {sentiment_result['avg_sentiment_score']:.4f} (range -1 to +1)
- Sentiment Label: {sentiment_result['sentiment_label']}
- Analyzed Items Count: {sentiment_result['items_count']}
- Sentiment Report Summary: {sentiment_result['gpt_summary'][:500]}

【Technical Indicators Data】
- Current Price: ${technical_result['current_price']:,.2f}
- RSI(14): {technical_result['rsi']:.1f}
- MACD: {technical_result['macd']:.4f} (Signal: {technical_result['macd_signal']:.4f})
- 7-Day SMA: ${technical_result['sma_7']:,.2f}
- 20-Day SMA: ${technical_result['sma_20']:,.2f}
- 50-Day SMA: ${technical_result['sma_50']:,.2f}
- Annualized Volatility: {technical_result['volatility']:.1f}%
- Technical Analysis Summary: {technical_result['gpt_analysis'][:500]}

Return ONLY one valid JSON object:
{{
    "thinking_summary": "3-6 concise bearish reasoning points in one paragraph",
    "bearish_strength": 0.0,
    "signal": "bearish|neutral|weak_bearish",
    "key_evidence": ["evidence1", "evidence2", "evidence3"],
    "downside_targets": {{"short_term": "...", "medium_term": "..."}},
    "risk_flags": ["risk1", "risk2"]
}}

Rules:
- bearish_strength in [0,1]
- no markdown and no extra text outside JSON
"""
    return ask_llm(prompt)
