# agents/bull_agent.py
"""
Agent 3: Bullish Analyst
- Receive sentiment data + technical data
- Build arguments from a bullish perspective
- Output bullish report
"""

from agents.llm_utils import ask_llm

def run_bull_agent(sentiment_result: dict, technical_result: dict) -> str:
    """
    Bull Agent: Analyze all data from a positive perspective and build the strongest bullish argument
    """
    print("\n" + "="*60)
    print("🐂 Agent 3: Bullish Analyst Starting...")
    print("="*60)

    prompt = f"""
You are a bullish BTC analyst. Build the strongest LONG case using BOTH sentiment and technical indicators.
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
    "thinking_summary": "3-6 concise bullish reasoning points in one paragraph",
    "bullish_strength": 0.0,
    "signal": "bullish|neutral|weak_bullish",
    "key_evidence": ["evidence1", "evidence2", "evidence3"],
    "targets": {{"short_term": "...", "medium_term": "..."}},
    "risk_flags": ["risk1", "risk2"]
}}

Rules:
- bullish_strength in [0,1]
- no markdown and no extra text outside JSON
"""
    return ask_llm(prompt)
