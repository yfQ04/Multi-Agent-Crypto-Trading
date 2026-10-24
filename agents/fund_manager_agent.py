# agents/fund_manager_agent.py
"""
Agent 8: Fund Manager Execution
- Approve and execute the final trading decision
"""

from agents.llm_utils import ask_llm

def run_fund_manager_agent(
    risk_assessment: str,
    trading_decision: str,
    technical_result: dict | None = None,
) -> str:
    """
    Fund Manager Agent: Approve execution of the final trading decision based on risk assessment
    """
    print("\n" + "="*60)
    print("💼 Agent 8: Fund Manager Execution Starting...")
    print("="*60)

    technical_result = technical_result or {}

    prompt = f"""
You are Agent 8: BTC Fund Manager and Execution Committee.
Your objective is to maximize risk-adjusted return and daily positive-PnL probability while keeping losses controlled; avoid unnecessary idle capital.
You MUST integrate decision + risk assessment + technical context, then produce one final executable plan.

Primary Responsibilities:
1) Validate whether the proposed trade is executable today.
2) Convert strategy into concrete risk-bounded execution instructions.
3) Override sizing/action if risk constraints are violated.
4) Keep capital productive: do not default to flat unless strict rejection criteria are met.
5) In favorable market regime, allow more aggressive but still risk-bounded execution.

Technical Context:
- current_price: {float(technical_result.get('current_price', 0.0)):.4f}
- RSI: {float(technical_result.get('rsi', 50.0)):.2f}
- MACD/signal: {float(technical_result.get('macd', 0.0)):.4f}/{float(technical_result.get('macd_signal', 0.0)):.4f}
- SMA_7/SMA_20/SMA_50: {float(technical_result.get('sma_7', 0.0)):.4f}/{float(technical_result.get('sma_20', 0.0)):.4f}/{float(technical_result.get('sma_50', 0.0)):.4f}
- volatility: {float(technical_result.get('volatility', 0.0)):.2f}%

Risk Assessment (Agent 6):
{risk_assessment}

Trading Decision (Agent 7):
{trading_decision}

Return ONLY one valid JSON object:
{{
    "thinking_summary": "brief execution reasoning with risk checks",
    "approval": "approved|modified|rejected",
    "final_action": "long|short|flat",
    "final_conviction": 0.0,
    "position_multiplier_override": 0.0,
    "order_plan": {{
        "order_type": "market|limit|staged_limit",
        "entry_policy": "how to enter",
        "max_slippage_bps": 0,
        "time_in_force": "IOC|GTC|DAY"
    }},
    "risk_plan": {{
        "stop_loss_rule": "clear stop logic",
        "take_profit_rule": "clear take-profit logic",
        "max_daily_loss_limit": "portfolio/day limit"
    }},
    "execution_steps": ["step1", "step2", "step3"],
    "kill_switch_conditions": ["condition1", "condition2"]
}}

Rules:
- final_conviction and position_multiplier_override must be in [0,1]
- choose flat ONLY when BOTH are true:
    1) estimated conviction < 0.35
    2) risk assessment indicates high uncertainty/conflict (e.g., tie/high uncertainty/conservative + high alert)
- if approval is rejected, final_action must be flat and final_conviction <= 0.25
- if approval is approved or modified and final_action is long/short:
    - final_conviction should normally be >= 0.45
    - position_multiplier_override should normally be >= 0.55 unless risk alert is high
- in clearly bullish regime with moderate/low uncertainty, prefer long or modified-long over flat
- if clearly bullish regime (SMA_7 > SMA_20 > SMA_50, RSI >= 52, MACD >= signal) and risk alert is not high:
    - if final_action is long, position_multiplier_override should usually be >= 0.65
    - avoid approval="rejected" unless risk controls are explicitly violated
- if volatility is high or risk controls conflict, prefer modified/rejected over blind approval
- execution_steps must be concrete and operational
- no markdown, no extra text outside JSON
"""
    return ask_llm(prompt)
