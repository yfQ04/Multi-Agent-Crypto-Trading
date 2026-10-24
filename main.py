# main.py
"""
Main Entry Point
- Orchestrate the flow of agents and strategies
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from agents.news_sentiment_agent import run_news_sentiment_agent
from agents.technical_agent import run_technical_agent_for_symbol
from agents.bull_agent import run_bull_agent
from agents.bear_agent import run_bear_agent
from agents.trader_agent import run_trader_agent
from agents.debate_agent import run_debate_agent
from agents.decision_agent import run_decision_agent
from agents.fund_manager_agent import run_fund_manager_agent

load_dotenv()

ALLOWED_SYMBOLS = {"BTC"}
BACKTEST_PERIOD = "1mo"
DATA_START_DATE = os.getenv("DATA_START_DATE", "2025-03-10")
DATA_END_DATE = os.getenv("DATA_END_DATE", "2026-03-10")
AGENT1_MARKET_DATA_PATH = "BTC.csv"
AGENT1_TWEETS_DATA_PATH = "bitcoin_tweets_cleaned.csv"
AGENT1_NEWS_DATA_PATH = "historical_news.csv"
UNIFIED_RUNNER = Path(__file__).resolve().parent / "runners" / "run_pipeline.py"
SINGLE_LLM_BASELINE_RUNNER = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "legacy_runners"
    / "stages"
    / "stage_single_llm_daily.py"
)


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.getenv(name, default)
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_symbols(raw: str) -> list[str]:
    if not raw:
        return []
    parsed = [s.strip().upper() for s in raw.split(",") if s.strip()]
    normalized = [s.split("-")[0] for s in parsed]
    filtered = [s for s in normalized if s in ALLOWED_SYMBOLS]
    if parsed and not filtered:
        return ["BTC"]
    return filtered


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(str(val).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(str(val).strip())
    except Exception:
        return default


def _run_python_script(script_path: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script_path), *args]
    print(f"[Pipeline] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _asset_tag(symbol: str) -> str:
    base = str(symbol or "").split("-")[0].strip().upper()
    return base if base else str(symbol or "BTC").replace("-", "_").upper()


def _date_tag(start_date: str, end_date: str) -> str:
    return f"{start_date}_{end_date}"


def _save_csv_no_overwrite(df, output_path: Path, overwrite: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        print(f"[Logs] Skip existing file (no overwrite): {output_path}")
        return
    df.to_csv(output_path, index=False)
    print(f"[Logs] Saved: {output_path}")


def _progress_delay_seconds() -> float:
    # Shared demo pacing knob for per-line progress logs.
    raw = os.getenv("AGENT_PROGRESS_DELAY_SEC", "0.25")
    try:
        value = float(str(raw).strip())
    except Exception:
        return 0.25
    return max(0.0, min(value, 2.0))


def _agent2_delay_seconds() -> float:
    # Optional override if Agent2 should be slower/faster than others.
    raw = os.getenv("AGENT2_PROGRESS_DELAY_SEC", "")
    if str(raw).strip() == "":
        return _progress_delay_seconds()
    try:
        value = float(str(raw).strip())
    except Exception:
        return _progress_delay_seconds()
    return max(0.0, min(value, 2.0))


def _paced_print(line: str, delay_s: float | None = None) -> None:
    print(line, flush=True)
    delay = _progress_delay_seconds() if delay_s is None else max(0.0, min(delay_s, 2.0))
    if delay > 0:
        time.sleep(delay)


def _print_agent2_daily_progress(df, status: str) -> None:
    if df is None or len(df) == 0 or "date" not in df.columns:
        return
    for _, row in df.iterrows():
        day = str(row.get("date", ""))
        close = row.get("Close", "")
        rsi = row.get("RSI_14", "")
        macd = row.get("MACD", "")
        mark = "✅" if str(status).upper() in {"SUCCEED", "FROM CACHE"} else "❌"
        _paced_print(
            f"[Agent2] 📈 {day} | {status} {mark} close={close} rsi={rsi} macd={macd}",
            delay_s=_agent2_delay_seconds(),
        )


def _find_existing_log_file(logs_root: Path, rel_path: str, pattern: str) -> Path | None:
    expected = logs_root / rel_path
    if expected.exists():
        return expected
    candidates = sorted(logs_root.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]


def _print_daily_csv_progress(csv_path: Path, agent_name: str, key_fields: list[str]) -> None:
    import pandas as pd

    if not csv_path.exists():
        print(f"[{agent_name}] Daily log not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[{agent_name}] Failed reading daily log {csv_path}: {e}")
        return

    if df.empty or "date" not in df.columns:
        print(f"[{agent_name}] Daily log has no rows/date: {csv_path}")
        return

    _paced_print(f"[{agent_name}] 📄 Daily log source: {csv_path}")
    for _, row in df.iterrows():
        day = str(row.get("date", ""))
        status = str(row.get("status", ""))
        parts = []
        for field in key_fields:
            if field in df.columns:
                parts.append(f"{field}={row.get(field, '')}")
        joined = " ".join(parts)
        ok_statuses = {"succeed", "success", "from cache"}
        mark = "✅" if status.strip().lower() in ok_statuses else "❌"
        if joined:
            _paced_print(f"[{agent_name}] {day} | {status} {mark} {joined}")
        else:
            _paced_print(f"[{agent_name}] {day} | {status} {mark}")


def _print_agent3_to_8_daily_progress(start_date: str, end_date: str) -> None:
    root = Path(__file__).resolve().parent
    logs_root = root / "artifacts" / "agent_logs"
    tag = _date_tag(start_date, end_date)

    specs = [
        {
            "agent": "Agent3-4",
            "rel": f"agent34/agent3_4_daily_llm_logs_{tag}.csv",
            "pattern": "agent34/agent3_4_daily_llm_logs_*.csv",
            "fields": ["agent", "signal", "strength"],
        },
        {
            "agent": "Agent5",
            "rel": f"agent5/agent5_daily_llm_logs_{tag}.csv",
            "pattern": "agent5/agent5_daily_llm_logs_*.csv",
            "fields": ["action", "strategy_type", "conviction", "position_size"],
        },
        {
            "agent": "Agent6",
            "rel": f"agent6/agent6_daily_llm_logs_{tag}.csv",
            "pattern": "agent6/agent6_daily_llm_logs_*.csv",
            "fields": ["winner", "confidence", "risk_level", "position_multiplier"],
        },
        {
            "agent": "Agent7",
            "rel": f"agent7/agent7_daily_llm_logs_{tag}.csv",
            "pattern": "agent7/agent7_daily_llm_logs_*.csv",
            "fields": ["action", "direction_preference", "conviction"],
        },
        {
            "agent": "Agent8",
            "rel": f"agent8/agent8_daily_llm_logs_{tag}.csv",
            "pattern": "agent8/agent8_daily_llm_logs_*.csv",
            "fields": ["approval", "final_action", "final_conviction", "position_multiplier_override"],
        },
    ]

    print("\n" + "=" * 60)
    print("📅 DAILY STATUS & RESULTS (AGENT3-8)")
    print("=" * 60)
    for spec in specs:
        path = _find_existing_log_file(logs_root, spec["rel"], spec["pattern"])
        if path is None:
            print(f"[{spec['agent']}] Daily log not found for tag={tag}")
            continue
        _print_daily_csv_progress(path, spec["agent"], spec["fields"])


def _print_single_llm_baseline_progress(start_date: str, end_date: str) -> None:
    root = Path(__file__).resolve().parent
    logs_root = root / "artifacts" / "agent_logs"
    tag = _date_tag(start_date, end_date)
    rel = f"agent9_single_llm/agent9_single_llm_daily_llm_logs_{tag}.csv"
    pattern = "agent9_single_llm/agent9_single_llm_daily_llm_logs_*.csv"
    path = _find_existing_log_file(logs_root, rel, pattern)
    if path is None:
        print(f"[SingleLLM] Daily log not found for tag={tag}")
        return
    print("\n" + "=" * 60)
    print("📅 DAILY STATUS & RESULTS (SINGLE LLM BASELINE)")
    print("=" * 60)
    _print_daily_csv_progress(
        path,
        "SingleLLM",
        ["action", "conviction", "model_name", "reason"],
    )


def _print_agent1_daily_progress(start_date: str, end_date: str) -> None:
    import pandas as pd

    root = Path(__file__).resolve().parent
    summary_csv = root / "artifacts" / "agent_logs" / "agent1" / "sentiment_daily_summary.csv"

    if not summary_csv.exists():
        print(f"[Agent1] Daily summary not found: {summary_csv}")
        return

    try:
        df = pd.read_csv(summary_csv)
    except Exception as e:
        print(f"[Agent1] Failed reading daily summary {summary_csv}: {e}")
        return

    if df.empty or "date" not in df.columns:
        print(f"[Agent1] Daily summary has no rows/date: {summary_csv}")
        return

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    st = pd.to_datetime(start_date, errors="coerce")
    ed = pd.to_datetime(end_date, errors="coerce")
    if pd.notna(st):
        df = df[df["date"] >= st]
    if pd.notna(ed):
        df = df[df["date"] <= ed]

    if df.empty:
        print(f"[Agent1] No rows found in date range: {start_date} -> {end_date}")
        return

    df = df.sort_values("date")
    _paced_print(f"[Agent1] 📄 Daily summary source: {summary_csv}")
    for _, row in df.iterrows():
        day = pd.to_datetime(row.get("date"), errors="coerce")
        day_str = day.strftime("%Y-%m-%d") if pd.notna(day) else str(row.get("date", ""))
        items = int(row.get("items_count", 0) or 0)
        avg_score = row.get("avg_sentiment_score", "")
        pos_ratio = row.get("positive_ratio", "")
        neg_ratio = row.get("negative_ratio", "")
        status = "FROM CACHE"
        mark = "✅"
        _paced_print(
            f"[Agent1] 📰 {day_str} | {status} {mark} items={items} avg_score={avg_score} pos_ratio={pos_ratio} neg_ratio={neg_ratio}"
        )


def _print_agent2_cached_progress(symbol: str, start_date: str, end_date: str) -> None:
    import pandas as pd

    root = Path(__file__).resolve().parent
    logs_root = root / "artifacts" / "agent_logs"
    asset = _asset_tag(symbol)
    tag = _date_tag(start_date, end_date)

    rel = f"agent2/agent2_daily_technical_indicators_{asset}_{tag}.csv"
    pattern = f"agent2/agent2_daily_technical_indicators_{asset}_*.csv"
    daily_path = _find_existing_log_file(logs_root, rel, pattern)
    if daily_path is None:
        print(f"[Agent2] Daily indicators log not found for {asset} tag={tag}")
        return

    try:
        df = pd.read_csv(daily_path)
    except Exception as e:
        print(f"[Agent2] Failed reading cached daily log {daily_path}: {e}")
        return

    if df.empty or "date" not in df.columns:
        print(f"[Agent2] Cached daily log has no rows/date: {daily_path}")
        return

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    st = pd.to_datetime(start_date, errors="coerce")
    ed = pd.to_datetime(end_date, errors="coerce")
    if pd.notna(st):
        df = df[df["date"] >= st]
    if pd.notna(ed):
        df = df[df["date"] <= ed]
    if df.empty:
        print(f"[Agent2] No rows found in date range: {start_date} -> {end_date}")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    _paced_print(f"[Agent2] 📄 Daily indicators source: {daily_path}")
    _print_agent2_daily_progress(df, status="FROM CACHE")


def _replay_agent1_to_8_from_cache(symbol: str, start_date: str, end_date: str) -> None:
    print("\n" + "=" * 60)
    print("♻️  CACHE REPLAY MODE: DAILY STATUS & RESULTS (AGENT1-8)")
    print("=" * 60)
    print("Replay only: no LLM calls, no rerun, only existing cached logs.")
    _print_agent1_daily_progress(start_date, end_date)
    _print_agent2_cached_progress(symbol, start_date, end_date)
    _print_agent3_to_8_daily_progress(start_date, end_date)
    _print_single_llm_baseline_progress(start_date, end_date)


def _has_agent1_cache_for_range(start_date: str, end_date: str) -> bool:
    import pandas as pd

    root = Path(__file__).resolve().parent
    summary_csv = root / "artifacts" / "agent_logs" / "agent1" / "sentiment_daily_summary.csv"
    if not summary_csv.exists():
        return False
    try:
        df = pd.read_csv(summary_csv)
    except Exception:
        return False
    if df.empty or "date" not in df.columns:
        return False
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    st = pd.to_datetime(start_date, errors="coerce")
    ed = pd.to_datetime(end_date, errors="coerce")
    if pd.notna(st):
        df = df[df["date"] >= st]
    if pd.notna(ed):
        df = df[df["date"] <= ed]
    return not df.empty


def _has_agent2_cache_for_range(symbol: str, start_date: str, end_date: str) -> bool:
    import pandas as pd

    root = Path(__file__).resolve().parent
    logs_root = root / "artifacts" / "agent_logs"
    asset = _asset_tag(symbol)
    tag = _date_tag(start_date, end_date)
    rel = f"agent2/agent2_daily_technical_indicators_{asset}_{tag}.csv"
    pattern = f"agent2/agent2_daily_technical_indicators_{asset}_*.csv"
    daily_path = _find_existing_log_file(logs_root, rel, pattern)
    if daily_path is None:
        return False
    try:
        df = pd.read_csv(daily_path)
    except Exception:
        return False
    if df.empty or "date" not in df.columns:
        return False
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    st = pd.to_datetime(start_date, errors="coerce")
    ed = pd.to_datetime(end_date, errors="coerce")
    if pd.notna(st):
        df = df[df["date"] >= st]
    if pd.notna(ed):
        df = df[df["date"] <= ed]
    return not df.empty


def _save_agent12_logs(
    symbol: str,
    start_date: str,
    end_date: str,
    technical_result: dict,
    sentiment_result: dict | None = None,
    overwrite: bool = False,
) -> None:
    import pandas as pd

    root = Path(__file__).resolve().parent
    logs_root = root / "artifacts" / "agent_logs"
    asset = _asset_tag(symbol)
    tag = _date_tag(start_date, end_date)

    # Agent1 logs (from sentiment_cache_readable.csv)
    if isinstance(sentiment_result, dict):
        agent1_dir = logs_root / "agent1"
        readable_path = agent1_dir / "sentiment_cache_readable.csv"
        if readable_path.exists():
            try:
                readable_df = pd.read_csv(readable_path)
                if "date" in readable_df.columns:
                    readable_df["date"] = pd.to_datetime(readable_df["date"], errors="coerce")
                    st = pd.to_datetime(start_date, errors="coerce")
                    ed = pd.to_datetime(end_date, errors="coerce")
                    if pd.notna(st):
                        readable_df = readable_df[readable_df["date"] >= st]
                    if pd.notna(ed):
                        readable_df = readable_df[readable_df["date"] <= ed]
                    readable_df["date"] = pd.to_datetime(readable_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                _save_csv_no_overwrite(
                    readable_df,
                    agent1_dir / f"agent1_sentiment_cache_readable_{asset}_{tag}.csv",
                    overwrite=overwrite,
                )
            except Exception as e:
                print(f"[Logs] Failed exporting Agent1 readable sentiment log: {e}")

    # Agent2 logs (daily indicators only)
    agent2_dir = logs_root / "agent2"
    tech_df = technical_result.get("df") if isinstance(technical_result, dict) else None
    if tech_df is not None and hasattr(tech_df, "reset_index"):
        export_df = tech_df.copy().reset_index()
        if len(export_df.columns) > 0 and export_df.columns[0] != "date":
            export_df = export_df.rename(columns={export_df.columns[0]: "date"})

        preferred_cols = [
            "date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "SMA_7",
            "SMA_20",
            "SMA_50",
            "EMA_12",
            "EMA_26",
            "MACD",
            "MACD_signal",
            "MACD_hist",
            "RSI_14",
            "Volatility",
            "Returns",
        ]
        keep_cols = [c for c in preferred_cols if c in export_df.columns]
        if keep_cols:
            export_df = export_df[keep_cols]
        daily_path = agent2_dir / f"agent2_daily_technical_indicators_{asset}_{tag}.csv"
        if daily_path.exists() and not overwrite:
            print(f"[Agent2] FROM CACHE: {daily_path}")
            _print_agent2_daily_progress(export_df, status="FROM CACHE")
        else:
            _save_csv_no_overwrite(
                export_df,
                daily_path,
                overwrite=overwrite,
            )
            _print_agent2_daily_progress(export_df, status="SUCCEED")


def run_daily_agent_pipeline(start_date: str, end_date: str, retries: int = 1, force_rerun: bool = True) -> None:
    """Run Agent3-8 daily pipeline through the unified runner entry."""
    if not UNIFIED_RUNNER.exists():
        raise FileNotFoundError(f"Unified runner script missing: {UNIFIED_RUNNER}")

    args = [
        "full",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--retries",
        str(retries),
    ]
    if force_rerun:
        args.append("--force-rerun")
    _run_python_script(UNIFIED_RUNNER, args)


def run_single_llm_baseline_pipeline(start_date: str, end_date: str, retries: int = 1, force_rerun: bool = False) -> None:
    """Run independent single-LLM daily baseline (not reusing multi-agent chain)."""
    if not SINGLE_LLM_BASELINE_RUNNER.exists():
        raise FileNotFoundError(f"Single LLM baseline runner missing: {SINGLE_LLM_BASELINE_RUNNER}")

    args = [
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--price-csv",
        AGENT1_MARKET_DATA_PATH,
        "--retries",
        str(retries),
    ]
    if force_rerun:
        args.append("--force-rerun")
    _run_python_script(SINGLE_LLM_BASELINE_RUNNER, args)

def main():
    print("="*60)
    print("🚀 STARTING MULTI-AGENT CRYPTO TRADING SYSTEM")
    print("="*60)
    
    # Multi-asset control
    # Example: SYMBOLS="BTC" RUN_LLM=0 python main.py
    symbols = _parse_symbols(os.getenv("SYMBOLS", "BTC"))
    if not symbols:
        symbols = ["BTC"]
    primary_symbol = symbols[0]
    event_every_n_days = max(1, _env_int("EVENT_EVERY_N_DAYS", 1))
    sentiment_lag_days = max(0, _env_int("SENTIMENT_LAG_DAYS", 1))
    sentiment_lookback_days = max(1, _env_int("SENTIMENT_LOOKBACK_DAYS", 7))
    trading_fee_bps = max(0.0, _env_float("TRADING_FEE_BPS", 10.0))
    sentiment_batch_per_day = max(1, _env_int("SENTIMENT_BATCH_PER_DAY", 10))
    use_sentiment_cache = _env_flag("USE_SENTIMENT_CACHE", default="1")
    sentiment_cache_file = os.getenv("SENTIMENT_CACHE_FILE", "artifacts/agent_logs/agent1/sentiment_cache.json")
    use_event_cache = _env_flag("USE_EVENT_CACHE", default="1")
    event_cache_dir = os.getenv("EVENT_CACHE_DIR", "artifacts/agent_chain_cache")
    save_daily_sentiment_calendar = _env_flag("SAVE_DAILY_SENTIMENT_CALENDAR", default="1")
    daily_sentiment_output_dir = os.getenv("DAILY_SENTIMENT_OUTPUT_DIR", "artifacts/daily_sentiment")
    enable_price_trigger = _env_flag("ENABLE_PRICE_TRIGGER", default="1")
    price_trigger_pct = max(0.0, _env_float("PRICE_TRIGGER_PCT", 0.015))

    # Step 1: (Optional) Run LLM agent workflow ONCE for the primary symbol.
    # Running LLM agents for multiple symbols may significantly increase API usage/cost.
    run_llm = _env_flag("RUN_LLM", default="0")
    run_daily_pipeline = _env_flag("RUN_DAILY_AGENT_PIPELINE", default="1")
    pipeline_retries = max(0, _env_int("PIPELINE_RETRIES", 1))
    pipeline_force_rerun = _env_flag("PIPELINE_FORCE_RERUN", default="0")
    run_single_llm_baseline = _env_flag("RUN_SINGLE_LLM_BASELINE", default="1")
    single_llm_baseline_retries = max(0, _env_int("SINGLE_LLM_BASELINE_RETRIES", 1))
    single_llm_baseline_force_rerun = _env_flag("SINGLE_LLM_BASELINE_FORCE_RERUN", default="0")
    run_legacy_chain_when_no_daily = _env_flag("RUN_LEGACY_CHAIN_WHEN_NO_DAILY", default="0")
    agent12_force_log_overwrite = _env_flag("AGENT12_FORCE_LOG_OVERWRITE", default="0")
    replay_cache_only = _env_flag("REPLAY_CACHE_ONLY", default="0")

    if replay_cache_only:
        _replay_agent1_to_8_from_cache(primary_symbol, DATA_START_DATE, DATA_END_DATE)
        print("\n" + "="*60)
        print("✅ MAIN PIPELINE FINISHED")
        print("="*60)
        print("Use visualization/plot_yearly_strategy_metrics.py for the 12-chart report.")
        return

    if run_llm:
        cache_first_ok = (
            (not pipeline_force_rerun)
            and (not agent12_force_log_overwrite)
            and use_sentiment_cache
            and _has_agent1_cache_for_range(DATA_START_DATE, DATA_END_DATE)
            and _has_agent2_cache_for_range(primary_symbol, DATA_START_DATE, DATA_END_DATE)
        )

        sentiment_data = None
        technical_primary = None

        if cache_first_ok:
            print("\n" + "="*60)
            print("♻️  CACHE HIT: AGENT1-2")
            print("="*60)
            print("RUN_LLM=1 but Agent1/2 cache is complete -> using FROM CACHE.")
            _print_agent1_daily_progress(DATA_START_DATE, DATA_END_DATE)
            _print_agent2_cached_progress(primary_symbol, DATA_START_DATE, DATA_END_DATE)
        else:
            sentiment_data = run_news_sentiment_agent(
                csv_path=AGENT1_TWEETS_DATA_PATH,
                news_csv_path=AGENT1_NEWS_DATA_PATH,
                start_date=DATA_START_DATE,
                end_date=DATA_END_DATE,
                days=31,
                per_day_batch_size=sentiment_batch_per_day,
                use_cache=use_sentiment_cache,
                cache_file=sentiment_cache_file,
            )

            technical_primary = run_technical_agent_for_symbol(
                symbol=primary_symbol,
                period=BACKTEST_PERIOD,
                interval="1d",
                use_llm=True,
                start_date=DATA_START_DATE,
                end_date=DATA_END_DATE,
                preferred_csv_path=AGENT1_MARKET_DATA_PATH,
            )

            _save_agent12_logs(
                symbol=primary_symbol,
                start_date=DATA_START_DATE,
                end_date=DATA_END_DATE,
                technical_result=technical_primary,
                sentiment_result=sentiment_data,
                overwrite=agent12_force_log_overwrite,
            )

        if run_daily_pipeline:
            print("\n" + "="*60)
            print("🧩 RUNNING UNIFIED DAILY AGENT PIPELINE (Agent3-8)")
            print("="*60)
            run_daily_agent_pipeline(
                start_date=DATA_START_DATE,
                end_date=DATA_END_DATE,
                retries=pipeline_retries,
                force_rerun=pipeline_force_rerun,
            )
            _print_agent3_to_8_daily_progress(DATA_START_DATE, DATA_END_DATE)

        if run_single_llm_baseline:
            print("\n" + "="*60)
            print("🧪 RUNNING SINGLE LLM BASELINE (Independent Strategy)")
            print("="*60)
            run_single_llm_baseline_pipeline(
                start_date=DATA_START_DATE,
                end_date=DATA_END_DATE,
                retries=single_llm_baseline_retries,
                force_rerun=single_llm_baseline_force_rerun,
            )
            _print_single_llm_baseline_progress(DATA_START_DATE, DATA_END_DATE)

        if (not run_daily_pipeline) and run_legacy_chain_when_no_daily:
            if sentiment_data is None or technical_primary is None:
                raise RuntimeError(
                    "RUN_LEGACY_CHAIN_WHEN_NO_DAILY=1 requires live Agent1/2 outputs. "
                    "Set PIPELINE_FORCE_RERUN=1 or AGENT12_FORCE_LOG_OVERWRITE=1 if cache is empty."
                )
            bull_report = run_bull_agent(sentiment_data, technical_primary)
            bear_report = run_bear_agent(sentiment_data, technical_primary)
            trader_plan = run_trader_agent(
                bull_report,
                bear_report,
                sentiment_result=sentiment_data,
                technical_result=technical_primary,
            )
            risk_team_result = run_debate_agent(
                bull_report,
                bear_report,
                trader_plan,
                sentiment_result=sentiment_data,
                technical_result=technical_primary,
            )
            trading_decision = run_decision_agent(
                risk_team_result,
                sentiment_result=sentiment_data,
                technical_result=technical_primary,
            )
            execution_report = run_fund_manager_agent(
                risk_team_result,
                trading_decision,
                technical_result=technical_primary,
            )

            print("\n" + "="*60)
            print(f"🎯 FINAL LIVE TRADING STRATEGY (AGENT DECISION) [{primary_symbol}]")
            print("="*60)
            print(execution_report)
            print("="*60)
    else:
        technical_primary = run_technical_agent_for_symbol(
            symbol=primary_symbol,
            period=BACKTEST_PERIOD,
            interval="1d",
            use_llm=False,
            start_date=DATA_START_DATE,
            end_date=DATA_END_DATE,
            preferred_csv_path=AGENT1_MARKET_DATA_PATH,
        )
        _save_agent12_logs(
            symbol=primary_symbol,
            start_date=DATA_START_DATE,
            end_date=DATA_END_DATE,
            technical_result=technical_primary,
            sentiment_result=None,
            overwrite=agent12_force_log_overwrite,
        )

        if run_daily_pipeline:
            _replay_agent1_to_8_from_cache(primary_symbol, DATA_START_DATE, DATA_END_DATE)

        if run_single_llm_baseline:
            print("\n" + "="*60)
            print("ℹ️  RUN_LLM=0: Skipping Single LLM baseline (requires real LLM)")
            print("="*60)

        print("\n" + "="*60)
        print("ℹ️  RUN_LLM=0: Skipping LLM agents (backtest-only mode)")
        print("="*60)

    print("\n" + "="*60)
    print("✅ MAIN PIPELINE FINISHED")
    print("="*60)
    print("Use visualization/plot_yearly_strategy_metrics.py for the 14-chart report.")

if __name__ == "__main__":
    main()
