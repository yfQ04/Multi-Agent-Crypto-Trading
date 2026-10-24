"""
Daily decision runner.

Purpose:
- Read the latest available date from BTC.csv (or a user-specified date).
- Run one-day pipeline for that date only.
- Keep cache-first and non-overwrite behavior by default.

Usage:
python runners/run_daily_decision.py
python runners/run_daily_decision.py --date 2026-04-22
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICE_CSV = PROJECT_ROOT / "BTC.csv"
MAIN_PY = PROJECT_ROOT / "main.py"


def _latest_date_from_csv(price_csv: Path) -> str:
    if not price_csv.exists():
        raise FileNotFoundError(f"Price CSV not found: {price_csv}")

    df = pd.read_csv(price_csv)
    if "date" not in df.columns:
        raise ValueError("Price CSV must contain 'date' column")

    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("No valid dates found in price CSV")

    return dates.max().strftime("%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-day (latest-date) decision pipeline")
    parser.add_argument(
        "--date",
        default="",
        help="Decision date (YYYY-MM-DD). If empty, auto-use latest date in BTC.csv",
    )
    parser.add_argument(
        "--price-csv",
        default=str(DEFAULT_PRICE_CSV),
        help="Price CSV path used for auto date detection",
    )
    parser.add_argument(
        "--symbol",
        default="BTC",
        help="Trading symbol (default: BTC)",
    )
    parser.add_argument(
        "--skip-single-llm",
        action="store_true",
        help="If set, do not run SingleLLM baseline",
    )
    parser.add_argument(
        "--skip-multi-agent",
        action="store_true",
        help="If set, do not run Agent3-8 daily pipeline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    price_csv = Path(args.price_csv)
    run_date = args.date.strip() if str(args.date).strip() else _latest_date_from_csv(price_csv)

    if not MAIN_PY.exists():
        raise FileNotFoundError(f"main.py not found: {MAIN_PY}")

    env = os.environ.copy()
    env["RUN_LLM"] = "1"
    env["RUN_DAILY_AGENT_PIPELINE"] = "0" if args.skip_multi_agent else "1"
    env["RUN_SINGLE_LLM_BASELINE"] = "0" if args.skip_single_llm else "1"
    env["PIPELINE_FORCE_RERUN"] = "0"
    env["SINGLE_LLM_BASELINE_FORCE_RERUN"] = "0"
    env["AGENT12_FORCE_LOG_OVERWRITE"] = "0"
    env["DATA_START_DATE"] = run_date
    env["DATA_END_DATE"] = run_date
    env["SYMBOLS"] = str(args.symbol).upper()

    cmd = [sys.executable, str(MAIN_PY)]

    print("=" * 60)
    print("DAILY DECISION RUNNER")
    print("=" * 60)
    print(f"Decision date: {run_date}")
    print("Cache policy: cache-first, no overwrite, no force rerun")
    print(f"Run Agent3-8: {'no' if args.skip_multi_agent else 'yes'}")
    print(f"Run SingleLLM: {'no' if args.skip_single_llm else 'yes'}")
    print("Note: Output decision at this date is intended for next-day action.")
    print("=" * 60)

    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)


if __name__ == "__main__":
    main()
