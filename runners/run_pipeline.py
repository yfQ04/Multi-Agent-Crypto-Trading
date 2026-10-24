"""
Unified runner entry for the daily multi-agent pipeline.

This file replaces split runner entry points in runners/.
Legacy stage scripts are archived under artifacts/legacy_runners/stages/.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_STAGES_DIR = PROJECT_ROOT / "artifacts" / "legacy_runners" / "stages"


def _run(script_path: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script_path), *args]
    print(f"[Runner] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_full(start_date: str, end_date: str, retries: int, force_rerun: bool) -> None:
    stages = [
        ("stage_bull_bear_daily.py", ["--start-date", start_date, "--end-date", end_date]),
        ("stage_trader_daily.py", ["--start-date", start_date, "--end-date", end_date]),
        (
            "stage_risk_daily.py",
            ["--start-date", start_date, "--end-date", end_date, "--retries", str(retries)],
        ),
        (
            "stage_decision_daily.py",
            ["--start-date", start_date, "--end-date", end_date, "--retries", str(retries)],
        ),
        (
            "stage_fund_daily.py",
            ["--start-date", start_date, "--end-date", end_date, "--retries", str(retries)],
        ),
    ]

    for script_name, stage_args in stages:
        script = LEGACY_STAGES_DIR / script_name
        if not script.exists():
            raise FileNotFoundError(f"Missing legacy stage script: {script}")
        args = list(stage_args)
        if force_rerun:
            args.append("--force-rerun")
        _run(script, args)


def run_window_refresh(window_start: str, window_end: str, full_start: str, full_end: str) -> None:
    full_tag = f"{full_start}_{full_end}"
    agent5_input_full = PROJECT_ROOT / "artifacts" / "agent_logs" / "agent5" / f"agent5_input_daily_{full_tag}.csv"
    agent5_log_full = PROJECT_ROOT / "artifacts" / "agent_logs" / "agent5" / f"agent5_daily_llm_logs_{full_tag}.csv"

    if not agent5_input_full.exists():
        raise FileNotFoundError(f"Missing Agent5 input: {agent5_input_full}")
    if not agent5_log_full.exists():
        raise FileNotFoundError(f"Missing Agent5 log: {agent5_log_full}")

    _run(
        LEGACY_STAGES_DIR / "stage_risk_daily.py",
        [
            "--start-date",
            window_start,
            "--end-date",
            window_end,
            "--agent5-input-csv",
            str(agent5_input_full),
            "--agent5-log-csv",
            str(agent5_log_full),
            "--force-rerun",
            "--retries",
            "1",
        ],
    )
    _run(
        LEGACY_STAGES_DIR / "stage_decision_daily.py",
        ["--start-date", window_start, "--end-date", window_end, "--force-rerun", "--retries", "1"],
    )
    _run(
        LEGACY_STAGES_DIR / "stage_fund_daily.py",
        ["--start-date", window_start, "--end-date", window_end, "--force-rerun", "--retries", "1"],
    )

    merge_cmd = [
        sys.executable,
        "-c",
        (
            "import os, pandas as pd; from pathlib import Path; "
            "ws=pd.to_datetime(os.environ['WINDOW_START']); we=pd.to_datetime(os.environ['WINDOW_END']); "
            "full_tag=os.environ['FULL_TAG']; window_tag=os.environ['WINDOW_TAG']; "
            "pairs=["
            "(Path(f'artifacts/agent_logs/agent6/agent6_daily_llm_logs_{full_tag}.csv'),Path(f'artifacts/agent_logs/agent6/agent6_daily_llm_logs_{window_tag}.csv'))," 
            "(Path(f'artifacts/agent_logs/agent7/agent7_daily_llm_logs_{full_tag}.csv'),Path(f'artifacts/agent_logs/agent7/agent7_daily_llm_logs_{window_tag}.csv'))," 
            "(Path(f'artifacts/agent_logs/agent8/agent8_daily_llm_logs_{full_tag}.csv'),Path(f'artifacts/agent_logs/agent8/agent8_daily_llm_logs_{window_tag}.csv'))]; "
            "for full_path,window_path in pairs: "
            " full_df=pd.read_csv(full_path); win_df=pd.read_csv(window_path); "
            " full_df['date']=pd.to_datetime(full_df['date'],errors='coerce'); win_df['date']=pd.to_datetime(win_df['date'],errors='coerce'); "
            " keep=full_df[(full_df['date']<ws)|(full_df['date']>we)].copy(); "
            " merged=pd.concat([keep,win_df],ignore_index=True).dropna(subset=['date']).sort_values('date'); "
            "\n"
            " if 'timestamp_utc' in merged.columns: "
            "  merged['timestamp_utc']=pd.to_datetime(merged['timestamp_utc'],errors='coerce',utc=True); "
            "  merged=merged.sort_values(['date','timestamp_utc']); "
            " merged=merged.groupby(merged['date'].dt.normalize(),as_index=False).tail(1); "
            " merged['date']=pd.to_datetime(merged['date']).dt.strftime('%Y-%m-%d'); merged.to_csv(full_path,index=False)"
        ),
    ]
    env = os.environ.copy()
    env["WINDOW_START"] = window_start
    env["WINDOW_END"] = window_end
    env["FULL_TAG"] = full_tag
    env["WINDOW_TAG"] = f"{window_start}_{window_end}"
    subprocess.run(merge_cmd, check=True, cwd=str(PROJECT_ROOT), env=env)

    _run(
        PROJECT_ROOT / "visualization" / "plot_yearly_strategy_metrics.py",
        ["--start-date", full_start, "--end-date", full_end],
    )


def print_status() -> None:
    logs_dir = PROJECT_ROOT / "artifacts" / "agent_logs"
    print("[Runner] Existing outputs:")
    for name in ["agent34", "agent5", "agent6", "agent7", "agent8"]:
        p = logs_dir / name
        if p.exists():
            count = len(list(p.glob("*.csv")))
            print(f"- {name}: {count} csv files")
        else:
            print(f"- {name}: missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified runner for multi-agent daily pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_full = sub.add_parser("full", help="Run Agent3-8 full daily pipeline")
    p_full.add_argument("--start-date", required=True)
    p_full.add_argument("--end-date", required=True)
    p_full.add_argument("--retries", type=int, default=1)
    p_full.add_argument("--force-rerun", action="store_true")

    p_win = sub.add_parser("window-refresh", help="Refresh Agent6-8 on a date window and merge back")
    p_win.add_argument("window_start")
    p_win.add_argument("window_end")
    p_win.add_argument("full_start")
    p_win.add_argument("full_end")

    sub.add_parser("status", help="Print existing artifact status")

    args = parser.parse_args()

    if args.mode == "full":
        run_full(args.start_date, args.end_date, retries=max(0, args.retries), force_rerun=bool(args.force_rerun))
        return
    if args.mode == "window-refresh":
        run_window_refresh(args.window_start, args.window_end, args.full_start, args.full_end)
        return
    if args.mode == "status":
        print_status()
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
