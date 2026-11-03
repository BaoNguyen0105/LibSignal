"""
Automated Federated Traffic Signal Control Benchmark Runner.

Executes a matrix of experiments combining different Agents, Trainers,
Client Selectors, Aggregators, Compressors, and Hardware/Network Profiles,
then aggregates the resulting metrics into a comparison table and CSV file.

Usage:
    python run_fl_benchmarks.py                  # Run all 8 benchmarks (10 episodes each)
    python run_fl_benchmarks.py --episodes 4     # Run all 8 benchmarks with 4 episodes
    python run_fl_benchmarks.py --only 1 3       # Run only experiments #1 and #3
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time


EXPERIMENTS = [
    {
        "id": 1,
        "prefix": "exp1_dqn_sync_baseline",
        "agent": "fl_dqn",
        "trainer": "sync",
        "selection": "all",
        "aggregation": "fedavg",
        "compression": "none",
        "profile": "simple",
        "extra_args": [],
    },
    {
        "id": 2,
        "prefix": "exp2_presslight_fedcs_fednova",
        "agent": "fl_presslight",
        "trainer": "sync",
        "selection": "fedcs",
        "aggregation": "fednova",
        "compression": "quantization",
        "profile": "tiered",
        "extra_args": ["--fraction", "0.5", "--weighted", "--bits", "8"],
    },
    {
        "id": 3,
        "prefix": "exp3_frap_fairfedcs_unbiased",
        "agent": "fl_frap",
        "trainer": "sync",
        "selection": "fair_fedcs",
        "aggregation": "unbiased",
        "compression": "adaptive",
        "profile": "wireless",
        "extra_args": ["--fraction", "0.5", "--fairness_weight", "0.3", "--weighted"],
    },
    {
        "id": 4,
        "prefix": "exp4_sac_resource_fedavgm",
        "agent": "fl_sac",
        "trainer": "sync",
        "selection": "resource",
        "aggregation": "fedavgm",
        "compression": "low_rank",
        "profile": "markov",
        "extra_args": ["--fraction", "0.5", "--weighted"],
    },
    {
        "id": 5,
        "prefix": "exp5_presslight_fedbuff",
        "agent": "fl_presslight",
        "trainer": "semi_async",
        "selection": "fedcs",
        "aggregation": "unbiased",
        "compression": "topk",
        "profile": "wireless",
        "extra_args": ["--fraction", "0.75", "--fairness_weight", "0.2", "--weighted"],
    },
    {
        "id": 6,
        "prefix": "exp6_dqn_async",
        "agent": "fl_dqn",
        "trainer": "async",
        "selection": "resource",
        "aggregation": "fedavg",
        "compression": "adaptive",
        "profile": "markov",
        "extra_args": ["--fraction", "0.75"],
    },
    {
        "id": 7,
        "prefix": "exp7_frap_step_fl",
        "agent": "fl_frap",
        "trainer": "step",
        "selection": "random",
        "aggregation": "fednova",
        "compression": "quantization",
        "profile": "tiered",
        "extra_args": ["--aggregate_steps", "90", "--fraction", "0.5", "--weighted", "--bits", "4"],
    },
    {
        "id": 8,
        "prefix": "exp8_sac_full_system_stack",
        "agent": "fl_sac",
        "trainer": "semi_async",
        "selection": "fair_fedcs",
        "aggregation": "fedavgm",
        "compression": "adaptive",
        "profile": "tiered",
        "extra_args": ["--fraction", "0.5", "--fairness_weight", "0.25", "--weighted"],
    },
]


def parse_kv_line(line: str) -> dict[str, float]:
    """Extracts key:value pairs from `[FL Episode/Round/Final Test Metrics]` lines."""
    metrics = {}
    for match in re.finditer(r"([a-zA-Z0-9_]+):([-+]?\d*\.?\d+)", line):
        metrics[match.group(1)] = float(match.group(2))
    return metrics


def run_cmd(cmd: list[str]) -> str:
    print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        captured.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return "".join(captured)


def main():
    parser = argparse.ArgumentParser(description="Run FL-TSC Benchmark Matrix")
    parser.add_argument("-e", "--episodes", type=int, default=10, help="Episodes per experiment")
    parser.add_argument("--network", type=str, default="cityflow4x4", help="Road network name")
    parser.add_argument("--only", type=int, nargs="*", default=None, help="Run only specific experiment IDs")
    parser.add_argument("--csv", type=str, default="data/output_data/fl_benchmark_summary.csv")
    args = parser.parse_args()

    selected_exps = [
        exp for exp in EXPERIMENTS
        if args.only is None or exp["id"] in args.only
    ]

    summary_rows = []
    for exp in selected_exps:
        base_cmd = [
            sys.executable,
            "federated_run.py",
            "-t", "fl_tsc",
            "-a", exp["agent"],
            "-n", args.network,
            "--prefix", exp["prefix"],
            "-e", str(args.episodes),
            "--trainer", exp["trainer"],
            "-s", exp["selection"],
            "-g", exp["aggregation"],
            "-c", exp["compression"],
            "-p", exp["profile"],
        ] + exp["extra_args"]

        print("=" * 86)
        print(
            f"[Experiment #{exp['id']}] agent={exp['agent']} | trainer={exp['trainer']} | "
            f"select={exp['selection']} | agg={exp['aggregation']} | "
            f"comp={exp['compression']} | profile={exp['profile']}"
        )
        print("=" * 86)

        t0 = time.time()
        train_out = run_cmd(base_cmd)
        test_out = run_cmd(base_cmd + ["--test"])
        wall_time_s = time.time() - t0

        last_round_metrics = {}
        for line in train_out.splitlines():
            if "[FL Round Metrics]" in line or "[FL FedBuff Round" in line or "[FL Async Update" in line:
                last_round_metrics.update(parse_kv_line(line))

        final_test_metrics = {}
        for line in test_out.splitlines():
            if "[FL Final Test Metrics]" in line:
                final_test_metrics.update(parse_kv_line(line))

        row = {
            "id": exp["id"],
            "prefix": exp["prefix"],
            "agent": exp["agent"],
            "trainer": exp["trainer"],
            "selection": exp["selection"],
            "aggregation": exp["aggregation"],
            "compression": exp["compression"],
            "profile": exp["profile"],
            "travel_time": final_test_metrics.get("travel_time", 0.0),
            "queue": final_test_metrics.get("queue", 0.0),
            "max_queue": final_test_metrics.get("max_queue", 0.0),
            "queue_fairness": final_test_metrics.get("queue_fairness", 0.0),
            "throughput": final_test_metrics.get("throughput", 0.0),
            "round_time_ms": last_round_metrics.get("round_time_ms", 0.0),
            "comm_volume_mb": last_round_metrics.get("comm_volume_mb", 0.0),
            "delivery_rate": last_round_metrics.get("delivery_rate", 0.0),
            "weight_drift": last_round_metrics.get("weight_drift", 0.0),
            "wall_time_s": round(wall_time_s, 2),
        }
        summary_rows.append(row)

    if summary_rows:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        fieldnames = list(summary_rows[0].keys())
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        print("\n" + "=" * 110)
        print("BENCHMARK COMPARISON SUMMARY")
        print("=" * 110)
        header = (
            f"{'ID':<3} | {'Agent':<13} | {'Trainer':<10} | {'Select':<10} | "
            f"{'Agg':<8} | {'Comp':<12} | {'Profile':<8} | {'ATT(s)':>8} | "
            f"{'MaxQ':>6} | {'FairJ':>6} | {'RndMs':>7} | {'CommMB':>6}"
        )
        print(header)
        print("-" * len(header))
        for r in summary_rows:
            print(
                f"{r['id']:<3} | {r['agent']:<13} | {r['trainer']:<10} | {r['selection']:<10} | "
                f"{r['aggregation']:<8} | {r['compression']:<12} | {r['profile']:<8} | "
                f"{r['travel_time']:>8.2f} | {r['max_queue']:>6.2f} | {r['queue_fairness']:>6.4f} | "
                f"{r['round_time_ms']:>7.1f} | {r['comm_volume_mb']:>6.2f}"
            )
        print("=" * 110)
        print(f"Saved full CSV results to: {args.csv}")


if __name__ == "__main__":
    main()

