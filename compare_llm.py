"""Compare frozen reflex and live LLM policies on evaluation seeds 11-15.

Run beside run.py and llm_master.py: python compare_llm.py --digit 3
Use --resume to reuse completed rows after an interrupted run, provided
the code, model and settings still match the saved metadata.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time


METRICS = ("cargo_delivered", "avg_ship_wait", "crowding_violations", "storm_knockouts")
COUNTS = ("invalid_outputs", "retries", "fallbacks", "calls", "transport_errors",
          "blocked_approvals")
AGENTS = ("reflex", "llm")
SEEDS = tuple(range(11, 16))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows):
    summary = []
    for agent in AGENTS:
        selected = [row for row in rows if row["agent"] == agent]
        record = {"agent": agent, "runs": len(selected)}
        for field in METRICS + COUNTS:
            values = [row[field] for row in selected]
            record[field + "_mean"] = statistics.mean(values)
            record[field + "_sd"] = statistics.stdev(values)
            if field in COUNTS:
                record[field + "_total"] = sum(values)
        seconds = sum(row["total_call_seconds"] for row in selected)
        calls = sum(row["calls"] for row in selected)
        record["total_call_seconds"] = seconds
        # Blank, rather than zero latency, for a policy making no model calls.
        record["pooled_seconds_per_call"] = seconds / calls if calls else None
        record["mean_run_wall_seconds"] = statistics.mean(row["run_wall_seconds"] for row in selected)
        summary.append(record)
    return summary


def paired_comparison(rows):
    indexed = {(row["agent"], row["seed"]): row for row in rows}
    result = []
    for field in METRICS:
        values = [indexed[("llm", seed)][field] - indexed[("reflex", seed)][field]
                  for seed in SEEDS]
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        # t(0.975, df=4) for five matched seeds. Interpret cautiously with n=5.
        half_width = 2.776445105 * sd / math.sqrt(len(values))
        result.append({"metric": field, "pairs": len(values),
                       "mean_difference_llm_minus_reflex": mean,
                       "sd_of_paired_differences": sd,
                       "approx_95_t_ci_low": mean - half_width,
                       "approx_95_t_ci_high": mean + half_width})
    return result


def make_table(summary):
    lines = ["Table 3. LLM and reflex comparison on matching seeds 11-15.",
             "Simulation metrics are mean ± sample standard deviation.", "",
             "| Agent | Cargo delivered | Ship wait | Crowding | Knock-outs |",
             "|---|---:|---:|---:|---:|"]
    for row in summary:
        cells = [f"{row[field + '_mean']:.2f} ± {row[field + '_sd']:.2f}" for field in METRICS]
        lines.append("| " + " | ".join([row["agent"], *cells]) + " |")
    lines += ["", "LLM reliability and cost totals across the five runs:", "",
              "| Measure | Total |", "|---|---:|"]
    llm = next(row for row in summary if row["agent"] == "llm")
    for field in COUNTS:
        lines.append(f"| {field} | {llm[field + '_total']} |")
    lines += [f"| Model-call seconds | {llm['total_call_seconds']:.2f} |",
              f"| Seconds per call, pooled | {llm['pooled_seconds_per_call']:.3f} |",
              "", "Call time includes loading and failed attempts, if any; replay is not used.",
              "Reflex makes zero model calls; its model-call latency is not applicable.",
              "Raw model timing and version metadata remain in logs/llm_log.jsonl.",
              "A valid-format action can still be ineffective or repetitive.",
              "Waiting includes elapsed, unfinished waits for ships still undocked at tick 600.",
              "Five seeds and one model provide limited evidence for general conclusions.",
              "Paired intervals use a t approximation with four degrees of freedom;",
              "inspect individual differences and distributional assumptions before interpreting them.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digit", type=int, choices=range(10), default=3)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for name in ("run.py", "llm_master.py"):
        if not (root / name).is_file():
            parser.error(f"Place compare_llm.py beside {name}.")
    out = root / "results" / "llm_comparison"
    out.mkdir(parents=True, exist_ok=True)
    log_path = root / "logs" / "llm_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    config = {"digit": args.digit, "model": args.model, "temperature": args.temperature,
              "seeds": list(SEEDS), "ticks": 600, "actions_per_turn": 1,
              "human_approval": "blocked by default",
              "source_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ("run.py", "llm_master.py")}}
    metadata_file = out / "metadata.json"
    checkpoint_file = out / "checkpoint.json"
    rows = []
    if args.resume and checkpoint_file.exists():
        if not metadata_file.exists() or json.loads(metadata_file.read_text())["config"] != config:
            parser.error("Saved code/settings do not match. Restore original settings before resuming.")
        rows = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        if not log_path.exists() and any(row["agent"] == "llm" for row in rows):
            parser.error("Restore logs/llm_log.jsonl before resuming; raw evidence is required.")
    elif checkpoint_file.exists():
        parser.error("Results already exist. Use --resume to reuse them without new calls.")
    else:
        metadata_file.write_text(json.dumps({"config": config,
            "started_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    completed = {(row["agent"], row["seed"]) for row in rows}
    for seed in SEEDS:
        for agent in AGENTS:
            if (agent, seed) in completed:
                print(f"Reusing completed {agent}, seed {seed}.", flush=True)
                continue
            print(f"\n[{len(rows) + 1}/10] Running {agent}, seed {seed}...", flush=True)
            command = [sys.executable, str(root / "run.py"), "--digit", str(args.digit),
                       "--agent", agent, "--seed", str(seed)]
            if agent == "llm":
                command += ["--model", args.model, "--temperature", str(args.temperature),
                            "--max-ticks", "600", "--actions-per-turn", "1",
                            "--log-file", str(log_path), "--verbose"]
            log_offset = log_path.stat().st_size if log_path.exists() else 0
            start = time.perf_counter()
            # stderr inherits the terminal so every LLM turn remains visible.
            process = subprocess.run(command, cwd=root, stdout=subprocess.PIPE,
                                     text=True, timeout=7200)
            wall = time.perf_counter() - start
            if process.returncode:
                parser.exit(1, "A run failed. Fix the cause and use --resume.\n")
            try:
                metrics = json.loads(process.stdout.strip())
                for field in METRICS + (COUNTS if agent == "llm" else ()):
                    value = metrics[field]
                    if (not isinstance(value, (int, float)) or isinstance(value, bool)
                            or not math.isfinite(value) or value < 0):
                        raise ValueError(f"Invalid {field}")
                if agent == "llm" and (metrics["live_calls"] != metrics["calls"]
                                       or metrics["replay_calls"] != 0):
                    raise ValueError("This comparison requires live model calls.")
            except (ValueError, KeyError, TypeError) as error:
                raise RuntimeError(f"Bad result for {agent}, seed {seed}: {process.stdout}") from error
            run_id = ""
            if agent == "llm":
                with log_path.open("rb") as handle:
                    handle.seek(log_offset)
                    events = [json.loads(line) for line in handle if line.strip()]
                endings = [event for event in events if event.get("event") == "run_end"]
                if len(endings) != 1:
                    raise RuntimeError("Could not identify this run's complete raw log.")
                run_id = endings[0]["run_id"]
            row = {"agent": agent, "seed": seed,
                   **{field: metrics[field] for field in METRICS},
                   **{field: metrics.get(field, 0) for field in COUNTS},
                   "total_call_seconds": metrics.get("total_call_seconds", 0),
                   "avg_call_seconds": metrics.get("avg_call_seconds", 0),
                   "run_wall_seconds": wall, "log_run_id": run_id}
            rows.append(row)
            completed.add((agent, seed))
            checkpoint_file.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            write_csv(out / "comparison_results.csv", rows)
            print(f"Completed: cargo={row['cargo_delivered']}, "
                  f"knock-outs={row['storm_knockouts']}", flush=True)

    summary = summarise(rows)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "paired_comparisons.csv", paired_comparison(rows))
    table = make_table(summary)
    (out / "table.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"Saved results in {out}")


if __name__ == "__main__":
    main()
