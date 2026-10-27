"""Evaluate the four rule-based harbour policies without changing settings.

Run beside run.py: python evaluate.py --digit 3
Defaults to the required evaluation seeds 11 through 40.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


AGENTS = ("baseline", "reflex", "model", "utility")
METRICS = ("cargo_delivered", "avg_ship_wait",
           "crowding_violations", "storm_knockouts")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows):
    summary = []
    for agent in AGENTS:
        selected = [row for row in rows if row["agent"] == agent]
        record = {"agent": agent, "runs": len(selected)}
        for metric in METRICS:
            values = [row[metric] for row in selected]
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_sd"] = statistics.stdev(values)
        summary.append(record)
    return summary


def compare_pairs(rows):
    indexed = {(row["agent"], row["seed"]): row for row in rows}
    seeds = sorted({row["seed"] for row in rows})
    comparisons = []
    for a, b in itertools.combinations(AGENTS, 2):
        for metric in METRICS:
            differences = [
                indexed[(a, seed)][metric] - indexed[(b, seed)][metric]
                for seed in seeds
            ]
            average = statistics.mean(differences)
            sd = statistics.stdev(differences)
            se = sd / math.sqrt(len(differences))
            comparisons.append({
                "agent_a": a,
                "agent_b": b,
                "metric": metric,
                "pairs": len(differences),
                "mean_difference_a_minus_b": average,
                "sd_of_paired_differences": sd,
                "standard_error": se,
                "approx_95_ci_low": average - 1.96 * se,
                "approx_95_ci_high": average + 1.96 * se,
                "abs_mean_difference_exceeds_paired_sd": abs(average) > sd,
            })
    return comparisons


def make_figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["agent"] for row in summary]
    titles = (
        "Cargo delivered — higher is better",
        "Ship waiting time — lower is better",
        "Crowding violations — lower is better",
        "Storm knock-outs — lower is better",
    )
    units = ("Cargo units", "Ticks", "Cell–tick violations", "Events")
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, metric, title, unit in zip(axes.flat, METRICS, titles, units):
        means = [row[f"{metric}_mean"] for row in summary]
        deviations = [row[f"{metric}_sd"] for row in summary]
        axis.bar(labels, means, yerr=deviations, capsize=4,
                 color=["#9ca3af", "#377eb8", "#4daf4a", "#984ea3"])
        axis.set_title(title, fontsize=10)
        axis.set_ylabel(unit)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle("Figure 1  Harbour policy performance\n"
                     "Bars show means; error bars show sample standard deviations")
    figure.tight_layout(rect=(0, 0.02, 1, 0.92))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_table(summary, start, end):
    lines = [
        f"Table 1. Harbour performance on seeds {start}–{end}.",
        "Values are mean ± sample standard deviation.",
        "",
        "| Agent | Cargo delivered | Average ship wait | Crowding violations | Storm knock-outs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        values = [
            f"{row[f'{metric}_mean']:.2f} ± {row[f'{metric}_sd']:.2f}"
            for metric in METRICS
        ]
        lines.append("| " + " | ".join([row["agent"], *values]) + " |")
    lines.extend([
        "",
        "Ship wait includes elapsed waits for ships still undocked at tick 600;",
        "these unfinished waits are censored and may grow after the run ends.",
        "Crowding counts cells containing more than three workers at each tick.",
        "",
        "Paired comparisons subtract agent B from agent A on matching seeds.",
        "Positive cargo differences favour A; negative differences in the other metrics favour A.",
        "The paired SD describes run-to-run variation in that difference.",
        "The 95% intervals are approximate normal intervals for the mean difference,",
        "not SD error bars, exact hypothesis tests or multiplicity-adjusted intervals.",
        "A mean difference smaller than the paired SD can still be consistent across seeds.",
        "All-zero observed crowding does not prove safety for every possible run.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digit", type=int, choices=range(10), default=3)
    parser.add_argument("--start", type=int, default=11)
    parser.add_argument("--end", type=int, default=40)
    parser.add_argument("--output", default="results/evaluation")
    args = parser.parse_args()
    if args.start < 1 or args.end <= args.start:
        parser.error("Choose at least two seeds: 1 <= start < end.")

    # Fail early with an installation hint, before running 120 simulations.
    try:
        import matplotlib
    except ImportError:
        parser.exit(1, "Install plotting support: python -m pip install matplotlib\n")

    root = Path(__file__).resolve().parent
    simulation = root / "run.py"
    if not simulation.is_file():
        parser.exit(1, "Place evaluate.py in the same folder as run.py.\n")
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = root / destination
    destination.mkdir(parents=True, exist_ok=True)

    seeds = range(args.start, args.end + 1)
    total = len(seeds) * len(AGENTS)
    rows = []
    for seed in seeds:
        for agent in AGENTS:
            print(f"[{len(rows) + 1}/{total}] {agent}, seed {seed}", flush=True)
            process = subprocess.run(
                [sys.executable, str(simulation), "--digit", str(args.digit),
                 "--agent", agent, "--seed", str(seed)],
                capture_output=True, text=True, cwd=root, timeout=120,
            )
            if process.returncode:
                raise RuntimeError(f"{agent}, seed {seed}:\n{process.stderr}")
            try:
                metrics = json.loads(process.stdout.strip())
                for metric in METRICS:
                    value = metrics[metric]
                    if (not isinstance(value, (int, float))
                            or isinstance(value, bool)
                            or not math.isfinite(value) or value < 0):
                        raise ValueError(f"Invalid {metric}: {value}")
            except (ValueError, KeyError, TypeError) as error:
                raise RuntimeError(f"Bad JSON for {agent}, seed {seed}: "
                                   f"{process.stdout}") from error
            rows.append({"agent": agent, "seed": seed,
                         **{metric: metrics[metric] for metric in METRICS}})
            # Checkpoint completed runs, including if a later run fails.
            write_csv(destination / "evaluation_results.csv", rows)

    summary = summarise(rows)
    write_csv(destination / "summary.csv", summary)
    write_csv(destination / "paired_comparisons.csv", compare_pairs(rows))
    table = make_table(summary, args.start, args.end)
    (destination / "table.md").write_text(table, encoding="utf-8")
    make_figure(summary, destination / "policy_comparison.png")
    metadata = {
        "digit": args.digit, "seeds": list(seeds),
        "python": sys.version, "agents": list(AGENTS),
        "simulation_sha256": hashlib.sha256(simulation.read_bytes()).hexdigest(),
        "note": "Use the unchanged supplied run.py; seeds 1-10 are development only.",
    }
    (destination / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\n" + table)
    print(f"Saved results in {destination}")


if __name__ == "__main__":
    main()
