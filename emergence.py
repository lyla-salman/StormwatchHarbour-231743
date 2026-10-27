"""Test berth switching without modifying the frozen run.py policies.

Run: python emergence.py --digit 3
Uses development seeds 1-10; the before/after experiment is separate from
the main evaluation. A 0.5 persistence bonus is a preselected intervention,
not a claim of optimal tuning.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from run import Harbour, WIDTH, HEIGHT, BERTHS


class ObservedHarbour(Harbour):
    def __init__(self, digit, seed, bonus):
        super().__init__(digit, seed, "utility")
        self.bonus = bonus
        self.switches = [0] * len(self.workers)
        self.trace = []
        self.heat = [[0] * WIDTH for _ in range(HEIGHT)]

    def berth_utility(self, worker, ship):
        score = super().berth_utility(worker, ship)
        if (worker["target_berth"] == ship["berth"]
                and worker["target_ship_id"] == ship["id"]):
            score += self.bonus
        return score

    def update_workers(self, tick):
        before = [(w["target_berth"], w["target_ship_id"], w["carrying"])
                  for w in self.workers]
        super().update_workers(tick)
        live = {ship["id"]: ship for ship in self.ships if ship["left"] > 0}
        for index, (worker, old) in enumerate(zip(self.workers, before)):
            old_berth, old_ship, was_carrying = old
            # Count conservative, voluntary berth changes, not initial
            # assignments, completed ships or mandatory storm evacuation.
            old_still_available = old_ship in live
            forced_by_storm = (worker["avoid_unsafe"]
                               and old_berth in ("B1", "B4"))
            if (not was_carrying and old_still_available
                    and not forced_by_storm
                    and old_berth is not None
                    and worker["target_berth"] is not None
                    and worker["target_berth"] != old_berth):
                self.switches[index] += 1

            x, y = worker["pos"]
            self.heat[y][x] += 1
            self.trace.append({
                "tick": tick, "worker": index,
                "x": x, "y": y,
                "target_berth": worker["target_berth"] or "none",
                "carrying": worker["carrying"],
                "cargo_delivered": self.cargo_delivered,
            })


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(original, changed, path, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worker_id = max(range(len(original.switches)),
                    key=lambda index: original.switches[index])
    maximum = max(max(map(max, original.heat)), max(map(max, changed.heat)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for column, (harbour, label) in enumerate(
            ((original, "Original utility"), (changed, "Persistence bonus 0.5"))):
        axis = axes[0, column]
        image = axis.imshow(harbour.heat, origin="upper", vmin=0,
                            vmax=maximum, cmap="magma", aspect="equal")
        axis.set_title(label)
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
        axis.scatter([0], [15], marker="s", facecolors="none",
                     edgecolors="cyan", s=60)
        for name, (x, y) in BERTHS.items():
            axis.text(x, y + 1.5, name, color="cyan", ha="center", fontsize=8)

        records = [row for row in harbour.trace if row["worker"] == worker_id]
        berth_numbers = {"none": 0, "B1": 1, "B2": 2, "B3": 3, "B4": 4}
        axis = axes[1, column]
        axis.step([row["tick"] for row in records],
                  [berth_numbers[row["target_berth"]] for row in records],
                  where="post", linewidth=0.8)
        axis.set_yticks(range(5), ["None", "B1", "B2", "B3", "B4"])
        axis.set_xlabel("Tick")
        axis.set_ylabel("Remembered target berth")
        axis.set_title(f"Worker {worker_id}: {harbour.switches[worker_id]} voluntary switches")
        axis.grid(alpha=0.2)
    fig.colorbar(image, ax=list(axes[0]), label="Worker–ticks per cell", shrink=0.8)
    fig.suptitle(f"Figure 2  Berth selection and occupancy on development seed {seed}\n"
                 "Worker selected for the most original voluntary switches; "
                 "trace includes all assignment changes")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digit", type=int, choices=range(10), default=3)
    args = parser.parse_args()
    try:
        import matplotlib
    except ImportError:
        parser.exit(1, "Install matplotlib: python -m pip install matplotlib\n")

    root = Path(__file__).resolve().parent
    output = root / "results" / "emergence"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    example = {}
    for seed in range(1, 11):
        for label, bonus in (("original", 0.0), ("persistence", 0.5)):
            print(f"Seed {seed}/10: {label}", flush=True)
            harbour = ObservedHarbour(args.digit, seed, bonus)
            metrics = harbour.run()
            rows.append({"variant": label, "seed": seed,
                         "voluntary_switches": sum(harbour.switches), **metrics})
            if seed == 1:
                example[label] = harbour
                write_csv(output / f"trace_{label}_seed1.csv", harbour.trace)
        write_csv(output / "experiment_results.csv", rows)

    metrics = ("voluntary_switches", "cargo_delivered", "avg_ship_wait",
               "crowding_violations", "storm_knockouts")
    summary = []
    for label in ("original", "persistence"):
        selected = [row for row in rows if row["variant"] == label]
        entry = {"variant": label, "seeds": len(selected)}
        for metric in metrics:
            values = [row[metric] for row in selected]
            entry[f"{metric}_mean"] = statistics.mean(values)
            entry[f"{metric}_sd"] = statistics.stdev(values)
        summary.append(entry)
    write_csv(output / "summary.csv", summary)

    paired = []
    for metric in metrics:
        differences = [
            next(row[metric] for row in rows
                 if row["seed"] == seed and row["variant"] == "persistence")
            - next(row[metric] for row in rows
                   if row["seed"] == seed and row["variant"] == "original")
            for seed in range(1, 11)
        ]
        paired.append({"metric": metric,
                       "mean_difference_persistence_minus_original": statistics.mean(differences),
                       "sd_of_paired_differences": statistics.stdev(differences)})
    write_csv(output / "paired_effects.csv", paired)
    make_figure(example["original"], example["persistence"],
                output / "emergence_comparison.png", 1)

    lines = ["Table 2. Rule change on ten matching development seeds (1–10).",
             "Values are mean ± sample standard deviation.", "",
             "| Variant | Voluntary switches | Cargo delivered | Ship wait | Crowding | Knock-outs |",
             "|---|---:|---:|---:|---:|---:|"]
    for entry in summary:
        cells = [f"{entry[f'{metric}_mean']:.2f} ± {entry[f'{metric}_sd']:.2f}"
                 for metric in metrics]
        lines.append("| " + " | ".join([entry["variant"], *cells]) + " |")
    table = "\n".join(lines) + "\n"
    (output / "table.md").write_text(table, encoding="utf-8")
    (output / "experiment_metadata.json").write_text(json.dumps({
        "seeds": list(range(1, 11)), "digit": args.digit,
        "persistence_bonus": 0.5,
        "original_simulation_sha256": hashlib.sha256((root / "run.py").read_bytes()).hexdigest(),
        "count_definition": "Berth changes while empty at tick start, old ship still has cargo at tick end, old berth not storm-excluded.",
        "caveats": ["The selected worker is illustrative, not a random sample.",
                    "Fewer switches alone do not prove better throughput.",
                    "Safety-driven switches and empty-ship replacements are excluded.",
                    "This intervention does not replace the frozen main evaluation policy."],
    }, indent=2), encoding="utf-8")
    print("\n" + table)
    print(f"Saved results in {output}")


if __name__ == "__main__":
    main()
