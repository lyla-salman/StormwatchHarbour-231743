Table 1. Harbour performance on seeds 11–40.
Values are mean ± sample standard deviation.

| Agent | Cargo delivered | Average ship wait | Crowding violations | Storm knock-outs |
|---|---:|---:|---:|---:|
| baseline | 0.00 ± 0.00 | 271.08 ± 3.39 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| reflex | 215.20 ± 81.96 | 189.49 ± 27.68 | 0.00 ± 0.00 | 3.30 ± 4.81 |
| model | 205.73 ± 3.85 | 194.05 ± 4.46 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| utility | 183.50 ± 4.41 | 216.39 ± 4.08 | 0.00 ± 0.00 | 0.00 ± 0.00 |

Ship wait includes elapsed waits for ships still undocked at tick 600;
these unfinished waits are censored and may grow after the run ends.
Crowding counts cells containing more than three workers at each tick.

Paired comparisons subtract agent B from agent A on matching seeds.
Positive cargo differences favour A; negative differences in the other metrics favour A.
The paired SD describes run-to-run variation in that difference.
The 95% intervals are approximate normal intervals for the mean difference,
not SD error bars, exact hypothesis tests or multiplicity-adjusted intervals.
A mean difference smaller than the paired SD can still be consistent across seeds.
All-zero observed crowding does not prove safety for every possible run.
