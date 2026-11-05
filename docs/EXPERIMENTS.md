# Experiment-to-file map

| Paper material | Entry point | Recorded scalar results |
|---|---|---|
| Main RMSE and quantile diagnostics; supplementary MAE | `reproduce_main.py` | `published_results/main.csv` |
| Client, round, depth, smoothing, training quadrature and crossing-penalty studies | `reproduce_supplement.py` | `published_results/supplementary.csv` |
| Same-fit prediction quadrature and gradient-workload matching | `reproduce_supplement.py` | `published_results/supplementary_details.json` plus the supplementary CSV |
| Shared Host initialization, optimizer-specific tuning, three- and twenty-round comparisons | `run_fair_baseline_confirmation.py` | `published_results/fair/` |
| Motor insurance and air quality | `reproduce_real.py` | `published_results/real.csv` |
| Application extremile profiles and centralized gaps | `reproduce_real.py` | `published_results/real_profiles.csv` |

Main simulation identifiers use `contaminated` for the standardized Gaussian-mixture error distribution. This identifier is retained to match the recorded seeds and configurations.

`main.csv` contains 9,600 method rows from 1,200 paired repetitions. The supplementary CSV contains 6,090 method rows from 2,280 records. Main and component configurations are intentionally separate. Real-data comparisons contain eight methods per dataset. The fair-comparison checkpoint CSV contains all 2,520 exported checkpoint rows.

Scores in the main and supplementary CSVs are per replication. CE and CR are stored as proportions; the paper displays percentages. In simulation, CE averages equally over the 128 Gauss--Legendre prediction levels; CR uses the 99-point diagnostic grid. In applications, both CE and CR use the diagnostic grid. IPS uses quadrature weights. `mean_absolute_centralized_extremile_gap` is agreement with the centralized fitted model.

The supporting files `continue_main_simulations*.py` are retained because the unchanged supplementary implementation imports their file-handling and validation helpers. Use the `reproduce_*.py` entry points for this release; they supply portable fixed configurations rather than requiring local development-history directories.
