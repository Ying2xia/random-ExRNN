# ExtremileFed

Reproducibility code for **ExtremileFed: Communication-Efficient Federated Neural Extremile Regression via Quantile-Process Learning**.

ExRNN learns a shared conditional quantile process and integrates it to estimate conditional extremiles. ExtremileFed fits the model using common-point client gradients and a corrected host objective. Clients retain their observations and exchange parameter and gradient vectors.

[中文说明](README_zh.md)

## Contents

| Directory | Contents |
|---|---|
| `code/extremilefed/` | Neural model, smoothed loss, extremile quadrature, CSL training, baselines, simulation and evaluation |
| `code/reproduce_*.py` | Portable entry points using the paper's fixed configurations |
| `code/run_fair_baseline_confirmation.py` | Independent baseline development and shared-initialization confirmation |
| `code/tests/` | Model, gradient, baseline, preprocessing and experiment tests |
| `configs/` | Fixed main, supplementary and application configurations; baseline selections |
| `published_results/` | Scalar results underlying the paper, without training histories or observation-level predictions |
| `data/` | Data preparation instructions and SHA-256 checksums |
| `docs/` | Experiment map, tested environment and file manifest |

The main simulation uses one hidden layer in all 12 settings. The supplementary component study uses its own fixed reference configurations for the five specified settings. Both configuration sets are included separately.

## Setup

The recorded environment is Python 3.9.25 with CPU PyTorch 2.8.0. Package versions are in `requirements.txt` and `docs/tested_environment.json`. The experiment launchers use POSIX process locks and have been checked on macOS; Linux uses the same launch interfaces.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run commands from the repository root. The portable entry points locate configuration files relative to their own location and accept explicit output paths.

## Quick checks

```sh
PYTHONPATH=code python -B -m unittest discover -s code/tests
python -B code/reproduce_main.py --smoke --run
python -B code/reproduce_supplement.py --smoke --max-tasks 3 --run
python -B code/run_fair_baseline_confirmation.py --smoke --workers 2
```

Smoke runs use small samples and short optimization schedules. Their outputs go into separate temporary or `outputs/smoke_*` locations.

## Main simulation

The plan contains 12 settings, 100 paired replications, 100,000 training observations, 5,000 test observations, 20 clients and eight methods. The entry point preserves the data and model seeds of the recorded main study.

```sh
# Inspect the full plan without fitting.
python -B code/reproduce_main.py

# Reproduce all methods; completed tasks are reused.
python -B code/reproduce_main.py --run --workers 4

# Run a smaller subset with the same full-size settings.
python -B code/reproduce_main.py --settings scenario1_normal --replications 1 --run
```

Each method uses the original single-method fitting and numerical-check functions. `outputs/main/` contains fresh records and parameter checkpoints. Choose a fresh `--output` directory when changing the task plan or code.

## Supplementary experiments

The fixed plan contains 2,280 records across five settings, with 30 replications per configuration. It includes depth, client-count, round-count, smoothing, training and prediction quadrature, crossing-penalty, and gradient-workload studies.

```sh
python -B code/reproduce_supplement.py
python -B code/reproduce_supplement.py --run --workers 4
python -B code/reproduce_supplement.py --suite budget --run --workers 4
```

Reference fits are recomputed from the included configurations, so these commands do not require a historical local result directory. `followup_depth.py` implements the fixed-width depth extension; the remaining numerical routines are shared with the main model.

## Shared-initialization baseline comparison

This experiment uses three settings, 100,000 training observations, 20 clients, three independent development datasets per setting, and 30 independent confirmation replications. Each method starts from the same trained Host parameter. Baselines are selected using validation IPS at round 3; checkpoints extend to round 20.

```sh
python -B code/run_fair_baseline_confirmation.py --dry-run
python -B code/run_fair_baseline_confirmation.py --workers 4
```

The numerical update functions are in `code/_fair_updates.py`. They are the same functions used by the recorded confirmation experiment. Only configuration and helper-file locations were adapted for this repository. Recorded baseline choices and protocol are in `configs/fair/`.

## Real-data applications

Obtain the two datasets using [data/README.md](data/README.md), then verify and evaluate the fixed settings:

```sh
python -B code/reproduce_real.py --dataset fremtpl2
python -B code/reproduce_real.py --dataset fremtpl2 --run
python -B code/reproduce_real.py --dataset beijing --run
```

The original chronological and stratified splits, preprocessing, selected model settings and evaluation functions are retained. Checksums identify the exact source files used for the archived results. Raw datasets, generated feature matrices and observation-level predictions are not included in this repository.

The original application-specific validation and tuning implementation is also included in `code/run_real_data_experiments.py`; its `--help` describes the separate development and evaluation stages.

## Recorded results

The scalar CSV files allow the main comparisons to be inspected without repeating the fits:

```sh
python -B code/summarize_results.py
python -B code/plot_results.py
```

The summary command writes means and Monte Carlo standard errors under `outputs/summaries/`. The plot command writes simulation RMSE and application profile figures under `outputs/figures/`. Metrics in the scalar CSVs use proportions for CE and CR; multiply by 100 to obtain the percentages shown in the paper. Communication uses MiB. Fitting times are recorded single-machine simulation times. The real-data profile CSV contains only seven-level aggregate predictions and centralized gaps.

[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) maps the files to the paper. [CITATION.cff](CITATION.cff) records the authors and manuscript title.
