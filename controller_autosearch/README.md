# ECHO Controller Simulation Framework

Simulates model-switching controllers for adaptive object detection on Jetson Orin Nano. Uses LSTM-predicted miss rates to dynamically select between YOLO11 nano/small/medium models, trading off energy consumption against detection quality.

## Files

| File | Description |
|------|-------------|
| `models.py` | Jetson Orin Nano hardware constants (latency, power, energy per frame), switching costs, and the `required_model()` oracle function |
| `controllers.py` | All 10 controller implementations: 5 baselines (AlwaysNano, AlwaysMedium, BestFixed, Random, Oracle) + 5 adaptive (Threshold, ThresholdHysteresis, BayesRiskMPC, DQN, Proxy) |
| `metrics.py` | `evaluate()` function that runs a controller over a prediction stream and computes energy, adequacy, switching, and model distribution metrics |
| `search.py` | Grid search functions for each adaptive controller type. Finds the best parameters that minimize energy subject to adequate_rate >= 0.85 |
| `budget.py` | Budget-constrained simulation: runs a controller under a fixed total energy budget, forcing degraded mode (nano-only) when budget is exhausted |
| `visualize.py` | Matplotlib plotting: timeline (miss rates + model bands), energy comparison bars, per-intersection grouped bars |
| `run_all.py` | Main entry point: runs everything end-to-end and saves results JSON + plots |

## Usage

```bash
# Single prediction file
python controller/run_all.py --predictions second_autosearch/results/predictions_plainlstm_4cam.csv

# All 5 prediction files
python controller/run_all.py --all

# Skip DQN (if PyTorch is unavailable)
python controller/run_all.py --all --skip-dqn
```

## Metrics

| Metric | Meaning |
|--------|---------|
| `avg_energy_mj` | Mean energy per 0.5s decision window (mJ). Includes 15 detector frames + switching overhead |
| `energy_savings_pct` | Percentage saved vs AlwaysMedium: `100 * (1 - E/E_medium)` |
| `oracle_gap` | `(E - E_oracle) / (E_medium - E_oracle)`. 0 = matches oracle, 1 = no better than always-medium |
| `adequate_rate` | Fraction of windows where selected model >= required model. Target: >= 0.85 |
| `under_provision_rate` | Fraction where model is too weak (detection quality suffers) |
| `over_provision_rate` | Fraction where model is stronger than needed (energy wasted) |
| `switches_per_100` | Model changes per 100 windows. Lower = more stable |

## Updating Jetson Benchmarks

Edit the `MODELS` dict and `SWITCH_COSTS` dict in `models.py`. All values are from real measurements on Jetson Orin Nano in 15W mode with TensorRT FP16 and clocks locked.

## Dependencies

- numpy, pandas, matplotlib, scipy
- torch (optional, only for DQN controller)
