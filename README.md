# eBus-Drone
Minimal runnable integrated eBus-drone experimental harness.

## Commands
- `python -m src.main --mode generate --config configs/default.yaml --instance medium --seed 1`
- `python -m src.main --mode offline --config configs/default.yaml --instance medium --seed 1`
- `python -m src.main --mode train --config configs/default.yaml --instance medium --method proposed --seed 1`
- `python -m src.main --mode eval --config configs/default.yaml --instance medium --method proposed --seed 1`
- `python -m src.main --mode benchmark --config configs/experiments/benchmark.yaml --instance medium --seeds 1 2 3`

Outputs are saved under `outputs/` including CSV/JSON summaries and assignments.
