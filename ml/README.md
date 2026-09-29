# ML behavior prediction — pipeline (branch: ml-behavior-prediction)

Predicts per-tick player acceleration (pure input accel, friction-cancelled) with a
tiny GRU-64 policy + (planned) tiny value head for release gating. See
`Amalgam/src/Features/Simulation/BehaviorPrediction/BehaviorPrediction.h` for the
runtime contract.

## Pipeline (data lives in ~/Downloads, code here)

1. `dl_demos.py` — 25 MGE SourceTV demos (~450MB) via Drive IDs in script.
2. `demodump_rs` — Rust extractor (`tf-demo-parser` crate): per-tick player state
   (origin/vel/eye/flags/health/life/team/class) + projectile snapshots
   (`.proj.csv` sidecars). Run: `dump_all.py` / `dump_proj.py`.
   - Gotcha: owner-table priority matters (NonLocal first for SourceTV), networked
     player velocity is stale in demos → derive velocity from origin.
3. `build_ds2.py` — SQLite store (`dataset.db`, Turso-compatible) + windows:
   66-tick warmup, origin-derived velocity, bidirectional AFK slice (±2s still →
   drop, 1s onset approach kept so dodge reactions survive), teleport split,
   movement-weighted fitness, horizons 100/200/300/500/1000ms → `train2/ev2.npz`.
4. `build_ds3.py` — same windows, labels = PURE INPUT accel (TF2 ground friction
   cancelled: `A_in = dV/dt + vhat*max(|v|,100)*4`; air unchanged) → `train3/ev3.npz`.
5. `train2.py` — GRU-64 policy, weighted MSE. `finetune.py` — multi-step rollout
   loss (fixes teacher-forcing bias). `eval_rollout.py` / `tail_eval.py` —
   autoregressive position error @ horizons vs coast/yawrate baselines.
6. `featurize.py` + `manifest.json` (in data dir) — config-hashed feature groups;
   input changes recompute only affected groups.

## Results (eval, soldier-only)

wMSE: coast 43.3 / yawrate 166 / GRU 16.3 (2.7x over coast).
Rollout position error hu @100/200/300/500/1000ms (finetuned v3, clean labels):
coast 4.9/11.6/22.3/47.0/122.3 vs model 4.0/9.1/17.2/35.8/82.3 — model wins
everywhere, gap widens at high curvature + long horizons.

Labels v3 (current): PURE INPUT accel — TF2 ground friction cancelled
(A_in = dV/dt + vhat*max(|v|,100)*4; air 2D unchanged), so the net learns
behavior (wishdir intent), not physics. Sim owns friction/gravity.
Windows identical to v2 (cache: only Y recomputed).

Note: single-step wMSE can look good while rollout lags coast (exposure bias) —
always judge by autoregressive multi-horizon position error. Finetune with
differentiable rollout loss before comparing.

## Next

- Train policy on v3 (pure-input) labels.
- Threat/FOV feature group from `.proj.csv` (dodge-keep rule, 90° FOV + 1s memory).
- Tiny value head: P(rollout final pos aligns) at min(T_limit, T_intercept).
- Correction-history inputs (past predictions per future tick) + per-tick/total
  divergence loss. Then int8 C++ forward, wire into RunTick/release gate.
