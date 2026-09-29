# STATUS — ML behavior prediction (session-proof notes, 2026-09-29)

Branch: `ml-behavior-prediction` (clean tree expected; `master` = upstream ee2ba31,
`projectile-aim-improvements` = heuristics branch, independent).

## Runtime contract
`Amalgam/src/Features/Simulation/BehaviorPrediction/BehaviorPrediction.h`:
per-tick expected accel (pure input accel, friction-cancelled) + 0..1 value head
= P(rollout matches truth) at min(T_limit, T_intercept). NOT wired into aimbot yet.

## Data (~/Downloads, NOT in repo)
- 25 MGE SourceTV demos (`mge_demos/`, ~450MB) + CSV dumps (`mge_csv/`, ~1.1GB + 650MB `.proj.csv` sidecars).
- `mge_ds/dataset.db` (236MB, soldier-only ticks) + `dataset_all.db` (726MB, all classes).
- Key files: `train3/eval3.npz` (soldier, pure-input labels), `trainA/evalA.npz`
  (all-class, 14-dim), `manifest.json` (config-hashed feature groups).
- Checkpoints: `gru64_v4.pt` = CHAMPION (soldier). `gru64_v3ii.pt` = previous best.

## What worked
- Origin-derived velocity (networked vel stale in demos), NonLocal table priority (SourceTV).
- Pure-input accel labels (friction cancelled) > total accel.
- Hitbox loss (log inside 24hu AABB / 4x quad outside) > rollout MSE > single-step MSE.
- Judge ONLY by autoregressive rollout hit-rate @24hu, never single-step loss.
- v4 hit-rate: 96.8/82.3/59.0/28.2/12.7 @100/200/300/500/1000ms vs coast 92.2/70.5/50.9/28.0/11.5.
- INT8 post-training quant preserves hit-rate (≤0.2pp). FP16 identical. INT8 = runtime choice.

## What failed (do NOT retry as-is)
- Residual-over-heuristic target: loses everywhere (rollout 47.1 vs 59.0 @300ms).
  Reason: target entangles intent + heuristic error + integrated state error.
- Static-broadcast context (corr1, ctx EMAs as frozen per-step copies): wins single-step,
  collapses rollout. Context must enter via init-hidden/FiLM or per-step features.
- FP16 CPU-AMP training on i7-4900MQ (Haswell, no AVX512-FP16): 31x SLOWER than fp32
  (4729 vs 150 ms/step → 31h vs 1h). Train FP32, export INT8. Killed 2026-09-29, zero progress lost
  (checkpoint-per-best-eval design; fp16 job died in epoch 0, nothing written).

## In flight — DONE 2026-09-29
- `train_all_fp32.py`: GRU(14,96) all-class → `gru_all_v5.pt` (eval 30.23). Superceded.
- Nine specialists (warm-started, horizon-weighted): scout/sniper/soldier/demo/
  heavy/pyro/spy/engy saved; medic skipped (too few windows). Sniper 5.89 best.
- Dense-66 soldier `gru_soldier_dense66.pt` (loss 22.31) = CURRENT CHAMPION.
  Hit-rate: 97.0/82.5/58.2/27.2/12.4 vs coast 92.4/70.7/50.8/28.1/10.9.
- C++ inference written (BehaviorPrediction.cpp, int8 weights in-repo), NOT WIRED.
  Verified: gates exact, int8 hit-rate safe, rounding documented.
- Next: hit-rate re-checks per specialist → value head on champion → RunTick wiring.

## Machine notes
- CPU i7-4900MQ 4c/8t, 16GB RAM. torch 2.12.1+cpu, py3.14. Use torch.set_num_threads(4).
- NEVER `Stop-Process -Name python3` (kills the tool shell); use `taskkill /PID <pid> /F`.
- Detached launches via Start-Process currently flaky; prefer foreground with long timeout.
- Checkpoints save on best-eval only → crashed runs lose nothing but time.
