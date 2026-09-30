# STATUS — ML behavior prediction (session-proof notes, 2026-09-29/30)

Branch: `ml-behavior-prediction` (clean tree expected; `master` = upstream ee2ba31,
`projectile-aim-improvements` = heuristics branch, independent).

## Runtime contract
`Amalgam/src/Features/Simulation/BehaviorPrediction/BehaviorPrediction.h`:
per-tick expected accel (pure input accel, friction-cancelled) + 0..1 value head
= P(rollout matches truth) at min(T_limit, T_intercept). NOT wired into aimbot yet.
Control split: AI owns odds, user owns bounded preset + panic-keybind override
(two sliders, never fully blind). Rings for all players, Amalgam owns viability.

## Data (~/Downloads, NOT in repo)
- 25 MGE SourceTV demos (`mge_demos/`, ~450MB) + CSV dumps (`mge_csv/`, ~1.1GB + 650MB `.proj.csv` sidecars).
- `mge_ds/dataset.db` (236MB, soldier-only ticks) + `dataset_all.db` (726MB, all classes).
- Key files: `train3/eval3.npz` (soldier, pure-input labels), `trainA/evalA.npz`
  (all-class, 14-dim), `trainB/evalB.npz` (all-class + impulse-split, 682k drops),
  `manifest.json` (config-hashed feature groups).
- Checkpoints: `gru_soldier_denseC.pt` = CHAMPION (soldier, combat-masked,
  friction-order). `gru_soldier_dense66.pt` superseded (kept in git history).
  `gru_all_v6.pt` (all-class friction finetune, eval 28.53). `gru64_v4.pt` old best.
- Threat features built (`trainA_ext.npz`: walk_age + 6 threat feats, 15% nonzero) —
  NOT yet in any training input (needs input-dim expansion = next arch step).

## What worked
- Origin-derived velocity (networked vel stale in demos), NonLocal table priority (SourceTV).
- Pure-input accel labels (friction cancelled) > total accel.
- Hitbox loss (log inside 24hu AABB / 4x quad outside) > rollout MSE > single-step MSE.
- Judge ONLY by autoregressive rollout hit-rate @24hu, never single-step loss.
- v4 hit-rate: 96.8/82.3/59.0/28.2/12.7 @100/200/300/500/1000ms vs coast 92.2/70.5/50.9/28.0/11.5.
- Engine-order rollout (friction-first vector addition): v6 ep1 eval 28.91 vs v5 best
  30.23 — same data/net/loss, fix isolated as cause. C++ mirrors it.
- Impulse-split (knockback = external force, not input): 682k drops, ep0 eval improves.
- Measured difficulty weights: coast err fits t^1.37 (R2=0.998), exponential rejected
  (R2=0.45). w(t)=(t/66)^1.37 floor 0.02. Dense-66 soldier beats coast 4/5 horizons.
- INT8 post-training quant preserves hit-rate (≤0.2pp). FP16 identical. INT8 = runtime choice.
- 8-modality real (55% near modes) but post-hoc magnet worthless (±0.2pp). Default raw.

## What failed (do NOT retry as-is)
- Residual-over-heuristic target: loses everywhere (rollout 47.1 vs 59.0 @300ms).
  Reason: target entangles intent + heuristic error + integrated state error.
- Static-broadcast context (corr1, ctx EMAs as frozen per-step copies): wins single-step,
  collapses rollout. Context must enter via init-hidden/FiLM or per-step features.
- FP16 CPU-AMP training on i7-4900MQ (Haswell, no AVX512-FP16): 31x SLOWER than fp32
  (4729 vs 150 ms/step → 31h vs 1h). Train FP32, export INT8. Killed 2026-09-29, zero progress lost
  (checkpoint-per-best-eval design; fp16 job died in epoch 0, nothing written).
- v2 champion file lost to naming bug (overwritten by v3i run); numbers survive in logs.
  Mitigation: distinct checkpoint names per line (v3i/v3ii/v4/v5/v6...), never reuse.

## Results ledger (eval hit-rate @24hu, 100/200/300/500/1000ms)
- coast (soldier): 92.4/70.7/50.8/28.1/10.9
- v3ii: 96.7/82.2/58.1/27.0/12.5
- v4 (hitbox finetune): 96.8/82.3/59.0/28.2/12.7
- dense66: 97.0/82.5/58.2/27.2/12.4 (B-eval: 97.5/85.5/59.4/27.2/13.0)
- denseC (friction unroll + combat mask): 97.8/86.4/61.8/27.7/13.1 full,
  98.0/86.3/61.8/28.3/14.0 combat-masked. CHAMPION. INT8-verified (≤0.4pp).
- all-class v6: eval loss 28.53 (hit-rate pending v6 line)
- specialists (loss): sniper 5.89 best, spy 57.23 hardest, medic skipped (thin data)

## Hard lessons (read before touching evals)
- EVAL DYNAMICS MUST MATCH TRAINING DYNAMICS. denseC looked collapsed
  (19.5/3.6 @500/1000ms) purely because it was evaluated with a frictionless
  rollout while trained with friction. Same weights, correct unroll: champion.
  Always: model+rollout combo is the unit, never the weights alone.
- Knockback: 33k hurt events extracted, 90% victim-resolved via health drops.
  Impulse-split caps already capture ~all of it (flagged ≈ clean distribution).
  Near-damage labels ARE hotter (p50 10k vs 870) — caps, not events, do the work.
- Data is clean: 0 NaNs, 0 dupes (PK), teams 2/3 only, 4 demos have zero
  projectiles (few player samples there, defaults cover). Rocket oz 100% NaN → 2D only.
- Threat features exist (15% nonzero, FOV 5%) but are NOT in any training input yet.

## In flight
- `finetune_all_fric.py` → `gru_all_v6.pt` (engine-order, all-class). Nearly done.
- NEXT: dense-B soldier (power-law weights + friction unroll, warm start v6):
  `train_denseSUF.py` with DS_SUFFIX=B, WARM_CKPT=gru_all_v6.pt → `gru_soldier_denseB.pt`.
- AFTER: threat inputs need input-dim expansion (14→20: threat 6 feats) = new arch step;
  then value head on champion → threshold curve → INT8 export → RunTick wiring.
- LTM-v1 builder + per-engagement accumulator: built, not yet evaluated as features.

## Machine notes
- CPU i7-4900MQ 4c/8t, 16GB RAM. torch 2.12.1+cpu, py3.14. Use torch.set_num_threads(4).
- NEVER `Stop-Process -Name python3` (kills the tool shell); use `taskkill /PID <pid> /F`.
- Detached launches via Start-Process currently flaky; prefer foreground with long timeout.
- Checkpoints save on best-eval only → crashed runs lose nothing but time.
