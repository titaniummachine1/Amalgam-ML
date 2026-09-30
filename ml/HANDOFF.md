# HANDOFF — ML behavior prediction (new-agent brief, 2026-09-30)

Branch: `ml-behavior-prediction`. Tree is CLEAN (this note is the only new file).
`master` = upstream ee2ba31, `projectile-aim-improvements` = separate heuristics branch.
Nothing is running. No background jobs. Machine is free.

## 1. What this is (30 seconds)

We predict where a TF2 player will be 0–1s ahead, to aim projectiles. A GRU policy
network predicts per-tick movement; a frozen-physics rollout integrates it; a value
head gates the shot. We beat the constant-velocity ("coast") baseline at almost
every horizon. Eval metric is ALWAYS autoregressive rollout hit-rate @24hu
(% of test cases where the predicted position lands inside the 48x48 AABB around
the true position) — never single-step loss. Full ledger: `ml/STATUS.md`.

## 2. Current best (do not regress)

- Champion: `~/Downloads/mge_ds/gru_threatsimB.pt` (soldier, GRU(20,96):
  14 base feats + 6 threat feats, threat simulated during rollout).
- Hit-rate @100/200/300/500/1000ms — ALL: sim 97.8/87.4/60.0/27.4/12.5
  vs coast 92.5/73.5/51.7/30.0/12.4. THREAT-ACTIVE: sim leads everywhere.
- 500ms (27.4 vs 28.1 coast) is the single open gap. Everything else beats coast.
- C++ inference exists (`Amalgam/src/Features/Simulation/BehaviorPrediction/`,
  int8 weights in-repo) but is NOT wired into the aimbot yet.
- Committed this session: PushState weapon-def param + full-66-tick-history rule
  (commit 98c678a).

## 3. The active decision: input-space policy (user-ordered, DO THIS NEXT)

The user directed a refactor: the policy should output game inputs
(forwardmove, sidemove ∈ [-1,1], yaw-delta additive) instead of raw accel,
with the rollout applying engine `Accelerate` math. Rationale: inputs are
bounded, WASD-structured (labels are 8-modal: 54.7% near modes vs 25% uniform),
and the rollout becomes a differentiable ProcessMovement-lite.

Status: design settled, NOT implemented. Implement in this order:
1. **Labels are free**: NO new labels needed for (fm, sm) — position loss
   supervises through the differentiable engine. dyaw HAS ground truth
   (eye-yaw deltas in DB); add it as a small aux loss ONLY where eye is fresh
   (eye std > 0.5 over window; demo eye goes stale).
2. **New head**: GRU trunk unchanged (warm-start from threatsim/denseC),
   head 96→2 becomes 96→3 (fm, sm, dyaw; all tanh). No single-step anchor on
   (fm, sm) — inputs are UNOBSERVABLE at steady state (constant velocity ⇒
   zero accel regardless of keys held), so any such anchor is wrongly supervised.
3. **Rollout**: per step — yaw += dyaw·YAW_SCALE; wishdir from (fm,sm) rotated
   by yaw offset into ego@T frame; engine Accelerate (ground accel 10, air 10 —
   assumption, documented); friction-first order (already in code); integrate.
   Feedback predicted dyaw (not zero) into GRU input.
4. **Measure first**: p99 of per-tick eye deltas sets DYAW_SCALE (tanh must cover it).
   Skip pitch: TF2 wishdir uses yaw only, pitch adds nothing to movement.
5. **Eval**: same dense hit-rate protocol. Compare vs threatsim champion.

## 4. Queued behind it (in order)

1. **Weapon sidecar** (design done, not implemented): extractor
   (`demodump_rs/src/main.rs`) must dump per-tick active-weapon itemdef to a
   `.weap.csv` sidecar. Resolved identifiers: player
   `DT_BaseCombatCharacter.m_hActiveWeapon` = 12867205056681323540,
   decode entity via `handle & 0xFFF`, read itemdef off the weapon entity
   (`DT_ScriptCreatedItem.m_iItemDefinitionIndex` = 13444166425656641027,
   confirmed on shotgun; shovel-class weapons use a different flattened table —
   resolve per entity by stable-small-int scan, procedure in this file's history:
   DUMP_WEAPON tick-gated map scan). Then: training windows split on itemdef
   change like class change; runtime already invalidates (PushState param landed).
   Rationale: loadout change respawns capabilities without moving (position-only
   detectors miss it when standing at spawn).
2. **Threat-sim + input-space combined** (after input-space proves out alone).
3. **Value head retrain** on the new champion (gated 500ms is the prize:
   denseC value head did 71%@12% coverage from 27% base).
4. **C++ port** of Accelerate + yaw integration into BehaviorPrediction.cpp,
   new weights header, RunTick/release wiring, in-game verification.

## 5. Gotchas (each paid for in full — do not relearn)

- EVAL DYNAMICS MUST MATCH TRAINING DYNAMICS. Friction-trained weights under a
  frictionless rollout faked a collapse (19.5/3.6). Model+rollout is one unit.
- Dense cache has 66 cols, horizons are [6,12,19,32,65]. Indexing the wrong one
  trains garbage silently — validate every eval script with a second implementation.
- Judge by rollout hit-rate, never single-step loss (static-context models win
  single-step and collapse in rollout: corr1, ctx).
- Checkpoints save on best-eval only; distinct names per line (v3i/v3ii/v4...).
  A naming collision once overwrote a champion — numbers survived only in logs.
- FP16 CPU-AMP is 31x SLOWER than fp32 on this Haswell i7-4900MQ. Train fp32,
  export int8 (hit-rate verified ≤0.4pp, 500ms even reads higher).
- NEVER `Stop-Process -Name python3` (kills the tool shell); `taskkill /PID x /F`.
- Eval set FROZEN (byte-identical 147,984 rebuild verified) — all numbers comparable.
- did_registry.json is append-only; all scripts migrated off positional dids.
- Residual-over-heuristic and magnet post-processing FAILED with reasons recorded
  in STATUS.md — do not retry as-is.

## 6. Map

- Code: `ml/*.py` (pipeline), `ml/STATUS.md` (full ledger), `ml/README.md` (metric defs).
- Runtime: `Amalgam/src/Features/Simulation/BehaviorPrediction/` (.h = contract).
- Data (~/Downloads, NOT in repo): `mge_ds/` (npz windows, .pt checkpoints, .db files),
  `mge_csv/` (dumps + .proj/.hurt sidecars), `mge_demos/` + `orange_demos/`.
- Scratch: `C:\Users\Terminatort8000\AppData\Local\Temp\opencode\` (temp scripts/logs).
- Machine: i7-4900MQ 4c/8t, 16GB, torch 2.12.1+cpu, py3.14, torch.set_num_threads(4).
