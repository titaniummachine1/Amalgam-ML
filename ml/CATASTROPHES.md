# DIARY OF CATASTROPHES — every failure, its cost, and the rule it bought.
Read before building anything new. Newest last. No blame, just prevention.

## 1. Dead `tf2/` dir nearly ported (2026-10-12)
Ported Friction/Accelerate/AirMove from `game/shared/tf2/` — which NO .vpc
compiles. Live code is `game/shared/tf/` (in server_tf.vpc).
Cost: ~1h spec work on dead code.
RULE: verify build membership (.vpc) before reading any leaked file.

## 2. Frozen STV columns trusted blindly (2026-10-12)
CSV `vx,vy,vz` repeat one baseline forever (speed pinned 320.0); eye and
maxspeed freeze the same way. Velocity IS networked (SDK NETVAR) but STV
snapshots don't update it.
Cost: validation PASS on garbage (capped=1.000); hours of confused analysis.
RULE: liveness-audit every column (changes/tick) before use. Derive velocity
from positions (engine precedent: EstimateAbsVelocity). Ledger tracks
per-column liveness; builder drops stale rows (gap-aware `load_csv_live`).

## 3. Teleport cut 300hu leaked spawn snaps (2026-10-12)
182hu respawn snap fabricated a -9458hu/s "velocity", poisoning inverse
dynamics and the wall-stop gate. Legit max ~60hu/tick.
Cost: gate 0.005 fiasco + days of suspect data in legacy builders.
RULE: teleport cut 80hu everywhere new. Legacy trainB used 300hu — do not
retrain champion inputs mid-experiment (comparability); tighten on rebuild.

## 4. Extractor repeats rows without updates (2026-10-12)
m2f~0.9: motion rows isolated by frozen repeats. Naive dp/row overestimated
speed by gap length (580hu/s phantoms, capped-everything).
Cost: validation FAIL loop until understood.
RULE: liveness = pos OR eye OR flags changed; v = dp/(gap*DT); pairs need
gap==1. Standing-while-aiming survives via eye. Single source:
`movesim.load_csv_live` — never reimplement.

## 5. abs-vs-relative frame mixup in eval (2026-10-12)
`eval_tx` compared ABSOLUTE rollout positions to RELATIVE stored truth:
0.4% hits regardless of model. Rollout gate passed (correct comparator).
Cost: a full false "model broken" diagnosis cycle + user fury, deserved.
RULE: sanity check (sim+exact packets replicates demo) must use identical
frames both sides; verify the COMPARATOR before judging the model. Oracle
(truth tokens -> sim) is the machinery check: 84.9% @7 eye-fresh.

## 6. np.savez appends .npz to tmp names (2026-10-12)
Atomic-save wrote `x.npz.tmp`, loader looked for `x.npz.tmp.npz`.
Cost: clobbered a good soup file with empty output, twice.
RULE: tmp names end in `.tmp.npz`; verify-load inside the script; os.replace.

## 7. Same-second edit+run executes stale bytecode (2026-10-12)
`__pycache__` mtime granularity: edited file, immediate run used old code.
Cost: phantom failures (soup-floor miss, broadcast-shape error).
RULE: `Remove-Item ml/__pycache__ -Recurse` before any run following an edit
in the same minute. Never parallelize dependent edit->run.

## 8. Window-random val split leaks via overlapping windows (2026-10-12)
In-train vacc 0.49 (memorization) vs held-out 0.132 (truth). 25-token acc
below majority is possible under exposure bias, but the gap is overfit.
Cost: false confidence from smoke + early training; verdict delayed.
RULE: in-train split is early-stopping ONLY. Honest verdict = held-out
DEMOS (eval_all.txt) via eval_tx. Token rebalancing + eye-gated rebuild next.

## 9. Driver listed never-built files (2026-10-12)
`build_all_rollouts` appended outputs for zero-window CSVs that saved nothing
(old code), crashing training with FileNotFound. Fixed: always-save empty
arrays + resume on loadable + rebuild lists from disk reality.
Cost: a dead training launch + diagnosis.
RULE: builders always emit (possibly empty) artifacts; drivers verify
existence after each unit; training lists regenerate from disk, never memory.

## 10. Interrupted turns hide completed work (ongoing)
Killed shells still commit/push/launch. Outputs lie ("no changes", missing
logs) while side effects landed (48ca9109, ddb2651f, tx_v1 training at ep7).
Cost: repeated/duplicate work, confusion about what ran.
RULE: verify by state (git log ml/main, PIDs, file mtimes), never by the
last command's output. `git status -sb` + upstream fixed to ml/main.
