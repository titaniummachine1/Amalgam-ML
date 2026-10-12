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
- Checkpoints: `gru_threatsimB.pt` = CHAMPION (soldier, threat-sim in rollout:
  97.8/87.4/60.0/27.4/12.5 full, threat-active validated). `gru_soldier_denseC.pt`
  previous best. `gru_all_v6.pt` (all-class friction finetune, eval 28.53).
- Threat features built (`trainA_ext.npz`: walk_age + 6 threat feats, 15% nonzero) —
  NOT yet in any training input (needs input-dim expansion = next arch step).
- Data expansion 2026-09-30: +21 MGE +3 orange (brygada, quality over demos.tf pubs).
  did_registry.json append-only (0-24 frozen, new 25-48); all scripts migrated off
  positional dids (which would have silently corrupted keys); eval set FROZEN
  (byte-identical 147,984 rebuild verified). Train 532k→928k windows; thin classes
  fixed: pyro 13k→51k, engy 11k→31k, medic 11k→24k, heavy 31k→39k. cls-0 excluded.

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
- INDEXING: dense cache has 66 cols, horizons are [6,12,19,32,65]. value_denseC
  trained once on WRONG columns (hi instead of mapped) — garbage labels, had to
  retrain. Duel scripts caught it. Always validate eval code against a second
  implementation before trusting headline numbers.
- Knockback: 33k hurt events extracted, 90% victim-resolved via health drops.
  Impulse-split caps already capture ~all of it (flagged ≈ clean distribution).
  Near-damage labels ARE hotter (p50 10k vs 870) — caps, not events, do the work.
- Data is clean: 0 NaNs, 0 dupes (PK), teams 2/3 only, 4 demos have zero
  projectiles (few player samples there, defaults cover). Rocket oz 100% NaN → 2D only.
- Threat features exist (15% nonzero, FOV 5%) but are NOT in any training input yet.

## 2026-09-30: input-space policy + data findings (in progress)

- NO VIEW ANGLES in SourceTV demos. m_angEyeAngles lives only in
  DT_TFLocalPlayerExclusive/DT_TFNonLocalPlayerExclusive, which the server never
  populates for remote players (verified via DUMP_TABLES: eye#=0 for every moving
  player). DB `eyey` is ONE frozen spawn constant per entity; the "ego@T frame"
  is really the velocity-heading frame; base feature slot 4 (dyaw) is identically
  zero. Consequences: no dyaw ground truth, no yaw rotation - packet output is
  (fm,sm) in the velocity frame only.
- m_flMaxspeed IS networked per tick (DT_BasePlayer, float). 240.0 on 100% of
  soldier rows. Lua class enum (Scout 1..Engineer 9) matches MAXV_CLS exactly
  (scout 400 fastest, heavy 230 slowest; only mismatch: spy 320 vs 300).
  Engine equilibrium 240*(1-4/66)=225 lands on the observed grounded plateau
  (mode 230) - the friction/stopspeed/accel constants are in the data's own
  units. Charge/boost RAISES maxspeed (seen: 332.5/345/384/400/825), so per-tick
  mspd is the correct wishspeed cap, not the class constant.
- MEASURED: the engine add-speed early-out `wishspeed-|v|` is incompatible with
  this dataset at a 240 cap (labels show +500..1500 hu/s^2 radial accel at the
  speed plateau; dropping the rule is +16pp hit-rate at 300ms on the champion's
  own direction). Input-space rollout = friction-first + accel ceiling
  (accel*dt*wishspeed), addspeed OFF; (fm,sm) bounded by radial clamp, NOT tanh
  (tanh compresses p90 magnitude by 19%).
- HARDENED extractor (the eyey lesson): ml/demodump_rs prints a per-entity LIVE
  ledger at end of every run (rows/pos#/eye#/mspd# + STALE/EYE-FROZEN flags);
  ml/verify_csv.py gates every dump (exact header, (ent,tick) unique, tick
  monotonic, liveness flags); build_mspd resolves columns BY NAME, asserts the
  header contract, refuses below 50% mspd coverage. Re-dumps are content-verified
  as keyed sets before swapping in (row ORDER is nondeterministic - HashMap).
- DATA: all 49 CSVs re-dumped with an `mspd` column (content-identical on shared
  columns; eval threatwin rebuilt BIT-IDENTICAL, so the frozen eval set is
  untouched). train-side caches are stale from the 40% data expansion; the old
  train caches are a verified bit-identical PREFIX (first 189565 soldier rows =
  exactly the champion's training rows) -> input-space A/B is data-matched.
- FILES: ml/inputspace.py (single-source rollout: train+eval share it),
  ml/train_inputspace.py (TAG=inputspaceA, addspeed off, feedback on, mspd on,
  4ep, warm start gru_threatsimB head/10), ml/hit_eval_input.py (champion
  protocol + reparam rows), ml/test_inputspace.py (engine formula vs longhand,
  ALL PASS), ml/build_mspd.py, ml/verify_csv.py.
- VELOCITY-FRAME packet result (gru_inputspaceinputspaceA.pt, 189565 matched rows):
  loss 22.11 (vs 21.56 champ) but hit-rate COLLAPSED 59.3->44.0 @300ms. Head
  outputs identical to init; GRU drifted recurrently into coasting. Lesson:
  position-loss-only training through 66 steps rewards conservative throttle.
  Fix = per-tick input supervision (below) + horizon reweighting.
- ENGINE CORRECTED: CGameMovement::Accelerate projects velocity ONTO WISHDIR
  (v.wh), NOT |v|. The magnitude version was wrong and cost 16pp @300ms. Dot
  rule verified vs longhand + reparam check (84.4 vs 78.0 @200ms). Throttle
  labels use W=max(|dv|/0.1515, v.what+|dv|) (dot-consistent; median 0.98).
  Wishdir rotation: forward=(c,s), right=(s,-c), D-positive sm (ego+y=left).
- INPUT LOSS (inputspace.py): quadratic on throttle distance (obs-gated),
  -log((1+cos)/2) on direction alignment, quadratic on yaw (fresh-gated), all
  x w(t). Pitch skipped (yaw-only wishdir). Smoke-tested (grads flow).
- C-CHAMPION RETRAIN hit a NaN-grad epoch (killed, ep0 ckpt safe). Forward clean
  on all batches; grads non-finite systematically in P.* harness (under
  investigation - suspect sqrt/norm/atan2 backward at zero). train_threatsim's
  own unroll clean on 20 batches. Restarted WITH skip-on-nonfinite guards
  (logs SKIP-NONFINITE batches for diagnosis).
- IN-GAME LOGGER (for morning DLL): Features/Misc/MoveLog (CreateMove hook,
  usercmd+state+cvars to %TEMP%/amalgam_movelog_*.csv, Vars toggle). User builds
  (5min), plays, sends CSV = ground-truth input->trajectory + cvar values.
  BUILT 2026-10-01 04:25: output/x64/Release/Amalgamx64Release.dll (3MB, x64 =
  tree is x64-first; x86 config has pre-existing breakage: Menu size_t/1ui64,
  ExceptionHandler x64 regs - fixed 2 Menu lines + weights np.float32 syntax +
  TargetState visibility to get x86 compiling, but x86 link still needs work;
  ship x64, confirm user runs 64-bit TF2).
- VERDICT 2026-10-01 ~06:30 (generation C, eye frame, 2500-row protocol):
  C-champ (accel) 98.0/84.2/54.0/24.9/11.6 vs packet (fm,sm,dyaw,pitch + input
  loss) 97.0/83.0/51.2/24.5/11.7 vs coast 92.5/73.6/52.0/30.1/12.3.
  Packet does NOT beat accel (-1..-3pp). B-champ (velocity frame) 60.0 @300ms
  still best overall; eye frame helps 100ms (+0.2) but hurts 300ms (-6).
  Coast still wins 500/1000ms. CONCLUSION: accel prediction wins today
  (user theory confirmed); packets queued (gap small, may close with tuning).
  Sign test (forced +/-psi) inconclusive - all variants equally bad (test broken,
  not evidence). packetC2 (warm from C-champ) running to check the gap.
  UPDATE packetC2 (warm from C): 97.1/83.2/53.8/24.9/11.6 - gap fully closed to
  0.2-1pp (was warm-start quality, not form). Packets now TIED, keep all
  structural wins (class-free, bounded, full packet). Frame matters more than
  form (6pp vs 1pp): next experiment is packets on B velocity frame.
- FORCES VERDICT 2026-10-06 (B, new 108k-row eval after expansion):
  champ 97.7/87.2/61.2/28.0/11.9 vs forces(29-dim, knock+flags+W2+momentum)
  97.1/87.5/60.4/26.2/10.9. Wins 200ms (+0.3, +1.1 on knockback rows), loses
  300/500/1000. k_imp=0 ablation: impulses ~neutral (+0.2 @200, -0.5 @500) -
  the gap is training mix (W2 coast-downweight suspect), not physics.
  Sidegrade, not a promotion; champion keeps the crown. Next: W2-without-coast
  ablation + value-gated coast blending for the 500ms prize.
- NaN HUNTS (two distinct mechanisms, both version-proofed):
  (a) NaN TARGETS poison through NaN*0=NaN despite validity masks (short-series
  data made invalid ticks common) -> nan_to_num all targets (dense/forces/packet).
  (b) torch.where does NOT protect 0/0 in BACKWARD: dist/appr with appr=0
  (perpendicular approach, routine) NaN'd every batch through the dead branch ->
  clamp the denominator (forward bit-identical). Same fix in train_threatsim.
  Plus eps-inside-sqrt (slen()) on all 16 length sites (torch-version-proof).
  UPDATE packetC2 (warm from C): 97.1/83.2/53.8/24.9/11.6 - gap fully closed to
  0.2-1pp (was warm-start quality, not form). Packets now TIED, keep all
  structural wins (class-free, bounded, full packet). Frame matters more than
  form (6pp vs 1pp): next experiment is packets on B velocity frame.
- C++ WIRING (denseC, for morning test): BehaviorPrediction_t gains m_vPredicted
  (world positions, ego->world rotation verified exact); PushState feed in
  CreateMove (all enemies, per tick); Predict override in AimbotProjectile
  (gated by Vars::Aimbot::Projectile::BehaviorPrediction); eval logs
  (amalgam_behaveval.csv predictions + MoveLog .enemies.csv actuals for offline
  hit-rate join). MoveLog extended with enemy positions. Menu checkboxes added.
  BUILT 16:18 output/x64/Release/Amalgamx64Release.dll (with wiring).
  ThreatsimB weights exported (weights_threatsimB.h, int8 roundtrip err 0.77 -
  needs hit-rate verification before wiring 20-dim + threat runtime).
  BUILT 2026-10-01 04:25: output/x64/Release/Amalgamx64Release.dll (3MB, x64 =
  tree is x64-first; x86 config has pre-existing breakage: Menu size_t/1ui64,
  ExceptionHandler x64 regs - fixed 2 Menu lines + weights np.float32 syntax +
  TargetState visibility to get x86 compiling, but x86 link still needs work;
  ship x64, confirm user runs 64-bit TF2).

## 2026-10-06: expansion #3, forces, momentum, in-game wiring

- Data: +15 demos (brygadagraczy.pl/sourcetv) -> train 526k->660k soldier windows
  (eval frozen). Filter audit per class (raw->ok->cand->imp->kept; kept/raw 6-9%,
  no class eaten; heavy lowest at 43% ok-rate, correctly - they stand still).
- Air cap 1500->10*MAXV=2400 (ate legit air-strafes); short series kept (67 min);
  loadout->input feature decided (sidecar first, no split; runtime stops resetting).
- CORRECTION to 09-30 notes: eye angles ARE sent (DT_TFNonLocalPlayerExclusive);
  the extractor's eye() had a last-wins bug shadowing live values with the stale
  Local baseline. Fixed -> re-dumped 49 CSVs (content-verified) -> dataset_eye.db
  (live eye+pitch) -> C generation (eye frame, dyaw+pitch labels, p99 14-16deg).
  C-champ 98.0/84.2/54.0/24.9/11.6; packetC2 tied 97.1/83.2/53.8/24.9/11.6.
  Frame matters more than form (eye frame costs 300ms vs velocity for BOTH).
- Known-forces rollout (2-pass, input last) + knockback sidecar (KB/KBW,
  K_fit=3.0 hu/s/dmg) + per-tick flags/health + W2 (blast 0, recovery x2) +
  momentum training (velocity aux 0.1). FORCES VERDICT: 97.1/87.5/60.4/26.2/10.9
  vs champ 97.7/87.2/61.2/28.0/11.9 - wins 200ms, loses long. k_imp ablation:
  impulses ~neutral; gap is training mix (W2 coast-downweight suspect).
- NaN hunts: (a) NaN targets via NaN*0 despite masks -> nan_to_num everywhere;
  (b) torch.where does NOT protect 0/0 in backward (dist/appr, appr=0 routine) ->
  clamp denominators (forward bit-identical). Plus slen() eps-inside-sqrt x16.
- Engine corrected: CGameMovement projects velocity ONTO WISHDIR (dot rule, not
  |v|); throttle labels use W=max(|dv|/0.1515, v.what+|dv|).
- In-game: MoveLog + BehaviorPrediction wiring + eval logs + menu, frame mismatch
  found+fixed (B saw frozen eye -> C++ feeds velocity frame + zero dyaw), hold-fire
  off by default, confirm-shot gate, splash backface culls both sides (muzzle-ref),
  shot logger. TF2 Oct-2 update broke KeyValues sig (upstream fixed, pulled).
- Fork: backup/master-pre-upstream-20261006 + splash-backface-cull (+21 lines)
  pushed; master reset pending user go-ahead.

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

## 2026-10-10 overnight (4th pull + attention bracket)
- DATA 2x: pulled 37 MGE + 11 orange from brygadagraczy.pl (10-08..10-10),
  dumped all 125 demos, didreg 128, dataset_all.db 121 live demos (42.5M ticks).
  Rebuilt trainB 3.24M windows (soldier 660k -> 1.34M) + evalB 159k + all B
  caches (dense66/threatwin/mspd) + NEW trainB/evalB_dense132 (132-tick targets,
  cov@132 0.88/0.94). Eval NOT byte-identical (builder evolved: short-series
  inclusion) -> champion RE-BASELINED on new files: 97.7/87.2/61.2/28.0/11.9
  (identical -> comparable). Fixed stale header contract (build_mspd/
  verify_csv now expect wcls/w0/w1/w2); dyaw never existed for B (eyep only in
  eye DB) -> skipped, matches history.
- IO: soldier rows scatter across 12GB X (page faults ~1.4s/batch) -> repack20.py
  (contiguous X20/HD66/HD132 + VD/S0 sidecars, resume-safe .done.json).
  train_attn66/train_correct132 dual-mode loaders (repack fast path w/
  done-guard, legacy fallback). NOTE: np.save .npy flaked Errno 22 twice
  (transient lock); S0 sidecar bypassed via threatwin direct load.
- VARIANT A (GRU20-96 + single-head attention over 66 warmup hiddens + coast
  suggestion cols, warm-started champion-exact incl. 22-dim zero-pad):
  ep1 wiped at 34% (4AM machine-wide python kill) + relaunched on repack.
- VARIANT C (correction-transformer 1L d32 2head, one pass over 132 ticks,
  friction-coast base AS INPUT, uncertainty ramp, absolute power loss):
  ep1 random-head 51/56/44 (destroys base) -> zero-init head; relative loss
  EXPLOITED via sign flips (1%!) -> absolute champion-formula loss; ramp
  recovered near field. ep1: 96/80/51/12/3, ep2: 96/79/45/26/4.5 (ALL
  @7/13/20/33/66). Beats coast @7/@13, loses to sim overall. Needs capacity
  (2L/d64?) or more epochs. Corrector verdict: promising near-field refiner,
  not a champion (yet).
- C++ (unbuilt): HitChanceML toggle (Vars.h + Menu.cpp) + veto gates at direct
  + splash record sites in CanHit (main path only; redirect path untouched),
  stub-pass-through (conf 1.0) -> zero behavior change until value head lands.
- Machine notes: shells get killed constantly (user stops? watchdog?) but
  CHILDREN SURVIVE as orphans -> launch-with-log + poll pattern works. 4AM wipe
  killed everything incl. DB writer (resumed cleanly: per-demo commits).
  Stop-Process -Id <my-own-pid> is safe (shell survives); never touch others' PIDs.

## 2026-10-11/12: sim-first era (offline truth loop, no game)

- DIRECTION (user-ordered): parser-only data path. DemoLoop live-capture built
  then REMOVED (commits ebec4c86 -> 48ca9109); training consumes STV-demo truth
  + ported engine formulas + BSP map soup. No game, no supervision needed.
- LEAK REFERENCE (read-only, NEVER committed): C:/gitProjects/tf2-source-code
  (sparse: game/shared, game/server, engine, public, raytrace, mathlib, common,
  tier0/1). Rule: numeric constants + independently written code only; never
  `git add` anything under it (DMCA risk to the repos). Sibling folder, not in repo.
- LIVE movement file: game/shared/tf/tf_gamemovement.cpp (in server_tf.vpc).
  game/shared/tf2/ is DEAD (no .vpc references it; caught before porting).
  Constants: accel 10, airaccel 10, friction 4, stop 100, gravity 800,
  aircap 30, jump 289, wish-thresh 40, back-clamp 0.9/100, dt 0.015.
  Ground = WalkMove (NOT WalkMove2); air = base AirMove; friction = base.
- CANONICAL NEW FILES (this era; everything else in ml/ is legacy — read but
  do not delete others' working scripts):
  ml/movesim.py (formula port) + ml/validate_movesim.py (free-space gate PASS:
  MGE-calibrated: 90-mass 0.35>0.30, round-trip res 0.000, capped 0.001) +
  ml/parity_movesim.py (Python<->Rust 20k cases BIT-IDENTICAL 0.0) +
  ml/mapsoup.py (BSP->collision soup) + ml/souptrace.py (ray + wall-stop gate) +
  ml/diag_stops.py, ml/dbg6.py, ml/dbg7.py (one-off forensics).
  C:/gitProjects/tf2-movesim/ (separate Rust workspace: movesim-core 6/6 unit
  tests PASS + rollout-cli parity harness). C++ runtime port comes at ship time.
- DATA HARDENING (byte-proven, all in-repo scripts updated):
  STV velocity columns are FROZEN (parser repeats baseline; bytes prove positions
  update, velocities repeat verbatim) -> derive velocity from positions x66.67
  (engine does the same: EstimateAbsVelocity precedent; arcs fit -400*DT^2 to 2%).
  STV flags include stale rows (airborne-but-motionless) -> quarantine logic.
  Teleport cut 300->80hu (182hu spawn snap fabricated -9458hu/s "velocity";
  legit max ~60hu/tick). Pipeline successors must adopt 80hu (trainB used 300hu;
  do NOT retrain champion inputs mid-experiment - comparability).
  Duck maxspeed x1/3 assumption; class table fallback when mspd missing.
- MAP SOUP (beta4b): 141,008 tris from 6594 solid brushes (PLAYERSOLID mask,
  bevels kept, outward-normal clipper); 30 illusionary + 4 door brushes excluded
  via model-face plane matching (near-miss empty); 0 displacements; 1491 static
  props (733 solid) = documented gap (meshes later); movers skipped+counted.
  File: ~/Downloads/mge_ds/soup_mge_training_v8_beta4b.npz (tris/contents/doors/meta).
- ARCHITECTURE (locked): sim owns physics+collisions (code, validated, cannot
  drift); map lives in soup (never in model); model learns ONLY behavior
  (intended inputs). Targets always observed truth (no self-training loop).
  Unpredictable ticks (jump/blast/teleport/knockback) = zero loss weight,
  context preserved. Dual rollouts (policy-inputs vs demo-derived-inputs)
  separate physics error from behavior error. Geometry-as-input (wall probes)
  is a LATER experiment, not a requirement. Transformer: early-exit AR
  (stop at intercept solution), binned-input tokens, confidence head, 5ms
  measured budget; must beat champion 97.7/87.2/61.5/28.0/11.9 on frozen eval.
- ANTI-CONFUSION RULES (Stockfish discipline): STATUS.md is the single source
  of truth (update every session); every new module docstring cites its spec
  (leak file:line) + gate + usage; dead code deleted same-day (never piles);
  one-off debug lives in %TEMP%/opencode (never repo); gates assert before
  downstream runs (verify_csv, parity 1e-9, validation PASS); training inputs
  pinned by filename; no two scripts own the same step (canonical list above).
