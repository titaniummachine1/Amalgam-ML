# DIARY OF CATASTROPHES — mistakes made, causes, fixes, preventions.
Update same-day, every time. Newest first. Read after compaction.

## 2026-10-12: abs-vs-rel eval bug (severity: CRITICAL, wasted full train+eval cycle)
- What: eval_tx compared ABSOLUTE rollout positions vs RELATIVE stored truth
  (POS saved relative to window start, never re-offset). Off by thousands of
  hu on every window -> 0.4% hits regardless of model. Declared the model dead
  before finding it.
- Why: rollout_gate (which passed) compares correctly; eval was written fast
  and never cross-checked against the gate on identical inputs.
- Fix: `truth = POS[gi] + p0`. Oracle (truth tokens -> sim) jumped 0.4% -> 84.9% @7.
- Prevention: ORACLE FIRST — truth-through-sim must pass before ANY model eval.
  Comparator frames (abs vs rel) must be asserted in code, not eyeballed.

## 2026-10-12: leaky in-train val hid total generalization failure
- What: vacc 0.49 (window-random 10% split shares overlapping history with
  train -> memorization) vs 0.132 on held-out demos. Trusted the inflated
  number for a full training run.
- Prevention: window-random splits are for early-stopping ONLY; the verdict is
  always held-out DEMOS (eval_all). Never quote leaky numbers as progress.

## 2026-10-12: train crashed on listed-but-missing npz
- What: chunk driver appended output paths before verifying existence; one CSV
  built 0 windows (old code saved nothing) yet got listed -> FileNotFound mid-train.
- Fix: disk-reality lists (train_all from files on disk, not chunk lists).
- Prevention: drivers append outputs ONLY after verified existence+load.
  Trainers skip missing files with a warning (hardening still OPEN for train_tx).

## 2026-10-12: eval split landed all 6 eval demos in chunk 0 (20 % 4 == 0)
- What: works, but fragile coincidence, not design. A different eval_every or
  chunk count silently unbalances or empties eval.
- Prevention: stratify splits explicitly (by map/class), assert eval non-empty
  and class-mixed in the driver.

## 2026-10-12: parallel edit+run races (multiple)
- What: same-turn edit + bash-run raced; runs executed stale code (np.savez
  .tmp gotcha misdiagnosed, phantom "already fixed" states, duplicate blocks).
- Prevention: NEVER parallelize dependent edit->run. One turn = edits; next
  turn = run. Stale .pyc after same-second edit+run -> always rm __pycache__.

## 2026-10-12: missing __main__ guards crash importers (twice)
- What: souptrace.py and train_tx.py ran main() on import (dbg6/eval_tx broke:
  gate executed with wrong argv; training launched on import).
- Prevention: every script with side effects gets `if __name__ == '__main__':`.
  Grep for bare `^main()` before commit.

## 2026-10-12: numpy/Windows file traps (three)
- savez appends .npz to tmp names (atomic-save broke); np.load keeps file
  handles open (Windows os.replace PermissionError -> use `with`); compressed
  zip writes stall under CPU starvation and get killed (use uncompressed +
  tmp + verify + os.replace).
- Prevention: the atomic-save pattern in mapsoup.py is canonical, copy it.

## 2026-10-12: wall-stop gate confounds (detector, not soup)
- What: 0.005 agreement; root causes found one by one: respawn/teleport snaps
  (182hu -> fake -9458hu/s), body-blocks (d=0-4hu!), dup-row freeze artifacts,
  stale flags. Soup exonerated (floor-hit + free-path + synthetic rays pass).
- Prevention: gate the DATA before the geometry (teleport cut, quarantine,
  liveness, persistence, proximity). Superseded by rollout_gate for sim
  fidelity; wall-stop kept for spot checks only.

## 2026-10-12: sweep tunneling + frame bugs (two, caught by selftest)
- (a) Single-slide returns p_hit+slid, tunneling past floors after wall grazes
  -> bump loop (re-sweep remainder, contact points, max 4).
- (b) Hit distance measured from offset ray origin, not segment start ->
  project hit point onto move dir for the fraction.
- Prevention: selftest with synthetic + real-soup cases runs on every change.
  Never trust untested geometry code.

## 2026-10-12: illusionary/door brush exclusion (matching direction mattered)
- What: subset test (keys<=ps) missed doors (6 sides vs 5 faces); overlap test
  (all model planes present) missed multi-brush illusionaries. Correct rule:
  nearly all of the BRUSH's planes in the model set (>=4 shared).
- Prevention: near-miss reporting (shared-plane counts) makes assignment
  quality visible instead of hoped.

## 2026-10-12: STV frozen velocity columns (caught pre-training)
- What: extractor repeats baseline vel (bytes prove positions update, vels
  repeat verbatim). Would have trained on frozen targets.
- Fix: derive from positions x66.667 (EstimateAbsVelocity precedent); arcs fit
  -400*DT^2 to 2% as proof. SDK NETVAR confirms velocity IS networked live;
  the freeze is STV/parser-side.
- Prevention: liveness ledger per column (the extractor's own lesson: it once
  trained months on frozen eye yaw). Byte-level spot checks before trust.

## 2026-10-12: teleport cut 300 -> 80hu
- What: pipeline-standard 300hu leaks 182hu spawn snaps (fake -9000hu/s).
  Legit max ~60hu/tick (4000hu/s blast).
- Prevention: cut values derive from physics maxima, not round numbers.
  trainB used 300hu (do NOT retrain mid-experiment; adopt 80hu for successors).

## 2026-10-12: dup-row stale repeats (m2f~0.9)
- What: extractor repeats rows without updates; naive dp/row overestimates
  speed by gap (580hu/s phantoms, capped-everything).
- Fix: liveness = pos OR eye OR flags changed; v = dp/(gap*DT); pairs need
  gap==1. Single source: movesim.load_csv_live (never reimplement).
- Prevention: distribution checks (speed histograms) before trusting derived
  quantities; capped-everything = data bug smell, not formula bug.

## 2026-10-12: dead tf2/ dir (near-catastrophe, caught pre-code)
- What: nearly ported movement from game/shared/tf2/tf_gamemovement.cpp; .vpc
  files prove it is DEAD (uncompiled). Live file: game/shared/tf/.
- Prevention: verify build inclusion (.vpc/CMake) before reading ANY leaked
  file as spec. Comments saying "new code" mean nothing.

## 2026-10-12: DemoLoop built then removed (direction churn)
- What: full DLL live-capture path (built, compiled, deployed) deleted same
  day when user ordered offline-only. Restorable in 1 cmd (git history).
- Prevention: cheap to restore; no lesson except: confirm direction before
  building runtime features (plan mode exists for this).

## OPEN (not yet bitten, watch closely)
- train_tx missing-file hardening (skips empty, crashes on missing).
- Orange v34 BSP absent (21 demos blocked; beta4b-only corpus).
- ep7 dead (shell wipe ~14:03); checkpoints safe; relaunch after transformer v1.
- Prop meshes (733 solid, deferred); Rust BVH (scale); jump sim in eval.
- Token imbalance (none-collapse) + frozen-eye windows (poison labels) — the
  current prime suspects for vacc 0.49 -> 0.132. Eye-gate + reweighting queued.
