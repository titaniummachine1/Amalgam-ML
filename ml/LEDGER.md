# LEDGER — what exists, what runs, what's dead. Read this first after compaction.

## Hard lessons (never repeat)
- abs-vs-rel (2026-10-12): eval_tx compared ABSOLUTE rollout positions against
  RELATIVE stored truth -> 0.4% hits regardless of model. Rollout gate passed
  because it compares correctly. Lesson: the sanity check (sim replicates demo
  with same packets) must use identical frames on both sides; verify the
  comparator before judging the model. Oracle (truth tokens -> sim) is the
  machinery check: 84.9% @7 on eye-fresh windows.
- Binning convention verified self-consistent (encode->decode median cos 0.98,
  zero ticks below 0.9 over 57k ticks). No sign/frame bug.
- Training targets are OBSERVED TRUTH only. No self-training loops (no drift).
- Sim-truth disagreement doctrine: same inputs through the sim MUST reproduce
  truth (rollout_gate enforces: 2.4hu @7, 7.7hu @33). Any excess deviation is a
  SIM bug or DATA bug — fix the sim/data, never tune the model to compensate.
  (STV stores no inputs; inverse dynamics recovers them. Capped-regime rows
  carry lower-bound inputs — masked/downweighted, never trusted exactly.)
- Unpredictable ticks (jump/blast/teleport/knockback) get ZERO loss weight, kept as context. Never reward, never penalty.
- Poison rule: tick sequences must be consecutive; gaps quarantined, never ingested.
- NO bot data for training. NO PR anywhere without measured proof.
- Leak (`C:/gitProjects/tf2-source-code`) is READ-ONLY reference, outside all repos. NEVER stage/commit any of it (DMCA).
- Budgets: 15ms/frame total sim; transformer ≤5ms for 132-tick decode (early-exit AR).
- Champion baseline (frozen eval, hit-rate @24hu, HOR=[7,13,20,33,66]): 97.7/87.2/61.2/28.0/11.9. Beat 3/5 to ship.

## ACTIVE — run these
| file | purpose | proof/gate |
|---|---|---|
| `ml/movesim.py` | TF2 movement formulas (live `tf/` file, build-verified; `tf2/` dir is DEAD) | free-space PASS 400k; jump 289; arcs fit -400*DT^2 to 2%; DT=0.015 |
| `ml/validate_movesim.py` | formula gate | PASS (90-mass 0.35, res 0.0, capped 0.001) |
| `ml/parity_movesim.py` | Python↔Rust differential | 20k cases bit-identical (0.0) |
| `ml/train_both27.py` | GRU-27 trainer (ep7: 132-horizon, blame, adaptive, jumpmask) | DIED ~2026-10-12 14:03 (shell wipe; best eval 382.16 ckpt safe). Relaunch AFTER transformer v1 (CPU). |
| `ml/hit_eval_both.py` | verdict vs champion | beats = ship |
| `ml/mapsoup.py` | BSP→collision soup (srctools, engine-exact contents) | beta4b: 141008 tris, 34 brushes excluded, doors tracked |
| `ml/build_rollout.py` | counterfactual dataset: 33+20 windows, ego-frame 25-tokenizer, masks, rollout errors | CORPUS DONE 2026-10-12: 107 beta4b CSVs -> 101 train + 6 eval demos; ~95k rows; masks 9.6/8.5/61/20.5. v1 = open-only + stride 2; stride-1 + wall rows + sweeper = v2 if needed. |
| `ml/dbg7.py` | miss forensics (done) | verdict: stops = body-blocks (d=0-4hu) + dup artifacts; soup exonerated (floor-hit + free-path pass) |
| `ml/souptrace.py` | wall-stop gate (SUPERSEDED by rollout_gate for sim fidelity) | FAIL was detector confounds, not soup. Keep for soup-geometry spot checks. |
| `ml/sweep.py` | gate-grade hull sweep (bump loop, ClipVelocity, step-up) | selftest PASS (synthetic + soup floor + free-path) |
| `ml/build_all_rollouts.py` | corpus driver (per-CSV npz, resume-safe, train/eval split by demo) | running 4 workers over 107 beta4b |
| `ml/train_tx.py` | tiny AR transformer (d64/2L/4H, 25 tokens + jump + confidence) | smoke learns (val 1.94->0.66 acc .84); multi-file corpus support |
| `ml/eval_tx.py` | AR eval, champion protocol (greedy decode, frozen eye, demo-grounded) | smoke runs; in-train val is window-random (early-stop only) — honest verdict = held-out demos; jump sim TODO |
| `ml/rollout_gate.py` | THE agreement gate: recovered-input rollouts vs truth (dataset engine) | PASS 230 seg: 2.4/3.9/5.8/7.7hu @7/13/20/33, beats frozen 3-8x |
| `ml/build_jumpmask.py`, `ml/build_w7.py` | jump mask, weapon cache (ep7 inputs) | built |
| `C:/gitProjects/tf2-movesim/` | separate Rust workspace (formulas+CLI, unsupervised-safe) | 6/6 tests, parity harness |

## ARCHIVED (`ml/_archive/`) — do NOT run, kept for forensics only
| file | why dead |
|---|---|
| `diag_stops.py` | superseded by `dbg7.py` (old detector, teleport hole) |
| `dbg6.py` | one-off; finding recorded (182hu spawn snap → fake -9458hu/s vel) |

## REMOVED (git history only)
- `DemoLoop.*`, `ml/ingest_demoloop.py` — offline-only direction; restore = 1 cmd.

## RUNBOOK (exact commands; run verbatim after compaction)
Corpus (beta4b, 107 CSVs, stride 2, eval every 20th demo):
  list:  powershell Get-ChildItem mge_csv *beta4b.dem.csv (no hurt/proj) -> beta4b_list.txt
  build: 4x Start-Process python ml/build_all_rollouts.py <list> <outdir=~/Downloads/mge_ds/rollout_beta4b> <soup> 2 20 <i> 4  (logs: %TEMP%/opencode/corpus_c<i>.log)
  merge: cat rollout_beta4b/train_list_c*.txt > train_all.txt ; cat *eval_list_c*.txt > eval_all.txt (dedupe)
Train (CPU, ~1.5h @30ep on ~100k rows):
  python ml/train_tx.py <train_all.txt> 12 <out.tx.pt>   (12 epochs v1; smoke used 30 on 1k rows)
Eval (champion protocol, held-out demos ONLY):
  python ml/eval_tx.py <out.tx.pt> $(cat eval_all.txt)   (compare @7/13/20 vs champ 97.7/87.2/61.2; FUT=20 caps horizons)
Verdict: beats frozen-input rollout (rollout_gate) AND approaches champ @7/13/20 -> ship track (int8 export + C++ port + quant gate <=0.5pp + 5ms measured); else diagnose via ERR/token-hist, next variant.
Never: train on eval_all demos; single-CSV smoke models for verdicts (overfit toy); wall rows (open-only v1).

## PENDING
- both27 ship (quant gate ≤0.5pp, C++ 27-dim swap, rebuild, push) after ep7 verdict.
  ep7 PAUSED 2026-10-12 for corpus build (CPU contention; checkpoints safe, resume anytime).
- Orange v34 (21 demos): BLOCKED on BSP (not on disk anywhere). beta4b corpus first; orange joins when BSP lands.
- Rollout dataset (dual rollouts, innovation + λ-blame, 80hu teleport cut!) → tiny early-exit AR transformer → geometry-probe inputs (later experiment).

## Known data facts (do not rediscover)
- STV vel/eye/maxspeed columns FREEZE (parser/extractor artifact); velocity is networked (SDK NETVAR) but STV snapshots don't update it. DERIVE velocity from positions (×66.667). Flags mostly live; row-1 transients suspect.
- Extractor repeats whole ROWS on ticks without updates (m2f~0.9 for duelists; ent 6 fully frozen). Naive dp/row overestimates speed by gap length (580hu/s phantoms, capped-everything). Liveness = pos OR eye OR flags changed; stale repeats DROPPED; v = dp/(gap*DT); pairs need gap==1. Standing-while-aiming survives via eye (true v=0). Single source: `movesim.load_csv_live` (validate/souptrace/dataset all use it — never reimplement).
- Teleport cut is 80hu, NOT 300 (182hu spawn snaps leak through 300). Pipeline builders still use 300hu — tighten on next dataset build.
- DT = 0.015 (66.67Hz), gravity 800, jump 289, accel/air 10, friction 4, stop 100, aircap 30.
- Corpus v2 2026-10-13 (`~/Downloads/mge_ds/rollout_v2/`, same 107 CSVs, 4 workers): eye-liveness gate only change
  (eye unique ≤2 over HIST+1 AND speed>100 → drop window). Result: 70 train files / 78217 rows + 6 eval / 10980 rows,
  31 empty (verified pre-existing duds — zero v1-productive demos killed). Masks 9.5/8.4/61/21, none-token 12.8%.
   `train_tx.py` rebalanced (inverse-sqrt class weights). v1 verdict: token-acc 0.132 held-out (none-collapse + frozen eye);
   v2 target = beat v1 @7/13/20 (83.1/42.9/23.7).
- VERDICT v2 2026-10-13 (same v2 eval set, 10980 rows): tx_v1 82.9/42.8/23.6 (tacc .133) vs tx_v2 **83.2/49.4/28.8**
  (tacc .124). Rebalance trades single-token acc for long-horizon hits (+6.6pp @13, +5.2pp @20) — REAL gain, not data artifact.
  Gap to champ (97.7/87.2/61.2) still large, worst @13. Next: bigger model (d96/4L/30ep) on v2, then stride-1 + sweeper corpus.
