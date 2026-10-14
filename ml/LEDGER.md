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
- VERDICT v3 2026-10-13 (d96/4L/4H/ff192, 30ep, same v2 eval): **82.5/46.8/26.6** (tacc .137, val .0607) vs v2 83.2/49.4/28.8.
  Bigger model does NOT win on 78k rows (better leaky-val, worse held-out hits; token-acc up, hits down — acc≠hits again).
- VERDICT v4 2026-10-13 (d64/2L, 30ep, stride-1 corpus 150631 rows): **83.1/49.3/28.3** (s1 held-out 21k rows).
  Doubling data: NO gain. Capacity + data both exhausted as levers.
- COAST VERDICT (frozen inputs, same physics/flags): **86.5/59.5/34.8** (s1) and 86.3/59.7/34.9 (v2 set) — beats v1/v2/v3/v4
  AND the binned oracle (84.9 @7) at EVERY horizon. Root cause: 25-bin quantization ceiling (22.5° octants, 2 speeds;
  bin centers cannot reproduce true inputs; residual > frozen drift). Training NOT useless — target formulation is.
  Next variant (building overnight): continuous future inputs (`ml/build_cont.py` → WC sidecars) → regression head
  (wishdir xy + wishspeed float) instead of 25 bins. Morning: continuous trainer + ceiling-proof number.
  Bottleneck = DATA not capacity. v2 (d64/2L) stays champion. Next: orange corpus (BSP secured from Brygada fastdl via
  join 54.38.205.24:28045, soup 32k tris) + stride-1 → combined ~170k rows → retrain d64.
  NOTE loss scale: v1 1.49 vs v2/v3 0.078/0.061 NOT comparable (inverse-sqrt rebalance shrinks effective scale ~20x —
  cls_w mean-1 over classes, batches dominated by low-w none). Only v2↔v3 comparable. vacc/hits are scale-free.
- NO-CHEAT CERTIFICATE 2026-10-13 (`ml/gate_wall.py`, beta4b, orange soup 32k tris unused — see below):
  wall-impact windows (speed>150, wall<250hu ahead, slowdown<60, trace-ground==demo-ground all ticks), recovered input
  packets replayed through WalkMove + hull sweep + down-traces, ZERO future truth: n=24, err med **1.6/2.1/3.6hu**
  @7/13/20 (p90 3.6/5.5/10.6; 61 candidates, 37 trace-skipped on ramps/edges). Sim replicates wall behavior like engine.
  Open-space gate (rollout_gate) 2.4/3.9/5.8hu still holds. Cheat-audit: eval AR loop reads TOK only at scoring
  (eval_tx.py:81) + truth POS (line 98); predictions come from net() only. No future leaks into model inputs, ever.
- ORANGE TRACK PARKED (hollow, verified not cheated): orange STV records entities every 3rd tick (tick-step hist 3:108139,
  consecutive pairs 0/109999) → inverse_*/rollout (DT-exact) cannot consume it; rollout_orange = 21 files, 0 rows.
  BSP (52MB, from Brygada fastdl) + soup (32k tris) kept; gap-aware (3xDT) physics = future work, NOT silent. beta4b MGE
  (every-tick) remains the only training data. v34 download adventure (gamebanana API, skial/lazypurple fastdl, Anubis
  walls) archived in DIARY.
- REAL SWEPT BOX 2026-10-13 (`ml/boxtrace.py` own slab code + `ml/build_brushes.py` sidecar + `ml/audit_brushes.py`,
  NO pasted engine text): TF hulls read from leak (stand 24/24/82, duck /62, bottom at origin; ground = full-hull
  trace origin→origin-2; same move order trace→clip→step→bump→ground). Parser phantoms found by demo-motion audit
  (brushes swallowing free-motion positions 5hu+ deep): 31 dropped (giants 6436/6438/6440/6442/6444/6447/6448 + 24
  small). Wall-gate with TRUE box: n=24, err med **1.7/2.1/3.3hu** @7/13/20, p90 3.7/5.6/9.9 (beats 9-ray 1.6/2.1/3.6
  on p90); unstick nudges ~0 (3 win, max 0.5hu = STV rounding only). Replay-from-truth is engine-faithful; the
  residual is the certificate, not a promise.
- USERCMD PARSER 2026-10-14 (`ml/demoin.py`): frame walk VERIFIED on 200k frames
  (all dem_packet, 93-byte header: cmd+tick+democmdinfo_t(76)+seq+seqack+len).
  Message ids from leak common/protocol.h:86-161 (net_Tick=3, svc_PacketEntities=26,
  clc_Move=9). svc_PacketEntities is SELF-DELIMITING (own bit length) so it is
  skippable without sendtables — earlier stale-INVAR skip was what killed the walk.
  EXHAUSTIVE bit-offset scan (no id assumption, nnew 1..15) over 50 frames: the ONLY
  recurring CLC_Move-shaped payload is at bit 48, inside net_Tick's own payload, with
  a constant length=178 across frames — provably a false positive. Combined with the
  engine (hltvclientstate.cpp:244 attaches the recorder to HLTV's SERVER-FACING
  channel; both net_chan.cpp recording hooks are on the incoming server->HLTV path;
  hltvserver.cpp:1957 "MOTODO HLTV must store user commands too") => these demos do NOT
  contain client inputs. NOTE: never claim absence from a broken parser — the first
  version of this conclusion was asserted while the parser was still dying at msgid 40.
- COLLISION ROOT CAUSE 2026-10-14 (`ml/boxtrace.py`): `load_brushes` defaulted to
  `drop=()`, so the 31 verified phantom brushes were LIVE in every collision run.
  Giant 6419 (1286x1300x2547) has the player 2163hu inside on Z -> every tick
  startedsolid -> unstick shoved the hull 1.5hu sideways -> zero forward progress,
  ~3hu/tick injected error. Also 10 infinite-volume (+-1e9) skybox brushes made
  "inside solid" true map-wide. Bisect proof: no-collision 0.18hu@7 vs collision
  21.3hu@7. Fixes: auto-load brush_drop list (cannot be silently ignored again),
  new `brush_solid_mge_training_v8_beta4b.npz` (6528 brushes, drops baked,
  infinities removed), 0.05hu hull vertical shrink, contact epsilon.
  ORACLE (`ml/oracle_tick.py`, true recovered inputs + swept-box collision, 12 wins):
  hit<24hu 100% at EVERY horizon; median 0.19/0.36/0.57/4.05/5.22/5.08/6.77hu at
  7/13/20/33/66/100/132 ticks. 1s=5.2hu, 2s=6.8hu. Tracer selftest PASS.
- COLLISION EVAL HARNESS 2026-10-14 (`ml/eval_collision.py`): scores coast/oracle/model
  in the REAL sim (swept box + ground trace), horizons 7/13/20/33/66/132.
  CRITICAL BUG FOUND AND FIXED: the main loop passed POSITION as velocity
  (`apply_dir(p, p, ...)`), so velocity became ~5600 units/tick and both
  controllers produced identical garbage (0.0% everywhere). Single-tick tests
  passed because they passed `v` correctly — a reminder to always test the
  harness end-to-end, not just its pieces.
  After the fix, on tfmge_2026-09-30__17-14 (60 windows):
    oracle 100.0% at EVERY horizon, median 0.23/0.44/0.73/1.31/2.34/1.31hu
    coast  100/100/0/0/0/0%, median 5.36/15.64/32.09/73.93/191.88/303.69hu
  GATE PASS. On 2026-10-05__16-54 the gate also passes but oracle is weaker
  (3.40hu @7, 14.3% @66) — window selection lands near spawns/rocket-jumps;
  UNRESOLVED, do not quote demo A's numbers as universal.
  Gate compares median error when hit-rates tie (coast is near-optimal at 7).
- INPUT DECODER 2026-10-14 (`ml/recover_keys.py`, `ml/recover_viterbi.py`,
  `ml/diag_candidates.py`): 9 yaw-relative candidates (8 + coast) through the
  real physics. Source-verified fixes vs tf_gamemovement.cpp:1747/2070:
  (a) diagonal wishspeed is 450*sqrt2 clamped after normalize, NOT min(450,max);
  (b) CheckVelocity clamp then BACK-SPEED clamp, in that order; (c) the demo's
  `mspd` column is unreliable (uniq 0/80/110/160/230/240/264/288) AND already
  duck-adjusted — re-applying DUCK_FRAC quartered it and made friction brake
  the player 240->70; maxspeed now comes from CLASS (verified: observed p95
  speed per class == class speed exactly, 1->400 2->300 3->240 5->320 7->300).
  Result: candidates now separate cleanly (best err 4.4 vs 9.6 for coast on a
  saturated tick) but per-tick match is still ~8hu/s median vs a 0.67 floor,
  99.6% of ticks fail. Cause NOT isolated: candidates demand wishspeed ~64 where
  a held key gives 240-450. Ruled out: collision (0.34%), direction (87.5% within
  10deg), class caps, duck scaling, clamp order. Remaining suspects: per-brush
  m_surfaceFriction (BSP has NO LUMP_SURFACE — see ml/bsp_friction.py, so this
  map's friction is unavailable), per-tick m_flMaxSpeed rewrites
  (ChargeMove/StunMove/grapple/forward-pull), or STV positions being
  interpolated rather than true physics state. DO NOT TRAIN ON DECODER OUTPUT.
- CORRECTION 2026-10-14: the earlier "per-tick replay residual 0.0000 hu/s"
  claim was WRONG — `inverse_ground` only assigns `info['res']` in its first
  branch; the capped/boosted branches return early leaving it at its default
  0.0, and 67% of ticks take those branches. Honest number: median 4.0 hu/s,
  67% of ticks fail to reproduce. Never report a metric whose field may be
  unset.
- CORPUS POISON 2026-10-14 (`ml/build_rollout.py`, ALL corpora since v2): `inverse_ground`
  returns a 3-vector wishdir, `inverse_air` a 2-vector; history line `x[k,11:13] = wd`
  threw ValueError on EVERY grounded-moving tick → window dropped. Old corpora =
  99.8% air, ZERO grounded wishes in 10149 rows (measured). Models v2–v5 NEVER SAW
  a ground strafe; coast (air-drift optimal) unbeatable by construction. Fix:
  `ravel()[:2]` + same ±2 intent-smoothing on history wishes as targets.
- TARGET DENOISE 2026-10-14 (`ml/build_cont.py`): raw per-tick STV wishes are white
  noise (lag-1 cos 0.037, 46% flip >90°/tick — STV jitter dominates 1-tick deltas).
  Boxcar ±2 (75ms) → lag-1 cos 0.842, flips 6%, moving frac .94→.66 (jitter
  zeroed). Smoothed-oracle replays to 64.4/53.3 @13/20 (vs coast 52.3/26.6 on the
  same file) — formulation sound, 12/27pp headroom for prediction.
- TRAIN HYGIENE 2026-10-14 (`ml/train_cont.py`): per-row-mean loss (was batch-sum /
  denom → air-heavy batches blew up, train=88 vs val=2.3 print), grad clip 1.0,
  FILE-split val (stride-1 random split leaked), teacher-noise ramp (AR-robustness),
  +8 feats (ego hp, enemy rel pos/vel/present — present 88%, med dist 437hu),
  NFEAT 14→22 (old ckpts obsolete).
- SOLIDITY ROOT CAUSE 2026-10-13 (`ml/mapsoup.py` NEVER-model drop): 23 excluded-model
  brushes were `func_brush` solidity=1 (`BRUSHSOLID_NEVER`, `modelentities.h:46-48`) — non-solid
  by engine definition; now dropped by construction. 7 giant phantoms proven orphans
  (6-plane boxes matching ZERO faces map-wide; 29–21791 deep-swallow frames). Canonical
  `brush_mge_training_v8_beta4b.npz` rebuilt (6569 brushes) + 31 audit drops; wall-gate
  re-certified identical **1.7/2.1/3.3hu** (p90 3.7/5.6/9.9). Still missing: 733 solid static
  props (points only, no meshes) — prop OBBs next.
- VERDICT v5 2026-10-13 (continuous ContAR d64/2L, 30ep, 150631 s1 rows + WC sidecars,
  best val 2.2852): **84.6/61.0/37.1** vs coast 86.5/59.5/34.8 on same s1 eval — FIRST model
  to beat coast @13 (+1.5pp) and @20 (+2.3pp); loses @7 (−1.9pp, coast near-optimal at 7
  ticks: exact-velocity continuation, model wish-noise hurts). Quantization ceiling broken
  at steering horizons. v5 = new champion. Next: eval→world (sweep_box+grounded_box),
  prop OBBs, orange gap-aware.
