# LEDGER — what exists, what runs, what's dead. Read this first after compaction.

## Rules (never violate)
- Training targets are OBSERVED TRUTH only. No self-training loops (no drift).
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
| `ml/train_both27.py` | GRU-27 trainer (ep7: 132-horizon, blame, adaptive, jumpmask) | running; best eval 382.16 |
| `ml/hit_eval_both.py` | verdict vs champion | beats = ship |
| `ml/mapsoup.py` | BSP→collision soup (srctools, engine-exact contents) | beta4b: 141008 tris, 34 brushes excluded, doors tracked |
| `ml/souptrace.py` | wall-stop agreement gate | FAIL under diagnosis (detector artifacts found: respawn snaps) |
| `ml/dbg7.py` | miss forensics (current) | in progress |
| `ml/build_jumpmask.py`, `ml/build_w7.py` | jump mask, weapon cache (ep7 inputs) | built |
| `C:/gitProjects/tf2-movesim/` | separate Rust workspace (formulas+CLI, unsupervised-safe) | 6/6 tests, parity harness |

## ARCHIVED (`ml/_archive/`) — do NOT run, kept for forensics only
| file | why dead |
|---|---|
| `diag_stops.py` | superseded by `dbg7.py` (old detector, teleport hole) |
| `dbg6.py` | one-off; finding recorded (182hu spawn snap → fake -9458hu/s vel) |

## REMOVED (git history only)
- `DemoLoop.*`, `ml/ingest_demoloop.py` — offline-only direction; restore = 1 cmd.

## PENDING
- both27 ship (quant gate ≤0.5pp, C++ 27-dim swap, rebuild, push) after ep7 verdict.
- Rollout dataset (dual rollouts, innovation + λ-blame, 80hu teleport cut!) → tiny early-exit AR transformer → geometry-probe inputs (later experiment).

## Known data facts (do not rediscover)
- STV vel/eye/maxspeed columns FREEZE (parser/extractor artifact); velocity is networked (SDK NETVAR) but STV snapshots don't update it. DERIVE velocity from positions (×66.667). Flags mostly live; row-1 transients suspect.
- Extractor repeats whole ROWS on ticks without updates (m2f~0.9 for duelists; ent 6 fully frozen). Naive dp/row overestimates speed by gap length (580hu/s phantoms, capped-everything). Liveness = pos OR eye OR flags changed; stale repeats DROPPED; v = dp/(gap*DT); pairs need gap==1. Standing-while-aiming survives via eye (true v=0). Single source: `movesim.load_csv_live` (validate/souptrace/dataset all use it — never reimplement).
- Teleport cut is 80hu, NOT 300 (182hu spawn snaps leak through 300). Pipeline builders still use 300hu — tighten on next dataset build.
- DT = 0.015 (66.67Hz), gravity 800, jump 289, accel/air 10, friction 4, stop 100, aircap 30.
