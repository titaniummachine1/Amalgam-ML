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
Rollout position error hu @100/200/300/500/1000ms, eval, by curvature band
(v3ii = pure-input labels + rollout-loss finetune; gru64_v3ii.pt):
all: coast 4.9/11.6/22.3/47.0/122.3 vs model 4.0/9.0/17.1/35.7/81.5
p95 (sharpest): coast 6.2/13.0/22.3/41.0/90.7 vs model 5.1/10.0/16.7/31.3/69.6
Model wins everywhere; gap widens with horizon + curvature. v2-finetuned (total
accel) tied v3ii on rollout, but v3 labels are the correct formulation (behavior
vs physics separated), so v3 is the line going forward.

Hitbox loss (phase 4, current champion v4 = gru64_v4.pt): AABB 48x48 Chebyshev,
L = ln((err+1)/25) inside (log reward, steep near perfect), L = 4*((err-24)/24)^2
outside (quadrupling penalty). Hit-rate @24hu:
coast 92.2/70.5/50.9/28.0/11.5 vs v3ii 96.7/82.2/58.1/27.0/12.5
vs v4 96.8/82.3/59.0/28.2/12.7 @100/200/300/500/1000ms.
v4 beats v3ii everywhere incl. 500ms. Binary hit labels fall out free for the
value head (inside-AABB at intercept horizon).

Dense-66 soldier (phase 5, gru_soldier_dense66.pt, loss 22.31): per-tick loss over
ALL 66 ticks, time weight w(t)=exp((t-66)/33) — exponential difficulty scaling,
anchored w(66)=1, w(1)=0.14. Measured coast difficulty growth: err=0.47*t^1.42.
Hit-rate: 97.0/82.5/58.2/27.2/12.4 vs coast 92.4/70.7/50.8/28.1/10.9 —
beats coast 4/5, 500ms still -0.9pp. INT8 weights preserve hit-rate.
Nine specialists (warm-started, horizon-weighted): sniper 5.89 leads; spy 57.23
hardest (cloak behavior unobservable). Medic skipped (too few windows).

C++ inference (BehaviorPrediction.cpp, NOT YET WIRED): GRU(14,96) int8-dequant
forward, training-faithful featurization (backward-diff vel, eye fallback,
per-class MAXV). Gate math exact 2.4e-07; int8 hit-rate proven; recurrent fp32
rounding documented, in-game verification required. Value head + RunTick wiring next.

Context group (phase 5): 32 static feats = multi-timescale EMA accel/vel/speed
(tau 0.25/1/4s, hard-reset on death/class/teleport gaps — never hold stale
history) + 3 opponent slots (ego dir, dist, present) + yaw-rate heuristic accel.
Single-step wMSE: raw+ctx 16.46 beats v4-base 16.79; residual+ctx 16.82.
ROLLOUT hit-rate verdict: rawctx 96.8/79.3/45.1/23.8/9.4, resctx
96.8/81.8/47.1/25.1/10.7 — BOTH LOSE to v4 (96.8/82.3/59.0/28.2/12.7).
Lessons: (a) residual-over-heuristic loses everywhere — raw accel target stays;
(b) static-broadcast context overfits single-step and poisons rollout; context
must enter via init-hidden/FiLM or per-step features, not identical copies at
every warmup step. Champion stays gru64_v4.pt.

Labels v3 (current): PURE INPUT accel — TF2 ground friction cancelled
(A_in = dV/dt + vhat*max(|v|,100)*4; air 2D unchanged), so the net learns
behavior (wishdir intent), not physics. Sim owns friction/gravity.
Windows identical to v2 (cache: only Y recomputed).

Note: single-step wMSE can look good while rollout lags coast (exposure bias) —
always judge by autoregressive multi-horizon position error. Finetune with
differentiable rollout loss before comparing.

## Metric definition (hit-rate @24hu — read this before quoting numbers)For each eval sample (a real 1s history + its real future): roll out the policy,
take predicted position at horizon h, compare to where the player ACTUALLY was.
err = max(|dx|,|dz|) (Chebyshev — matches the square box, not a circle).
Hit = err < 24hu, i.e. the predicted point falls inside the 48x48 AABB centered
on the actual position = a projectile aimed there connects (volume ignored).
Hit-rate = fraction of TEST CASES that hit (not fraction of time).
"57.8% @300ms" = in ~58 of 100 real situations, the 300ms prediction lands in-box.
Coast = same procedure with constant-velocity rollout. Deltas vs coast are the score.

## Magnet experiment (8-direction pull — honest negative)
Human input IS 8-modal around yaw-forward (WASD combos): 54.7% of label mass
within +-11.25deg of a mode vs 25% uniform (`ml/modality.py`). But post-hoc soft
pull of output direction toward nearest 45deg mode (s = 0.05/0.15/0.3/0.5,
magnitude untouched, `ml/magnet_sweep.py`) changes hit-rate by +-0.2pp noise
only (s=0: 96.9/82.0/58.0/28.5/14.1). Verdict: policy already sits on the modes;
output shaping adds nothing. A modal prior would have to live IN training
(mixture-density) — noted, not scheduled. Default stays raw.

## Next

- Correction-history as STATIC features: FAILED (honest negative). corr model
  (29 inputs, past 1-step errors from frozen champion as fixed context) scores
  single-step wMSE 19.8 vs 16.8 and collapses in rollout (hit-rate 84/51/13/4/0
  vs v3ii 97/82/58/27/13). Lesson: past errors decorrelate too fast to help as
  frozen context; self-correction needs LIVE prediction buffer during the net's
  OWN rollout (scheduled sampling over segments, phase 3) — `ml/hit_eval.py`
  proves the metric, `build_corr1.py` stays as the feature-group prototype.
- Hit-rate @24hu (half hitbox = guaranteed hit) is the ship metric:
  coast 92/71/51/28/12 vs v3ii 97/82/58/27/13 @100/200/300/500/1000ms.
  v3ii already worth wiring; value head covers 500ms+ uncertainty.
- Train policy on v3 (pure-input) labels.
- Threat/FOV feature group from `.proj.csv` (dodge-keep rule, 90° FOV + 1s memory).
- Tiny value head: P(rollout final pos aligns) at min(T_limit, T_intercept).
- Correction-history inputs (past predictions per future tick) + per-tick/total
  divergence loss. Then int8 C++ forward, wire into RunTick/release gate.
