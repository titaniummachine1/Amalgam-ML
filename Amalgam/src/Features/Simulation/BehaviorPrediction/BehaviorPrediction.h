#pragma once
#include "../../../SDK/SDK.h"

// ML behavior prediction contract (branch: ml-behavior-prediction).
// Independent of projectile-aim-improvements; combine later via gating.
//
// Model: single-rollout policy (MCTS with 1 simulation). GRU-64, ~15k params, int8 CPU.
//   Policy head: per-tick 2D accel vector (horizontal; vertical stays in physics),
//                rolled out autoregressively (GRU hidden carried, no history re-encode).
//   Value head : TINY (few hundred params, NOT a chess-like evaluator). Sees warmup
//                hidden + rollout endpoint stats only. Outputs 0..1 = P(predicted final
//                position aligns with actual future) at min(sim time limit, intercept T).
//                Trained on hindsight rollout-match, NOT aggregated per-tick formulas.
//   Warmup : feed ~1s (66 ticks) of real MoveData history -> hidden state.
//   Rollout: per tick, PredictStep returns expected acceleration for THIS tick.
// Simulation assumptions (frozen world):
//   ONLY the simulated player and OUR projectile move. Everything else frozen.
//   Projectile enters the player's knowledge iff within history or inside their
//   90 deg FOV during the prediction window (incl. seen-within-1s memory).
// Output contract:
//   m_vAccel      - expected acceleration vector this tick, egocentric yaw frame,
//                   scaled by class max speed (integrate via existing RunTick/ProcessMovement).
//   m_flConfidence - 0..1. Low = do not trust rollout (evasion / unseen regime).
//                    Consumers (AimbotProjectile release logic, splash radius shrink)
//                    gate on this: hold shot / shrink radius when confidence is low.
// Training data: SourceTV demos (server-authoritative trajectories) -> SQLite store
//   (Turso-compatible file), AFK sliced bidirectionally (+-2s still => drop, 1s onset
//   approach always kept so dodge reactions survive), movement-weighted fitness,
//   teleport-split. Metric that matters: multi-horizon POSITION error
//   (100/200/300/500/1000ms), not just accel MSE.
// Memory architecture (ONE ring, not two tiers — verified cheap enough):
//   Per-player STM ring: 2s (132 ticks) x 15 floats (pos3, vel3, flags, eye2,
//   class, accel-pred2, accel-actual2, conf) x fp32 = ~8KB/player, ~250KB for
//   32 players. Tracked for EVERYONE (needed for FOV/threat inputs anyway).
//   FULL prediction (warmup + rollout + value) runs ONLY on the selected target,
//   constantly re-predicted (fresh input for release gating + splash shrink).
//   Everyone else: ring maintenance only, no inference. No PC melting.
//   Reset ring on death / class-change / teleport — never leak stale history.
//   Warmup length under test (22/33/66-tick ablation on frozen champion);
//   2s hardcoded unless ablation says otherwise.
// LTM (long-term, per-ENGAGEMENT not lifetime): fixed-size accumulator (~16 floats,
//   O(1) update per tick) answering "how did we do + how to adjust":
//     v1 observable-only (buildable offline, no predictions needed): strafe bias
//       (L/R tendency), reversal rate, speed/accel variance, airtime frac,
//       engagement length, dodge-reaction count (accel onset within 300ms of
//       projectile entering 90deg FOV).
//     v2 prediction-aware (needs live buffer / scheduled sampling): EMA of own
//       past accel/position errors, calibration (predicted vs actual hit rate),
//       per-horizon reliability. Feeds value head directly; FiLM-conditions policy.
//   Scope rule: LTM covers the current battle with THIS player. Lifetime stats
//   across matches are explicitly out of scope (memory explosion, no transfer).
// Correction history (phase 3): every tick predicts 1-2s ahead (rollout until
//   intercept T or timeout). Each future tick records, from every past prediction:
//   age-k predicted position + predicted accel. At tick T the policy sees, about NOW,
//   the distribution of what past selves predicted (mean/lag-k errors = self-correction
//   signal). Training: sequential segments with prediction buffer (scheduled sampling),
//   loss = per-tick position divergence + final endpoint divergence.
//   (Frozen-champion corr features FAILED as static context: corr1 rollout collapse.
//   Live-buffer only.)
// Label discipline: PURE INPUT acceleration only. Friction/gravity, knockback
// impulses, and all other non-input forces are excluded from the label
// (ground: A_in = dV/dt + vhat*max(|v|,100)*4; air 2D: A_in = dV/dt;
// impulse-split at build: ground |A|>14*MAXV, air |A|>1500 -> drop sample).
// Runtime mirrors with a per-state magnitude clamp (direction preserved).
// Simulation owns physics; policy learns behavior (1-2s context).
// Prediction runs ALWAYS, not only when firing: the would-be target (same
//   targeting logic) is predicted constantly, so hiddens are warm, prediction
//   history exists, and the firing solution is ready the instant the user IS
//   prepared to shoot. Idle prediction is pure profit (rings + cached hiddens).
// Control split (user vs AI — misuse-proofing):
//   AI owns per-shot odds (value head) — the user cannot judge these, so they are
//   never user-tunable, only displayed (live confidence + recent gated hit-rate).
//   User owns ONE aggressiveness preset (Patient/Balanced/Aggressive), mapped to a
//   threshold inside a bounded range (never 0%: floor prevents spam-and-blame).
//   Default = knee of the measured coverage/hit-rate curve. Auto mode picks the
//   threshold from live calibration. Urgency modulates within the preset's band.
//   Rationale: if the user can set 0% and miss everything, they will call it trash.
//   Presets + floors + visible feedback keep every miss explainable.
//   (32 or 100+ — ~8KB each, always affordable). Amalgam owns ALL viability
//   logic (teammates, dormancy, ignore lists, FOV) via its existing selection.
//   This module outputs exactly two things per predicted target: predicted accel
//   (policy) + P(successful hit) (value). The existing menu hit-chance threshold
//   gates autoshoot below it — no new gating logic here. Zero tracking latency:
//   rings are always warm, hidden states cached, confidence decides shoot/hold.
// Multi-target budget (config var, default 1): ranked target list reuses the
//   existing aimbot target selection (FOV/dist/priority). Target #1 predicted
//   every tick; #2/#3 at reduced rate or on beefy PCs every tick.
//   Affordability trick: cache GRU hidden per tracked target across frames and
//   update incrementally (1 step/tick from the ring). Full re-warmup ONLY on
//   target switch / death / teleport / class change. Per-target per-frame cost
//   then ~= rollout only (~2-4ms int8), NOT warmup+rollout. 3 targets ~= 10-15ms.
//   Weak PCs: budget 1 (selected target only). Beefy PCs: budget 2-3, better
//   target-switch responsiveness + pre-computed firing solutions.
// Confidence-gated early prediction: a newly selected target does NOT wait for a
//   full 2s warmup. Ablation shows 0.33s warmup ~= 1s hit-rate, so: start rolling
//   out as soon as the ring has ~22 ticks, and let the VALUE head decide — high
//   confidence = shootable now, low confidence = keep warming up. Warmup length
//   is a latency/accuracy knob driven by confidence, not a fixed gate.
// Prediction count follows aimbot selection: however many targets the aimbot
//   tracks (up to entity limit), rings exist for all of them for free; full
//   prediction runs on the top-N by rank/budget. No separate target list.
// Sim-feedback loop: every rollout on a tracked target appends to ITS prediction
//   history (predicted pos/accel per future tick). When those ticks arrive, the
//   observed-vs-predicted deltas become live correction input (phase-3 scheduled
//   sampling in training; prediction buffer at runtime). Predicting a target
//   literally improves its future predictions — another reason to spend budget
//   on the top-N rather than the top-1 only.
// Hook points (not wired yet):
//   - CMovementSimulation::RunTick: replace/augment m_flAverageYaw input with rollout.
//   - CAimbotProjectile: skip release when rollout confidence < threshold.
//   - Splash radius: shrink proportional to (1 - confidence), replaces LC yaw-jitter heuristic.
struct BehaviorPrediction_t
{
	Vec3 m_vAccel = {};      // policy head: expected acceleration this tick (world units/tick^2, egocentric)
	float m_flConfidence = 1.f; // value head 0..1: this rollout is what's gonna happen. Gate release on it.
};

class CBehaviorPrediction
{
public:
	void Reset(int iEntIndex = -1);
	// Per-tick push of fresh server state for a tracked player (cheap, always on).
	// Ticks are assumed consecutive: caller must Reset() on death/dormancy gaps.
	// Teleport (>300hu step) and class change hard-reset internally.
	void PushState(int iEntIndex, const Vec3& vOrigin, float flEyeYawDeg, bool bGrounded, int iClass);
	// Roll out up to iTicks ahead (cap 132). Writes per-step accel (ego hu/s^2)
	// into pOut, returns steps written. Confidence is 1.0 until the value head
	// lands (phase 2); gate on policy only until then.
	// m_iBudget = how many ranked targets get full prediction per tick.
	int Predict(int iEntIndex, int iTicks, BehaviorPrediction_t* pOut, int iMaxOut);
	int m_iBudget = 1;
	// Validity: movement history is meaningless until walk begins (post-spawn freeze).
	bool IsHistoryValid(int iEntIndex) const;
	// Ticks since walk began (-1 if never). Drives warmup gating + confidence later.
	int TicksSinceWalk(int iEntIndex) const;
	// LTM readout in ml/build_ltm1.py order/scale (8 feats). False if engagement <8 ticks.
	bool GetLTM(int iEntIndex, float ouFeat[8]) const;

private:
	struct TargetState;
	std::unordered_map<int, TargetState> m_mTargets;
	bool m_bWeightsReady = false;
	float m_Wih[3 * 96 * 14] = {};
	float m_Whh[3 * 96 * 96] = {};
	float m_Bih[3 * 96] = {};
	float m_Bhh[3 * 96] = {};
	float m_HeadW[2 * 96] = {};
	float m_HeadB[2] = {};
	void EnsureWeights();
};

ADD_FEATURE(CBehaviorPrediction, BehaviorPrediction);
