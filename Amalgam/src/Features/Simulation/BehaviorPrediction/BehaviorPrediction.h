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
// Correction history (phase 3): every tick predicts 1-2s ahead (rollout until
//   intercept T or timeout). Each future tick records, from every past prediction:
//   age-k predicted position + predicted accel. At tick T the policy sees, about NOW,
//   the distribution of what past selves predicted (mean/lag-k errors = self-correction
//   signal). Training: sequential segments with prediction buffer (scheduled sampling),
//   loss = per-tick position divergence + final endpoint divergence.
// Label discipline: PURE INPUT acceleration only. Friction/gravity and all other
//   non-input forces are cancelled in the label (ground: A_in = dV/dt + vhat*max(|v|,100)*4;
//   air 2D: A_in = dV/dt). Simulation owns physics; policy learns behavior (1-2s context).
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
	// TODO(ml): Warmup(hidden, history), PredictStep(static ctx, dynamic feats) -> BehaviorPrediction_t
	// TODO(ml): native int8 GRU forward, no ONNX runtime in-process (15ms budget for 132-step rollout)
};

ADD_FEATURE(CBehaviorPrediction, BehaviorPrediction);
