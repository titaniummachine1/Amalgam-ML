#pragma once
#include "../../../SDK/SDK.h"

// ML behavior prediction contract (branch: ml-behavior-prediction).
// Independent of projectile-aim-improvements; combine later via gating.
//
// Model: single-rollout policy (MCTS with 1 simulation).
//   Policy head: per-tick accel vector, rolled out autoregressively (GRU hidden carried).
//   Value head : 0..1 confidence that THIS rollout is exactly what's gonna happen,
//                trained on hindsight rollout-match (endpoint within X hu of truth),
//                NOT aggregated from per-tick formulas.
//   Warmup : feed ~1s (66 ticks) of real MoveData history -> hidden state.
//   Rollout: per tick, PredictStep returns expected acceleration for THIS tick.
// Output contract:
//   m_vAccel      - expected acceleration vector this tick, egocentric yaw frame,
//                   scaled by class max speed (integrate via existing RunTick/ProcessMovement).
//   m_flConfidence - 0..1. Low = do not trust rollout (evasion / unseen regime).
//                    Consumers (AimbotProjectile release logic, splash radius shrink)
//                    gate on this: hold shot / shrink radius when confidence is low.
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
