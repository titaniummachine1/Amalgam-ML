#pragma once
#include "../../../SDK/SDK.h"

// ML behavior prediction contract (branch: ml-behavior-prediction).
// Independent of projectile-aim-improvements; combine later via gating.
//
// Model: GRU(64) + FiLM static context -> accel + uncertainty heads (int8 C++ forward).
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
	Vec3 m_vAccel = {};      // expected acceleration this tick (world units/tick^2, egocentric)
	float m_flConfidence = 1.f; // 0..1, 1 = fully trusted
};

class CBehaviorPrediction
{
public:
	void Reset(int iEntIndex = -1);
	// TODO(ml): Warmup(hidden, history), PredictStep(static ctx, dynamic feats) -> BehaviorPrediction_t
	// TODO(ml): native int8 GRU forward, no ONNX runtime in-process (15ms budget for 132-step rollout)
};

ADD_FEATURE(CBehaviorPrediction, BehaviorPrediction);
