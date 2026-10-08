#pragma once
#include "../../../SDK/SDK.h"

// CSmartJump: minecraft-style auto jump (Vars::Misc::Movement::AutoJump).
// Intent wishdir, slide-projection probe loop with our live AABB (1u out,
// fixed probe top, sweep down). Sheer walls are slid along (up to 8),
// steps ignored, then the smallest sufficient jump fires once per obstacle:
// regular, else the scripted ctap sequence, else nothing. Traces only, no
// movement sims: the live player is only ever READ (origin, hull), never
// written; the only writes are jump/duck bits in the cmd.
class CSmartJump
{
private:
	// Grab state: obstacle top recorded when a jump fires; while ascending
	// past it, a confirmed surface below releases duck (ledge grab: the
	// unduck stands you onto the surface instead of falling back down).
	bool m_bGrabArmed = false;
	float m_flGrabTop = 0.f;
	// Ctap sequence: 0 idle, 1 = crouched, jump fires next tick. Same-tick
	// DUCK+JUMP from standing never reaches the ducked-liftoff path (hull
	// still transitioning), so the duck jump is scripted like the lua:
	// full crouch tick first, then unduck+jump from ducked state.
	int m_iDuckSeq = 0;
	float m_flSeqNeed = 0.f;
	// Fire cooldown: grinding an obstacle must not machine-gun jump/duck
	// every tick (crouch flicker freezes the body anim). Absolute floor
	// of 10 ticks between fire starts; the need-edge gate in RunAuto
	// decides the rest. Stairs still climb: every step is a new edge.
	int m_iLastFireTick = 0;
	// Edge latch: the need height that caused the last fire. The same wall
	// held afterward stays quiet (no machine-gun); a new height, a new
	// obstacle, or a patient retry re-arms.
	float m_flLastNeed = -1.f;

	// (no movement sims: probe top and jump type use static game facts)
	// Obstacle probe: 1u into the wall along vPush, 72u up, hull-trace
	// down with our AABB. Returns height above feet, or -1 when steppable
	// (<= sv_stepsize), taller than 72u, or open air. Never jumps on -1.
	float StepNeed(CTFPlayer* pLocal, const Vec3& vFeet, const Vec3& vPush);
	// Records the obstacle top for the ledge grab on jump press.
	void ArmGrab(CTFPlayer* pLocal, float flNeed);

public:
	// Ctap tick two (armed by auto): crouched last tick, launch now.
	// Returns true when it handled the tick (skip the normal logic).
	bool RunSeq(CTFPlayer* pLocal, CUserCmd* pCmd);
	void RunAuto(CTFPlayer* pLocal, CUserCmd* pCmd);
	// Airborne companion to auto jump: ledge grab only (the tuck itself
	// lives in CMisc::AirDuck, which runs earlier). Ascending past the
	// recorded obstacle top with a confirmed surface below releases duck,
	// standing you onto the surface instead of falling back down.
	// Never touches duck while the rocket-jump sequencer runs.
	void RunAir(CTFPlayer* pLocal, CUserCmd* pCmd);
};

ADD_FEATURE(CSmartJump, SmartJump);
