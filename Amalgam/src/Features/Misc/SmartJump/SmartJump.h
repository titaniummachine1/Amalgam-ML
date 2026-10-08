#pragma once
#include "../../../SDK/SDK.h"

// CSmartJump: minecraft-style jump assistance, one mode at a time
// (Vars::Misc::Movement::SmartJump: Off / Auto jump / Smart jump).
// - Auto jump (reactive): intent wishdir, slide-projection probe loop with
//   our live AABB (1u out, top derived from the measured arc, sweep down).
//   Sheer walls are slid along (up to 8), steps ignored, then the smallest
//   sufficient jump fires: regular, else duck, else nothing. All heights are
//   engine-measured per tick (live gravity, real duck behavior), no constants.
// - Smart jump (predictive): coasts the current motion with the aimbot's
//   movement sim up to the measured jump peak, probes the wall found there,
//   and jumps on the latest tick whose measured arc still clears it.
// - Prediction never touches the live player beyond the shared sim's
//   save/sim/restore (same facility the aimbot uses on enemies): SimGuard
//   restores on every path, including init-fail, and re-saves the exact
//   live fields Initialize mutates. Probes are read-only SDK traces.
class CSmartJump
{
private:
	// Grab state: obstacle top recorded when a jump fires; while ascending
	// past it, a confirmed surface below releases duck (ledge grab: the
	// unduck stands you onto the surface instead of falling back down).
	bool m_bGrabArmed = false;
	float m_flGrabTop = 0.f;

	// Engine-measured jump arc with extra buttons held (0 or IN_DUCK).
	// Returns the apex tick (0 = no useful arc); fills pArc[1..apex] with
	// heights above the start origin. Uses the aimbot's movement sim with
	// full restore; live gravity and real duck behavior, no constants.
	int SimJumpArc(CTFPlayer* pLocal, CUserCmd* pCmd, int iExtraButtons, float* pArc, int iMaxTicks);
	// Obstacle probe: 1u into the wall along vPush, 72u up, hull-trace
	// down with our AABB. Returns height above feet, or -1 when steppable
	// (<= sv_stepsize), taller than 72u, or open air. Never jumps on -1.
	float StepNeed(CTFPlayer* pLocal, const Vec3& vFeet, const Vec3& vPush);
	// Records the obstacle top for the ledge grab on jump press.
	void ArmGrab(CTFPlayer* pLocal, float flNeed);

public:
	void RunAuto(CTFPlayer* pLocal, CUserCmd* pCmd);
	void RunSmart(CTFPlayer* pLocal, CUserCmd* pCmd);
	// Airborne companion to the jump modes: ledge grab only (the tuck itself
	// lives in CMisc::AirDuck, which runs earlier). Ascending past the
	// recorded obstacle top with a confirmed surface below releases duck,
	// standing you onto the surface instead of falling back down.
	// Never touches duck while the rocket-jump sequencer runs.
	void RunAir(CTFPlayer* pLocal, CUserCmd* pCmd);
};

ADD_FEATURE(CSmartJump, SmartJump);
