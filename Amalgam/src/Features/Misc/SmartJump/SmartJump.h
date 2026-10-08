#pragma once
#include "../../../SDK/SDK.h"

// CSmartJump: minecraft-style jump assistance, one mode at a time
// (Vars::Misc::Movement::SmartJump: Off / Auto jump / Smart jump).
// - Auto jump (reactive): fires only when already moving into a wall you
//   can't step over. Probes obstacle height (1u in, 72u up, hull down);
//   sheer walls (down-trace startsolid / fraction 0 up top) are ignored so
//   walking into a flat wall does nothing. Picks the smallest sufficient
//   jump by measurement: regular, else duck jump, else nothing.
// - Smart jump (predictive): coasts the current wishdir up to the jump peak
//   and jumps on the latest tick whose arc still clears the obstacle.
class CSmartJump
{
private:
	// Simulates a jump pressed on the first tick with the given buttons.
	// Returns the apex tick (0 = no useful arc); fills pArc[1..apex]
	// with heights above the start origin. Duck posture comes from the
	// buttons, so ctap arcs are measured, not assumed.
	int JumpArc(CTFPlayer* pLocal, CUserCmd* pCmd, int iButtons, float* pArc, int iMaxTicks);
	// Obstacle probe: 1u into the wall along vPush, 72u up, hull-trace
	// down with our AABB. Returns height above feet, or -1 when steppable
	// (<= sv_stepsize), taller than 72u, or open air. Never jumps on -1.
	float StepNeed(CTFPlayer* pLocal, const Vec3& vFeet, const Vec3& vPush);

public:
	void RunAuto(CTFPlayer* pLocal, CUserCmd* pCmd);
	void RunSmart(CTFPlayer* pLocal, CUserCmd* pCmd);
};

ADD_FEATURE(CSmartJump, SmartJump);
