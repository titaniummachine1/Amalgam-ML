#include "SmartJump.h"

#include "../../Aimbot/AutoRocketJump/AutoRocketJump.h"
#include "../../Ticks/Ticks.h"
#include "../../../Utils/BootLog/BootLog.h"

// sv_stepsize default: the engine auto-steps anything at/below this, no jump needed
#define SJ_STEP 18.f
// extra clearance so the jump doesn't clip the lip on the way over
#define SJ_CLEAR 4.f
// Traces only, no movement sims: static game facts. Regular jump apex
// (72u) decides regular jumps; the fixed probe top (duck-jump reach)
// bounds what the slide loop can see.
// TF2 stand/duck hull diff: the unduck mount the grab performs after liftoff
#define SJ_REGULAR_TOP 72.f
#define SJ_DUCK_LIFT 20.f
#define SJ_PROBE_TOP 91.f

// (movement sims removed: traces only, the live player is never written.
// StableTicks below stands down while warp/doubletap rewrites usercmds.)

// (SimJumpArc removed: probe top and jump type use static game facts.)

// Tick-shifting (warp / doubletap fire / recharge / speedhack) rewrites
// usercmds after we run: any jump pressed mid-shift executes at the wrong
// time or dies. The module stands down until ticks are stable again.
static bool StableTicks()
{
	return !F::Ticks.m_bRecharge && !F::Ticks.m_bWarp && !F::Ticks.m_bSpeedhack
		&& !F::Ticks.m_bDoubletap && F::Ticks.m_iShiftedGoal == F::Ticks.m_iShiftedTicks;
}

float CSmartJump::StepNeed(CTFPlayer* pLocal, const Vec3& vFeet, const Vec3& vPush)
{
	const Vec3 vInto = { vFeet.x + vPush.x, vFeet.y + vPush.y, vFeet.z };
	const Vec3 vTop = vInto + Vec3(0, 0, 72);
	CGameTrace trace = {};
	CTraceFilterWorldAndPropsOnly filter = {};
	filter.pSkip = pLocal;
	SDK::TraceHull(vTop, vTop - Vec3(0, 0, 144), pLocal->m_vecMins(), pLocal->m_vecMaxs(), pLocal->SolidMask(), &filter, &trace);
	if (!trace.DidHit() || trace.startsolid || trace.allsolid)
		return -1.f; // sheer wall above / open air: not a jumpable obstacle
	const float flNeed = trace.endpos.z - vFeet.z;
	if (flNeed <= SJ_STEP)
		return -1.f; // steppable, the engine walks over it
	return flNeed;
}

void CSmartJump::ArmGrab(CTFPlayer* pLocal, float flNeed)
{
	m_flGrabTop = pLocal->m_vecOrigin().z + flNeed;
	m_bGrabArmed = true;
}

bool CSmartJump::RunSeq(CTFPlayer* pLocal, CUserCmd* pCmd)
{
	if (!m_iDuckSeq)
		return false;
	if (!StableTicks())
	{
		m_iDuckSeq = 0; // shift started mid-sequence: abort, re-detect when stable
		return false;
	}
	m_iDuckSeq = 0;
	if (!pLocal->m_hGroundEntity())
		return true;
	pCmd->buttons &= ~IN_DUCK;
	pCmd->buttons |= IN_JUMP;
	ArmGrab(pLocal, m_flSeqNeed);
	BootLog("SJUMP fire duck");
	return true;
}

void CSmartJump::RunAuto(CTFPlayer* pLocal, CUserCmd* pCmd)
{
	// Tick-shift in progress (warp / doubletap fire / recharge): usercmds
	// get rewritten downstream, a press now would fire wrong or die.
	if (!StableTicks())
		return;
	// grounded + intent only. no speed gate: pressing into an obstacle from
	// standstill must fire too (minecraft parity).
	if (!pLocal->m_hGroundEntity())
		return;
	if (!(pCmd->buttons & (IN_FORWARD | IN_BACK | IN_MOVELEFT | IN_MOVERIGHT)))
		return;
	// Fire cooldown: grinding an obstacle must not machine-gun jump/duck
	// into it every tick (crouch flicker freezes the body anim). One fire
	// start per 10 ticks max; stairs still climb, walls get calm retries.
	const int iTick = I::GlobalVars->tickcount;
	if (m_iLastFireTick && iTick - m_iLastFireTick < 10)
		return;

	// wishdir from move intent rotated by yaw (NOT velocity: brushing a wall
	// means velocity points along it while intent points into/over it).
	// forwardmove/sidemove are view-relative; this converts to world.
	Vec3 vForward, vRight;
	Math::AngleVectors(pCmd->viewangles, &vForward, &vRight, nullptr);
	vForward.Normalize2D(), vRight.Normalize2D();
	Vec3 vWish = { vForward.x * pCmd->forwardmove + vRight.x * pCmd->sidemove,
		vForward.y * pCmd->forwardmove + vRight.y * pCmd->sidemove, 0.f };
	if (vWish.Length2DSqr() < 0.001f)
		return;
	vWish.Normalize2D();
	const Vec3 vOrigWish = vWish;

	const Vec3 vOrigin = pLocal->m_vecOrigin();
	CTraceFilterWorldAndPropsOnly filter = {};
	filter.pSkip = pLocal;

	// Fixed probe top (traces only): duck-jump reach bounds what the loop
	// below can see; anything taller is given up as unjumpable.
	const float flProbeTop = SJ_PROBE_TOP;

	// Slide-projection loop with OUR CURRENT AABB and nothing else: probe
	// ahead, and when the probe is inside a wall at obstacle height, project
	// the wish parallel to that wall's face and probe again. Corners/curves
	// resolve over iterations (8 max); a step (<= 18u) stops the loop
	// immediately, never slides. Read-only: SDK traces only.
	float flNeed = -1.f;
	for (int i = 0; i < 8; i++)
	{
		// standard probe, 1u out along (possibly slid) wish: top derived
		// from the measured arc, sweep down past the feet to catch ground
		const Vec3 vTop = vOrigin + Vec3(vWish.x, vWish.y, 0) + Vec3(0, 0, flProbeTop);
		CGameTrace trace = {};
		SDK::TraceHull(vTop, vTop - Vec3(0, 0, flProbeTop + 72), pLocal->m_vecMins(), pLocal->m_vecMaxs(), pLocal->SolidMask(), &filter, &trace);
		if (!trace.DidHit())
			return; // open air/gap ahead: nothing to jump over
		if (!trace.startsolid && !trace.allsolid)
		{
			flNeed = trace.endpos.z - vOrigin.z;
			if (flNeed <= SJ_STEP)
				return; // steppable: engine walks over it, no slide needed
			break; // jumpable obstacle found along current wish
		}

		// probe is inside a wall: fall back, get its face at feet level...
		CGameTrace wall = {};
		SDK::TraceHull(vOrigin + Vec3(0, 0, 1), vOrigin + Vec3(vWish.x, vWish.y, 0) * 4.f + Vec3(0, 0, 1),
			pLocal->m_vecMins(), pLocal->m_vecMaxs(), pLocal->SolidMask(), &filter, &wall);
		if (!wall.DidHit() || wall.plane.normal.z >= 0.707f)
			return; // no wall face (pillar edge etc.): don't invent slides
		Vec3 vN = { wall.plane.normal.x, wall.plane.normal.y, 0.f };
		if (vN.Length2DSqr() < 0.001f)
			return;
		vN.Normalize2D();

		// ...and slide the wish parallel to it (projection picks the side
		// closest to intent by construction). wish stays unit length.
		const float flD = vWish.x * vN.x + vWish.y * vN.y;
		vWish.x -= vN.x * flD;
		vWish.y -= vN.y * flD;
		if (vWish.Length2DSqr() < 0.01f)
		{
			// dead head-on: pick the perpendicular closest to original intent
			const Vec3 vP1 = { -vN.y, vN.x, 0.f }, vP2 = { vN.y, -vN.x, 0.f };
			vWish = (vP1.x * vOrigWish.x + vP1.y * vOrigWish.y) > (vP2.x * vOrigWish.x + vP2.y * vOrigWish.y) ? vP1 : vP2;
		}
		vWish.Normalize2D();
	}
	if (flNeed < 0.f)
	{
		m_flLastNeed = -1.f; // open air again: the next obstacle is a new edge
		return; // 8 walls, no jumpable face: give up (doorway etc.)
	}

	// Edge-triggered firing (no machine-gun): a new obstacle, a new height
	// on it, or a patient retry every 45 ticks fires once; the same wall
	// held afterward stays quiet. Stairs still climb: every step is a new
	// height edge.
	const bool bNew = m_flLastNeed < 0.f || fabsf(flNeed - m_flLastNeed) > 4.f;
	const bool bPatient = m_iLastFireTick && iTick - m_iLastFireTick > 45;
	if (!bNew && !bPatient)
		return;

	// Smallest sufficient, static game facts: a regular jump rises 72u, so
	// anything within it goes regular; taller needs the real ctap sequence
	// (full crouch tick, then launch), capped at duck-jump reach.
	if (flNeed + SJ_CLEAR <= SJ_REGULAR_TOP || (pCmd->buttons & IN_DUCK))
	{
		pCmd->buttons |= IN_JUMP;
		ArmGrab(pLocal, flNeed);
		m_flLastNeed = flNeed;
		m_iLastFireTick = iTick;
		BootLog("SJUMP fire auto");
		return;
	}
	// Ctap tick one: full crouch, no jump. Launch follows next tick from
	// ducked state, then tuck + grab mount the rest. Never restart mid-flight.
	if (!m_iDuckSeq && flNeed + SJ_CLEAR <= SJ_REGULAR_TOP + SJ_DUCK_LIFT && flNeed <= SJ_PROBE_TOP)
	{
		pCmd->buttons |= IN_DUCK;
		pCmd->buttons &= ~IN_JUMP;
		m_iDuckSeq = 1;
		m_flSeqNeed = flNeed;
		m_flLastNeed = flNeed;
		m_iLastFireTick = iTick;
		ArmGrab(pLocal, flNeed);
	}
}

// (Smart jump removed: the edge-triggered auto jump covers arrivals too.)

void CSmartJump::RunAir(CTFPlayer* pLocal, CUserCmd* pCmd)
{
	// Ledge grab only; the tuck lives in CMisc::AirDuck (runs earlier).
	if (!Vars::Misc::Movement::AutoJump.Value)
	{
		m_bGrabArmed = false;
		m_iDuckSeq = 0;
		return;
	}
	if (pLocal->m_hGroundEntity() || F::AutoRocketJump.IsRunning())
	{
		// landed: grab window over. sequencer owns duck mid-jump.
		m_bGrabArmed = false;
		return;
	}
	m_iDuckSeq = 0; // airborne: sequence either fired or walked off, clear

	// ascending past the recorded obstacle top with a confirmed surface
	// below? release duck: the unduck stands you onto it instead of falling
	// back. evaluated ascending-only so jumpbug descents are never stripped.
	if (m_bGrabArmed && pLocal->m_vecVelocity().z > 0.f && pLocal->m_vecOrigin().z > m_flGrabTop)
	{
		const Vec3 vPos = pLocal->m_vecOrigin();
		CGameTrace trace = {};
		CTraceFilterWorldAndPropsOnly filter = {};
		filter.pSkip = pLocal;
		SDK::TraceHull(Vec3(vPos.x, vPos.y, m_flGrabTop + 1), Vec3(vPos.x, vPos.y, m_flGrabTop - 10),
			pLocal->m_vecMins(), pLocal->m_vecMaxs(), pLocal->SolidMask(), &filter, &trace);
		if (trace.DidHit() && !trace.startsolid)
		{
			pCmd->buttons &= ~IN_DUCK;
			m_bGrabArmed = false;
		}
	}
	else if (pLocal->m_vecOrigin().z < m_flGrabTop - 100.f)
		m_bGrabArmed = false; // fell far below: stale window, disarm
}
