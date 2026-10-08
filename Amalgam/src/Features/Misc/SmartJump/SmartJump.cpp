#include "SmartJump.h"

#include "../../Aimbot/AutoRocketJump/AutoRocketJump.h"
#include "../../Simulation/MovementSimulation/MovementSimulation.h"
#include "../../../Utils/BootLog/BootLog.h"

// sv_stepsize default: the engine auto-steps anything at/below this, no jump needed
#define SJ_STEP 18.f
// extra clearance so the arc doesn't clip the lip on the way over
#define SJ_CLEAR 4.f
#define SJ_MAXTICKS 66
// TF2 stand/duck hull diff: same-tick duck+jump starts the feet this much
// higher (Valve duck-jump doc). Used only as probe margin, never as arc truth.
#define SJ_DUCK_LIFT 20.f

// Exact live fields MoveSim.Initialize mutates. Snapshot before every sim,
// restored after on ALL paths (its early-outs run no restore themselves).
struct LiveBackup_t
{
	Vec3 vVel;
	int iFlags, iState, iMoveType;
	bool bDucked, bDucking, bInDuckJump;
	float flDucktime, flDuckJumpTime;
};

static LiveBackup_t SaveLive(CTFPlayer* pLocal)
{
	return { pLocal->m_vecVelocity(), pLocal->m_fFlags(), pLocal->m_nPlayerState(), pLocal->m_MoveType(),
		pLocal->m_bDucked(), pLocal->m_bDucking(), pLocal->m_bInDuckJump(),
		pLocal->m_flDucktime(), pLocal->m_flDuckJumpTime() };
}

static void LoadLive(CTFPlayer* pLocal, const LiveBackup_t& t)
{
	pLocal->m_vecVelocity() = t.vVel;
	pLocal->m_fFlags() = t.iFlags;
	pLocal->m_nPlayerState() = t.iState;
	pLocal->m_MoveType() = t.iMoveType;
	pLocal->m_bDucked() = t.bDucked;
	pLocal->m_bDucking() = t.bDucking;
	pLocal->m_bInDuckJump() = t.bInDuckJump;
	pLocal->m_flDucktime() = t.flDucktime;
	pLocal->m_flDuckJumpTime() = t.flDuckJumpTime;
}

// RAII around the aimbot's movement sim: restores on every exit path,
// and never sims failed storage (Initialize returns true with m_bFailed
// set on the stuck path: simming that poisons decisions AND skips restore).
struct SimGuard
{
	CTFPlayer* m_pPlayer;
	LiveBackup_t m_tLive;
	MoveStorage m_tStorage = {};
	bool m_bOk = false;

	SimGuard(CTFPlayer* pLocal) : m_pPlayer(pLocal), m_tLive(SaveLive(pLocal))
	{
		m_bOk = F::MoveSim.Initialize(pLocal, m_tStorage, false, false) && !m_tStorage.m_bFailed;
	}
	~SimGuard()
	{
		F::MoveSim.Restore(m_tStorage);
		LoadLive(m_pPlayer, m_tLive);
	}
};

int CSmartJump::SimJumpArc(CTFPlayer* pLocal, CUserCmd* pCmd, int iExtraButtons, float* pArc, int iMaxTicks)
{
	SimGuard g(pLocal);
	if (!g.m_bOk)
		return 0;
	const int iMoveButtons = (pCmd->buttons | iExtraButtons) & ~(IN_ATTACK | IN_ATTACK2);
	auto& mv = g.m_tStorage.m_MoveData;
	mv.m_vecViewAngles = pCmd->viewangles;
	mv.m_flForwardMove = pCmd->forwardmove;
	mv.m_flSideMove = pCmd->sidemove;
	mv.m_flUpMove = pCmd->upmove;
	mv.m_nButtons = iMoveButtons | IN_JUMP;
	mv.m_nOldButtons = iMoveButtons & ~IN_JUMP;
	g.m_tStorage.m_bBunnyHop = false;
	const float flStartZ = mv.m_vecAbsOrigin.z;
	float flLastZ = flStartZ;
	int iApex = 0;
	for (int i = 1; i <= iMaxTicks; i++)
	{
		F::MoveSim.RunTick(g.m_tStorage, false);
		const float flZ = mv.m_vecAbsOrigin.z;
		pArc[i] = flZ - flStartZ;
		if (i > 1 && flZ <= flLastZ)
			break; // apex reached, arc only falls from here
		flLastZ = flZ;
		iApex = i;
	}
	return iApex;
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

void CSmartJump::RunAuto(CTFPlayer* pLocal, CUserCmd* pCmd)
{
	// grounded + intent only. no speed gate: pressing into an obstacle from
	// standstill must fire too (minecraft parity).
	if (!pLocal->m_hGroundEntity())
		return;
	if (!(pCmd->buttons & (IN_FORWARD | IN_BACK | IN_MOVELEFT | IN_MOVERIGHT)))
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

	// Arc first, engine truth: regular-posture measurement sets the probe
	// height (measured apex - 1 + duck lift so duck-clearable tops are seen)
	// and later decides the jump type. Live gravity, real duck behavior.
	float flArc[SJ_MAXTICKS + 1] = {};
	const int iApex = SimJumpArc(pLocal, pCmd, 0, flArc, SJ_MAXTICKS);
	const float flProbeTop = (iApex >= 2 ? flArc[iApex] + SJ_DUCK_LIFT : 91.f) - 1.f;

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
		return; // 8 walls, no jumpable face: give up (doorway etc.)

	// smallest sufficient jump, measured not assumed: regular arc clears
	// anything within itself; the ascending tuck (AirDuck) + ledge grab
	// (RunAir) mounts the rest, so duck is only pressed when the measured
	// regular arc can't reach even with clearance margin.
	bool bRegular = false;
	if (iApex >= 2)
	{
		for (int t = 1; t <= iApex; t++)
		{
			// pressed against the obstacle the arc rises in place and
			// StepMove mounts the ledge once high enough, so any clearing
			// tick counts
			if (flArc[t] >= flNeed + SJ_CLEAR) { bRegular = true; break; }
		}
	}
	if (bRegular || (pCmd->buttons & IN_DUCK))
		pCmd->buttons |= IN_JUMP;
	else
	{
		// measure the duck posture the same way; the engine's real duck-jump
		// (origin lift included) decides, not a constant
		float flDuckArc[SJ_MAXTICKS + 1] = {};
		const int iDuckApex = SimJumpArc(pLocal, pCmd, IN_DUCK, flDuckArc, SJ_MAXTICKS);
		bool bDuck = false;
		for (int t = 1; t <= iDuckApex; t++)
		{
			if (flDuckArc[t] >= flNeed + SJ_CLEAR) { bDuck = true; break; }
		}
		if (!bDuck)
			return; // neither posture clears it
		pCmd->buttons |= IN_JUMP | IN_DUCK;
	}
	ArmGrab(pLocal, flNeed);
	BootLog("SJUMP fire auto");
}

void CSmartJump::RunSmart(CTFPlayer* pLocal, CUserCmd* pCmd)
{
	// grounded + moving with intent only; airborne/standing still need no prediction
	if (!pLocal->m_hGroundEntity())
		return;
	if (!(pCmd->buttons & (IN_FORWARD | IN_BACK | IN_MOVELEFT | IN_MOVERIGHT)))
		return;
	Vec3 vVel = pLocal->m_vecVelocity();
	const float flSpeed = vVel.Length2D();
	if (flSpeed < 50.f)
		return;
	Vec3 vCoastDir = { vVel.x, vVel.y, 0.f };
	vCoastDir.Normalize2D();

	// Arc first, engine truth: sets the coast window (peak + 2) and later
	// the latest-tick firing. Live gravity, no constants.
	float flArc[SJ_MAXTICKS + 1] = {};
	const int iApex = SimJumpArc(pLocal, pCmd, 0, flArc, SJ_MAXTICKS);
	if (iApex < 2)
		return; // jump goes nowhere (ceiling etc.)

	// Coast with the aimbot's movement sim: identical inputs, no jump. Never
	// past the peak: arrival while falling is dead air, the window ends at
	// the apex. SimGuard restores on every path; failed storage is never simmed.
	const int iWindow = iApex < SJ_MAXTICKS ? iApex : SJ_MAXTICKS;
	int iWallTick = -1;
	Vec3 vWallPos = {};
	{
		SimGuard g(pLocal);
		if (!g.m_bOk)
			return;
		auto& mv = g.m_tStorage.m_MoveData;
		mv.m_vecViewAngles = pCmd->viewangles;
		mv.m_flForwardMove = pCmd->forwardmove;
		mv.m_flSideMove = pCmd->sidemove;
		mv.m_flUpMove = pCmd->upmove;
		const int iMoveButtons = pCmd->buttons & ~(IN_JUMP | IN_ATTACK | IN_ATTACK2);
		mv.m_nButtons = iMoveButtons;
		mv.m_nOldButtons = iMoveButtons;
		g.m_tStorage.m_bBunnyHop = false;
		Vec3 vPrev = mv.m_vecAbsOrigin;
		int iStuck = 0;
		for (int i = 1; i <= iWindow; i++)
		{
			F::MoveSim.RunTick(g.m_tStorage, false);
			if (!g.m_tStorage.m_bDirectMove)
				break; // left the ground: no obstacle ahead on this path
			Vec3 vCur = mv.m_vecAbsOrigin;
			// progress along the coast direction, not raw displacement:
			// sliding along a wall still moves, head-on contact doesn't
			const float flAlong = (vCur.x - vPrev.x) * vCoastDir.x + (vCur.y - vPrev.y) * vCoastDir.y;
			vPrev = vCur;
			if (flAlong < 2.f)
			{
				if (++iStuck >= 2) { iWallTick = i; vWallPos = vCur; break; }
			}
			else
				iStuck = 0;
		}
	}
	if (iWallTick < 0)
		return; // clear path through the whole window

	// Wall plane at contact; ramps/steps (engine-handled) are not walls.
	CTraceFilterWorldAndPropsOnly filter = {};
	filter.pSkip = pLocal;
	CGameTrace trace = {};
	SDK::TraceHull(vWallPos - Vec3(vCoastDir.x, vCoastDir.y, 0) * 4.f + Vec3(0, 0, 1),
		vWallPos + Vec3(vCoastDir.x, vCoastDir.y, 0) * 16.f + Vec3(0, 0, 1),
		pLocal->m_vecMins(), pLocal->m_vecMaxs(), pLocal->SolidMask(), &filter, &trace);
	if (!trace.DidHit() || trace.plane.normal.z >= 0.707f)
		return;

	const float flNeed = StepNeed(pLocal, vWallPos, Vec3(-trace.plane.normal.x, -trace.plane.normal.y, 0));
	if (flNeed < 0.f)
		return;

	// Fire exactly when jumping NOW clears but jumping NEXT tick wouldn't:
	// arrival ticks shrink by one per real tick, so next-tick airtime is
	// iArrive - 1. Regular posture only: anything needing duck falls
	// through to auto on arrival.
	const int iArrive = iWallTick - 1; // contact happened ~1 tick before stuck registered
	const bool bClearsNow = iArrive >= 1 && iArrive <= iApex && flArc[iArrive] >= flNeed + SJ_CLEAR;
	const bool bClearsNext = iArrive - 1 >= 1 && iArrive - 1 <= iApex && flArc[iArrive - 1] >= flNeed + SJ_CLEAR;
	if (iArrive < 1 || (bClearsNow && !bClearsNext))
	{
		pCmd->buttons |= IN_JUMP;
		ArmGrab(pLocal, flNeed);
		BootLog("SJUMP fire smart");
	}
}

void CSmartJump::RunAir(CTFPlayer* pLocal, CUserCmd* pCmd)
{
	// Ledge grab only; the tuck lives in CMisc::AirDuck (runs earlier).
	if (Vars::Misc::Movement::SmartJump.Value == Vars::Misc::Movement::SmartJumpEnum::Off)
	{
		m_bGrabArmed = false;
		return;
	}
	if (pLocal->m_hGroundEntity() || F::AutoRocketJump.IsRunning())
	{
		// landed: grab window over. sequencer owns duck mid-jump.
		m_bGrabArmed = false;
		return;
	}

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
