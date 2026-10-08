#include "SmartJump.h"

#include "../../Simulation/MovementSimulation/MovementSimulation.h"
#include "../../Aimbot/AutoRocketJump/AutoRocketJump.h"

// sv_stepsize default: the engine auto-steps anything at/below this, no jump needed
#define SJ_STEP 18.f
// extra clearance so the arc doesn't clip the lip on the way over
#define SJ_CLEAR 4.f
#define SJ_MAXTICKS 66

int CSmartJump::JumpArc(CTFPlayer* pLocal, CUserCmd* pCmd, int iButtons, float* pArc, int iMaxTicks)
{
	MoveStorage tJump = {};
	if (!F::MoveSim.Initialize(pLocal, tJump, false, false))
		return 0;
	tJump.m_MoveData.m_vecViewAngles = pCmd->viewangles;
	tJump.m_MoveData.m_flForwardMove = pCmd->forwardmove;
	tJump.m_MoveData.m_flSideMove = pCmd->sidemove;
	tJump.m_MoveData.m_flUpMove = pCmd->upmove;
	tJump.m_MoveData.m_nButtons = iButtons | IN_JUMP;
	tJump.m_MoveData.m_nOldButtons = iButtons & ~IN_JUMP;
	tJump.m_bBunnyHop = false;
	const float flStartZ = tJump.m_MoveData.m_vecAbsOrigin.z;
	float flLastZ = flStartZ;
	int iApex = 0;
	for (int i = 1; i <= iMaxTicks; i++)
	{
		F::MoveSim.RunTick(tJump, false);
		const float flZ = tJump.m_MoveData.m_vecAbsOrigin.z;
		pArc[i] = flZ - flStartZ;
		if (i > 1 && flZ <= flLastZ)
			break; // apex reached, arc only falls from here
		flLastZ = flZ;
		iApex = i;
	}
	F::MoveSim.Restore(tJump);
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
	// grounded + moving with intent only; airborne/standing still need no help
	if (!pLocal->m_hGroundEntity())
		return;
	if (!(pCmd->buttons & (IN_FORWARD | IN_BACK | IN_MOVELEFT | IN_MOVERIGHT)))
		return;
	Vec3 vVel = pLocal->m_vecVelocity();
	if (vVel.Length2D() < 50.f)
		return;
	Vec3 vDir = { vVel.x, vVel.y, 0.f };
	vDir.Normalize2D();

	// Single probe: AABB 1u ahead along motion, 72u up, hull sweep down.
	// startsolid up top = sheer wall (don't act weird walking into walls);
	// open air = nothing there; <= step height = walkable. One trace.
	const Vec3 vOrigin = pLocal->m_vecOrigin();
	const Vec3 vTop = vOrigin + Vec3(vDir.x, vDir.y, 0) + Vec3(0, 0, 72);
	CGameTrace trace = {};
	CTraceFilterWorldAndPropsOnly filter = {};
	filter.pSkip = pLocal;
	SDK::TraceHull(vTop, vTop - Vec3(0, 0, 144), pLocal->m_vecMins(), pLocal->m_vecMaxs(), pLocal->SolidMask(), &filter, &trace);
	if (!trace.DidHit() || trace.startsolid || trace.allsolid)
		return;
	const float flNeed = trace.endpos.z - vOrigin.z;
	if (flNeed <= SJ_STEP)
		return;

	const int iMoveButtons = pCmd->buttons & ~(IN_JUMP | IN_ATTACK | IN_ATTACK2);

	// smallest sufficient jump, measured: regular first (pressed against the
	// obstacle the arc rises in place and StepMove mounts the ledge once
	// high enough, so any clearing tick counts)...
	float flArc[SJ_MAXTICKS + 1] = {};
	const int iApex = JumpArc(pLocal, pCmd, iMoveButtons, flArc, SJ_MAXTICKS);
	for (int t = 1; t <= iApex; t++)
	{
		if (flArc[t] >= flNeed + SJ_CLEAR)
		{
			pCmd->buttons |= IN_JUMP;
			ArmGrab(pLocal, flNeed);
			return;
		}
	}

	// ...then duck jump, only when it actually clears and isn't needed otherwise
	if (!(pCmd->buttons & IN_DUCK))
	{
		float flDuckArc[SJ_MAXTICKS + 1] = {};
		const int iDuckApex = JumpArc(pLocal, pCmd, iMoveButtons | IN_DUCK, flDuckArc, SJ_MAXTICKS);
		for (int t = 1; t <= iDuckApex; t++)
		{
			if (flDuckArc[t] >= flNeed + SJ_CLEAR)
			{
				pCmd->buttons |= IN_JUMP | IN_DUCK;
				ArmGrab(pLocal, flNeed);
				return;
			}
		}
	}
}

void CSmartJump::RunSmart(CTFPlayer* pLocal, CUserCmd* pCmd)
{
	// grounded + moving with intent only; airborne/standing still need no prediction
	if (!pLocal->m_hGroundEntity())
		return;
	if (!(pCmd->buttons & (IN_FORWARD | IN_BACK | IN_MOVELEFT | IN_MOVERIGHT)))
		return;
	Vec3 vVel = pLocal->m_vecVelocity();
	if (vVel.Length2D() < 50.f)
		return;

	Vec3 vCoastDir = { vVel.x, vVel.y, 0.f };
	vCoastDir.Normalize2D();

	// coast inputs: current wishdir, trigger buttons stripped (jump is timed, not held)
	const int iMoveButtons = pCmd->buttons & ~(IN_JUMP | IN_ATTACK | IN_ATTACK2);

	// Jump probe first: the coast window is bounded by the jump peak, never further.
	float flArc[SJ_MAXTICKS + 1] = {};
	const int iApex = JumpArc(pLocal, pCmd, iMoveButtons, flArc, SJ_MAXTICKS);
	if (iApex < 2)
		return; // jump goes nowhere (ceiling etc.)

	// Coast sim: identical inputs, no jump. Stops early on airborne
	// (walked off an edge: nothing to jump over).
	int iWallTick = -1;
	Vec3 vWallPos = {};
	Vec3 vWallNormal = {};
	{
		MoveStorage tCoast = {};
		if (!F::MoveSim.Initialize(pLocal, tCoast, false, false))
			return;
		tCoast.m_MoveData.m_vecViewAngles = pCmd->viewangles;
		tCoast.m_MoveData.m_flForwardMove = pCmd->forwardmove;
		tCoast.m_MoveData.m_flSideMove = pCmd->sidemove;
		tCoast.m_MoveData.m_flUpMove = pCmd->upmove;
		tCoast.m_MoveData.m_nButtons = iMoveButtons;
		tCoast.m_MoveData.m_nOldButtons = iMoveButtons;
		tCoast.m_bBunnyHop = false;
		Vec3 vPrev = tCoast.m_MoveData.m_vecAbsOrigin;
		int iStuck = 0;
		const int iWindow = iApex + 2 < SJ_MAXTICKS ? iApex + 2 : SJ_MAXTICKS;
		for (int i = 1; i <= iWindow; i++)
		{
			F::MoveSim.RunTick(tCoast, false);
			if (!tCoast.m_bDirectMove)
				break; // left the ground: no obstacle ahead on this path
			Vec3 vCur = tCoast.m_MoveData.m_vecAbsOrigin;
			const float flAlong = (vCur.x - vPrev.x) * vCoastDir.x + (vCur.y - vPrev.y) * vCoastDir.y;
			vPrev = vCur;
			// progress along the coast direction, not raw displacement:
			// sliding along a wall still moves, head-on contact doesn't
			if (flAlong < 2.f)
			{
				if (++iStuck >= 2) { iWallTick = i; vWallPos = vCur; break; }
			}
			else
				iStuck = 0;
		}
		F::MoveSim.Restore(tCoast);
	}
	if (iWallTick < 0)
		return; // clear path through the whole window

	// Wall plane at contact; ramps/steps (engine-handled) are not walls.
	CGameTrace trace = {};
	CTraceFilterWorldAndPropsOnly filter = {};
	filter.pSkip = pLocal;
	SDK::TraceHull(vWallPos - vCoastDir * 4.f + Vec3(0, 0, 1), vWallPos + vCoastDir * 16.f + Vec3(0, 0, 1),
		pLocal->m_vecMins(), pLocal->m_vecMaxs(), pLocal->SolidMask(), &filter, &trace);
	if (!trace.DidHit() || trace.plane.normal.z >= 0.707f)
		return;
	vWallNormal = trace.plane.normal;

	const float flNeed = StepNeed(pLocal, vWallPos, Vec3(-vWallNormal.x, -vWallNormal.y, 0));
	if (flNeed < 0.f)
		return;

	// Latest-tick rule: first airtime tick clearing need + margin.
	// Waiting one more tick leaves airtime iArrive - 1, which still clears
	// iff iArrive - 1 >= iClear. Otherwise this tick is the last valid one.
	int iClear = -1;
	for (int t = 1; t <= iApex; t++)
	{
		if (flArc[t] >= flNeed + SJ_CLEAR) { iClear = t; break; }
	}
	if (iClear < 0)
		return; // even the apex doesn't clear it
	const int iArrive = iWallTick - 1; // contact happened ~1 tick before stuck registered
	const int iAt = iArrive < iApex ? iArrive : iApex;
	if (iArrive < 1 || (iArrive <= iClear + 1 && flArc[iAt] >= flNeed + SJ_CLEAR))
	{
		pCmd->buttons |= IN_JUMP;
		ArmGrab(pLocal, flNeed);
	}
}

void CSmartJump::RunAir(CTFPlayer* pLocal, CUserCmd* pCmd)
{
	if (Vars::Misc::Movement::SmartJump.Value == Vars::Misc::Movement::SmartJumpEnum::Off
		|| !Vars::Misc::Movement::AirDuck.Value)
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

	const float flVZ = pLocal->m_vecVelocity().z;
	if (flVZ > 0.f)
	{
		// ascending: tuck so the hull clears lips. falling is left alone
		// (jumpbug unduck timing, fast-fall edgebugs stay yours).
		pCmd->buttons |= IN_DUCK;

		// ledge grab: past the obstacle top with a confirmed surface below?
		// releasing duck stands you onto it instead of falling back down.
		if (m_bGrabArmed && pLocal->m_vecOrigin().z > m_flGrabTop)
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
	}
	else if (pLocal->m_vecOrigin().z < m_flGrabTop - 100.f)
		m_bGrabArmed = false; // fell far below: stale window, disarm
}
