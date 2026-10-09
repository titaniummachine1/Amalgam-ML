#include "AutoTrickstab.h"

#include "../AimbotGlobal/AimbotGlobal.h"
#include "../../Simulation/MovementSimulation/MovementSimulation.h"
#include "../../Ticks/Ticks.h"
#include "../../Misc/Misc.h"
#include "../../Backtrack/Backtrack.h"
#include "../../../SDK/VarsMelee.h"

// Verbatim Lua constants (Auto_Trickstab TF2 table).
#define TRICKSTAB_SCORE_DIST 120.f
#define TRICKSTAB_YAW_W 0.7f
#define TRICKSTAB_DIST_W 0.3f
#define TRICKSTAB_EXTENSION 450.f
#define TRICKSTAB_STAIRSTAB_H 82.f
#define TRICKSTAB_MANUAL_THRESH 100.f
#define TRICKSTAB_ACQUIRE_RANGE 500.f
#define TRICKSTAB_MINK_PAD 1.f
#define TRICKSTAB_DEG2RAD 0.0174532925f
#define TRICKSTAB_RAD2DEG 57.29578f
#define TRICKSTAB_MOVE_SPEED 450.f // full press; the engine clamps to class max internally
#define TRICKSTAB_PLAN_FLOOR 14 // planning/viz horizon while recharging (claim still needs the bank)
#define TRICKSTAB_HIST_DEPTH 4 // walk states: newest + up to 3 older ghosts, first confirmed stab stops the chain

// Yaw from vFrom to vTo, degrees.
static float TrickstabYaw(const Vec3& vFrom, const Vec3& vTo)
{
	return atan2f(vTo.y - vFrom.y, vTo.x - vFrom.x) * TRICKSTAB_RAD2DEG;
}

static float TrickstabNorm(float flAngle)
{
	while (flAngle > 180.f) flAngle -= 360.f;
	while (flAngle < -180.f) flAngle += 360.f;
	return flAngle;
}

// Lua CheckYawDelta, exact: strict inequalities, excludes 0.
static bool TrickstabYawDelta(float flA, float flB)
{
	const float flD = TrickstabNorm(flA - flB);
	return (flD > 0.f && flD < 90.f - 1.f)
		|| (flD < 0.f && flD > -(90.f - 1.f));
}

// Faithful flank walk: each sim owns ONE side (-1 left, +1 right, 0
// center) and chases a MOVING goal recomputed every tick from its live
// pos vs the foe's predicted pose — flank waypoint (approach-relative,
// truly abeam whatever he faces) while frontal, behind-point once abeam.
// Input is coast-lookahead (1 tick of drift, pure math, engine untouched),
// applied through REAL engine ticks with real collision. The coast
// predicts, only real ticks move. Contact breaks the walk (past the exit
// if we started overlapped): walls slide engine-honest, squeezing through
// him is impossible by construction. The replay runs this same formula on
// live state, so sim == execution and left visibly goes left.
static bool SimWalkTick(CTFPlayer* pLocal, int iSide, const Vec3& vBackOff, const Vec3& vFarC, int iTicks, const std::vector<Vec3>& vFoePath, float flFoeYaw, bool bBreakHull, bool bDirect, const Vec3& vOMin, const Vec3& vOMax, const Vec3& vTMin, const Vec3& vTMax, std::vector<Vec3>& vPathOut, Vec3& vEndOut, Vec3& vLastDirOut)
{
	vPathOut.clear();
	vEndOut = {};
	vLastDirOut = {};
	if (!pLocal || iTicks <= 0 || vFoePath.empty())
		return false;
	MoveStorage tStorage = {};
	if (!F::MoveSim.Initialize(pLocal, tStorage, false, false, false))
	{
		F::MoveSim.Restore(tStorage);
		return false;
	}
	auto& mv = tStorage.m_MoveData;
	const int iFoeN = (int)vFoePath.size();
	const float flDT = I::GlobalVars->interval_per_tick;
	Vec3 vFoeFwd = {};
	Math::AngleVectors({ 0.f, flFoeYaw, 0.f }, &vFoeFwd);
	vFoeFwd.Normalize2D();
	const float flFlankW = vOMax.y + vTMax.y + TRICKSTAB_MINK_PAD + 1.f;
	const auto bHullIn = [&](const Vec3& vP, const Vec3& vFoeK) -> bool
	{
		const Vec3 vEMin = vFoeK + (vTMin - vOMax);
		const Vec3 vEMax = vFoeK + (vTMax - vOMin);
		return vP.x > vEMin.x && vP.x < vEMax.x
			&& vP.y > vEMin.y && vP.y < vEMax.y
			&& vP.z > vEMin.z && vP.z < vEMax.z;
	};
	bool bOut = !bHullIn(mv.m_vecAbsOrigin, vFoePath[0]); // overlapped starts must exit first
	bool bOrbited = false; // square orbit: behind -> far corner -> tangential exit
	for (int i = 0; i < iTicks; i++)
	{
		if (tStorage.m_bFailed)
			break;
		const Vec3 vFoeK = (i < iFoeN) ? vFoePath[i] : vFoePath[iFoeN - 1];
		// Moving goal, square orbit: frontal -> wrap the ASSIGNED flank
		// (approach-relative right of the approach dir — truly abeam
		// whatever he faces); abeam/behind -> behind-point; rounded it ->
		// far corner (diagonal: the chord clears the hull) -> tangential
		// exit. Overshoot orbits instead of snapping back. Above/below ->
		// center.
		Vec3 vDest = vFoeK + vBackOff;
		{
			Vec3 vToFoe = vFoeK - mv.m_vecAbsOrigin;
			vToFoe.z = 0.f;
			const float flToLen = vToFoe.Length2D();
			const bool bHigh = fabsf(mv.m_vecAbsOrigin.z - vFoeK.z) >= TRICKSTAB_STAIRSTAB_H;
			bool bFrontal = false;
			if (iSide != 0 && !bHigh && flToLen > 1.f)
			{
				const Vec3 vAppN = vToFoe * (1.f / flToLen);
				// In his face ⟺ he looks at me ⟺ dot ≈ -1 ⟹ wrap my flank.
				// Behind him ⟺ dot ≈ +1 ⟹ finish behind.
				if (vAppN.Dot(vFoeFwd) < -0.3f)
				{
					bFrontal = true;
					const Vec3 vRight = Vec3(vAppN.y, -vAppN.x, 0.f);
					vDest = vFoeK + vRight * ((float)iSide * flFlankW);
				}
			}
			// Ghosts walk ONLY directly (no orbit: the record blocks nothing,
			// routing is unnecessary — straight is cheapest and exact).
			if (!bFrontal && !bDirect)
			{
				if (!bOrbited && (vDest - mv.m_vecAbsOrigin).Length2D() < 20.f)
					bOrbited = true; // rounded the behind-point: keep orbiting
				if (bOrbited)
					vDest = vFoeK + vFarC;
			}
		}
		// Direction, not position: steer AT the point, never stop AT it.
		// Overshoot self-corrects next tick (the goal recomputes live from
		// the new pos, i.e. a new best point every tick). Only contact or
		// engine failure ends the walk — arrival never does.
		Vec3 vToD = vDest - mv.m_vecAbsOrigin;
		vToD.z = 0.f;
		// COAST 1 tick (pure math): friction + gravity on the sim's live
		// velocity, then input from THERE at the offset. Kills drift before
		// it happens — the perfect-input half; the tick below is the
		// fidelity half.
		Vec3 vCoastPos = mv.m_vecAbsOrigin;
		Vec3 vCoastVel = mv.m_vecVelocity;
		{
			const float flSp = vCoastVel.Length2D();
			if (flSp > 0.f)
			{
				const float flDrop = flSp * 4.0f * flDT;
				const float flNew = std::max(flSp - flDrop, 0.f);
				vCoastVel.x *= flNew / flSp;
				vCoastVel.y *= flNew / flSp;
			}
			if (!(pLocal->m_fFlags() & FL_ONGROUND))
				vCoastVel.z -= 800.f * flDT;
			vCoastPos = vCoastPos + vCoastVel * flDT;
		}
		Vec3 vDir = vDest - vCoastPos;
		vDir.z = 0.f;
		float flLen = vDir.Length2D();
		if (flLen < 1.f)
		{
			vDir = vToD;
			flLen = vToD.Length2D();
			if (flLen < 1.f)
			{
				if (vLastDirOut.Length2D() < 0.5f)
					break; // on the point with no heading yet: give up
				vDir = vLastDirOut; // hold heading through the point
				flLen = 1.f;
			}
		}
		vDir = vDir * (1.f / flLen);
		vLastDirOut = vDir;
		const float flYaw = atan2f(vDir.y, vDir.x) * TRICKSTAB_RAD2DEG;
		mv.m_vecViewAngles = { 0.f, flYaw, 0.f };
		mv.m_flForwardMove = TRICKSTAB_MOVE_SPEED;
		mv.m_flSideMove = 0.f;
		// Air mirror of the replay's Directional autostrafe (user setting):
		// grounded, both sides no-op; airborne, both rotate moves by the same
		// formula from the same state. Legit never fires on planted ticks
		// (no mousedx), Off skips both — setting gates sim and replay alike.
		// (Single jump-press edge tick ignored: steady bhop hold is identical.)
		if (Vars::Misc::Movement::AutoStrafe.Value == Vars::Misc::Movement::AutoStrafeEnum::Directional
			&& !tStorage.m_pPlayer->m_hGroundEntity() && mv.m_vecVelocity.Length2D() >= 1.f)
		{
			Vec3 vCurD = Math::VectorAngles(mv.m_vecVelocity);
			const float flDelta = Math::NormalizeAngle(flYaw - vCurD.y);
			if (fabsf(flDelta) <= Vars::Misc::Movement::AutoStrafeMaxDelta.Value)
			{
				const float flTurnScale = Math::RemapVal(Vars::Misc::Movement::AutoStrafeTurnScale.Value, 0.f, 1.f, 0.9f, 1.f);
				const float flRot = Math::Deg2Rad((flDelta > 0.f ? -90.f : 90.f) + flDelta * flTurnScale);
				mv.m_flForwardMove = cosf(flRot) * TRICKSTAB_MOVE_SPEED;
				mv.m_flSideMove = sinf(flRot) * TRICKSTAB_MOVE_SPEED;
			}
		}
		F::MoveSim.RunTick(tStorage, true);
		// Ghost records never break the walk on the record pose (it blocks
		// nothing — his live body is elsewhere and the engine collides that
		// honestly). Present body always breaks: squeeze-through impossible.
		if (bBreakHull)
		{
			if (bHullIn(mv.m_vecAbsOrigin, vFoeK))
			{
				if (bOut)
					break; // walked into him: stop, honestly short
			}
			else
				bOut = true;
		}
	}
	vEndOut = mv.m_vecAbsOrigin;
	vPathOut = tStorage.m_vPath;
	bool bOk = !tStorage.m_bFailed && (int)vPathOut.size() >= 2;
	F::MoveSim.Restore(tStorage);
	return bOk;
}

bool CAutoTrickstab::SimDir(CTFPlayer* pLocal, float flYawDeg, int iTicks, const Vec3& vLeadPos, const Vec3& vOMin, const Vec3& vOMax, const Vec3& vTMin, const Vec3& vTMax, DirResult_t& tOut)
{
	tOut = DirResult_t();
	if (!pLocal || iTicks <= 0)
		return false;
	tOut.vDir = { cosf(flYawDeg * TRICKSTAB_DEG2RAD), sinf(flYawDeg * TRICKSTAB_DEG2RAD), 0.f };

	MoveStorage tStorage = {};
	if (!F::MoveSim.Initialize(pLocal, tStorage, false, false, false))
	{
		F::MoveSim.Restore(tStorage);
		return false;
	}
	auto& mv = tStorage.m_MoveData;
	mv.m_vecViewAngles = { 0.f, flYawDeg, 0.f };
	mv.m_flForwardMove = TRICKSTAB_MOVE_SPEED;
	mv.m_flSideMove = 0.f;
	mv.m_flUpMove = 0.f;
	// Current velocity INHERITED (never zeroed: that would teleport). The
	// engine accel steers toward wishdir from the real state — full press,
	// friction, walls. 100% calculated, never coast.
	// Exact Minkowski contact break (their box absorbs both bodies, live
	// hulls): strict inside = collide. Touching faces = free.
	const Vec3 vEMin = vLeadPos + (vTMin - vOMax);
	const Vec3 vEMax = vLeadPos + (vTMax - vOMin);
	for (int i = 0; i < iTicks; i++)
	{
		if (tStorage.m_bFailed)
			break;
		F::MoveSim.RunTick(tStorage);
		const Vec3& vP = mv.m_vecAbsOrigin;
		if (vP.x > vEMin.x && vP.x < vEMax.x
			&& vP.y > vEMin.y && vP.y < vEMax.y
			&& vP.z > vEMin.z && vP.z < vEMax.z)
			break;
	}
	tOut.vEnd = mv.m_vecAbsOrigin;
	tOut.vPath = tStorage.m_vPath;
	tOut.bValid = !tStorage.m_bFailed && (int)tOut.vPath.size() >= 2;
	F::MoveSim.Restore(tStorage);
	return tOut.bValid;
}

bool CAutoTrickstab::ResolveNeutral(CTFPlayer* pLocal, int iTicks, Vec3& vOut)
{
	vOut = pLocal->m_vecOrigin();
	if (!pLocal || iTicks <= 0)
		return false;
	MoveStorage tStorage = {};
	if (!F::MoveSim.Initialize(pLocal, tStorage, false, false, false))
	{
		F::MoveSim.Restore(tStorage);
		return false;
	}
	// Zero input: init may seed moves from stored records, which is exactly
	// what "no input" is not. View kept (irrelevant with zero moves).
	auto& mv = tStorage.m_MoveData;
	mv.m_flForwardMove = 0.f;
	mv.m_flSideMove = 0.f;
	mv.m_flUpMove = 0.f;
	// Inherit velocity: zero-input coast from real state (Lua exact).
	for (int i = 0; i < iTicks; i++)
	{
		if (tStorage.m_bFailed)
			break;
		F::MoveSim.RunTick(tStorage);
	}
	vOut = mv.m_vecAbsOrigin;
	const bool bOk = !tStorage.m_bFailed;
	F::MoveSim.Restore(tStorage);
	return bOk;
}

float CAutoTrickstab::Wishdir(const Vec3& vResolved, const Vec3& vMyPos, const Vec3& vLeadPos, const Vec3& vCorner)
{
	// Extension direction from live pos (arrival must stay impossible).
	Vec3 vTo = (vLeadPos + vCorner) - vMyPos;
	vTo.z = 0.f;
	const float flToLen = vTo.Length2D();
	const Vec3 vExt = (flToLen > 1.f) ? vTo * (1.f / flToLen) : Vec3(1.f, 0.f, 0.f);
	const Vec3 vD = (vLeadPos + vCorner) + vExt * TRICKSTAB_EXTENSION;
	// Residual from the RESOLVED state, steered from live pos unrotated.
	const Vec3 vR = vD - vResolved;
	if (vR.Length2D() < 1.f)
		return TrickstabYaw(vMyPos, vLeadPos + vCorner);
	return atan2f(vR.y, vR.x) * TRICKSTAB_RAD2DEG;
}

// Pure aimbot-unified gates (CanBackstab dot core + CanHit range/LOS
// decomposition, live weapon range). No prediction, no side effects — the
// firing path stays the authority, this is its scoring twin.
bool CAutoTrickstab::StabAt(const Vec3& vSimEye, const Vec3& vFoePos, const Vec3& vFoeEye, float flFoeYaw, float flOwnerYaw, float flRange, const Vec3& vSwingMins, const Vec3& vSwingMaxs, CTFPlayer* pLocal, CTFPlayer* pTarget)
{
	// Dot core, verbatim thresholds (CanBackstab).
	Vec3 vToTarget = (vFoePos - vSimEye).To2D();
	const float flDist = vToTarget.Normalize();
	if (flDist < 0.2974f) // sqrt(0.0884): inside them, no meaningful angle
		return false;
	const float flCompDist = PLAYER_ORIGIN_COMPRESSION / 2;
	const float flExtra = 2.f * flCompDist / flDist;
	// Perspective-free: the owner is assumed to aim exactly as the real
	// aimbot would (CalcAngle sim eye -> foe, same as Aim()), and ALL THREE
	// CanBackstab dots are tested with it — behind-them, owner-facing
	// (0.5 + extra, the one the old sim skipped), facestab guard.
	// Live view (flOwnerYaw) is NOT consulted: rotation never blocks a sim.
	Vec3 vFoeF;
	Math::AngleVectors({ 0.f, flFoeYaw, 0.f }, &vFoeF);
	vFoeF.Normalize2D();
	const Vec3 vAimAng = Math::CalcAngle(vSimEye, vFoePos); // what Aim() would output here
	Vec3 vOwnerF;
	Math::AngleVectors(vAimAng, &vOwnerF);
	vOwnerF.Normalize2D();
	if (!(vToTarget.Dot(vFoeF) > 0.f + 0.0031f + flExtra)) // behind them?
		return false;
	if (!(vToTarget.Dot(vOwnerF) > 0.5f + flExtra)) // facing them (aimed, real gate)?
		return false;
	if (!(vFoeF.Dot(vOwnerF) > -0.3f + 0.0031f)) // facestab guard (foe fwd vs OUR aim fwd, same as CanBackstab)
		return false;
	(void)flOwnerYaw;
	// Range + LOS (CanHit decomposition, live range/hull).
	const Vec3 vBoxMins = vFoePos + pTarget->m_vecMins();
	const Vec3 vBoxMaxs = vFoePos + pTarget->m_vecMaxs();
	const Vec3 vClosest = {
		vSimEye.x < vBoxMins.x ? vBoxMins.x : (vSimEye.x > vBoxMaxs.x ? vBoxMaxs.x : vSimEye.x),
		vSimEye.y < vBoxMins.y ? vBoxMins.y : (vSimEye.y > vBoxMaxs.y ? vBoxMaxs.y : vSimEye.y),
		vSimEye.z < vBoxMins.z ? vBoxMins.z : (vSimEye.z > vBoxMaxs.z ? vBoxMaxs.z : vSimEye.z) };
	if ((vSimEye - vClosest).Length() > flRange)
		return false;
	const float flLen = (vClosest - vSimEye).Length();
	const Vec3 vDir = (flLen > 0.001f) ? (vClosest - vSimEye) * (1.f / flLen) : Vec3(1.f, 0.f, 0.f);
	CTraceFilterHitscan filter = {};
	filter.pSkip = pLocal;
	CGameTrace trace = {};
	SDK::TraceHull(vSimEye, vSimEye + vDir * flRange, {}, {}, MASK_SOLID, &filter, &trace);
	if (trace.fraction < 1.f && trace.m_pEnt != pTarget)
	{
		SDK::TraceHull(vSimEye, vSimEye + vDir * flRange, vSwingMins, vSwingMaxs, MASK_SOLID, &filter, &trace);
		if (trace.m_pEnt != pTarget)
			return false;
	}
	return true;
}

// 2D segment vs AABB: does the straight walk ENTER the box from outside?
// Starts-inside exempt (face-hug walks sim honestly). Z ignored: the walk
// filter is purely horizontal, touching faces are free.
static bool TrickstabSegBlocked(const Vec3& vA, const Vec3& vB, const Vec3& vMin, const Vec3& vMax)
{
	if (vA.x > vMin.x && vA.x < vMax.x && vA.y > vMin.y && vA.y < vMax.y)
		return false;
	const Vec3 vD = vB - vA;
	float flT0 = 0.f, flT1 = 1.f;
	for (int iAxis = 0; iAxis < 2; iAxis++)
	{
		const float flO = (iAxis == 0) ? vA.x : vA.y;
		const float flD = (iAxis == 0) ? vD.x : vD.y;
		const float flMn = (iAxis == 0) ? vMin.x : vMin.y;
		const float flMx = (iAxis == 0) ? vMax.x : vMax.y;
		if (fabsf(flD) < 1e-8f)
		{
			if (flO <= flMn || flO >= flMx)
				return false;
		}
		else
		{
			float flTA = (flMn - flO) / flD, flTB = (flMx - flO) / flD;
			if (flTA > flTB) { const float flT = flTA; flTA = flTB; flTB = flT; }
			flT0 = std::max(flT0, flTA);
			flT1 = std::min(flT1, flTB);
			if (flT0 > flT1)
				return false;
		}
	}
	return true;
}

// Behind-margin reader: how squarely the hit sits behind the foe
// (~60 deg cone at >= 0.5). Shared by selection and the claim gate.
static float TrickstabBehind(const Vec3& vHitPos, const Vec3& vHitFoe, float flFoeYaw)
{
	Vec3 vF;
	Math::AngleVectors({ 0.f, flFoeYaw, 0.f }, &vF);
	vF.Normalize2D();
	Vec3 vT = (vHitFoe - vHitPos);
	vT.z = 0.f;
	const float flL = vT.Length2D();
	return (flL > 1.f) ? (vT * (1.f / flL)).Dot(vF) : -1.f;
}

float CAutoTrickstab::ScorePath(CTFPlayer* pLocal, CTFPlayer* pTarget, const std::vector<Vec3>& vPath, const std::vector<Vec3>& vFoePath, const Vec3& vLeadPos, const Vec3& vLeadEye, float flFoeYaw, float flOwnerYaw, float flRange, const Vec3& vSwingMins, const Vec3& vSwingMaxs, Vec3& vHitPosOut, Vec3& vHitEyeOut, Vec3& vHitFoeOut, int& iHitTickOut, int& iHitCountOut, std::vector<char>& vHitsOut)
{
	float flBest = -1.f;
	vHitPosOut = {};
	vHitEyeOut = {};
	vHitFoeOut = {};
	iHitTickOut = -1;
	iHitCountOut = 0;
	vHitsOut.assign(vPath.size(), 0);
	const Vec3 vLocalEye = pLocal->m_vecViewOffset();
	// Per-tick enemy eye: lead eye minus lead feet = the stance at record
	// time, carried along the extrapolated feet. Live view offset is NOT
	// used here: a mid-duck live stance would offset every tick by 20u.
	const Vec3 vFoeEyeOff = vLeadEye - vLeadPos;
	// Earliest margin-verified stab wins: first tick passing StabAt AND the
	// 0.5 behind cone stops the scan (shortest warp, cheapest scoring).
	// Grazes never shadow clean stabs — they fail the margin here, and the
	// claim re-gates it anyway. Score is computed for same-tick tie-breaks.
	for (size_t k = 0; k < vPath.size(); k++)
	{
		const Vec3 vEye = vPath[k] + vLocalEye;
		// Dual prediction: test against the enemy's pose THIS tick.
		const Vec3 vFoe = (k < vFoePath.size()) ? vFoePath[k] : vPath[k];
		const Vec3 vFoeEye = vFoe + vFoeEyeOff;
		if (!StabAt(vEye, vFoe, vFoeEye, flFoeYaw, flOwnerYaw, flRange, vSwingMins, vSwingMaxs, pLocal, pTarget))
			continue;
		if (TrickstabBehind(vPath[k], vFoe, flFoeYaw) < 0.5f)
			continue; // side-window graze: not a stab, keep scanning
		vHitsOut[k] = 1;
		iHitCountOut = 1;
		// Lua ranking at the winning tick (tie-breaks only now).
		const float flSpyYaw = TrickstabYaw(vEye, vLeadEye);
		const float flYawDiff = fabsf(TrickstabNorm(flSpyYaw - flFoeYaw));
		const float flYawComp = std::max(0.f, 1.f - flYawDiff / 90.f);
		const float flDistComp = std::max(0.f, 1.f - (vPath[k] - vFoe).Length() / TRICKSTAB_SCORE_DIST);
		flBest = TRICKSTAB_YAW_W * flYawComp + TRICKSTAB_DIST_W * flDistComp;
		vHitPosOut = vPath[k];
		vHitEyeOut = vEye;
		vHitFoeOut = vFoe;
		iHitTickOut = (int)k;
		break;
	}
	return flBest;
}

void CAutoTrickstab::PlantMoves(CTFPlayer* pLocal, CUserCmd* pCmd, const Vec3& vWishDir)
{
	if (!pLocal || !pCmd)
		return;
	// World wishdir -> view-relative moves. View itself is NEVER touched
	// (R3): decompose against the final angles so content stays consistent
	// no matter who wrote the angles before us. AntiAim is frozen during
	// commit ticks, so plant == execution there; during assist its rate
	// bounds any wobble.
	const float flYaw = pCmd->viewangles.y * TRICKSTAB_DEG2RAD;
	const float flC = cosf(flYaw), flS = sinf(flYaw);
	const float flFwd = vWishDir.x * flC + vWishDir.y * flS;
	const float flSide = vWishDir.x * flS - vWishDir.y * flC;
	pCmd->forwardmove = std::clamp(flFwd * TRICKSTAB_MOVE_SPEED, -TRICKSTAB_MOVE_SPEED, TRICKSTAB_MOVE_SPEED);
	pCmd->sidemove = std::clamp(flSide * TRICKSTAB_MOVE_SPEED, -TRICKSTAB_MOVE_SPEED, TRICKSTAB_MOVE_SPEED);
	pCmd->buttons |= IN_FORWARD;
	// R1: enabled strafe is USED. Legit needs mouse delta (absent on planted
	// ticks, no-ops itself); Directional converts our plant to optimal air
	// strafe; grounded it returns early. Jump state left alone (air duck
	// untouched, R2).
	if (Vars::Misc::Movement::AutoStrafe.Value != Vars::Misc::Movement::AutoStrafeEnum::Off)
		F::Misc.AutoStrafe(pLocal, pCmd);
}

void CAutoTrickstab::Run(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CUserCmd* pCmd)
{
	m_bHasBest = false;
	m_bBestHittable = false;
	m_iTotalPoints = 0;
	m_iMinTicks = -1;
	if (!Vars::Aimbot::Melee::AutoTrickstab.Value || !pLocal || !pWeapon || !pCmd)
		return;
	if (pWeapon->GetWeaponID() != TF_WEAPON_KNIFE)
		return;
	if (!pLocal->IsAlive())
		return;
	// Commit latch: claimed target + its validated path, replayed waypoint
	// per warp tick so execution matches the sim that guaranteed the stab.
	// Function-statics: no header change. Cleared on claim and commit end.
	static int s_iCommitEnt = -1;
	static std::vector<Vec3> s_vCommitPath;
	static Vec3 s_vCommitOff = {};
	static int s_iCommitSide = 0;
	static Vec3 s_vFarC = {};
	static bool s_bOrbited = false;
	static bool s_bCommitOld = false; // winner is a ghost record, not the present
	static Vec3 s_vCommitFoe = {}; // foe pos at the winning hit (ghost anchor)
	static Vec3 s_vCommitAnchor = {}; // winner lead pos (ghost anchor or live lead)
	static float s_flCommitYaw = 0.f; // foe yaw of the winning state
	static Vec3 s_vLastDir = { 1.f, 0.f, 0.f }; // hold-heading: direction never dies at a point
	// Candidates: same best-first order as the firing aimbot (distance
	// sort, MaxTargets count), inside the 500u radius. MaxTargets 1 = best
	// target only, 2 = two players, and so on.
	std::vector<CTFPlayer*> vCands;
	for (auto pEntity : H::Entities.GetGroup(EntityEnum::PlayerEnemy))
	{
		if (!pEntity || pEntity->IsDormant())
			continue;
		auto pEnemy = pEntity->As<CTFPlayer>();
		if (!pEnemy || !pEnemy->IsAlive())
			continue;
		if (F::AimbotGlobal.ShouldIgnore(pEntity, pLocal, pWeapon))
			continue;
		vCands.push_back(pEnemy);
	}
	const Vec3 vMyPos0 = pLocal->m_vecOrigin();
	std::sort(vCands.begin(), vCands.end(), [&](CTFPlayer* a, CTFPlayer* b) {
		return vMyPos0.DistToSqr(a->m_vecOrigin()) < vMyPos0.DistToSqr(b->m_vecOrigin());
	});
	if ((int)vCands.size() > Vars::Aimbot::General::MaxTargets.Value)
		vCands.resize((size_t)Vars::Aimbot::General::MaxTargets.Value);
	while (!vCands.empty() && vMyPos0.DistToSqr(vCands.back()->m_vecOrigin()) > TRICKSTAB_ACQUIRE_RANGE * TRICKSTAB_ACQUIRE_RANGE)
		vCands.pop_back();
	// Exec target: latched claim while committing/bonus, else nearest.
	CTFPlayer* pExec = nullptr;
	if ((m_bCommitting || m_iBonusLeft > 0) && s_iCommitEnt >= 0)
	{
		if (auto pE = I::ClientEntityList->GetClientEntity(s_iCommitEnt))
		{
			if (CTFPlayer* pC = pE->As<CTFPlayer>(); pC && !pC->IsDormant() && pC->IsAlive())
				pExec = pC;
		}
		if (!pExec)
		{
			// Target lost mid-commit: stop steering, bank the bonus, let the
			// in-flight warp drain. Never steer a wrong enemy.
			m_bCommitting = false;
			m_iCommitLeft = 0;
			m_iBonusLeft = 3;
			s_iCommitEnt = -1;
			s_vCommitPath.clear();
			s_bCommitOld = false;
			F::Ticks.HoldRecharge(13, m_iClaimTick);
		}
	}
	if (!pExec && !vCands.empty())
		pExec = vCands.front();
	if (!pExec)
	{
		s_iCommitEnt = -1;
		s_vCommitPath.clear();
		s_bCommitOld = false;
		return;
	}
	// Razorback on the exec target: the firing path refuses, so we refuse.
	if (Vars::Aimbot::Melee::IgnoreRazorback.Value)
	{
		CUtlVector<CBaseEntity*> itemList;
		int iBackstabShield = SDK::AttribHookValue(0, "set_blockbackstab_once", pExec, &itemList);
		if (iBackstabShield && itemList.Count())
		{
			if (CBaseEntity* pEntity = itemList.Element(0))
			{
				if (pEntity->ShouldDraw())
					return;
			}
		}
	}
	// Exec lead: newest valid RECORD when available (record-time truth),
	// else live. Drives commit/bonus steering for the latched target.
	Vec3 vLeadX = pExec->m_vecOrigin();
	Vec3 vLeadEyeX = vLeadX + pExec->m_vecViewOffset();
	Vec3 vFoeVelX = pExec->m_vecVelocity();
	float flFoeYawX = H::Entities.GetEyeAngles(pExec->entindex()).y;
	if (std::vector<TickRecord*> vRecsX = {}; F::Backtrack.GetRecords(pExec, vRecsX))
	{
		vRecsX = F::Backtrack.GetValidRecords(vRecsX, nullptr);
		if (!vRecsX.empty() && !vRecsX.front()->m_bInvalid)
		{
			const TickRecord* pRecX = vRecsX.front();
			vLeadX = pRecX->m_vOrigin;
			vLeadEyeX = pRecX->m_vEyePos;
			vFoeVelX = pRecX->m_vVelocity;
			flFoeYawX = pRecX->m_vEyeAngles.y;
		}
	}
	if (auto pNetChanX = I::EngineClient->GetNetChannelInfo())
	{
		const float flLeadTX = (std::max(pNetChanX->GetLatency(FLOW_OUTGOING), 0.f) + std::max(pNetChanX->GetLatency(FLOW_INCOMING), 0.f)) * 0.5f
			+ H::Entities.GetChoke(pExec->entindex()) * I::GlobalVars->interval_per_tick;
		vLeadX = vLeadX + vFoeVelX * flLeadTX;
		vLeadEyeX = vLeadEyeX + vFoeVelX * flLeadTX;
	}
	const int iBank = F::Ticks.m_iShiftedTicks;
	// Drain (fakelag-held ticks ahead of the burst): wallclock the foe walks
	// while our burst waits. Banked ticks excluded — only held-backlog counts.
	const int iDrain = std::max(I::ClientState->chokedcommands - F::Ticks.m_iShiftedTicks, 0);
	// Live weapon range/hull (CanHit decomposition, R10 unified predicate).
	float flRange = SDK::AttribHookValue(pWeapon->GetSwingRange(), "melee_range_multiplier", pWeapon);
	float flHull = SDK::AttribHookValue(18, "melee_bounds_multiplier", pWeapon);
	if (pLocal->m_flModelScale() > 1.0f)
	{
		flRange *= pLocal->m_flModelScale();
		flHull *= pLocal->m_flModelScale();
	}
	const Vec3 vSwingMins = { -flHull, -flHull, -flHull };
	const Vec3 vSwingMaxs = { flHull, flHull, flHull };
	const Vec3 vMyPos = pLocal->m_vecOrigin();
	const float flOwnerYaw = pCmd->viewangles.y;
	// Commit lifecycle + bonus run WITHOUT bank (bonus ticks are live-sent).
	// Everything below this point is sim-bound: bank 0 -> fully idle.
	if (m_bCommitting)
	{
		m_iCommitAge++;
		if (!F::Ticks.m_bWarp || m_iCommitAge > 90)
		{
			m_bCommitting = false;
			m_iCommitLeft = 0;
			m_iBonusLeft = 3;
			// Warp executed: blackout recharge until damage/kill confirms
			// or 13 ticks pass (skipped if damage landed since the claim).
			F::Ticks.HoldRecharge(13, m_iClaimTick);
		}
		else if (m_iCommitLeft > 0)
			m_iCommitLeft--;
	}
	if (!m_bCommitting && m_iBonusLeft > 0)
	{
		const float flBackRadB = (flFoeYawX + 180.f) * TRICKSTAB_DEG2RAD;
		const float flHxb = pLocal->m_vecMaxs().x + pExec->m_vecMaxs().x;
		Vec3 vBack = (vLeadX + Vec3(cosf(flBackRadB), sinf(flBackRadB), 0.f) * flHxb) - vMyPos;
		vBack.z = 0.f;
		if (vBack.Length2D() >= 1.f)
		{
			s_vLastDir = vBack * (1.f / vBack.Length2D());
			PlantMoves(pLocal, pCmd, s_vLastDir);
		}
		else
			PlantMoves(pLocal, pCmd, s_vLastDir); // hold heading through the point
		m_iBonusLeft--;
	}
	// Planning horizon: the bank when funded, a floor when recharging so
	// attempts stay visible (no more "no sims at all") at 0 bank. The CLAIM
	// stays honest: iBank >= m_iMinTicks still gates every warp.
	const int iTicks = std::max(iBank, TRICKSTAB_PLAN_FLOOR);
	// No executable-range early-out: sims draw as attempts out to the
	// radius so targets stay visible; the claim stays honest via
	// iBank >= m_iMinTicks (a warp that cannot arrive never fires).
	// ---- Per-target planning over the aimbot-ordered candidates ----
	// Best valid stab across targets wins the warp; every target draws.
	CTFPlayer* pBestEnt = nullptr;
	float flBestScoreG = -1.f;
	int iBestTickG = INT_MAX;
	bool bMargG = false;
	bool bOldG = false;
	Vec3 vBestHitPosG = {}, vBestHitEyeG = {}, vBestHitFoeG = {};
	Vec3 vBestDestG = {};
	Vec3 vBestOffG = {};
	int iBestSideG = 0;
	Vec3 vBestFarCG = {};
	Vec3 vLeadB = {}, vLeadEyeB = {};
	Vec3 vCornersB[4] = {};
	float flFoeYawB = 0.f;
	bool bStairB = false;
	int iBestMinTG = -1;
	std::vector<Vec3> vBestPathG;
	for (CTFPlayer* pT : vCands)
	{
		CTFPlayer* pTarget = pT;
		// Razorback per target: skip this one only, not the whole tick.
		if (Vars::Aimbot::Melee::IgnoreRazorback.Value)
		{
			CUtlVector<CBaseEntity*> itemListT;
			int iShieldT = SDK::AttribHookValue(0, "set_blockbackstab_once", pTarget, &itemListT);
			if (iShieldT && itemListT.Count())
			{
				if (CBaseEntity* pEntT = itemListT.Element(0))
				{
					if (pEntT->ShouldDraw())
						continue;
				}
			}
		}
		// Lead base per target: newest valid RECORD when available
		// (record-time truth), else live; latency/choke compensated.
		Vec3 vLead = pTarget->m_vecOrigin();
		Vec3 vLeadEye = vLead + pTarget->m_vecViewOffset();
		Vec3 vFoeVel = pTarget->m_vecVelocity();
		float flFoeYaw = H::Entities.GetEyeAngles(pTarget->entindex()).y;
		std::vector<TickRecord*> vRecsT;
		if (F::Backtrack.GetRecords(pTarget, vRecsT))
		{
			vRecsT = F::Backtrack.GetValidRecords(vRecsT, nullptr);
			if (!vRecsT.empty() && !vRecsT.front()->m_bInvalid)
			{
				const TickRecord* pRec = vRecsT.front();
				vLead = pRec->m_vOrigin;
				vLeadEye = pRec->m_vEyePos;
				vFoeVel = pRec->m_vVelocity;
				flFoeYaw = pRec->m_vEyeAngles.y;
			}
		}
		// No latency/choke push on the planning lead: the newest valid record
		// IS the truth and the swing backtracks to records anyway.
		// Extrapolating a "present" from past + stale velocity chased
		// phantoms through turns and ping spikes — planning walks at truth,
		// hitreg rewinds to meet it. (Exec aim below stays live-compensated:
		// the forced view must track the live body for FOV gates.)
	// Walk states, newest first: the compensated lead, then ONE older ghost
	// (static past — the swing backtracks to it). Same foe, not extra
	// targets. Ghosts only run when the newest fails: zero extra cost in
	// the common case, juke-punish when it matters.
	struct RecWalk_t { Vec3 vPos, vEye; float flYaw; };
	RecWalk_t aRW[TRICKSTAB_HIST_DEPTH] = {};
	int nRW = 0;
	aRW[nRW++] = { vLead, vLeadEye, flFoeYaw };
	if (!vRecsT.empty())
	{
		const bool bFrontUsed = !vRecsT.front()->m_bInvalid;
		const float flBaseT = bFrontUsed ? vRecsT.front()->m_flSimTime : I::GlobalVars->curtime;
		const float flGap = 3.f * I::GlobalVars->interval_per_tick;
		const size_t nStart = bFrontUsed ? 1u : 0u;
		for (size_t i = nStart; i < vRecsT.size() && nRW < TRICKSTAB_HIST_DEPTH; i++)
		{
			const TickRecord* pR = vRecsT[i];
			if (!pR || pR->m_bInvalid)
				continue;
			if (flBaseT - pR->m_flSimTime < flGap)
				continue; // too close to the lead: same walk, skip
			// Rewind budget: the swing must still reach the record after the
			// walk AND the drain. Fakelag eats the budget first — heavy choke
			// correctly allows no ghosts (unrewindable), light choke keeps them.
			const float flGhostBudget = 0.12f - (float)iDrain * I::GlobalVars->interval_per_tick;
			if (flGhostBudget <= 0.02f || I::GlobalVars->curtime - pR->m_flSimTime > flGhostBudget)
				continue; // too stale to still rewind at swing time
			aRW[nRW++] = { pR->m_vOrigin, pR->m_vEyePos, pR->m_vEyeAngles.y };
		}
	}
	// Minkowski corners: expanded half-extents + 1u graze padding. Standing:
	// +/-(24+24+1) = +/-49. Exact, duck-aware, rotation-independent.
	const Vec3 vOMin = pLocal->m_vecMins(), vOMax = pLocal->m_vecMaxs();
	const Vec3 vTMin = pTarget->m_vecMins(), vTMax = pTarget->m_vecMaxs();
	const float flHx = vOMax.x + vTMax.x + TRICKSTAB_MINK_PAD;
	const float flHy = vOMax.y + vTMax.y + TRICKSTAB_MINK_PAD;
	const Vec3 vCorners[4] = {
		{ -flHx, flHy, 0.f }, { flHx, flHy, 0.f },
		{ -flHx, -flHy, 0.f }, { flHx, -flHy, 0.f } };
	// Per-state walks: newest lead first, ghost(s) after. Newest-first
	// priority with early-out: a valid present walk ends the target.
	for (int iRW = 0; iRW < nRW; iRW++)
	{
		const bool bHist = (iRW > 0);
		if (bHist)
		{
			vLead = aRW[iRW].vPos;
			vLeadEye = aRW[iRW].vEye;
			flFoeYaw = aRW[iRW].flYaw;
			vFoeVel = {};
			// Ghost prefilters, DISTANCE FIRST (pure math, pre-sim):
			Vec3 vBF = {};
			Math::AngleVectors({ 0.f, flFoeYaw, 0.f }, &vBF);
			vBF.Normalize2D();
			const Vec3 vSpotH = vLead + Vec3(-vBF.x, -vBF.y, 0.f) * (vOMax.x + vTMax.x + TRICKSTAB_MINK_PAD + 1.f);
			// Facing: behind him ⟺ he doesn't face me ⟹ direct walk, SKIP
			// the distance filter. Frontal: the spot is exact-walkable only
			// inside walk radius + stab range + hitbox — else not worth the
			// math, let alone the sim.
			Vec3 vToMeH = vMyPos - vLead;
			vToMeH.z = 0.f;
			const float flToMeHL = vToMeH.Length2D();
			const bool bBehindH = flToMeHL < 1.f || (vToMeH * (1.f / flToMeHL)).Dot(vBF) < -0.3f;
			if (!bBehindH && (vMyPos - vSpotH).Length2D() > (float)iTicks * TRICKSTAB_MOVE_SPEED * I::GlobalVars->interval_per_tick + 120.f + flRange + flHull)
				continue;
			// OCCLUSION SECOND: his LIVE 48-box must not block the straight
			// walk to his own ghost (2D). Starts-inside exempt (face-hug:
			// the sim decides honestly). Blocked ⟹ skip, never sim.
			const Vec3 vLiveH = pTarget->m_vecOrigin();
			const Vec3 vHMin = vLiveH + (vTMin - vOMax);
			const Vec3 vHMax = vLiveH + (vTMax - vOMin);
			if (TrickstabSegBlocked(vMyPos, vSpotH, vHMin, vHMax))
				continue;
		}
	// Behind-point offset: +1 padded arrival (49), same honesty as corners.
	Vec3 vFoeFwdT = {};
	Math::AngleVectors({ 0.f, flFoeYaw, 0.f }, &vFoeFwdT);
	vFoeFwdT.Normalize2D();
	const Vec3 vBackOff = Vec3(-vFoeFwdT.x, -vFoeFwdT.y, 0.f) * (vOMax.x + vTMax.x + TRICKSTAB_MINK_PAD + 1.f);
	// Side picker: warp planning is ALWAYS automatic (closer side to the
	// back). Manual (A/D game input) is assist-only and handled in the
	// assist branch below — it never steers a warp or its commit.
	float flLeft = FLT_MAX, flRight = FLT_MAX;
	for (int i = 0; i < 4; i++)
	{
		const float flTestYaw = TrickstabYaw(vLead + vCorners[i], vLead);
		const float flYawDiff = fabsf(TrickstabNorm(flTestYaw - flFoeYaw));
		if (vCorners[i].y > 0.f)
			flRight = std::min(flRight, flYawDiff);
		else
			flLeft = std::min(flLeft, flYawDiff);
	}
	// Warp planning: automatic only. Manual input is assist-only (below).
	const bool bBestLeft = (flLeft < flRight);
	int iOptimal = -1;
	float flBestYaw = FLT_MAX;
	for (int i = 0; i < 4; i++)
	{
		if ((vCorners[i].y < 0.f) != bBestLeft)
			continue;
		const float flTestYaw = TrickstabYaw(vLead + vCorners[i], vLead);
		const float flYawDiff = fabsf(TrickstabNorm(flTestYaw - flFoeYaw));
		if (flYawDiff < flBestYaw)
		{
			flBestYaw = flYawDiff;
			iOptimal = i;
		}
	}
	// Other side: yaw-optimal corner over there, same as optimal.
	// (First-corner shortcut removed: both sims fly side-optimal.)
	int iOther = -1;
	float flOtherYaw = FLT_MAX;
	for (int i = 0; i < 4; i++)
	{
		if ((vCorners[i].y < 0.f) == bBestLeft) // same side as best: skip
			continue;
		const float flTestYawO = TrickstabYaw(vLead + vCorners[i], vLead);
		const float flYawDiffO = fabsf(TrickstabNorm(flTestYawO - flFoeYaw));
		if (flYawDiffO < flOtherYaw)
		{
			flOtherYaw = flYawDiffO;
			iOther = i;
		}
	}
	const bool bStairstab = fabsf(vMyPos.z - vLead.z) >= TRICKSTAB_STAIRSTAB_H;
	// Dual prediction: enemy extrapolated once (friction + gravity), reused
	// by all paths. Exact on flat ground; full enemy MoveSim is the
	// documented upgrade path if tests show divergence.
	const float flDT = I::GlobalVars->interval_per_tick;
	std::vector<Vec3> vFoePath;
	vFoePath.reserve(iTicks + 1);
	{
		Vec3 vP = vLead, vV = vFoeVel;
		const bool bGround = bHist ? true : ((pTarget->m_fFlags() & FL_ONGROUND) != 0); // ghost is static past: never falls
		// Drain shift (fakelag): the burst executes D ticks late while the
		// foe walks on — start his path where he'll BE at execution, not
		// where the record was. Ghosts stay put (static past). Zero shift
		// unchoked: identical to before.
		for (int iD = 0; iD < iDrain && !bHist; iD++)
		{
			if (bGround)
			{
				const float flSpD = vV.Length2D();
				if (flSpD < 100.f) { vV.x = vV.y = 0.f; }
				else { const float flNewD = std::max(flSpD - flSpD * 4.f * flDT, 0.f); vV.x *= flNewD / flSpD; vV.y *= flNewD / flSpD; }
			}
			else
				vV.z -= 800.f * flDT;
			vP = vP + vV * flDT;
		}
		vFoePath.push_back(vP); // tick 0 = lead at execution (matches vPath[0])
		for (int i = 0; i < iTicks; i++)
		{
			if (bGround)
			{
				const float flSp = vV.Length2D();
				if (flSp < 100.f)
				{
					vV.x = vV.y = 0.f;
				}
				else
				{
					const float flNew = std::max(flSp - flSp * 4.f * flDT, 0.f);
					vV.x *= flNew / flSp;
					vV.y *= flNew / flSp;
				}
			}
			else
				vV.z -= 800.f * flDT;
			vP = vP + vV * flDT;
			vFoePath.push_back(vP);
		}
	}
	// Per-target faithful walks: each sim chases its own fixed offset for
	// the banked ticks and re-steers every tick via coast-lookahead.
	Vec3 vBestDest = {}; // validated stab spot of the winning sim
	Vec3 vBestOff = {}; // fixed offset (center walks; replay fallback)
	int vBestSide = 0; // winning side: -1 left, +1 right, 0 center
	Vec3 vBestFarC = {}; // orbit corner of the winning sim (replay mirrors it)
	struct SimTarget_t { bool bCenter; int iCorner; };
	SimTarget_t aTargets[3] = {};
	int iTargets = 0;
	// Ghost: ONE direct walk at the behind-ghost spot. No flanks, no orbit —
	// the record blocks nothing, routing is unnecessary, straight is cheapest.
	// Present: two flank walks always (optimal, then the other — fastest stab
	// wins); the direct center walk ONLY inside the 90-degree behind cone.
	// No cone, no center: walking at his face is never a plan.
	const float flBackYawT = flFoeYaw + 180.f;
	const bool bBehindCone = fabsf(TrickstabNorm(TrickstabYaw(vLead, vMyPos) - flBackYawT)) <= 90.f;
	if (bHist)
		aTargets[iTargets++] = { true, -1 };
	else if (bStairstab)
		aTargets[iTargets++] = { true, -1 };
	else
	{
		if (iOptimal >= 0 && iTargets < 3)
			aTargets[iTargets++] = { false, iOptimal };
		if (iOther >= 0 && iOther != iOptimal && iTargets < 3)
			aTargets[iTargets++] = { false, iOther };
		if (bBehindCone && iTargets < 3)
			aTargets[iTargets++] = { true, -1 };
	}
	DirResult_t aRes[3] = {};
	int iDirs = 0;
	bool bIsSide[3] = {}; // true = left/right corner sim, false = center (hidden)
	float flBestScore = -1.f;
	int iBestTick = INT_MAX;
	int iBest = -1;
	CTraceFilterHitscan wallFilter = {};
	wallFilter.pSkip = pLocal;
	// Leader pruning: no later sim can beat the global best tick (equal
	// ticks still run — score breaks those ties).
	int iHor = iTicks;
	if (iBestTickG < INT_MAX)
		iHor = std::min(iTicks, iBestTickG + 1);
	// Broadphase: cheap math walk finds the candidate stab tick; real engine
	// ticks confirm only up to candidate+slack. Math models momentum, ground
	// friction + capped accel (air: capped strafe), one-iteration wall slide
	// and hull stop (present) — optimistic where cheap, honest where it
	// matters. Returns earliest tick passing StabAt+margin, or -1. The
	// attempt path is drawn red with zero engine ticks spent.
	const auto MathFirstHit = [&](int iSide, const Vec3& vFarC, int iH, Vec3& vEndOut, std::vector<Vec3>& vMathPathOut) -> int
	{
		vMathPathOut.clear();
		vEndOut = {};
		if (!pLocal || !pTarget || iH <= 0 || vFoePath.empty())
			return -1;
		const float flMDT = I::GlobalVars->interval_per_tick;
		Vec3 vFoeFM = {};
		Math::AngleVectors({ 0.f, flFoeYaw, 0.f }, &vFoeFM);
		vFoeFM.Normalize2D();
		const float flFlWM = vOMax.y + vTMax.y + TRICKSTAB_MINK_PAD + 1.f;
		const float flMaxSp = SDK::MaxSpeed(pLocal);
		const Vec3 vEyeOffL = pLocal->m_vecViewOffset();
		const Vec3 vFoeEyeOffM = vLeadEye - vLead;
		const Vec3 vLiveM = pTarget->m_vecOrigin();
		const Vec3 vHMinL = vLiveM + (vTMin - vOMax);
		const Vec3 vHMaxL = vLiveM + (vTMax - vOMin);
		const bool bAirM = !(pLocal->m_fFlags() & FL_ONGROUND);
		Vec3 vPos = vMyPos;
		Vec3 vVel = pLocal->m_vecVelocity();
		vVel.z = 0.f;
		Vec3 vLastM = {};
		const int iFoeN = (int)vFoePath.size();
		const auto bHullRec = [&](const Vec3& vP, const Vec3& vFoeK) -> bool
		{
			const Vec3 vEMin = vFoeK + (vTMin - vOMax);
			const Vec3 vEMax = vFoeK + (vTMax - vOMin);
			return vP.x > vEMin.x && vP.x < vEMax.x
				&& vP.y > vEMin.y && vP.y < vEMax.y
				&& vP.z > vEMin.z && vP.z < vEMax.z;
		};
		bool bOutM = !bHullRec(vPos, vFoePath[0]);
		bool bOrbM = false;
		vMathPathOut.push_back(vPos);
		CTraceFilterHitscan mFilter = {};
		mFilter.pSkip = pLocal;
		for (int i = 0; i < iH; i++)
		{
			const Vec3 vFoeK = (i < iFoeN) ? vFoePath[i] : vFoePath[iFoeN - 1];
			// Same moving goal, same triggers (frontal / orbit / center).
			Vec3 vDest = vFoeK + vBackOff;
			if (!bHist)
			{
				Vec3 vToF = vFoeK - vPos;
				vToF.z = 0.f;
				const float flToL = vToF.Length2D();
				const bool bHighM = fabsf(vPos.z - vFoeK.z) >= TRICKSTAB_STAIRSTAB_H;
				bool bFrM = false;
				if (iSide != 0 && !bHighM && flToL > 1.f)
				{
					const Vec3 vAppN = vToF * (1.f / flToL);
					if (vAppN.Dot(vFoeFM) < -0.3f)
					{
						bFrM = true;
						vDest = vFoeK + Vec3(vAppN.y, -vAppN.x, 0.f) * ((float)iSide * flFlWM);
					}
				}
				if (!bFrM)
				{
					if (!bOrbM && (vDest - vPos).Length2D() < 20.f)
						bOrbM = true;
					if (bOrbM)
						vDest = vFoeK + vFarC;
				}
			}
			// Coast-lookahead input (same formula as the real walk).
			Vec3 vCPos = vPos + vVel * flMDT;
			Vec3 vDir = vDest - vCPos;
			vDir.z = 0.f;
			float flLen = vDir.Length2D();
			if (flLen < 1.f)
			{
				vDir = vDest - vPos;
				vDir.z = 0.f;
				flLen = vDir.Length2D();
				if (flLen < 1.f)
				{
					if (vLastM.Length2D() < 0.5f)
						break;
					vDir = vLastM;
					flLen = 1.f;
				}
			}
			vDir = vDir * (1.f / flLen);
			vLastM = vDir;
			// Integrate momentum: ground friction + capped accel, air capped.
			if (!bAirM)
			{
				const float flFric = std::max(0.f, 1.f - 4.0f * flMDT);
				vVel.x *= flFric;
				vVel.y *= flFric;
				const float flAlong = vVel.x * vDir.x + vVel.y * vDir.y;
				const float flAdd = std::clamp(flMaxSp - flAlong, 0.f, 4500.f * flMDT);
				vVel.x += vDir.x * flAdd;
				vVel.y += vDir.y * flAdd;
			}
			else
			{
				const float flAlong = vVel.x * vDir.x + vVel.y * vDir.y;
				const float flAdd = std::clamp(30.f - flAlong, 0.f, 30.f);
				vVel.x += vDir.x * flAdd;
				vVel.y += vDir.y * flAdd;
			}
			Vec3 vStep = vVel * flMDT;
			vStep.z = 0.f;
			Vec3 vNew = vPos + vStep;
			// One-iteration wall slide (broadphase approx of engine slide).
			{
				CGameTrace wtr = {};
				SDK::TraceHull(vPos + Vec3(0.f, 0.f, 18.f), vNew + Vec3(0.f, 0.f, 18.f), vOMin, vOMax, MASK_SOLID, &mFilter, &wtr);
				if (wtr.fraction < 1.f)
				{
					Vec3 vN = { wtr.plane.normal.x, wtr.plane.normal.y, 0.f };
					const float flNL = vN.Length2D();
					if (flNL > 1e-6f)
					{
						vN = vN * (1.f / flNL);
						const float flDot = vStep.x * vN.x + vStep.y * vN.y;
						Vec3 vSlide = vStep - vN * flDot;
						vNew = vPos + vStep * wtr.fraction + vSlide * (1.f - wtr.fraction);
					}
					else
						vNew = vPos + vStep * wtr.fraction;
				}
			}
			if (!bHist)
			{
				// Present: hull entry stops the math walk (mirror break).
				if (bHullRec(vNew, vFoeK))
				{
					if (bOutM)
					{
						vMathPathOut.push_back(vNew);
						vEndOut = vNew;
						return -1;
					}
				}
				else
					bOutM = true;
			}
			else if (vNew.x > vHMinL.x && vNew.x < vHMaxL.x && vNew.y > vHMinL.y && vNew.y < vHMaxL.y)
			{
				// Ghost: slide the LIVE body (min-penetration push-out, 2D).
				const float flDx = std::min(vNew.x - vHMinL.x, vHMaxL.x - vNew.x);
				const float flDy = std::min(vNew.y - vHMinL.y, vHMaxL.y - vNew.y);
				if (flDx < flDy)
					vNew.x = (vNew.x - vHMinL.x < vHMaxL.x - vNew.x) ? vHMinL.x : vHMaxL.x;
				else
					vNew.y = (vNew.y - vHMinL.y < vHMaxL.y - vNew.y) ? vHMinL.y : vHMaxL.y;
			}
			vPos = vNew;
			vPos.z = vMyPos.z; // frozen (broadphase)
			vMathPathOut.push_back(vPos);
			// Score this tick: the real predicate (traces only on near-hits).
			const Vec3 vEye = vPos + vEyeOffL;
			const Vec3 vFoeEye = vFoeK + vFoeEyeOffM;
			if (StabAt(vEye, vFoeK, vFoeEye, flFoeYaw, flOwnerYaw, flRange, vSwingMins, vSwingMaxs, pLocal, pTarget)
				&& TrickstabBehind(vPos, vFoeK, flFoeYaw) >= 0.5f)
			{
				vEndOut = vPos;
				return i;
			}
		}
		vEndOut = vPos;
		return -1;
	};
	const auto SimOne = [&](const SimTarget_t& t) -> int
	{
		if (iDirs >= 3)
			return -1;
		const int iSide = t.bCenter ? 0 : ((vCorners[t.iCorner].y < 0.f) ? -1 : +1);
		// Orbit continuation past the behind-point: the far corner (other
		// side's behind corner; center takes the optimal one).
		const Vec3 vFarC = t.bCenter ? vCorners[(iOptimal >= 0) ? iOptimal : 0]
			: ((t.iCorner == iOptimal && iOther >= 0) ? vCorners[iOther] : vCorners[(iOptimal >= 0) ? iOptimal : 0]);
		DirResult_t& r = aRes[iDirs];
		// Two-phase: cheap math finds the candidate tick first. No math hit:
		// draw the attempt red, spend zero engine ticks. Math hit: confirm
		// with real ticks up to candidate+slack; extend honestly on miss.
		// Present math-negative still runs the full real walk (a broadphase
		// miss must never cost a stab); ghost math-negative stays engine-free.
		Vec3 vMathEnd = {};
		std::vector<Vec3> vMathPath;
		const int kStar = MathFirstHit(iSide, vFarC, iHor, vMathEnd, vMathPath);
		if (kStar < 0 && bHist)
		{
			r.bValid = true;
			r.bHittable = false;
			r.flScore = -1.f;
			r.iHitTick = -1;
			r.iHitCount = 0;
			r.vPath = vMathPath;
			r.vEnd = vMathEnd;
			r.vHits.assign(r.vPath.size(), 0);
			const int iIdxN = iDirs++;
			bIsSide[iIdxN] = !t.bCenter;
			return iIdxN;
		}
		const int iCap = (kStar >= 0) ? std::min(iHor, kStar + 3) : iHor;
		bool bWalked = SimWalkTick(pLocal, iSide, vBackOff, vFarC, iCap, vFoePath, flFoeYaw, !bHist, bHist, vOMin, vOMax, vTMin, vTMax, r.vPath, r.vEnd, r.vDir)
			&& !r.vPath.empty();
		if (bWalked)
		{
			r.flScore = ScorePath(pLocal, pTarget, r.vPath, vFoePath, vLead, vLeadEye, flFoeYaw, flOwnerYaw, flRange, vSwingMins, vSwingMaxs, r.vHitPos, r.vHitEye, r.vHitFoe, r.iHitTick, r.iHitCount, r.vHits);
			r.bHittable = r.flScore >= 0.f;
			bWalked = r.bHittable && r.iHitTick <= iCap;
		}
		if (!bWalked)
		{
			// Math optimistic (accel/collision) or engine failed early:
			// honest full-horizon rerun before giving up on the side.
			if (!SimWalkTick(pLocal, iSide, vBackOff, vFarC, iHor, vFoePath, flFoeYaw, !bHist, bHist, vOMin, vOMax, vTMin, vTMax, r.vPath, r.vEnd, r.vDir))
				return -1;
			if (r.vPath.empty())
				return -1;
			r.flScore = ScorePath(pLocal, pTarget, r.vPath, vFoePath, vLead, vLeadEye, flFoeYaw, flOwnerYaw, flRange, vSwingMins, vSwingMaxs, r.vHitPos, r.vHitEye, r.vHitFoe, r.iHitTick, r.iHitCount, r.vHits);
			r.bHittable = r.flScore >= 0.f;
		}
		r.bValid = true;
		// Ghost record: score outside-ghost runs only (segmented). No
		// entry/exit kills — the record pose blocks nothing (his live body
		// is elsewhere; the engine collides that honestly) — but
		// inside-ghost ticks score degenerate angles, so each outside run
		// scores separately and the best absolute tick wins.
		if (bHist && r.bHittable && !vFoePath.empty() && !r.vPath.empty())
		{
			const size_t nN = r.vPath.size();
			float flGB = -1.f;
			Vec3 vGHP = {}, vGHE = {}, vGHF = {};
			int iGHT = -1, iGHC = 0;
			std::vector<char> vGH(nN, 0);
			size_t nA = 0;
			bool bRun = false;
			for (size_t k = 0; k <= nN; k++)
			{
				bool bOutK = false;
				if (k < nN)
				{
					const Vec3 vFoeK = (k < vFoePath.size()) ? vFoePath[k] : vFoePath.back();
					const Vec3 vEMin = vFoeK + (vTMin - vOMax);
					const Vec3 vEMax = vFoeK + (vTMax - vOMin);
					const Vec3& vP = r.vPath[k];
					bOutK = !(vP.x > vEMin.x && vP.x < vEMax.x
						&& vP.y > vEMin.y && vP.y < vEMax.y
						&& vP.z > vEMin.z && vP.z < vEMax.z);
					if (bOutK && !bRun) { nA = k; bRun = true; }
				}
				if ((k >= nN || !bOutK) && bRun)
				{
					bRun = false;
					if (k > nA + 1 && nA < vFoePath.size())
					{
						std::vector<Vec3> vPT(r.vPath.begin() + nA, r.vPath.begin() + k);
						std::vector<Vec3> vFT(vFoePath.begin() + nA, vFoePath.begin() + std::min(k, vFoePath.size()));
						Vec3 vHP = {}, vHE = {}, vHF = {};
						int iHT = -1, iHC = 0;
						std::vector<char> vHH;
						const float flS = ScorePath(pLocal, pTarget, vPT, vFT, vLead, vLeadEye, flFoeYaw, flOwnerYaw, flRange, vSwingMins, vSwingMaxs, vHP, vHE, vHF, iHT, iHC, vHH);
						if (flS >= 0.f)
						{
							iGHC += iHC;
							if (flS > flGB + 1e-6f || (fabsf(flS - flGB) <= 1e-6f && (int)nA + iHT < iGHT))
							{
								flGB = flS;
								vGHP = vHP; vGHE = vHE; vGHF = vHF;
								iGHT = (int)nA + iHT;
							}
							for (size_t j = 0; j < vHH.size() && nA + j < nN; j++)
							{
								if (vHH[j]) vGH[nA + j] = 1;
							}
						}
					}
				}
			}
			r.flScore = flGB;
			r.bHittable = flGB >= 0.f;
			r.vHitPos = vGHP; r.vHitEye = vGHE; r.vHitFoe = vGHF;
			r.iHitTick = iGHT; r.iHitCount = iGHC; r.vHits = vGH;
		}
		// Phase guard, present body only (ghosts scored above): no
		// through-body hit ever scores. Bare hull, per tick:
		//  - start OUTSIDE + ENTER later: squeeze gaps that don't exist and
		//    straight-through shortcuts. Only the honest prefix counts.
		//  - start INSIDE (face-hug/wedged): inside ticks can never stab —
		//    live bodies would have pushed apart. Only ticks AFTER first
		//    exit count: walk out his back and it's green, never exit and
		//    it's dead. A green sim is always executable, full stop.
		if (!bHist && r.bHittable && !vFoePath.empty() && !r.vPath.empty())
		{
			const size_t nN = r.vPath.size();
			std::vector<char> vIn(nN, 0);
			for (size_t k = 0; k < nN; k++)
			{
				const Vec3 vFoeK = (k < vFoePath.size()) ? vFoePath[k] : vFoePath.back();
				const Vec3 vEMin = vFoeK + (vTMin - vOMax);
				const Vec3 vEMax = vFoeK + (vTMax - vOMin);
				const Vec3& vP = r.vPath[k];
				vIn[k] = (vP.x > vEMin.x && vP.x < vEMax.x
					&& vP.y > vEMin.y && vP.y < vEMax.y
					&& vP.z > vEMin.z && vP.z < vEMax.z) ? 1 : 0;
			}
			if (vIn[0])
			{
				size_t nExit = nN;
				for (size_t k = 1; k < nN; k++)
				{
					if (!vIn[k]) { nExit = k; break; }
				}
				if (nExit >= nN || nExit >= vFoePath.size())
				{
					r.bHittable = false;
					r.flScore = -1.f;
					r.iHitCount = 0;
					r.iHitTick = -1;
					r.vHits.assign(nN, 0);
				}
				else
				{
					std::vector<Vec3> vPT(r.vPath.begin() + nExit, r.vPath.end());
					std::vector<Vec3> vFT(vFoePath.begin() + nExit, vFoePath.end());
					r.flScore = ScorePath(pLocal, pTarget, vPT, vFT, vLead, vLeadEye, flFoeYaw, flOwnerYaw, flRange, vSwingMins, vSwingMaxs, r.vHitPos, r.vHitEye, r.vHitFoe, r.iHitTick, r.iHitCount, r.vHits);
					r.bHittable = r.flScore >= 0.f;
					if (r.bHittable)
					{
						r.iHitTick += (int)nExit; // suffix-relative -> absolute (warp length)
						std::vector<char> vFull(nN, 0);
						for (size_t j = 0; j < r.vHits.size() && nExit + j < nN; j++)
							vFull[nExit + j] = r.vHits[j];
						r.vHits = vFull;
					}
				}
			}
			else
			{
				int iEntry = -1;
				for (size_t k = 1; k < nN; k++)
				{
					if (vIn[k]) { iEntry = (int)k; break; }
				}
				if (iEntry > 0)
				{
					if (iEntry < 2)
					{
						r.bHittable = false;
						r.flScore = -1.f;
						r.iHitCount = 0;
						r.iHitTick = -1;
						r.vHits.assign(nN, 0);
					}
					else
					{
						std::vector<Vec3> vPT(r.vPath.begin(), r.vPath.begin() + iEntry);
						std::vector<Vec3> vFT(vFoePath.begin(), vFoePath.begin() + std::min((size_t)iEntry, vFoePath.size()));
						r.flScore = ScorePath(pLocal, pTarget, vPT, vFT, vLead, vLeadEye, flFoeYaw, flOwnerYaw, flRange, vSwingMins, vSwingMaxs, r.vHitPos, r.vHitEye, r.vHitFoe, r.iHitTick, r.iHitCount, r.vHits);
						r.bHittable = r.flScore >= 0.f;
					}
				}
			}
		}
		m_iTotalPoints += r.iHitCount;
		CGameTrace wallTrace = {};
		SDK::TraceHull(vMyPos + Vec3(0.f, 0.f, 18.f), r.vEnd + Vec3(0.f, 0.f, 18.f), {}, {}, MASK_SOLID, &wallFilter, &wallTrace);
		r.bHitWall = wallTrace.fraction < 0.8f && !(wallTrace.m_pEnt && wallTrace.m_pEnt->IsPlayer());
		const int iIdx = iDirs++;
		bIsSide[iIdx] = !t.bCenter;
		// Fastest stab wins the side: earliest hit tick (fewest warp ticks),
		// ties broken by score. Both sides always fully run first.
		if (r.bValid && r.bHittable && (r.iHitTick < iBestTick || (r.iHitTick == iBestTick && r.flScore > flBestScore + 1e-6f)))
		{
			flBestScore = r.flScore;
			iBestTick = r.iHitTick;
			iBest = iIdx;
			vBestDest = r.vHitPos; // the validated stab spot, not an extension
			vBestOff = vBackOff; // center fallback for the replay
			vBestSide = iSide; // replay runs this side's moving goal
			vBestFarC = vFarC; // replay mirrors this orbit corner
		}
		return iIdx;
	};
	int iOptimalIdx = -1;
	for (int i = 0; i < iTargets; i++)
		iOptimalIdx = SimOne(aTargets[i]);
	(void)iOptimalIdx;
	// Cross-target snapshot: margin-passing sims beat grazes, then the
	// fastest stab wins (fewest warp ticks), ties by score.
	if (iBest >= 0)
	{
		const bool bMargT = TrickstabBehind(aRes[iBest].vHitPos, aRes[iBest].vHitFoe, flFoeYaw) >= 0.5f;
		if ((bMargT && !bMargG) || (bMargT == bMargG && (iBestTick < iBestTickG || (iBestTick == iBestTickG && flBestScore > flBestScoreG + 1e-6f))))
		{
			bMargG = bMargT;
			bOldG = bHist;
			flBestScoreG = flBestScore;
			iBestTickG = iBestTick;
			pBestEnt = pTarget;
			vBestHitPosG = aRes[iBest].vHitPos;
			vBestHitEyeG = aRes[iBest].vHitEye;
			vBestHitFoeG = aRes[iBest].vHitFoe;
			vBestDestG = vBestDest;
			vBestOffG = vBestOff;
			iBestSideG = vBestSide;
			vBestFarCG = vBestFarC;
			iBestMinTG = aRes[iBest].iHitTick + 1;
			vBestPathG = aRes[iBest].vPath;
			vLeadB = vLead;
			vLeadEyeB = vLeadEye;
			for (int i = 0; i < 4; i++)
				vCornersB[i] = vCorners[i];
			flFoeYawB = flFoeYaw;
			bStairB = bStairstab;
		}
	}
	// Draw this target's walks (flanks, plus center inside the behind cone),
	// this tick only: one path plus end marker each. Green = stab found,
	// reddish = none (walks that bump him land here — red, never warp).
	if (Vars::Aimbot::Melee::TrickstabDraw.Value)
	{
		const float flUntilT = I::GlobalVars->curtime;
		for (int i = 0; i < iDirs; i++)
		{
			if (aRes[i].vPath.empty())
				continue;
			Color_t tColT = !aRes[i].bValid ? Color_t(255, 255, 255, 55)
				: aRes[i].bHittable ? Color_t(0, 255, 0) : Color_t(255, 70, 70);
			for (size_t k = 0; k + 1 < aRes[i].vPath.size(); k++)
				G::LineStorage.emplace_back(std::pair<Vec3, Vec3>(aRes[i].vPath[k], aRes[i].vPath[k + 1]), flUntilT, tColT, true);
			const Vec3 vMarkT = aRes[i].bHittable ? aRes[i].vHitPos : aRes[i].vEnd;
			G::LineStorage.emplace_back(std::pair<Vec3, Vec3>(vMarkT, vMarkT + Vec3(0, 0, 8)), flUntilT, tColT, true);
		}
	}
	// Newest-first priority with ghost chain: the first CONFIRMED stab
	// (newest or ghost) ends the target — later states never run. A failed
	// state (math-negative, confirm-miss) falls through to the next ghost.
	// Bounded cost, same foe, no remaining ticks checked after success.
	if (iBest >= 0)
		break;
	} // end rec-state loop
	} // end candidate loop
	// Members from the cross-target snapshot (assist + claim read these).
	if (pBestEnt)
	{
		m_bHasBest = true;
		Vec3 vDirBest = vBestDestG - vMyPos;
		vDirBest.z = 0.f;
		float flLenBest = vDirBest.Length2D();
		m_vBestDir = (flLenBest >= 1.f) ? vDirBest * (1.f / flLenBest) : Vec3(1.f, 0.f, 0.f);
		m_bBestHittable = true;
		m_iMinTicks = iBestMinTG;
	}
	else
	{
		// No scored winner anywhere: assist may still position toward the
		// exec target, but nothing is hittable and nothing claims.
		m_bHasBest = (m_iTotalPoints > 0);
		m_bBestHittable = false;
		m_iMinTicks = -1;
		vBestDestG = vLeadX;
	}
	// Assist positions OUTSIDE warp toward the snapshot best; manual (A/D)
	// picks the assist side here and ONLY here. The warp below replays the
	// validated path and never reads manual input.
	const bool bAssist = Vars::Aimbot::Melee::TrickstabAssist.Value;
	if (m_bCommitting)
	{
		// Aim FIRST: PlantMoves decomposes the world wishdir against the
		// shipped viewangles, so the basis must already be final. Planting
		// against stale angles and aiming after rotated every warp tick by
		// the per-tick view delta — the drift that ended warps out of the
		// backstab angle stabbing air.
		{
			// Ghost winner: aim the record eye (the swing backtracks to it).
			// Present winner: live lead eye led by the drain (fakelag) so the
			// sent angles meet the foe at execution, not at planning.
			const Vec3 vAimEye = s_bCommitOld ? (s_vCommitFoe + pExec->m_vecViewOffset())
				: (vLeadEyeX + vFoeVelX * ((float)iDrain * I::GlobalVars->interval_per_tick));
			const Vec3 vAim = Math::CalcAngle(pLocal->GetShootPos(), vAimEye);
			pCmd->viewangles.x = vAim.x;
			pCmd->viewangles.y = vAim.y;
		}
		// (Live fallback-direction update removed: m_vCommitDir is unread —
		// the mirrored moving goal below does all routing. No 450 anywhere.)
		// Faithful replay: the sim's moving goal + orbit on live state —
		// flank waypoint of the latched side while frontal, behind-point
		// once abeam, far corner once rounded — re-anchored to the LIVE
		// lead every tick, coast-corrected input like the sim. Overshoot
		// orbits; the point never stops the walk (hold heading).
		{
			// Ghost winner: replay anchors to the latched record (pos + yaw) —
			// the exact frame the sim validated. Present winner: live lead.
			const Vec3 vAnchorR = s_bCommitOld ? s_vCommitAnchor : vLeadX;
			const float flYawR = s_bCommitOld ? s_flCommitYaw : flFoeYawX;
			Vec3 vFoeFX = {};
			Math::AngleVectors({ 0.f, flYawR, 0.f }, &vFoeFX);
			vFoeFX.Normalize2D();
			Vec3 vToFX = vAnchorR - vMyPos;
			vToFX.z = 0.f;
			const float flToFL = vToFX.Length2D();
			Vec3 vGoalD = vAnchorR + Vec3(-vFoeFX.x, -vFoeFX.y, 0.f) * (pLocal->m_vecMaxs().x + pExec->m_vecMaxs().x + TRICKSTAB_MINK_PAD + 1.f);
			const bool bHighR = fabsf(vMyPos.z - vAnchorR.z) >= TRICKSTAB_STAIRSTAB_H;
			bool bFrontalR = false;
			if (s_iCommitSide != 0 && !bHighR && flToFL > 1.f)
			{
				const Vec3 vAppNR = vToFX * (1.f / flToFL);
				if (vAppNR.Dot(vFoeFX) < -0.3f)
				{
					bFrontalR = true;
					const Vec3 vRightR = Vec3(vAppNR.y, -vAppNR.x, 0.f);
					vGoalD = vLeadX + vRightR * ((float)s_iCommitSide * (pLocal->m_vecMaxs().y + pExec->m_vecMaxs().y + TRICKSTAB_MINK_PAD + 1.f));
				}
			}
			if (!bFrontalR)
			{
				if (!s_bOrbited && (vGoalD - vMyPos).Length2D() < 20.f)
					s_bOrbited = true; // rounded it live: keep orbiting like the sim
				if (s_bOrbited)
					vGoalD = vAnchorR + s_vFarC;
			}
			const float flRDT = I::GlobalVars->interval_per_tick;
			Vec3 vCoastP = vMyPos, vCoastV = pLocal->m_vecVelocity();
			{
				const float flSp = vCoastV.Length2D();
				if (flSp > 0.f)
				{
					const float flDrop = flSp * 4.0f * flRDT;
					const float flNew = std::max(flSp - flDrop, 0.f);
					vCoastV.x *= flNew / flSp;
					vCoastV.y *= flNew / flSp;
				}
				if (!(pLocal->m_fFlags() & FL_ONGROUND))
					vCoastV.z -= 800.f * flRDT;
				vCoastP = vCoastP + vCoastV * flRDT;
			}
			Vec3 vD = vGoalD - vCoastP;
			vD.z = 0.f;
			float flDL = vD.Length2D();
			if (flDL < 1.f)
			{
				vD = vGoalD - vMyPos;
				vD.z = 0.f;
				flDL = vD.Length2D();
			}
			if (flDL >= 1.f)
			{
				s_vLastDir = vD * (1.f / flDL);
				PlantMoves(pLocal, pCmd, s_vLastDir);
			}
			else
				PlantMoves(pLocal, pCmd, s_vLastDir); // hold heading through the point
		}
	}
	else if (bAssist && m_bHasBest && iBank > 0)
	{
		// Manual A/D selects the assist side; no input = automatic side.
		// Warp planning never sees this.
		Vec3 vAsDest = vBestDestG;
		if (pBestEnt)
		{
			const float flSM = pCmd->sidemove;
			const bool bMan = Vars::Aimbot::Melee::TrickstabSide.Value == Vars::Aimbot::Melee::TrickstabSideEnum::Manual
				&& fabsf(flSM) >= TRICKSTAB_MANUAL_THRESH;
			float flAL = FLT_MAX, flAR = FLT_MAX;
			for (int i = 0; i < 4; i++)
			{
				const float flTY = TrickstabYaw(vLeadB + vCornersB[i], vLeadB);
				const float flYD = fabsf(TrickstabNorm(flTY - flFoeYawB));
				if (vCornersB[i].y > 0.f)
					flAR = std::min(flAR, flYD);
				else
					flAL = std::min(flAL, flYD);
			}
			const bool bLeftAs = bMan ? (flSM < 0.f) : (flAL < flAR);
			int iAsOpt = -1;
			float flAsY = FLT_MAX;
			for (int i = 0; i < 4; i++)
			{
				if ((vCornersB[i].y < 0.f) != bLeftAs)
					continue;
				const float flTY = TrickstabYaw(vLeadB + vCornersB[i], vLeadB);
				const float flYD = fabsf(TrickstabNorm(flTY - flFoeYawB));
				if (flYD < flAsY)
				{
					flAsY = flYD;
					iAsOpt = i;
				}
			}
			// No 450 extension: steer AT the corner itself. Direction is
			// recomputed live every tick and arrival never stops the walk,
			// so the far-point crutch only diluted the math — and dragged
			// assist 450 past the fight.
			if (iAsOpt >= 0)
				vAsDest = vLeadB + vCornersB[iAsOpt];
		}
		Vec3 vDir = vAsDest - vMyPos;
		vDir.z = 0.f;
		float flLen = vDir.Length2D();
		if (flLen >= 1.f)
		{
			s_vLastDir = vDir * (1.f / flLen);
			PlantMoves(pLocal, pCmd, s_vLastDir);
		}
		else
			PlantMoves(pLocal, pCmd, s_vLastDir); // hold heading through the point
	}
	// Commit claim: scored best + live re-test at the hit state + not
	// currently stabbing + >= 1 margin-verified point + bank covers exact length.
	// Notably NOT gated on: generic Warp.Value (Warp() yields to the commit
	// latch, so they share the goal field without fighting) or m_bRecharge
	// (the commit flag suppresses recharge; blocking mid-top-up refused
	// funded claims). m_bWarp overlap and DT/speedhack still exclude.
	if (!m_bCommitting && m_bBestHittable && m_iMinTicks > 0
		&& !F::Ticks.m_bWarp
		&& !F::Ticks.m_bDoubletap && !F::Ticks.m_bSpeedhack
		&& m_iTotalPoints >= 1 && iBank >= m_iMinTicks
		&& pBestEnt)
	{
		// Behind-margin gate: the winning hit must sit clearly behind the
		// foe (~60 deg cone, dot >= 0.5), never a side-window graze. Grazes
		// fund warps that terminate at the face and hug it — no stab, no
		// warp, simple. (StabAt keeps its >0 hemisphere for sim breadth;
		// the claim is strict.)
		const float flBehind = TrickstabBehind(vBestHitPosG, vBestHitFoeG, flFoeYawB);
		// Always warp on HIT: live walk-up ability no longer blocks the claim.
		if (flBehind >= 0.5f
			&& StabAt(vBestHitEyeG, vBestHitFoeG, vBestHitFoeG + pBestEnt->m_vecViewOffset(), flFoeYawB, flOwnerYaw, flRange, vSwingMins, vSwingMaxs, pLocal, pBestEnt))
		{
			m_bCommitting = true;
			m_vCommitDir = vBestDestG; // fallback only; the offset chase does the routing
			m_iClaimTick = I::GlobalVars->tickcount;
			m_iCommitLeft = m_iMinTicks;
			m_iCommitAge = 0;
			// Latch the validated walk: same target, same side, same orbit.
			s_iCommitEnt = pBestEnt->entindex();
			s_vCommitPath = vBestPathG;
			s_vCommitOff = vBestOffG;
			s_iCommitSide = iBestSideG;
			s_vFarC = vBestFarCG;
			s_bOrbited = false;
			s_bCommitOld = bOldG;
			s_vCommitFoe = vBestHitFoeG;
			s_vCommitAnchor = vLeadB; // winner lead (ghost anchor when old)
			s_flCommitYaw = flFoeYawB; // winner yaw (ghost yaw when old)
			F::Ticks.m_bWarp = true;
			F::Ticks.m_iShiftedGoal = F::Ticks.ReleaseGoal(m_iMinTicks);
		}
	}
}
