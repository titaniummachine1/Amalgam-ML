#include "AutoTrickstab.h"

#include "../../Simulation/MovementSimulation/MovementSimulation.h"
#include "../../Ticks/Ticks.h"
#include "../../Misc/Misc.h"
#include "../../../SDK/VarsMelee.h"

// Verbatim Lua constants (Auto_Trickstab TF2 table).
#define TRICKSTAB_SCORE_DIST 120.f
#define TRICKSTAB_YAW_W 0.7f
#define TRICKSTAB_DIST_W 0.3f
#define TRICKSTAB_EXTENSION 450.f
#define TRICKSTAB_STAIRSTAB_H 82.f
#define TRICKSTAB_MANUAL_THRESH 100.f
#define TRICKSTAB_ACQUIRE_RANGE 225.f
#define TRICKSTAB_MINK_PAD 1.f
#define TRICKSTAB_DEG2RAD 0.0174532925f
#define TRICKSTAB_RAD2DEG 57.29578f
#define TRICKSTAB_MOVE_SPEED 450.f

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
	Vec3 vOwnerF, vFoeF;
	Math::AngleVectors({ 0.f, flOwnerYaw, 0.f }, &vOwnerF);
	Math::AngleVectors({ 0.f, flFoeYaw, 0.f }, &vFoeF);
	vOwnerF.Normalize2D();
	vFoeF.Normalize2D();
	if (!(vToTarget.Dot(vFoeF) > 0.f + 0.0031f + flExtra)) // behind them?
		return false;
	if (!(vToTarget.Dot(vOwnerF) > 0.5f + flExtra)) // facing them?
		return false;
	if (!(vFoeF.Dot(vOwnerF) > -0.3f + 0.0031f)) // facestab guard
		return false;
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

float CAutoTrickstab::ScorePath(CTFPlayer* pLocal, CTFPlayer* pTarget, const std::vector<Vec3>& vPath, const std::vector<Vec3>& vFoePath, const Vec3& vLeadEye, float flFoeYaw, float flOwnerYaw, float flRange, const Vec3& vSwingMins, const Vec3& vSwingMaxs, Vec3& vHitPosOut, Vec3& vHitEyeOut, Vec3& vHitFoeOut, int& iHitTickOut, int& iHitCountOut, std::vector<char>& vHitsOut)
{
	float flBest = -1.f;
	vHitPosOut = {};
	vHitEyeOut = {};
	vHitFoeOut = {};
	iHitTickOut = -1;
	iHitCountOut = 0;
	vHitsOut.assign(vPath.size(), 0);
	const Vec3 vLocalEye = pLocal->m_vecViewOffset();
	const Vec3 vFoeEyeOff = pTarget->m_vecViewOffset();
	for (size_t k = 0; k < vPath.size(); k++)
	{
		const Vec3 vEye = vPath[k] + vLocalEye;
		// Dual prediction: test against the enemy's pose THIS tick.
		const Vec3 vFoe = (k < vFoePath.size()) ? vFoePath[k] : vPath[k];
		const Vec3 vFoeEye = vFoe + vFoeEyeOff;
		if (!StabAt(vEye, vFoe, vFoeEye, flFoeYaw, flOwnerYaw, flRange, vSwingMins, vSwingMaxs, pLocal, pTarget))
			continue;
		vHitsOut[k] = 1;
		iHitCountOut++;
		// Lua ranking: eye-anchored yaw + feet distance, tie -> earlier.
		// (Direction fixed: sim -> enemy vs facing, ~0 behind. The old code
		// measured enemy -> sim, which reads 180 behind and never scored.)
		const float flSpyYaw = TrickstabYaw(vEye, vLeadEye);
		const float flYawDiff = fabsf(TrickstabNorm(flSpyYaw - flFoeYaw));
		const float flYawComp = std::max(0.f, 1.f - flYawDiff / 90.f);
		const float flDistComp = std::max(0.f, 1.f - (vPath[k] - vFoe).Length() / TRICKSTAB_SCORE_DIST);
		const float flScore = TRICKSTAB_YAW_W * flYawComp + TRICKSTAB_DIST_W * flDistComp;
		if (flScore > flBest + 1e-6f || (fabsf(flScore - flBest) <= 1e-6f && (int)k < iHitTickOut))
		{
			flBest = flScore;
			vHitPosOut = vPath[k];
			vHitEyeOut = vEye;
			vHitFoeOut = vFoe;
			iHitTickOut = (int)k;
		}
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
	// Nearest live enemy inside acquisition range (Lua: 225).
	CTFPlayer* pTarget = nullptr;
	float flBest = std::numeric_limits<float>::max();
	for (auto pEntity : H::Entities.GetGroup(EntityEnum::PlayerEnemy))
	{
		if (!pEntity || pEntity->IsDormant())
			continue;
		auto pEnemy = pEntity->As<CTFPlayer>();
		if (!pEnemy || !pEnemy->IsAlive())
			continue;
		const float flD = pLocal->m_vecOrigin().DistToSqr(pEnemy->m_vecOrigin());
		if (flD < flBest)
		{
			flBest = flD;
			pTarget = pEnemy;
		}
	}
	if (!pTarget || flBest > TRICKSTAB_ACQUIRE_RANGE * TRICKSTAB_ACQUIRE_RANGE)
		return;
	// Razorback: the firing path refuses, so the sim refuses once, not per tick.
	if (Vars::Aimbot::Melee::IgnoreRazorback.Value)
	{
		CUtlVector<CBaseEntity*> itemList;
		int iBackstabShield = SDK::AttribHookValue(0, "set_blockbackstab_once", pTarget, &itemList);
		if (iBackstabShield && itemList.Count())
		{
			if (CBaseEntity* pEntity = itemList.Element(0))
			{
				if (pEntity->ShouldDraw())
					return;
			}
		}
	}
	// Lead: half ping + choke, the pose the server will judge.
	Vec3 vLead = pTarget->m_vecOrigin();
	const Vec3 vFoeVel = pTarget->m_vecVelocity();
	if (auto pNetChan = I::EngineClient->GetNetChannelInfo())
	{
		const float flLeadT = (std::max(pNetChan->GetLatency(FLOW_OUTGOING), 0.f) + std::max(pNetChan->GetLatency(FLOW_INCOMING), 0.f)) * 0.5f
			+ H::Entities.GetChoke(pTarget->entindex()) * I::GlobalVars->interval_per_tick;
		vLead = vLead + vFoeVel * flLeadT;
	}
	const int iBank = F::Ticks.m_iShiftedTicks;
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
	const Vec3 vLeadEye = vLead + pTarget->m_vecViewOffset();
	const float flFoeYaw = H::Entities.GetEyeAngles(pTarget->entindex()).y;
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
		}
		else if (m_iCommitLeft > 0)
			m_iCommitLeft--;
	}
	if (!m_bCommitting && m_iBonusLeft > 0)
	{
		const float flBackRadB = (flFoeYaw + 180.f) * TRICKSTAB_DEG2RAD;
		const float flHxb = pLocal->m_vecMaxs().x + pTarget->m_vecMaxs().x;
		Vec3 vBack = (vLead + Vec3(cosf(flBackRadB), sinf(flBackRadB), 0.f) * flHxb) - vMyPos;
		vBack.z = 0.f;
		if (vBack.Length2D() >= 1.f)
		{
			vBack = vBack * (1.f / vBack.Length2D());
			PlantMoves(pLocal, pCmd, vBack);
		}
		m_iBonusLeft--;
	}
	// Sim horizon is ALWAYS the bank (no planning floor): bank 0 -> no sims,
	// no viz, no assist, no claim. Lua-faithful (0 ticks = empty paths).
	if (iBank <= 0)
		return;
	const int iTicks = iBank;
	// Executable-range gate: bank carry + stab range + bonus reach. Beyond
	// this the warp physically cannot arrive — drawing there is lying.
	{
		const float flStep = SDK::MaxSpeed(pLocal) * I::GlobalVars->interval_per_tick;
		const float flMaxReach = iBank * flStep + flRange + 3.f * flStep;
		if (flBest > flMaxReach * flMaxReach)
			return;
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
	const Vec3 vCenter = {};
	// Side picker (R7): Manual reads GAME INPUT (cmd sidemove, never keys);
	// no input falls back to Automatic (closer side to the back, tie right).
	// Lua code sites agree D-positive = right; the stray comment claiming
	// the opposite is disregarded in favor of both call sites + SDK.
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
	const float flSideMove = pCmd->sidemove;
	const bool bManualIn = Vars::Aimbot::Melee::TrickstabSide.Value == Vars::Aimbot::Melee::TrickstabSideEnum::Manual
		&& fabsf(flSideMove) >= TRICKSTAB_MANUAL_THRESH;
	const bool bBestLeft = bManualIn ? (flSideMove < 0.f) : (flLeft < flRight);
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
	int iOther = -1;
	for (int i = 0; i < 4; i++)
	{
		if ((vCorners[i].y < 0.f) == bBestLeft)
		{
			iOther = i;
			break;
		}
	}
	const float flBackYaw = flFoeYaw + 180.f;
	const bool bWithinBackAngle = fabsf(TrickstabNorm(TrickstabYaw(vLead, vMyPos) - flBackYaw)) <= 90.f;
	const bool bStairstab = fabsf(vMyPos.z - vLead.z) >= TRICKSTAB_STAIRSTAB_H;
	// Dual prediction: enemy extrapolated once (friction + gravity), reused
	// by all paths. Exact on flat ground; full enemy MoveSim is the
	// documented upgrade path if tests show divergence.
	const float flDT = I::GlobalVars->interval_per_tick;
	std::vector<Vec3> vFoePath;
	vFoePath.reserve(iTicks);
	{
		Vec3 vP = vLead, vV = vFoeVel;
		const bool bGround = (pTarget->m_fFlags() & FL_ONGROUND) != 0;
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
	// Neutral resolve, full engine fidelity (R: two sims per tick, linear).
	// One physics truth: walls/slides/steps/ledges inside P1 by construction.
	Vec3 vResolved = vMyPos;
	ResolveNeutral(pLocal, iTicks, vResolved);
	struct SimTarget_t { bool bCenter; int iCorner; };
	SimTarget_t aTargets[3] = {};
	int iTargets = 0;
	if (bStairstab)
		aTargets[iTargets++] = { true, -1 };
	else if (iOptimal >= 0)
		aTargets[iTargets++] = { false, iOptimal };
	DirResult_t aRes[3] = {};
	int iDirs = 0;
	float flBestScore = -1.f;
	int iBestTick = INT_MAX;
	int iBest = -1;
	CTraceFilterHitscan wallFilter = {};
	wallFilter.pSkip = pLocal;
	const auto SimOne = [&](const SimTarget_t& t) -> int
	{
		if (iDirs >= 3)
			return -1;
		const Vec3 vCorner = t.bCenter ? vCenter : vCorners[t.iCorner];
		const float flWishYaw = Wishdir(vResolved, vMyPos, vLead, vCorner);
		DirResult_t& r = aRes[iDirs];
		// Stored regardless of outcome: failed/stalled paths still draw
		// (that's the "shows it all" rule). Only clean paths can win.
		SimDir(pLocal, flWishYaw, iTicks, vLead, vOMin, vOMax, vTMin, vTMax, r);
		if (r.vPath.empty())
			return -1;
		r.flScore = ScorePath(pLocal, pTarget, r.vPath, vFoePath, vLeadEye, flFoeYaw, flOwnerYaw, flRange, vSwingMins, vSwingMaxs, r.vHitPos, r.vHitEye, r.vHitFoe, r.iHitTick, r.iHitCount, r.vHits);
		r.bHittable = r.flScore >= 0.f;
		m_iTotalPoints += r.iHitCount;
		CGameTrace wallTrace = {};
		SDK::TraceHull(vMyPos + Vec3(0.f, 0.f, 18.f), r.vEnd + Vec3(0.f, 0.f, 18.f), {}, {}, MASK_SOLID, &wallFilter, &wallTrace);
		r.bHitWall = wallTrace.fraction < 0.8f && !(wallTrace.m_pEnt && wallTrace.m_pEnt->IsPlayer());
		const int iIdx = iDirs++;
		if (r.bValid && r.bHittable && (r.flScore > flBestScore + 1e-6f || (fabsf(r.flScore - flBestScore) <= 1e-6f && r.iHitTick < iBestTick)))
		{
			flBestScore = r.flScore;
			iBestTick = r.iHitTick;
			iBest = iIdx;
		}
		return iIdx;
	};
	int iOptimalIdx = -1;
	for (int i = 0; i < iTargets; i++)
		iOptimalIdx = SimOne(aTargets[i]);
	// Other side ALWAYS (R5 axis-aligned; matador lives here). Lua's
	// wall-gated conditional stays as a comment, not a gate.
	if (!bStairstab && iOther >= 0)
		SimOne({ false, iOther });
	if (!bStairstab && bWithinBackAngle)
		SimOne({ true, -1 });
	if (!iDirs)
		return;
	// Fallback: points exist but no scored winner — steer by geometry.
	if (iBest < 0 && m_iTotalPoints > 0)
	{
		Vec3 vFallback;
		if (bStairstab)
		{
			const float flBackRad = flBackYaw * TRICKSTAB_DEG2RAD;
			vFallback = (vLead + Vec3(cosf(flBackRad), sinf(flBackRad), 0.f) * flHx) - vMyPos;
		}
		else if (iOptimal >= 0)
			vFallback = (vLead + vCorners[iOptimal]) - vMyPos;
		else
			return;
		vFallback.z = 0.f;
		const float flLen = vFallback.Length2D();
		if (flLen < 1.f)
			return;
		m_bHasBest = true;
		m_vBestDir = vFallback * (1.f / flLen);
		m_bBestHittable = false;
		m_iMinTicks = -1;
	}
	else if (iBest >= 0)
	{
		m_bHasBest = true;
		m_vBestDir = aRes[iBest].vDir;
		m_bBestHittable = aRes[iBest].bHittable;
		m_iMinTicks = aRes[iBest].iHitTick + 1; // index -> warp length
	}
	// else: no winner — fields stay clear, but the sims still draw below.
	// Assist (R6/R8): toggle drives. Commit drives on its frozen geometry
	// even if this tick unscored; assist never plants a stale direction.
	const bool bAssist = Vars::Aimbot::Melee::TrickstabAssist.Value;
	if (m_bCommitting || (bAssist && m_bHasBest))
		PlantMoves(pLocal, pCmd, m_vBestDir);
	// Commit claim: scored best + live re-test at the hit state + not
	// currently stabbing + >= 3 points + bank covers exact length.
	// Notably NOT gated on: generic Warp.Value (Warp() yields to the commit
	// latch, so they share the goal field without fighting) or m_bRecharge
	// (the commit flag suppresses recharge; blocking mid-top-up refused
	// funded claims). m_bWarp overlap and DT/speedhack still exclude.
	if (!m_bCommitting && m_bBestHittable && m_iMinTicks > 0
		&& !F::Ticks.m_bWarp
		&& !F::Ticks.m_bDoubletap && !F::Ticks.m_bSpeedhack
		&& m_iTotalPoints >= 3 && iBank >= m_iMinTicks)
	{
		const int iHit = (iBest >= 0) ? iBest : 0;
		const Vec3 vFoeNow = (iBest >= 0) ? aRes[iBest].vHitFoe : vLead;
		if (StabAt(aRes[iHit].vHitEye, vFoeNow, vFoeNow + pTarget->m_vecViewOffset(), flFoeYaw, flOwnerYaw, flRange, vSwingMins, vSwingMaxs, pLocal, pTarget)
			&& !StabAt(pLocal->GetShootPos(), pTarget->m_vecOrigin(), pTarget->m_vecOrigin() + pTarget->m_vecViewOffset(), flFoeYaw, flOwnerYaw, flRange, vSwingMins, vSwingMaxs, pLocal, pTarget))
		{
			m_bCommitting = true;
			m_iCommitLeft = m_iMinTicks;
			m_iCommitAge = 0;
			F::Ticks.m_bWarp = true;
			F::Ticks.m_iShiftedGoal = std::max(iBank - m_iMinTicks, 0);
		}
	}
	// Draw EVERYTHING the sims produced (toggle in Visuals > Prediction):
	// clean paths bright, stalled ones dim, per-tick dots, marker + feet
	// link on the winner. No winner / no hits still draws — the rule is:
	// if a sim ran, you see it.
	if (!Vars::Aimbot::Melee::TrickstabDraw.Value)
		return;
	const float flUntil = I::GlobalVars->curtime + 1.f;
	for (int i = 0; i < iDirs; i++)
	{
		if (aRes[i].vPath.empty())
			continue;
		const bool bIsBest = (i == iBest);
		Color_t tCol = !aRes[i].bValid ? Color_t(255, 255, 255, 55)
			: bIsBest ? (aRes[i].bHittable ? Color_t(0, 255, 0) : Color_t(255, 255, 0))
			: Color_t(255, 255, 255, 110);
		for (size_t k = 0; k + 1 < aRes[i].vPath.size(); k++)
			G::LineStorage.emplace_back(std::pair<Vec3, Vec3>(aRes[i].vPath[k], aRes[i].vPath[k + 1]), flUntil, tCol, true);
		// Per-tick dots: green = hittable here, faint red = not. Feet level,
		// same storage as prediction lines (draw tab shows both together).
		for (size_t k = 0; k < aRes[i].vPath.size(); k++)
		{
			const bool bHit = k < aRes[i].vHits.size() && aRes[i].vHits[k];
			const Vec3 vDot = aRes[i].vPath[k];
			const Color_t tDot = bHit ? Color_t(0, 255, 0) : Color_t(255, 60, 60, 70);
			G::LineStorage.emplace_back(std::pair<Vec3, Vec3>(vDot, vDot + Vec3(0, 0, bHit ? 6 : 3)), flUntil, tDot, true);
		}
		if (bIsBest)
		{
			const Vec3 vMark = aRes[i].bHittable ? aRes[i].vHitPos : aRes[i].vEnd;
			G::LineStorage.emplace_back(std::pair<Vec3, Vec3>(vMark, vMark + Vec3(0, 0, 8)), flUntil, tCol, true);
			G::LineStorage.emplace_back(std::pair<Vec3, Vec3>(vMark, vLead), flUntil, m_bCommitting ? Color_t(0, 255, 255) : Color_t(255, 255, 0), true);
		}
	}
}
