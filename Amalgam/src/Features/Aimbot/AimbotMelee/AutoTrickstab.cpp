#include "AutoTrickstab.h"

#include "../../Simulation/MovementSimulation/MovementSimulation.h"
#include "../../../SDK/VarsMelee.h"

// LUT-faithful port of Auto_Trickstab (qawe) direction logic; SIMULATION is
// Amalgam engine (MoveSim) instead of Lua math. Deliberate deviations (all
// noted): hull radii for hitbox radii (mapped fields), no TARGET_EXTENSION /
// coast pass (engine integrates internally), no manual A/D side override or
// yaw-delta side preference (movement-commitment phase), no TF2.* constant
// table (live sv_* via engine), trace target is GetCenter (not per-point
// clamp; scoring only, the stab itself still aims clamp-point via aimbot).
#define TRICKSTAB_RANGE_GATE 600.f
#define TRICKSTAB_HORIZON_MIN 8
#define TRICKSTAB_HORIZON_MAX 24
#define TRICKSTAB_DEG2RAD 0.0174532925f

// Direction -> corner subset LUT, verbatim shape (0-indexed corners:
// TL,TR,BL,BR at +/-hitbox, [4] = center; -1 pads short rows).
static const int TRICKSTAB_LUT[3][3][3] = {
	{ {4,3,0}, {4,3,1}, {4,2,1} },
	{ {4,1,0}, {4,-1,-1}, {4,2,3} },
	{ {4,1,2}, {4,0,2}, {4,0,3} },
};

bool CAutoTrickstab::SimDir(CTFPlayer* pLocal, float flYawDeg, int iTicks, DirResult_t& tOut)
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
	mv.m_flForwardMove = 450.f;
	mv.m_flSideMove = 0.f;
	mv.m_flUpMove = 0.f;
	for (int i = 0; i < iTicks; i++)
	{
		if (tStorage.m_bFailed)
			break;
		F::MoveSim.RunTick(tStorage);
	}
	tOut.vEnd = mv.m_vecAbsOrigin;
	tOut.vPath = tStorage.m_vPath;
	tOut.bValid = !tStorage.m_bFailed && (int)tOut.vPath.size() >= 2;
	F::MoveSim.Restore(tStorage);
	return tOut.bValid;
}

int CAutoTrickstab::ScorePath(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CTFPlayer* pTarget, const Vec3& vTargetCenter, float flTargetYaw, const std::vector<Vec3>& vPath, Vec3& vHitPosOut)
{
	float flRange = SDK::AttribHookValue(pWeapon->GetSwingRange(), "melee_range_multiplier", pWeapon);
	if (pLocal->m_flModelScale() > 1.0f)
		flRange *= pLocal->m_flModelScale();
	const float flHull = SDK::AttribHookValue(18, "melee_bounds_multiplier", pWeapon);
	const Vec3 vSwingMins = { -flHull, -flHull, -flHull };
	const Vec3 vSwingMaxs = { flHull, flHull, flHull };
	const float flRad = flTargetYaw * TRICKSTAB_DEG2RAD;
	const Vec3 vTargetFwd = { cosf(flRad), sinf(flRad), 0.f };
	CTraceFilterHitscan filter = {};
	filter.pSkip = pLocal;
	for (size_t k = 0; k < vPath.size(); k++)
	{
		const Vec3 vEye = vPath[k] + pLocal->m_vecViewOffset();
		Vec3 vToT = vTargetCenter - vEye;
		vToT.z = 0.f;
		const float flDist = (vTargetCenter - vEye).Length();
		if (flDist < 0.001f || flDist > flRange)
			continue;
		if ((vToT * (1.f / flDist)).Dot(vTargetFwd) <= 0.f)
			continue; // not behind
		CGameTrace trace = {};
		SDK::TraceHull(vEye, vTargetCenter, vSwingMins, vSwingMaxs, MASK_SOLID, &filter, &trace);
		if (trace.m_pEnt && trace.m_pEnt == pTarget)
		{
			vHitPosOut = vPath[k];
			return (int)k;
		}
	}
	return -1;
}

void CAutoTrickstab::Run(CTFPlayer* pLocal, CTFWeaponBase* pWeapon)
{
	m_bHasBest = false;
	m_bBestHittable = false;
	if (!Vars::Aimbot::Melee::AutoTrickstab.Value || !pLocal || !pWeapon)
		return;
	if (pWeapon->GetWeaponID() != TF_WEAPON_KNIFE)
		return;
	if (!pLocal->IsAlive())
		return;
	// Nearest live enemy player (frozen-world assumption: evaluated once).
	// Beyond warp-relevant range the fan is pointless: skip all sims.
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
	if (!pTarget || flBest > TRICKSTAB_RANGE_GATE * TRICKSTAB_RANGE_GATE)
		return;
	// Lag-compensated target pos (half ping * velocity), like the reference:
	// the warp transmits across latency, score where they'll be, not are.
	Vec3 vLead = pTarget->m_vecOrigin();
	if (auto pNetChan = I::EngineClient->GetNetChannelInfo())
	{
		const float flLeadT = (std::max(pNetChan->GetLatency(FLOW_OUTGOING), 0.f) + std::max(pNetChan->GetLatency(FLOW_INCOMING), 0.f)) * 0.5f;
		vLead = vLead + pTarget->m_vecVelocity() * flLeadT;
	}
	// Relative quadrant (verbatim: 1u buffer, vertical gate zeroes it) then
	// the corner subset LUT. Corners at +-combined hull radii, world axes.
	const float flHitbox = pLocal->m_vecMaxs().x + pTarget->m_vecMaxs().x;
	const float flVertRange = pTarget->m_vecMaxs().z - pTarget->m_vecMins().z;
	const Vec3 vMy = pLocal->m_vecOrigin();
	const float flDx = vLead.x - vMy.x, flDy = vLead.y - vMy.y, flDz = vLead.z - vMy.z;
	const int iVert = (fabsf(flDz) > flVertRange) ? 1 : 0;
	const int iQx = (((flDx > flHitbox - 1.f) ? 1 : 0) - ((flDx < -flHitbox + 1.f) ? 1 : 0)) * (1 - iVert);
	const int iQy = (((flDy > flHitbox - 1.f) ? 1 : 0) - ((flDy < -flHitbox + 1.f) ? 1 : 0)) * (1 - iVert);
	const Vec3 vCorners[5] = {
		{ -flHitbox, flHitbox, 0.f }, { flHitbox, flHitbox, 0.f },
		{ -flHitbox, -flHitbox, 0.f }, { flHitbox, -flHitbox, 0.f },
		{ 0.f, 0.f, 0.f } };
	const Vec3 vTargetCenter = pTarget->GetCenter();
	const float flTargetYaw = H::Entities.GetEyeAngles(pTarget->entindex()).y;
	const int iTicks = std::clamp(Vars::Doubletap::TickLimit.Value, TRICKSTAB_HORIZON_MIN, TRICKSTAB_HORIZON_MAX);
	DirResult_t aRes[3] = {};
	int iDirs = 0;
	for (int k = 0; k < 3; k++)
	{
		const int iCorner = TRICKSTAB_LUT[iQx + 1][iQy + 1][k];
		if (iCorner < 0 || iCorner > 4)
			break;
		const Vec3 vPoint = vLead + vCorners[iCorner];
		const float flYaw = atan2f(vPoint.y - vMy.y, vPoint.x - vMy.x) * 57.29578f;
		if (!SimDir(pLocal, flYaw, iTicks, aRes[iDirs]))
			continue;
		aRes[iDirs].iHitTick = ScorePath(pLocal, pWeapon, pTarget, vTargetCenter, flTargetYaw, aRes[iDirs].vPath, aRes[iDirs].vHitPos);
		aRes[iDirs].bHittable = aRes[iDirs].iHitTick >= 0;
		aRes[iDirs].flScore = aRes[iDirs].bHittable
			? (vTargetCenter - aRes[iDirs].vHitPos).Length()
			: (pTarget->m_vecOrigin() - aRes[iDirs].vEnd).Length();
		iDirs++;
	}
	if (!iDirs)
		return;
	// Best: earliest hittable tick, then nearest; else nearest end overall.
	int iBest = -1;
	for (int iPass = 0; iPass < 2 && iBest < 0; iPass++)
	{
		int iBestTick = INT_MAX;
		float flS = std::numeric_limits<float>::max();
		for (int i = 0; i < iDirs; i++)
		{
			if (!aRes[i].bValid)
				continue;
			if (iPass == 0)
			{
				if (!aRes[i].bHittable)
					continue;
				if (aRes[i].iHitTick < iBestTick || (aRes[i].iHitTick == iBestTick && aRes[i].flScore < flS))
				{
					iBestTick = aRes[i].iHitTick;
					flS = aRes[i].flScore;
					iBest = i;
				}
			}
			else if (aRes[i].flScore < flS)
			{
				flS = aRes[i].flScore;
				iBest = i;
			}
		}
	}
	if (iBest < 0)
		return;
	m_bHasBest = true;
	m_vBestDir = aRes[iBest].vDir;
	m_bBestHittable = aRes[iBest].bHittable;
	// Draw: all valid paths dim, best green (hittable) / yellow, hit marker +
	// target link. Short expiry: refreshed per tick, self-cleaning.
	const float flUntil = I::GlobalVars->curtime + 1.f;
	for (int i = 0; i < iDirs; i++)
	{
		if (!aRes[i].bValid || (int)aRes[i].vPath.size() < 2)
			continue;
		const bool bIsBest = (i == iBest);
		Color_t tCol = bIsBest
			? (aRes[i].bHittable ? Color_t(0, 255, 0) : Color_t(255, 255, 0))
			: Color_t(255, 255, 255, 40);
		for (size_t k = 0; k + 1 < aRes[i].vPath.size(); k++)
			G::LineStorage.emplace_back(std::pair<Vec3, Vec3>(aRes[i].vPath[k], aRes[i].vPath[k + 1]), flUntil, tCol, true);
		if (bIsBest)
		{
			const Vec3 vMark = aRes[i].bHittable ? aRes[i].vHitPos : aRes[i].vEnd;
			G::LineStorage.emplace_back(std::pair<Vec3, Vec3>(vMark, vMark + Vec3(0, 0, 5)), flUntil, tCol, true);
			G::LineStorage.emplace_back(std::pair<Vec3, Vec3>(vMark, vTargetCenter), flUntil, Color_t(255, 255, 0), true);
		}
	}
}
