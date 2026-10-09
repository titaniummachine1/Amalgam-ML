#include "AutoTrickstab.h"

#include "../../Simulation/MovementSimulation/MovementSimulation.h"
#include "../../../SDK/VarsMelee.h"

// Fan constants. Spread is user-tunable later (no menu yet, phase 1);
// vertical band picks center-only on stairs (lateral fan is useless there).
#define TRICKSTAB_FAN_HALF_DEG 45.f
#define TRICKSTAB_VERTICAL_BAND 40.f
#define TRICKSTAB_HORIZON_MIN 8
#define TRICKSTAB_HORIZON_MAX 24
#define TRICKSTAB_DEG2RAD 0.0174532925f

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

bool CAutoTrickstab::ScoreDir(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CTFPlayer* pTarget, const DirResult_t& tDir)
{
	// Behind-dot (CanBackstab core shape): victim facing vs us-at-end.
	Vec3 vToT = pTarget->m_vecOrigin() - tDir.vEnd;
	vToT.z = 0.f;
	const float flDist = (pTarget->m_vecOrigin() - tDir.vEnd).Length();
	float flRange = SDK::AttribHookValue(pWeapon->GetSwingRange(), "melee_range_multiplier", pWeapon);
	if (pLocal->m_flModelScale() > 1.0f)
		flRange *= pLocal->m_flModelScale();
	if (flDist < 0.001f || flDist > flRange)
		return false;
	vToT = vToT * (1.f / flDist);
	const float flTargetYaw = H::Entities.GetEyeAngles(pTarget->entindex()).y;
	const float flRad = flTargetYaw * TRICKSTAB_DEG2RAD;
	const Vec3 vTargetFwd = { cosf(flRad), sinf(flRad), 0.f };
	if (vToT.Dot(vTargetFwd) <= 0.f)
		return false; // not behind
	// Visibility from sim end (swing hull): no wall between.
	CTraceFilterHitscan filter = {};
	filter.pSkip = pLocal;
	CGameTrace trace = {};
	const float flHull = SDK::AttribHookValue(18, "melee_bounds_multiplier", pWeapon);
	const Vec3 vSwingMins = { -flHull, -flHull, -flHull };
	const Vec3 vSwingMaxs = { flHull, flHull, flHull };
	SDK::TraceHull(tDir.vEnd + pLocal->m_vecViewOffset(), pTarget->GetCenter(), vSwingMins, vSwingMaxs, MASK_SOLID, &filter, &trace);
	return trace.m_pEnt && trace.m_pEnt == pTarget;
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
	if (!pTarget)
		return;
	// Fan: to-target center, +/-45; center-only across floors.
	const Vec3 vTo = pTarget->m_vecOrigin() - pLocal->m_vecOrigin();
	const float flBaseYaw = atan2f(vTo.y, vTo.x) * 57.29578f;
	const float flDz = pTarget->m_vecOrigin().z - pLocal->m_vecOrigin().z;
	float aYaws[3] = { flBaseYaw, 0.f, 0.f };
	int iDirs = 1;
	if (fabsf(flDz) <= TRICKSTAB_VERTICAL_BAND)
	{
		aYaws[0] = flBaseYaw - TRICKSTAB_FAN_HALF_DEG;
		aYaws[1] = flBaseYaw;
		aYaws[2] = flBaseYaw + TRICKSTAB_FAN_HALF_DEG;
		iDirs = 3;
	}
	const int iTicks = std::clamp(Vars::Doubletap::TickLimit.Value, TRICKSTAB_HORIZON_MIN, TRICKSTAB_HORIZON_MAX);
	DirResult_t aRes[3] = {};
	for (int i = 0; i < iDirs; i++)
	{
		if (!SimDir(pLocal, aYaws[i], iTicks, aRes[i]))
			continue;
		aRes[i].bHittable = ScoreDir(pLocal, pWeapon, pTarget, aRes[i]);
		aRes[i].flScore = (pTarget->m_vecOrigin() - aRes[i].vEnd).Length();
	}
	// Best: hittable first (nearest end), else nearest end overall.
	int iBest = -1;
	for (int iPass = 0; iPass < 2 && iBest < 0; iPass++)
	{
		float flS = std::numeric_limits<float>::max();
		for (int i = 0; i < iDirs; i++)
		{
			if (!aRes[i].bValid)
				continue;
			if (iPass == 0 && !aRes[i].bHittable)
				continue;
			if (aRes[i].flScore < flS)
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
	// Draw: all valid paths dim, best green (hittable) / yellow, end marker
	// + link to target. Short expiry: refreshed per tick, self-cleaning.
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
			G::LineStorage.emplace_back(std::pair<Vec3, Vec3>(aRes[i].vEnd, aRes[i].vEnd + Vec3(0, 0, 5)), flUntil, tCol, true);
			G::LineStorage.emplace_back(std::pair<Vec3, Vec3>(aRes[i].vEnd, pTarget->m_vecOrigin()), flUntil, Color_t(255, 255, 0), true);
		}
	}
}
