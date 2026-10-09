#pragma once
#include "../../../SDK/SDK.h"

// Auto trickstab, phase 1 (visualize only): every tick, simulate warp-fan
// directions with the ENGINE sim (MoveSim, same facility as the aimbot),
// score each by trickstab quality (ends behind the enemy in swing range,
// visible), keep the best, draw all paths + best. NO movement writes, NO
// firing, NO warp execution this phase.
// Lua technique (SimulateDash/optimal-wishdir family): direction of best pos,
// re-evaluated every tick; left/right fan, center-only when the target is
// below/above (stairs: lateral fan is useless).
class CAutoTrickstab
{
private:
	struct DirResult_t
	{
		Vec3 vDir = {};
		Vec3 vEnd = {};
		std::vector<Vec3> vPath = {};
		bool bValid = false;
		bool bHittable = false;
		float flScore = 0.f;
	};

	// Engine sim along a world yaw for iTicks. Full restore on ALL paths.
	// Returns false when the sim failed/stalled (direction unusable for warp).
	bool SimDir(CTFPlayer* pLocal, float flYawDeg, int iTicks, DirResult_t& tOut);
	// Trickstab quality at sim end vs frozen target (CanBackstab core shape:
	// behind-dot + swing range + visibility). Pure math, no writes.
	bool ScoreDir(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CTFPlayer* pTarget, const DirResult_t& tDir);

public:
	void Run(CTFPlayer* pLocal, CTFWeaponBase* pWeapon);

	// Best direction this tick (for later phases: movement + warp execution).
	Vec3 m_vBestDir = {};
	bool m_bHasBest = false;
	bool m_bBestHittable = false;
};

ADD_FEATURE(CAutoTrickstab, AutoTrickstab);
