#pragma once
#include "../../../SDK/SDK.h"

// Auto trickstab, phase 1 (visualize only): every tick, simulate warp-fan
// directions with the ENGINE sim (MoveSim, same facility as the aimbot),
// score each by earliest hittable tick (behind + range + visibility per path
// point), keep the best, draw all paths + best. NO movement writes, NO
// firing, NO warp execution this phase.
// LUT ported VERBATIM from Auto_Trickstab (qawe): relative-quadrant (-1/0/1
// per axis, 1u buffer, vertical gate zeroes it) -> corner subset table.
// SIMULATION is Amalgam engine instead of Lua math. Deliberate deviations
// (all noted): hull radii for hitbox radii (mapped fields), no TARGET_
// EXTENSION/coast pass (engine integrates internally), no manual A/D side
// override or yaw-delta side preference (movement-commitment phase), no
// TF2.* constant table (live sv_* via engine).
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
		int iHitTick = -1;
		int iHitCount = 0;
		Vec3 vHitPos = {};
		float flScore = 0.f;
	};

	// Engine sim along a world yaw for iTicks. Full restore on ALL paths.
	// Stops AT live opponents (combined-radii contact break, Lua-faithful:
	// warp skirts to behind, never through). Returns false when the sim
	// failed/stalled elsewhere (direction unusable for warp).
	bool SimDir(CTFPlayer* pLocal, float flYawDeg, int iTicks, const Vec3& vTargetPos, float flHitbox, DirResult_t& tOut);
	// Earliest hittable path index, mirroring Auto_Trickstab CanBackstabFromPos
	// (AABB closest-point + BACKSTAB_RANGE sphere + yaw-delta + LOS, geometry
	// first). Amalgam's own CanBackstab stays the authority for live firing
	// decisions; this ranks sim-future states only. Returns -1, counts hits.
	int ScorePath(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CTFPlayer* pTarget, const Vec3& vLeadPos, const std::vector<Vec3>& vPath, Vec3& vHitPosOut, int& iHitCountOut);

public:
	void Run(CTFPlayer* pLocal, CTFWeaponBase* pWeapon);

	// Best direction this tick (for later phases: movement + warp execution).
	Vec3 m_vBestDir = {};
	bool m_bHasBest = false;
	bool m_bBestHittable = false;
};

ADD_FEATURE(CAutoTrickstab, AutoTrickstab);
