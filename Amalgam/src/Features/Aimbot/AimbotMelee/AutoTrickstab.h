#pragma once
#include "../../../SDK/SDK.h"

// Auto trickstab: mirrors Auto_Trickstab (qawe) CalculateTrickstab path
// selection — best side by yaw-delta (Manual picker forces via game input,
// no input falls back to Automatic), optimal-side corner, other side always
// (axis-aligned, rotation-independent; matador lives here), center only when
// within backstab angle (stairstab: center only) — but with Amalgam-native
// upgrades, all frozen by spec:
//
// - Minkowski collision: we are a point, the target box absorbs both bodies
//   (expanded = theirMin-ourMax .. theirMax-ourMin, live hulls). Corners at
//   expanded + 1u graze padding (never touch, never slow). Contact break =
//   exact strict-inside test, 3D, 6 comparisons.
// - Wishdir: neutral ENGINE resolve (bank ticks, zero input) -> residual to
//   the live corner + 450 extension, applied unrotated at live pos. One
//   physics truth, linear cost, never N^2.
// - Scoring: gates unified with the aimbot firing path (CanBackstab dot core
//   + live weapon range + CanHit LOS decomposition, pure: no prediction side
//   effects), ranking by Lua weights (0.7 yaw + 0.3 distance, tie -> early).
// - Dual prediction: the enemy is extrapolated per tick (friction + gravity)
//   and every test runs against its predicted pose, not a static lead.
// - Assist: while the toggle holds, the best wishdir is planted into cmd
//   (view-relative, zero view writes; autostrafe routed when enabled).
// - Commit: exact-minTicks warp via public Ticks fields, AA frozen for the
//   commit ticks only, 3 live-sent back-aimed bonus ticks, recharge gates.
//
// NO firing here. The aimbot stays the firing authority.
class CAutoTrickstab
{
private:
	struct DirResult_t
	{
		Vec3 vDir = {}; // dash wishdir (yaw unit vector, world)
		Vec3 vEnd = {};
		std::vector<Vec3> vPath = {}; // sim feet origins
		bool bValid = false;
		bool bHitWall = false;
		bool bHittable = false;
		float flScore = -1.f; // path-best weighted score
		int iHitTick = -1;    // index into vPath of the path-best score
		int iHitCount = 0;
		std::vector<char> vHits = {}; // per-tick hit flags (viz dots)
		Vec3 vHitPos = {};  // feet at the path-best tick
		Vec3 vHitEye = {};  // eye at the path-best tick
		Vec3 vHitFoe = {};  // predicted enemy feet at the path-best tick
	};

	// Engine dash sim along a world yaw. Full restore on ALL paths. Stops AT
	// the opponent via the exact Minkowski predicate (never through).
	bool SimDir(CTFPlayer* pLocal, float flYawDeg, int iTicks, const Vec3& vLeadPos, const Vec3& vOMin, const Vec3& vOMax, const Vec3& vTMin, const Vec3& vTMax, DirResult_t& tOut);
	// Neutral resolve: bank ticks, zero input. Returns the resolved origin.
	bool ResolveNeutral(CTFPlayer* pLocal, int iTicks, Vec3& vOut);
	// Wishdir yaw from the residual (resolved -> live corner + 450).
	float Wishdir(const Vec3& vResolved, const Vec3& vMyPos, const Vec3& vLeadPos, const Vec3& vCorner);
	// Pure per-tick stab test (aimbot-unified gates). No side effects.
	bool StabAt(const Vec3& vSimEye, const Vec3& vFoePos, const Vec3& vFoeEye, float flFoeYaw, float flOwnerYaw, float flRange, const Vec3& vSwingMins, const Vec3& vSwingMaxs, CTFPlayer* pLocal, CTFPlayer* pTarget);
	// Weighted Lua ranking for one path. Fills path-best fields + per-tick
	// hit flags (for viz dots).
	float ScorePath(CTFPlayer* pLocal, CTFPlayer* pTarget, const std::vector<Vec3>& vPath, const std::vector<Vec3>& vFoePath, const Vec3& vLeadPos, const Vec3& vLeadEye, float flFoeYaw, float flOwnerYaw, float flRange, const Vec3& vSwingMins, const Vec3& vSwingMaxs, Vec3& vHitPosOut, Vec3& vHitEyeOut, Vec3& vHitFoeOut, int& iHitTickOut, int& iHitCountOut, std::vector<char>& vHitsOut);
	// Plant a world wishdir into cmd without touching view (view-relative
	// decompose against the final angles). Autostrafe routed when enabled.
	void PlantMoves(CTFPlayer* pLocal, CUserCmd* pCmd, const Vec3& vWishDir);

public:
	void Run(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CUserCmd* pCmd);
	bool IsCommitting() const { return m_bCommitting; }
	// Auto warp (non-knife melee): straight warp-to-hit sharing the commit
	// latch below. Latch/audit members double for both modes; the stab-only
	// path replay (orbit/behind) is bypassed for direct commits.
	bool IsWarpDirect() const { return m_bWarpDirect; }
	bool WarpPressDue();
	int WarpEnt() const { return m_iWarpEnt; }
	Vec3 WarpDest() const { return m_vWarpDest; }
	bool WarpWantsCharge() const { return m_bWarpCharge; }

	// Best direction this tick (assist steering + warp commit).
	Vec3 m_vBestDir = {};
	bool m_bHasBest = false;
	bool m_bBestHittable = false;
	int m_iTotalPoints = 0; // commit gate: >= 3
	int m_iMinTicks = -1;   // warp length (hit index + 1)

private:
	bool m_bCommitting = false;
	bool m_bWarpDirect = false; // direct warp-to-hit commit (no orbit/behind)
	int m_iWarpEnt = -1;        // warp target (commit lock + press stamping)
	Vec3 m_vWarpDest = {};      // warp arrival feet (travel goal)
	bool m_bWarpCharge = false; // sim assumed charge speeds: live arm follows
	int m_iWarpPressAt = 0;   // stopped-time ticks to hold the swing (Full; Normal 0)
	void WarpDirect(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CUserCmd* pCmd,
		const std::vector<CTFPlayer*>& vCands, float flRange, float flHull,
		const Vec3& vSwingMins, const Vec3& vSwingMaxs, const Vec3& vMyPos, int iBank);
	int m_iClaimTick = 0; // tickcount at claim (damage-since-claim check)
	Vec3 m_vCommitDir = {}; // frozen at claim: warp steering ignores fresh
	// selection AND manual input (assist positions, warp executes).
	int m_iCommitLeft = 0;
	int m_iCommitAge = 0;
	int m_iBonusLeft = 0;
};

ADD_FEATURE(CAutoTrickstab, AutoTrickstab);
