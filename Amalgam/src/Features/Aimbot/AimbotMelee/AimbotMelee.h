#pragma once
#include "../../../SDK/SDK.h"

#include "../AimbotGlobal/AimbotGlobal.h"

class CAimbotMelee
{
private:
	void UpdateInfo(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CUserCmd* pCmd, const std::vector<Target_t>& vTargets);
	bool CanBackstab(CBaseEntity* pTarget, CTFPlayer* pLocal, Vec3 vEyeAngles);
	int CanHit(Target_t& tTarget, CTFPlayer* pLocal, CTFWeaponBase* pWeapon);
	
	bool Aim(const Vec3& vCurAngle, const Vec3& vToAngle, Vec3& vOut, int iMethod = Vars::Aimbot::General::AimType.Value);
	void Aim(CUserCmd* pCmd, Vec3& vAngles, int iMethod = Vars::Aimbot::General::AimType.Value);

	bool FindNearestBuildPoint(CBaseObject* pBuilding, CTFPlayer* pLocal, Vec3& vPoint);
	bool RunSapper(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CUserCmd* pCmd);

	Vec3 m_vEyePos = {};
	float m_flRange = 0.f;
	bool m_bShouldSwing = false;
	bool m_bSimulatedLocal = false;

	int m_iSwingTicks = 0;
	int m_iSimulatedTicks = 0;
	int m_iDoubletapTicks = 0;

	enum class ChargeState { Idle, Tracking, Charge };
	ChargeState m_eChargeState = ChargeState::Idle;
	int m_iChargeTicks = 0;
	int m_iChargeTarget = -1;

	std::unordered_map<int, std::deque<TickRecord>> m_mRecordMap;
	std::unordered_map<int, std::vector<Vec3>> m_mPaths;

public:
	void Run(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CUserCmd* pCmd);
	// Charge-reach tracking/charge in flight: DT/warp must stand down or the
	// IN_ATTACK2 tick executes inside compressed shift-time (see Ticks).
	bool IsChargeBusy() const { return m_eChargeState != ChargeState::Idle; }
};

ADD_FEATURE(CAimbotMelee, AimbotMelee);