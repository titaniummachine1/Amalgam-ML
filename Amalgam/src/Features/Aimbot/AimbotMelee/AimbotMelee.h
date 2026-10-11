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
	// Scratch live-position record: contested in CanHit every tick so a
	// hittable live foe is never passed over for history. Per-target
	// scratch would dangle; one member reused sequentially is safe
	// because only the finally-chosen target's record is consumed.
	TickRecord m_tLiveRecord = {};

	int m_iSwingTicks = 0;
	int m_iSimulatedTicks = 0;
	int m_iDoubletapTicks = 0;

	enum class ChargeState { Idle, Tracking, Aim, Charge };
	ChargeState m_eChargeState = ChargeState::Idle;
	int m_iChargeTicks = 0;
	int m_iChargeTarget = -1;
	// M2 fired for THIS swing (Aim merged snap+press, then Idles): the
	// condition lands 1-2 ticks later, but range/sim must treat the swing
	// as charged from the trigger. Cleared when the swing ends; meter
	// gating (rules check meter > 0) auto-clears at charge end.
	bool m_bChargeFired = false;
	// Aim angles captured at swing press (arm). Reapplied on the aim-snap
	// tick so the smack resolves with the swing's aim (qawe-faithful).
	Vec3 m_vChargeAim = {};
	// Coverage hold: tickcount when holding the swing for bank completion
	// (0 = not holding). Instant damage needs bank >= smack delay.
	int m_iSwingHoldTick = 0;
	// Smack-countdown jump detector: last observed smack time + countdown.
	// A drop >1 between observations means compressed time (warp/DT dump),
	// and any precise window would fire late -> press now instead.
	float m_flLastSmackTime = -1.f;
	int m_iLastSmackTicks = 0;

	std::unordered_map<int, std::deque<TickRecord>> m_mRecordMap;
	std::unordered_map<int, std::vector<Vec3>> m_mPaths;
	bool m_bWarpSolving = false; // warp solution in flight: skip FOV gates

public:
	void Run(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CUserCmd* pCmd);
	// Firing solution FROM a warp-dest eye through the real firing path
	// (same gates, same angle math, same record choice as the swing that
	// will execute). True when verified: angle/pos/record written back.
	bool SolveWarpAim(Target_t& tTarget, CTFPlayer* pLocal, CTFWeaponBase* pWeapon, const Vec3& vEye, float flRangeOverride = -1.f);
	// Per-tick validation signal for the shift driver: true when this
	// tick's firing loop CanHit-validated a target (any result). Lets the
	// driver tell manual air-swings (no solution, never dump) apart from
	// combat swings (validated, dump). Reset at the top of every Run.
	bool m_bFiringSolution = false;
	// Coverage-gated frozen world: true only when the dump truly covers
	// the full smack window (pre-press: bank spans it and the dump will
	// fire; mid-swing: a dump is executing, batch-atomic). Short bank =
	// honest live prediction (foe simmed, records windowed) for a swing
	// that fires NOW with a partial instant head and a live tail.
	bool m_bDtValidated = false;
	// Range validated at press: mid-swing recomputation can only shrink
	// it while the swing still flies (dump covers: frozen world, the
	// validated reach is the truth). A live tail keeps recomputed range.
	float m_flPressRange = 0.f;
	// Latch: the in-flight swing was aimbot-initiated (validated press).
	// Separates our swings (failsafe may re-aim them) from manual swings
	// (wall tests etc. — never touch their angles). Set at press, cleared
	// whenever no smack is in flight (self-heals phantom presses too).
	bool m_bAimbotSwing = false;
	// Manual swings: shift tick_count onto the closest valid record so
	// un-aimed hits still land on lag-compensated history.
	void BacktrackToCrosshair(CTFPlayer* pLocal, CUserCmd* pCmd);
private:
	std::pair<float, float> GetClosestRecord(std::vector<TickRecord*> vRecords);
public:
	// Charge-reach tracking/charge in flight: DT/warp must stand down or the
	// IN_ATTACK2 tick executes inside compressed shift-time (see Ticks).
	bool IsChargeBusy() const { return m_eChargeState != ChargeState::Idle; }
	// Arm charge tracking from a warp press (no aimbot target in hand).
	void ArmChargeForWarp(CUserCmd* pCmd, int iEntIndex);
};

ADD_FEATURE(CAimbotMelee, AimbotMelee);