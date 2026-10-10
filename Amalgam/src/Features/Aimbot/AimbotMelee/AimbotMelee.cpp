#include "AimbotMelee.h"

#include "../Aimbot.h"
#include "AutoTrickstab.h"
#include "../AutoRocketJump/AutoRocketJump.h"
#include "../../../SDK/VarsMelee.h"
#include "../../Simulation/MovementSimulation/MovementSimulation.h"
#include "../../EnginePrediction/EnginePrediction.h"
#include "../../Ticks/Ticks.h"
#include "../../Visuals/Visuals.h"
#include "../../AntiCheatCompatibility/AntiCheatCompatibility.h"
#include "../../CritHack/CritHack.h"

static constexpr float kChargeReachDistance = 128.f;
static constexpr float kCritRefillCombatDistance = 500.f;
static constexpr float kCritRefillSafeDistance = 700.f;

static inline bool AimFriendlyBuilding(CBaseObject* pBuilding)
{
	if (!pBuilding->m_bMiniBuilding() && pBuilding->m_iUpgradeLevel() != 3 || pBuilding->m_iHealth() < pBuilding->m_iMaxHealth() || pBuilding->m_bHasSapper())
		return true;

	if (pBuilding->IsSentrygun())
	{
		int iShells, iMaxShells, iRockets, iMaxRockets; pBuilding->As<CObjectSentrygun>()->GetAmmoCount(iShells, iMaxShells, iRockets, iMaxRockets);
		if (iShells < iMaxShells || iRockets < iMaxRockets)
			return true;
	}

	return false;
}

static inline std::vector<Target_t> GetTargets(CTFPlayer* pLocal, CTFWeaponBase* pWeapon)
{
	std::vector<Target_t> vTargets;

	const Vec3 vLocalPos = F::Ticks.GetShootPos();
	const Vec3 vLocalAngles = I::EngineClient->GetViewAngles();

	// Rage killaura / trickstab commit: no FOV culling anywhere below,
	// hittability decides. Seeded straight from the center so the no-FOV
	// path never reads uninitialized FOV-function outputs (bone lookup
	// yields nothing when records are missing) and skips the per-enemy
	// bone loop. A latched trickstab commit counts as no-FOV so a small
	// AimFOV can never starve the exec target and break the stab.
	const bool bNoFov = Vars::Aimbot::Melee::Rage.Value
		|| F::AutoTrickstab.IsCommitting();

	if (Vars::Aimbot::General::Target.Value & Vars::Aimbot::General::TargetEnum::Players)
	{
		auto eGroup = !SDK::FriendlyFire() || Vars::Aimbot::General::Ignore.Value & Vars::Aimbot::General::IgnoreEnum::Team ? EntityEnum::PlayerEnemy : EntityEnum::PlayerAll;
		if (Vars::Aimbot::Melee::WhipTeam.Value &&
			!SDK::FriendlyFire() && SDK::AttribHookValue(0, "speed_buff_ally", pWeapon) > 0)
			eGroup = EntityEnum::PlayerAll;

		for (auto pEntity : H::Entities.GetGroup(eGroup))
		{
			if (F::AimbotGlobal.ShouldIgnore(pEntity, pLocal, pWeapon))
				continue;

			float flFOVTo; Vec3 vPos, vAngleTo;
			if (bNoFov)
			{
				vPos = pEntity->GetCenter();
				vAngleTo = Math::CalcAngle(vLocalPos, vPos);
				flFOVTo = Math::CalcFov(vLocalAngles, vAngleTo);
			}
			else if (!F::AimbotGlobal.PlayerBoneInFOV(pEntity->As<CTFPlayer>(), vLocalPos, vLocalAngles, flFOVTo, vPos, vAngleTo))
				continue;

			float flDistTo = vLocalPos.DistToSqr(vPos);
			bool bTeam = pEntity->m_iTeamNum() == pLocal->m_iTeamNum();
			int iPriority = F::AimbotGlobal.GetPriority(pEntity->entindex());
			if (bTeam && !SDK::FriendlyFire())
				iPriority = 0;
			vTargets.emplace_back(pEntity, TargetEnum::Player, vPos, vAngleTo, flFOVTo, flDistTo, iPriority);
		}
	}

	{
		auto eGroup = EntityEnum::Invalid;
		if (Vars::Aimbot::General::Target.Value & Vars::Aimbot::General::TargetEnum::Building)
			eGroup = EntityEnum::BuildingEnemy;
		bool bWrench = pWeapon->GetWeaponID() == TF_WEAPON_WRENCH, bSapper = SDK::AttribHookValue(0, "set_dmg_apply_to_sapper", pWeapon);
		if (Vars::Aimbot::Healing::AutoRepair.Value && (bWrench || bSapper))
			eGroup = eGroup != EntityEnum::Invalid ? EntityEnum::BuildingAll : EntityEnum::BuildingTeam;
		for (auto pEntity : H::Entities.GetGroup(eGroup))
		{
			if (F::AimbotGlobal.ShouldIgnore(pEntity, pLocal, pWeapon))
				continue;

			bool bTeam = pEntity->m_iTeamNum() == pLocal->m_iTeamNum();
			if (bTeam && (bWrench && !AimFriendlyBuilding(pEntity->As<CBaseObject>()) || bSapper && !pEntity->As<CBaseObject>()->m_bHasSapper()))
				continue;

			float flFOVTo; Vec3 vPos, vAngleTo;
			if (bNoFov)
			{
				vPos = pEntity->GetCenter();
				vAngleTo = Math::CalcAngle(vLocalPos, vPos);
				flFOVTo = Math::CalcFov(vLocalAngles, vAngleTo);
			}
			else if (!F::AimbotGlobal.EntityCenterInFOV(pEntity, vLocalPos, vLocalAngles, flFOVTo, vPos, vAngleTo))
				continue;

			int iPriority = 0;
			if (bTeam)
			{
				int iOwner = pEntity->As<CBaseObject>()->m_hBuilder().GetEntryIndex();
				switch (Vars::Aimbot::Healing::HealPriority.Value)
				{
				case Vars::Aimbot::Healing::HealPriorityEnum::PrioritizeFriends:
					if (iOwner == I::EngineClient->GetLocalPlayer() || H::Entities.IsFriend(iOwner) || H::Entities.InParty(iOwner))
						iPriority = std::numeric_limits<int>::max();
					break;
				case Vars::Aimbot::Healing::HealPriorityEnum::PrioritizeTeam:
					iPriority = std::numeric_limits<int>::max();
				}
			}

			float flDistTo = vLocalPos.DistToSqr(vPos);
			vTargets.emplace_back(pEntity, pEntity->IsSentrygun() ? TargetEnum::Sentry : pEntity->IsDispenser() ? TargetEnum::Dispenser : TargetEnum::Teleporter, vPos, vAngleTo, flFOVTo, flDistTo, iPriority);
		}
	}

	if (Vars::Aimbot::General::Target.Value & Vars::Aimbot::General::TargetEnum::NPCs)
	{
		for (auto pEntity : H::Entities.GetGroup(EntityEnum::WorldNPC))
		{
			if (F::AimbotGlobal.ShouldIgnore(pEntity, pLocal, pWeapon))
				continue;

			float flFOVTo; Vec3 vPos, vAngleTo;
			if (bNoFov)
			{
				vPos = pEntity->GetCenter();
				vAngleTo = Math::CalcAngle(vLocalPos, vPos);
				flFOVTo = Math::CalcFov(vLocalAngles, vAngleTo);
			}
			else if (!F::AimbotGlobal.EntityCenterInFOV(pEntity, vLocalPos, vLocalAngles, flFOVTo, vPos, vAngleTo))
				continue;

			float flDistTo = vLocalPos.DistToSqr(vPos);
			vTargets.emplace_back(pEntity, TargetEnum::NPC, vPos, vAngleTo, flFOVTo, flDistTo);
		}
	}

	return vTargets;
}



static inline int GetSwingTime(CTFWeaponBase* pWeapon, bool bVar = true)
{
	return pWeapon->GetWeaponID() == TF_WEAPON_KNIFE ? 0
		: bVar ? Vars::Aimbot::Melee::SwingTicks.Value
		: ceilf(pWeapon->GetSmackDelay() / TICK_INTERVAL);
}

void CAimbotMelee::UpdateInfo(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CUserCmd* pCmd, const std::vector<Target_t>& vTargets)
{
	m_mRecordMap.clear(); m_mPaths.clear();
	m_vEyePos = pLocal->GetShootPos();
	m_flRange = pWeapon->GetSwingRange();
	m_iSimulatedTicks = GetSwingTime(pWeapon), m_iSwingTicks = GetSwingTime(pWeapon, false);
	m_iDoubletapTicks = F::Ticks.GetTicks(pWeapon);

	// Charge-tracking swing timing (melee DT): while a charge-reach swing
	// is in flight, the sim window is the smack countdown, not the full
	// swing time. No new options: reuses ChargeReach + existing timing.
	const bool bChargeTrackingSwing = Vars::Aimbot::Melee::ChargeReach.Value
		&& pLocal->m_iClass() == TF_CLASS_DEMOMAN
		&& m_eChargeState == ChargeState::Tracking
		&& pWeapon->m_flSmackTime() > 0.f;
	if (bChargeTrackingSwing)
	{
		auto pNetChan = I::EngineClient->GetNetChannelInfo();
		const int iOutLatencyTicks = pNetChan ? std::max(TIME_TO_TICKS(std::max(pNetChan->GetLatency(FLOW_OUTGOING), 0.f)), 0) : 0;
		const int iChokeMargin = std::max(I::ClientState ? I::ClientState->chokedcommands : 0, 1);
		const int iChargeWindow = std::max(iOutLatencyTicks + iChokeMargin, 2);
		const int iTicksToSmack = TIME_TO_TICKS(std::max(pWeapon->m_flSmackTime() - I::GlobalVars->curtime, 0.f));
		m_iSimulatedTicks = std::max(iTicksToSmack, 1);
		m_iSwingTicks = std::max(iTicksToSmack - iChargeWindow, 1);
	}

	// Predict exactly to the smack, never past it. Instant smack (0 left)
	// skips the forward sim via the gate below and aims live/live.
	// Pre-swing the full delay is the horizon.
	const int iSmackLeft = pWeapon->m_flSmackTime() > 0.f
		? TIME_TO_TICKS(std::max(pWeapon->m_flSmackTime() - I::GlobalVars->curtime, 0.f))
		: m_iSwingTicks;
	m_iSimulatedTicks = std::min(m_iSimulatedTicks, iSmackLeft);

	m_bShouldSwing = m_iDoubletapTicks <= m_iSwingTicks || Vars::Doubletap::AntiWarp.Value && pLocal->m_hGroundEntity();
	m_bSimulatedLocal = false;

	if ((!Vars::Aimbot::Melee::SwingPrediction.Value || !m_iSimulatedTicks) && !m_iDoubletapTicks || !G::CanPrimaryAttack || (pWeapon->m_flSmackTime() > 0.f && !bChargeTrackingSwing))
	{
		m_iSimulatedTicks = m_iSwingTicks = 0;
		return;
	}

	std::unordered_map<int, MoveStorage> mMoveStorage;

	// Charging local: coast, not full AI. Locks the charge course (facing
	// yaw, full forward, charge speed cap) so predictions match the locked
	// ~750 straight instead of strafing fiction.
	const bool bCharging = pLocal->m_iClass() == TF_CLASS_DEMOMAN
		&& pLocal->InCond(TF_COND_SHIELD_CHARGE);
	F::MoveSim.Initialize(pLocal, mMoveStorage[pLocal->entindex()], false, !m_iDoubletapTicks && !bCharging, false);
	if (bCharging)
	{
		auto& tLocalStorage = mMoveStorage[pLocal->entindex()];
		const float flChargeSpeed = std::max(SDK::MaxSpeed(pLocal), 750.f);
		tLocalStorage.m_MoveData.m_flMaxSpeed = flChargeSpeed;
		tLocalStorage.m_MoveData.m_flClientMaxSpeed = flChargeSpeed;
		tLocalStorage.m_MoveData.m_flForwardMove = 450.f;
		tLocalStorage.m_MoveData.m_flSideMove = 0.f;
		tLocalStorage.m_MoveData.m_flUpMove = 0.f;
		tLocalStorage.m_MoveData.m_vecViewAngles = { 0.f, pCmd->viewangles.y, 0.f };
	}
	for (auto& tTarget : vTargets)
		F::MoveSim.Initialize(tTarget.m_pEntity, mMoveStorage[tTarget.m_pEntity->entindex()], false, true, false);

	int iMax = std::max(m_iSimulatedTicks, m_iDoubletapTicks), iTicks = iMax; bool bSwung = false;
	Vec3 vLocalOrigin = mMoveStorage[pLocal->entindex()].m_MoveData.m_vecAbsOrigin;
	for (int i = 0; i < iTicks; i++) // intended for plocal to collide with targets
	{
		{
			auto& tMoveStorage = mMoveStorage[pLocal->entindex()];
			if (!bSwung && (!m_iDoubletapTicks || Vars::Doubletap::AntiWarp.Value && pLocal->m_hGroundEntity() || iMax - i <= m_iSwingTicks))
			{
				iTicks = std::min(i + m_iSwingTicks, iMax), bSwung = true;
				if (!m_iSwingTicks)
					break;
				// (old strip-charge hack removed: charging is coast-simmed
				// above with zero live writes; stripping lied to both sim
				// and live state)
			}
			if (m_iDoubletapTicks && Vars::Doubletap::AntiWarp.Value && pLocal->m_hGroundEntity())
				F::Ticks.AntiWarp(pLocal, pCmd->viewangles.y, tMoveStorage.m_MoveData.m_flForwardMove, tMoveStorage.m_MoveData.m_flSideMove, iMax - i - 1);

			F::MoveSim.RunTick(tMoveStorage);

			vLocalOrigin = tMoveStorage.m_MoveData.m_vecAbsOrigin;
			m_bSimulatedLocal = true;
		}

		if (i < m_iSimulatedTicks - m_iDoubletapTicks)
		{
			for (auto& tTarget : vTargets)
			{
				auto& tMoveStorage = mMoveStorage[tTarget.m_pEntity->entindex()];
				if (tMoveStorage.m_bFailed)
					continue;

				F::MoveSim.RunTick(tMoveStorage);
				if (Vars::Aimbot::Melee::SwingPredictLag.Value && !tMoveStorage.m_bPredictNetworked)
					continue;

				Vec3 vOrigin = Vars::Aimbot::Melee::SwingPredictLag.Value ? tMoveStorage.m_vPredictedOrigin : tMoveStorage.m_MoveData.m_vecAbsOrigin;
				m_mRecordMap[tTarget.m_pEntity->entindex()].emplace_front(
					tTarget.m_pEntity->m_flSimulationTime() + TICKS_TO_TIME(i + 1), vOrigin, tTarget.m_pEntity->m_vecMins(), tTarget.m_pEntity->m_vecMaxs()
				);
			}
		}
	}
	m_vEyePos = vLocalOrigin + pLocal->m_vecViewOffset();
	m_flRange = pWeapon->GetSwingRange();

	// Extend melee range to charge-reach distance when Demoman has full charge and exploit is ready
	if (Vars::Aimbot::Melee::ChargeReach.Value &&
		pLocal->m_iClass() == TF_CLASS_DEMOMAN &&
		pLocal->m_flChargeMeter() >= 100.f &&
		!pLocal->InCond(TF_COND_SHIELD_CHARGE))
	{
		m_flRange = kChargeReachDistance;
	}

	if (Vars::Visuals::Prediction::SwingLines.Value && Vars::Visuals::Prediction::PlayerPath.Value)
	{
		for (auto& [iIndex, tMoveStorage] : mMoveStorage)
			m_mPaths[iIndex] = tMoveStorage.m_vPath;

		const bool bAlwaysDraw = !Vars::Aimbot::General::AutoShoot.Value || Vars::Debug::Info.Value;
		if (bAlwaysDraw)
		{
			G::LineStorage.clear();
			G::BoxStorage.clear();
			G::PathStorage.clear();

			for (auto& vPath : m_mPaths | std::views::values)
			{
				float flDuration = Vars::Visuals::Prediction::PlayerDrawDuration.Value;
				if (Vars::Colors::PlayerPathIgnoreZ.Value.a)
					G::PathStorage.emplace_back(vPath, !flDuration ? -int(vPath.size()) : I::GlobalVars->curtime + flDuration, Vars::Colors::PlayerPathIgnoreZ.Value, Vars::Visuals::Prediction::PlayerPath.Value);
				if (Vars::Colors::PlayerPath.Value.a)
					G::PathStorage.emplace_back(vPath, !flDuration ? -int(vPath.size()) : I::GlobalVars->curtime + flDuration, Vars::Colors::PlayerPath.Value, Vars::Visuals::Prediction::PlayerPath.Value, true);
			}
		}
	}

	for (auto& tMoveStorage : mMoveStorage | std::views::values)
		F::MoveSim.Restore(tMoveStorage);
}

bool CAimbotMelee::CanBackstab(CBaseEntity* pTarget, CTFPlayer* pLocal, Vec3 vEyeAngles)
{
	if (!pTarget->IsPlayer() || pTarget->m_iTeamNum() == pLocal->m_iTeamNum())
		return false;

	if (Vars::Aimbot::Melee::IgnoreRazorback.Value)
	{
		CUtlVector<CBaseEntity*> itemList;
		int iBackstabShield = SDK::AttribHookValue(0, "set_blockbackstab_once", pTarget, &itemList);
		if (iBackstabShield && itemList.Count())
		{
			CBaseEntity* pEntity = itemList.Element(0);
			if (pEntity && pEntity->ShouldDraw())
				return false;
		}
	}

	Vec3 vEyePos = m_vEyePos;
	const float flCompDist = PLAYER_ORIGIN_COMPRESSION / 2;
	const float flSqCompDist = 0.0884f;

	if (auto pCmd = G::CurrentUserCmd;
		!m_bSimulatedLocal && pCmd->viewangles != vEyeAngles && G::CanPrimaryAttack)
	{	// repredict, prevent prediction error potentially causing miss
		CUserCmd tOldCmd = *pCmd;
		Vec3 vOldAngles = I::EngineClient->GetViewAngles();
		int iOldAttacking = G::Attacking;
		bool bOldSilent = G::PSilentAngles;
		F::EnginePrediction.End(pLocal, pCmd);

		G::Attacking = true;
		Aim(pCmd, vEyeAngles);
		F::Ticks.Start(pLocal, pCmd);
		vEyePos = pLocal->GetShootPos();
		F::EnginePrediction.End(pLocal, pCmd);

		*pCmd = tOldCmd;
		I::EngineClient->SetViewAngles(vOldAngles);
		G::Attacking = iOldAttacking;
		G::PSilentAngles = bOldSilent;
		F::Ticks.Start(pLocal, pCmd);
	}

	Vec3 vToTarget = (pTarget->GetAbsOrigin() - vEyePos).To2D();
	const float flDist = vToTarget.Normalize();
	if (flDist < flSqCompDist)
		return false;

	const float flExtra = 2.f * flCompDist / flDist; // account for origin compression
	float flPosVsTargetViewMinDot = 0.f + 0.0031f + flExtra;
	float flPosVsOwnerViewMinDot = 0.5f + flExtra;
	float flViewAnglesMinDot = -0.3f + 0.0031f; // 0.00306795676297 ?

	auto fTestDots = [&](Vec3 vTargetAngles)
	{
		Vec3 vOwnerForward; Math::AngleVectors(vEyeAngles, &vOwnerForward);
		vOwnerForward.Normalize2D();

		Vec3 vTargetForward; Math::AngleVectors(vTargetAngles, &vTargetForward);
		vTargetForward.Normalize2D();

		const float flPosVsTargetViewDot = vToTarget.Dot(vTargetForward); // Behind?
		const float flPosVsOwnerViewDot = vToTarget.Dot(vOwnerForward); // Facing?
		const float flViewAnglesDot = vTargetForward.Dot(vOwnerForward); // Facestab?

		return flPosVsTargetViewDot > flPosVsTargetViewMinDot && flPosVsOwnerViewDot > flPosVsOwnerViewMinDot && flViewAnglesDot > flViewAnglesMinDot;
	};

	Vec3 vTargetAngles = { 0.f, H::Entities.GetEyeAngles(pTarget->entindex()).y, 0.f };
	bool bStab = false;
	if (!(Vars::Aimbot::Melee::BackstabFlags.Value & Vars::Aimbot::Melee::BackstabFlagsEnum::AccountPing))
		bStab = fTestDots(vTargetAngles);
	else
	{
		if (Vars::Aimbot::Melee::BackstabFlags.Value & Vars::Aimbot::Melee::BackstabFlagsEnum::DoubleTest && !fTestDots(vTargetAngles))
			bStab = false;
		else
		{
			vTargetAngles.y += H::Entities.GetDeltaAngles(pTarget->entindex()).y;
			bStab = fTestDots(vTargetAngles);
		}
	}
	if (!bStab && Vars::Aimbot::Melee::AutoTrickstab.Value && F::AutoTrickstab.IsCommitting())
	{
		// Commit ghost second chance: the warp may arrive behind a backtrack
		// record while the foe's LIVE facing vetoes the dots (spun target).
		// The swing's tick_count rewinds to records, so record dots are the
		// truth — test the newest valid record yaw before refusing the stab.
		std::vector<TickRecord*> vRecsB;
		if (F::Backtrack.GetRecords(pTarget, vRecsB))
		{
			vRecsB = F::Backtrack.GetValidRecords(vRecsB, nullptr);
			if (!vRecsB.empty() && !vRecsB.front()->m_bInvalid)
				bStab = fTestDots({ 0.f, vRecsB.front()->m_vEyeAngles.y, 0.f });
		}
	}

	return bStab;
}

int CAimbotMelee::CanHit(Target_t& tTarget, CTFPlayer* pLocal, CTFWeaponBase* pWeapon)
{
	if (Vars::Aimbot::General::Ignore.Value & Vars::Aimbot::General::IgnoreEnum::Unsimulated && H::Entities.GetChoke(tTarget.m_pEntity->entindex()) > Vars::Aimbot::General::TickTolerance.Value)
		return false;

	// No-FOV aim: rage killaura or a latched trickstab commit. Same
	// condition as target selection so the exec target always flows
	// through and the stab can't be starved by a small AimFOV.
	const bool bNoFov = Vars::Aimbot::Melee::Rage.Value
		|| F::AutoTrickstab.IsCommitting();

	float flRange = SDK::AttribHookValue(m_flRange, "melee_range_multiplier", pWeapon);
	float flHull = SDK::AttribHookValue(18, "melee_bounds_multiplier", pWeapon);
	if (pLocal->m_flModelScale() > 1.0f)
	{
		flRange *= pLocal->m_flModelScale();
		flHull *= pLocal->m_flModelScale();
	}
	if (pWeapon->GetWeaponID() == TF_WEAPON_WRENCH && tTarget.m_pEntity->m_iTeamNum() == pLocal->m_iTeamNum())
	{
		flRange = 70;
		flHull = 18;
	}
	Vec3 vSwingMins = { -flHull, -flHull, -flHull };
	Vec3 vSwingMaxs = { flHull, flHull, flHull };
	auto& vSimRecords = m_mRecordMap[tTarget.m_pEntity->entindex()];

	std::vector<TickRecord*> vRecords = {};
	if (F::Backtrack.GetRecords(tTarget.m_pEntity, vRecords))
	{
		std::vector<int> vTimeMods = {};
		switch (Vars::Aimbot::Melee::SwingValidateMode.Value)
		{
		case Vars::Aimbot::Melee::SwingValidateModeEnum::Both:
			if (m_iSimulatedTicks != m_iSwingTicks)
				vTimeMods.push_back(m_iSimulatedTicks);
			[[fallthrough]];
		case Vars::Aimbot::Melee::SwingValidateModeEnum::Swing:
			vTimeMods.push_back(m_iSwingTicks);
			break;
		case Vars::Aimbot::Melee::SwingValidateModeEnum::Simulated:
			vTimeMods.push_back(m_iSimulatedTicks);
			break;
		}

		for (auto& tRecord : vSimRecords)
			vRecords.push_back(&tRecord);
		for (auto iTimeMod : vTimeMods)
			vRecords = F::Backtrack.GetValidRecords(vRecords, pLocal, true, -TICKS_TO_TIME(iTimeMod));
		if (vRecords.empty())
			return false;
	}
	else
	{
		F::Backtrack.m_tRecord = { tTarget.m_pEntity->m_flSimulationTime(), tTarget.m_pEntity->m_vecOrigin(), tTarget.m_pEntity->m_vecMins(), tTarget.m_pEntity->m_vecMaxs() };
		if (!tTarget.m_pEntity->SetupBones(F::Backtrack.m_tRecord.m_aBones, MAXSTUDIOBONES, BONE_USED_BY_ANYTHING, tTarget.m_pEntity->m_flSimulationTime()))
			return false;

		vRecords = { &F::Backtrack.m_tRecord };
	}

	// Execution eye: swings execute now, so firing solutions originate live.
	// The sim still predicts TARGETS to smack time (records below).
	const Vec3 vExecEye = pLocal->GetShootPos();
	CGameTrace trace = {};
	CTraceFilterHitscan filter = {};
	filter.pSkip = pLocal;

	const Vec3 vSelectAngle = tTarget.m_vAngleTo; // selection (bone) angle, pre-clamp
	// Record-independent gate: hoisted (identical outcome, runs once).
	// The clamp point on steep geometry can sit outside AimFOV while the
	// target is hittable. No-FOV aim skips it.
	if (!bNoFov && !F::AimbotGlobal.ShouldAimAtAngle(vSelectAngle))
		return false;
	const Vec3 vViewAngles = G::CurrentUserCmd->viewangles;
	const bool bKnifeStab = Vars::Aimbot::Melee::AutoBackstab.Value && pWeapon->GetWeaponID() == TF_WEAPON_KNIFE;
	// Backtrack-to-crosshair: EVERY valid record is tested; the hittable one
	// closest to the crosshair wins — not the nearest, not the newest. Ties
	// keep the earlier (fresher, deque-front) record via strict less-than,
	// which is the younger-ticks-are-stabler rule made explicit.
	TickRecord* pBest = nullptr;
	Vec3 vBestPos = {}, vBestAngle = {};
	float flBestFov = FLT_MAX;
	TickRecord* pFall = nullptr;
	Vec3 vFallPos = {}, vFallAngle = {};
	for (auto pRecord : vRecords)
	{
		// Closest point on the target AABB to our eye. Point-blank (eye
		// inside the box) degenerates to a zero-vector angle: keep the
		// selection bone pos instead. Knife keeps center x/y: backstab
		// detection needs center-based angles, that path is untouched.
		const Vec3 vClamp = vExecEye.Clamp(pRecord->m_vOrigin + pRecord->m_vMins, pRecord->m_vOrigin + pRecord->m_vMaxs);
		if (vClamp.DistToSqr(vExecEye) >= 1.f)
			tTarget.m_vPos = vClamp;
			if (bKnifeStab)
				tTarget.m_vPos.x = pRecord->m_vOrigin.x, tTarget.m_vPos.y = pRecord->m_vOrigin.y;
			Aim(G::CurrentUserCmd->viewangles, Math::CalcAngle(vExecEye, tTarget.m_vPos), tTarget.m_vAngleTo);

		Vec3 vRestoreOrigin = tTarget.m_pEntity->GetAbsOrigin();
		Vec3 vRestoreMins = tTarget.m_pEntity->m_vecMins();
		Vec3 vRestoreMaxs = tTarget.m_pEntity->m_vecMaxs();

		tTarget.m_pEntity->SetAbsOrigin(pRecord->m_vOrigin);
		tTarget.m_pEntity->m_vecMins() = pRecord->m_vMins + PLAYER_ORIGIN_COMPRESSION;
		tTarget.m_pEntity->m_vecMaxs() = pRecord->m_vMaxs - PLAYER_ORIGIN_COMPRESSION;

		Vec3 vForward; Math::AngleVectors(tTarget.m_vAngleTo, &vForward);
		Vec3 vTraceEnd = vExecEye + vForward * flRange;

		SDK::TraceHull(m_vEyePos, vTraceEnd, {}, {}, MASK_SOLID, &filter, &trace);
		bool bReturn = trace.m_pEnt == tTarget.m_pEntity;
		if (!bReturn)
		{
			SDK::TraceHull(m_vEyePos, vTraceEnd, vSwingMins, vSwingMaxs, MASK_SOLID, &filter, &trace);
			bReturn = trace.m_pEnt == tTarget.m_pEntity;
		}

		tTarget.m_pEntity->SetAbsOrigin(vRestoreOrigin);
		tTarget.m_pEntity->m_vecMins() = vRestoreMins;
		tTarget.m_pEntity->m_vecMaxs() = vRestoreMaxs;

		if (bReturn)
		{
			const float flFov = Math::CalcAngle(vExecEye, tTarget.m_vPos).DeltaAngle(vViewAngles).Length2DSqr();
			if (flFov < flBestFov)
			{
				// Knife verified only on improvement (cheaper than before:
				// the repredict inside runs only for a new leader).
				if (bKnifeStab && !CanBackstab(tTarget.m_pEntity, pLocal, tTarget.m_vAngleTo))
					continue;

				flBestFov = flFov;
				pBest = pRecord;
				vBestPos = tTarget.m_vPos;
				vBestAngle = tTarget.m_vAngleTo;
			}
		}
		else switch (Vars::Aimbot::General::AimType.Value)
		{
		case Vars::Aimbot::General::AimTypeEnum::Smooth:
		case Vars::Aimbot::General::AimTypeEnum::Assistive:
		{
			if (pFall)
				break;

			auto vAngle = Math::CalcAngle(vExecEye, tTarget.m_vPos);

			Math::AngleVectors(vAngle, &vForward);
			vTraceEnd = vExecEye + vForward * flRange;

			SDK::TraceHull(m_vEyePos, vTraceEnd, vSwingMins, vSwingMaxs, MASK_SOLID, &filter, &trace);
			if (trace.m_pEnt == tTarget.m_pEntity)
			{
				pFall = pRecord;
				vFallPos = tTarget.m_vPos;
				vFallAngle = vAngle;
			}
		}
		}
	}

	// Sweep fallback (Rijin height sweep, crosshair-ranked): when the clamp
	// point finds nothing, walk the box feet->head. Entity moved once per
	// record; traces per candidate; the winner alone gets the full knife
	// verify. Zero common-case cost: runs only when the cheap path missed.
	if (!pBest)
	{
		for (auto pRecord : vRecords)
		{
			Vec3 vRestoreOrigin = tTarget.m_pEntity->GetAbsOrigin();
			Vec3 vRestoreMins = tTarget.m_pEntity->m_vecMins();
			Vec3 vRestoreMaxs = tTarget.m_pEntity->m_vecMaxs();

			tTarget.m_pEntity->SetAbsOrigin(pRecord->m_vOrigin);
			tTarget.m_pEntity->m_vecMins() = pRecord->m_vMins + PLAYER_ORIGIN_COMPRESSION;
			tTarget.m_pEntity->m_vecMaxs() = pRecord->m_vMaxs - PLAYER_ORIGIN_COMPRESSION;

			for (float flMod = 0.f; flMod <= 1.f; flMod += 0.05f)
			{
				const Vec3 vPoint = pRecord->m_vOrigin + (pRecord->m_vMins + pRecord->m_vMaxs) * flMod;
				const Vec3 vAngle = Math::CalcAngle(vExecEye, vPoint);
				if (!bNoFov && !F::AimbotGlobal.ShouldAimAtAngle(vAngle))
					continue;

				Vec3 vForward; Math::AngleVectors(vAngle, &vForward);
				Vec3 vTraceEnd = vExecEye + vForward * flRange;

				SDK::TraceHull(m_vEyePos, vTraceEnd, {}, {}, MASK_SOLID, &filter, &trace);
				bool bHit = trace.m_pEnt == tTarget.m_pEntity;
				if (!bHit)
				{
					SDK::TraceHull(m_vEyePos, vTraceEnd, vSwingMins, vSwingMaxs, MASK_SOLID, &filter, &trace);
					bHit = trace.m_pEnt == tTarget.m_pEntity;
				}
				if (!bHit)
					continue;

				const float flFov = vAngle.DeltaAngle(vViewAngles).Length2DSqr();
				if (flFov < flBestFov)
				{
					if (bKnifeStab && !CanBackstab(tTarget.m_pEntity, pLocal, vAngle))
						continue;

					flBestFov = flFov;
					pBest = pRecord;
					vBestPos = vPoint;
					vBestAngle = vAngle;
				}
			}

			tTarget.m_pEntity->SetAbsOrigin(vRestoreOrigin);
			tTarget.m_pEntity->m_vecMins() = vRestoreMins;
			tTarget.m_pEntity->m_vecMaxs() = vRestoreMaxs;
		}
	}

	if (pBest)
	{
		tTarget.m_vPos = vBestPos;
		tTarget.m_vAngleTo = vBestAngle;
		tTarget.m_pRecord = pBest;
		tTarget.m_bBacktrack = tTarget.m_iTargetType == TargetEnum::Player;

		return true;
	}

	if (pFall)
	{
		tTarget.m_vPos = vFallPos;
		tTarget.m_vAngleTo = vFallAngle;
		tTarget.m_pRecord = pFall;

		return 2;
	}

	return false;
}



bool CAimbotMelee::Aim(const Vec3& vCurAngle, const Vec3& vToAngle, Vec3& vOut, int iMethod)
{
	/*
	if (Vec3* pHoldAngle = F::Ticks.GetShootAngle())
	{
		vOut = *pHoldAngle;
		return true;
	}
	*/

	bool bReturn = false;
	switch (iMethod)
	{
	case Vars::Aimbot::General::AimTypeEnum::Plain:
	case Vars::Aimbot::General::AimTypeEnum::Silent:
	case Vars::Aimbot::General::AimTypeEnum::Locking:
		vOut = vToAngle;
		break;
	case Vars::Aimbot::General::AimTypeEnum::Smooth:
		vOut = vCurAngle.LerpAngle(vToAngle, Vars::Aimbot::General::AssistStrength.Value / 100.f);
		bReturn = true;
		break;
	case Vars::Aimbot::General::AimTypeEnum::Assistive:
		Vec3 vMouseDelta = G::CurrentUserCmd->viewangles.DeltaAngle(G::LastUserCmd->viewangles);
		Vec3 vTargetDelta = vToAngle.DeltaAngle(G::LastUserCmd->viewangles);
		float flMouseDelta = vMouseDelta.Length2DSqr(), flTargetDelta = vTargetDelta.Length2DSqr();
		vTargetDelta = vTargetDelta.Normalized() * sqrtf(std::min(flMouseDelta, flTargetDelta));
		vOut = vCurAngle - vMouseDelta + vMouseDelta.LerpAngle(vTargetDelta, Vars::Aimbot::General::AssistStrength.Value / 100.f);
		bReturn = true;
		break;
	}

	if (iMethod != Vars::Aimbot::General::AimTypeEnum::Silent || F::AntiCheatCompatibility.Active())
		Math::ClampAngles(vOut);
	return bReturn;
}

// assume angle calculated outside with other overload
void CAimbotMelee::Aim(CUserCmd* pCmd, Vec3& vAngles, int iMethod)
{
	bool bUnsure = F::Ticks.IsTimingUnsure();
	switch (iMethod)
	{
	case Vars::Aimbot::General::AimTypeEnum::Plain:
		if (G::Attacking != 1 && !bUnsure)
			break;
		[[fallthrough]];
	case Vars::Aimbot::General::AimTypeEnum::Smooth:
	case Vars::Aimbot::General::AimTypeEnum::Assistive:
		pCmd->viewangles = vAngles;
		I::EngineClient->SetViewAngles(vAngles);
		break;
	case Vars::Aimbot::General::AimTypeEnum::Silent:
		if (G::Attacking == 1 || bUnsure)
		{
			SDK::FixMovement(pCmd, vAngles);
			pCmd->viewangles = vAngles;
			G::PSilentAngles = true;
		}
		break;
	case Vars::Aimbot::General::AimTypeEnum::Locking:
		SDK::FixMovement(pCmd, vAngles);
		pCmd->viewangles = vAngles;
		G::SilentAngles = true;
	}
}

static inline void DrawVisuals(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CUserCmd* pCmd, Target_t& tTarget, std::unordered_map<int, std::vector<Vec3>>& mPaths)
{
	G::AimTarget = { tTarget.m_pEntity->entindex(), I::GlobalVars->tickcount };
	G::AimPoint = { tTarget.m_vPos, I::GlobalVars->tickcount };

	bool bPath = Vars::Visuals::Prediction::SwingLines.Value && Vars::Visuals::Prediction::PlayerPath.Value;
	//bool bLine = Vars::Visuals::Line::TracersEnabled.Value;
	bool bBoxes = Vars::Visuals::Hitbox::BonesEnabled.Value & Vars::Visuals::Hitbox::BonesEnabledEnum::OnShot;
	bool bRealPath = Vars::Visuals::Prediction::RealPath.Value;
	if (bPath /*|| bLine*/ || bBoxes || bRealPath)
	{
		if (pCmd->buttons & IN_ATTACK && G::CanPrimaryAttack && pWeapon->m_flSmackTime() < 0.f)
		{
			G::LineStorage.clear();
			G::BoxStorage.clear();
			G::PathStorage.clear();

			if (bPath)
			{
				float flDuration = Vars::Visuals::Prediction::PlayerDrawDuration.Value;
				if (Vars::Colors::PlayerPathIgnoreZ.Value.a)
				{
					G::PathStorage.emplace_back(mPaths[pLocal->entindex()], !flDuration ? -int(mPaths[pLocal->entindex()].size()) : I::GlobalVars->curtime + flDuration, Vars::Colors::PlayerPathIgnoreZ.Value, Vars::Visuals::Prediction::PlayerPath.Value);
					G::PathStorage.emplace_back(mPaths[tTarget.m_pEntity->entindex()], !flDuration ? -int(mPaths[tTarget.m_pEntity->entindex()].size()) : I::GlobalVars->curtime + flDuration, Vars::Colors::PlayerPathIgnoreZ.Value, Vars::Visuals::Prediction::PlayerPath.Value);
				}
				if (Vars::Colors::PlayerPath.Value.a)
				{
					G::PathStorage.emplace_back(mPaths[pLocal->entindex()], !flDuration ? -int(mPaths[pLocal->entindex()].size()) : I::GlobalVars->curtime + flDuration, Vars::Colors::PlayerPath.Value, Vars::Visuals::Prediction::PlayerPath.Value, true);
					G::PathStorage.emplace_back(mPaths[tTarget.m_pEntity->entindex()], !flDuration ? -int(mPaths[tTarget.m_pEntity->entindex()].size()) : I::GlobalVars->curtime + flDuration, Vars::Colors::PlayerPath.Value, Vars::Visuals::Prediction::PlayerPath.Value, true);
				}
			}
			if (int iSwingTime = GetSwingTime(pWeapon, false); bRealPath && iSwingTime)
			{
				F::Aimbot.Store(pLocal, iSwingTime);
				F::Aimbot.Store(tTarget.m_pEntity, iSwingTime);
			}
		}
		if (G::Attacking == 1)
		{
			//if (bLine)
			//{
			//	Vec3 vEyePos = pLocal->GetShootPos();
			//	float flDist = vEyePos.DistTo(tTarget.m_vPos);
			//	Vec3 vForward; Math::AngleVectors(tTarget.m_vAngleTo, &vForward);

			//	if (Vars::Colors::LineIgnoreZ.Value.a)
			//		G::LineStorage.emplace_back(std::pair<Vec3, Vec3>(vEyePos, vEyePos + vForward * flDist), I::GlobalVars->curtime + Vars::Visuals::Line::DrawDuration.Value, Vars::Colors::LineIgnoreZ.Value);
			//	if (Vars::Colors::Line.Value.a)
			//		G::LineStorage.emplace_back(std::pair<Vec3, Vec3>(vEyePos, vEyePos + vForward * flDist), I::GlobalVars->curtime + Vars::Visuals::Line::DrawDuration.Value, Vars::Colors::Line.Value, true);
			//}
			if (bBoxes)
			{
				auto vBoxes = F::Visuals.GetHitboxes(tTarget.m_pRecord->m_aBones, tTarget.m_pEntity->As<CBaseAnimating>());
				G::BoxStorage.insert(G::BoxStorage.end(), vBoxes.begin(), vBoxes.end());

				//if (Vars::Colors::BoneHitboxEdgeIgnoreZ.Value.a || Vars::Colors::BoneHitboxFaceIgnoreZ.Value.a)
				//	G::BoxStorage.emplace_back(tTarget.m_pRecord->m_vOrigin, tTarget.m_pRecord->m_vMins, tTarget.m_pRecord->m_vMaxs, Vec3(), I::GlobalVars->curtime + Vars::Visuals::Hitbox::DrawDuration.Value, Vars::Colors::BoneHitboxEdgeIgnoreZ.Value, Vars::Colors::BoneHitboxFaceIgnoreZ.Value);
				//if (Vars::Colors::BoneHitboxEdge.Value.a || Vars::Colors::BoneHitboxFace.Value.a)
				//	G::BoxStorage.emplace_back(tTarget.m_pRecord->m_vOrigin, tTarget.m_pRecord->m_vMins, tTarget.m_pRecord->m_vMaxs, Vec3(), I::GlobalVars->curtime + Vars::Visuals::Hitbox::DrawDuration.Value, Vars::Colors::BoneHitboxEdge.Value, Vars::Colors::BoneHitboxFace.Value, true);
			}
		}
	}
}

std::pair<float, float> CAimbotMelee::GetClosestRecord(std::vector<TickRecord*> vRecords)
{
	float flSimTime = -1.f;
	float flMaxDist = 300.f;
	for (auto pRecord : vRecords)
	{
		float flDist = pRecord->m_vOrigin.DistTo(m_vEyePos);
		if (flDist < flMaxDist)
		{
			flMaxDist = flDist;
			flSimTime = pRecord->m_flSimTime;
		}
	}
	return { flMaxDist, flSimTime };
}

void CAimbotMelee::BacktrackToCrosshair(CTFPlayer* pLocal, CUserCmd* pCmd)
{
	float flBestSimTime = -1.f;
	float flMaxDist = 300.f;
	m_vEyePos = pLocal->GetShootPos();
	for (auto pEntity : H::Entities.GetGroup(EntityEnum::PlayerEnemy))
	{
		if (!pEntity->As<CTFPlayer>()->IsAlive()
			|| pEntity->As<CTFPlayer>()->IsAGhost()
			|| pEntity->As<CTFPlayer>()->IsInvulnerable()
			|| pEntity->GetAbsOrigin().DistTo(m_vEyePos) > 1000.f)
			continue;

		std::vector<TickRecord*> vRecords;
		if (!F::Backtrack.GetRecords(pEntity, vRecords))
			continue;

		vRecords = F::Backtrack.GetValidRecords(vRecords);
		auto tInfo = GetClosestRecord(vRecords);
		if (tInfo.first < flMaxDist)
		{
			flBestSimTime = tInfo.second;
			flMaxDist = tInfo.first;
		}
	}

	if (flBestSimTime == -1.f)
		return;

	pCmd->tick_count = TIME_TO_TICKS(flBestSimTime + F::Backtrack.GetFakeInterp());
}

void CAimbotMelee::Run(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CUserCmd* pCmd)
{
	// Auto trickstab warp-fan viz (phase 1: simulate + draw only, no writes).
	F::AutoTrickstab.Run(pLocal, pWeapon, pCmd);
	// Charge reach follow-up: fires IN_ATTACK2 in the latency-aware window
	// before smack. Runs before all early returns so the charge still
	// executes if the target leaves normal melee range mid-swing.
	if (Vars::Aimbot::Melee::ChargeReach.Value
		&& pLocal->m_iClass() == TF_CLASS_DEMOMAN && pWeapon->m_flSmackTime() > 0.f)
	{
		switch (m_eChargeState)
		{
		case ChargeState::Tracking:
			if (pLocal->InCond(TF_COND_SHIELD_CHARGE)
				|| pLocal->m_flChargeMeter() < 100.f)
			{
				m_eChargeState = ChargeState::Idle;
				m_iChargeTarget = -1;
			}
			else
			{
				m_iChargeTicks++;
				const int iTicksToSmack = TIME_TO_TICKS(std::max(pWeapon->m_flSmackTime() - I::GlobalVars->curtime, 0.f));
				// Full smack length for the elapsed-since-swing math below.
				const int iSmackFull = GetSwingTime(pWeapon, false);
				// Smack-domain trigger (qawe-faithful): Early = 2 ticks after
				// swing start; Late = 2 ticks before smack lands. The countdown
				// advances with EXECUTED game time, so press-tick dumps and
				// fakelag bursts can't desync it the way hook-run counting
				// could (a dump collapses the whole window into one observation
				// and the old counter never reached Early's press tick: the
				// "Early breaks charges" bug). Smack already landed is handled
				// by the outer gate (aborts, saves the meter).
				const bool bEarly = Vars::Aimbot::Melee::ChargeTiming.Value == Vars::Aimbot::Melee::ChargeTimingEnum::Early;
				const bool bPressNow = bEarly
					? (iSmackFull - iTicksToSmack >= 2)
					: (iTicksToSmack <= 2);
				if (bPressNow)
					m_eChargeState = ChargeState::Aim;
				else if (m_iChargeTicks > 25)
				{
					m_eChargeState = ChargeState::Idle;
					m_iChargeTarget = -1;
				}
			}
			break;
		case ChargeState::Aim:
			// Snap view to the press-time aim: the smack resolves with the
			// swing's angles, not drifted camera or stale backlog. Camera
			// rotates this tick (qawe-faithful); press follows next tick.
			if (pWeapon->m_flSmackTime() <= 0.f)
			{
				m_eChargeState = ChargeState::Idle;
				m_iChargeTarget = -1;
			}
			else
			{
				pCmd->viewangles.x = m_vChargeAim.x;
				pCmd->viewangles.y = m_vChargeAim.y;
				m_eChargeState = ChargeState::Charge;
			}
			break;
		case ChargeState::Charge:
			// One-shot press: mark consumed but HOLD the state one extra tick
			// so shift drivers (Doubletap/Warp, gated on busy) can't start a
			// fresh shift on the press tick and steal the tailored goal below.
			if (m_iChargeTarget == -1)
			{
				m_eChargeState = ChargeState::Idle;
				break;
			}
			pCmd->buttons |= IN_ATTACK2;
			G::SendPacket = true;
			// Collapse the remaining smack delay into the fresh charge: the
			// swing tail rides the shift, impact lands instantly mid-charge,
			// ranged. The press block can't run now (smack in flight), so
			// without this explicit latch the ride-along never fires. Never
			// inside a warp/stab commit: the commit owns the goal.
			if (!F::Ticks.m_bDoubletap && F::Ticks.GetTicks(pWeapon) > 0 && !F::AutoTrickstab.IsCommitting())
			{
				const int iTicksToSmack = TIME_TO_TICKS(std::max(pWeapon->m_flSmackTime() - I::GlobalVars->curtime, 0.f));
				F::Ticks.m_bDoubletap = true;
				F::Ticks.m_iShiftedGoal = F::Ticks.ReleaseGoal(std::max(iTicksToSmack, 1));
			}
			m_iChargeTarget = -1; // consumed; state stays Charge one extra tick
			break;
		case ChargeState::Idle:
		default:
			break;
		}
	}
	else if (pWeapon->m_flSmackTime() <= 0.f
		|| !Vars::Aimbot::Melee::ChargeReach.Value
		|| pLocal->m_iClass() != TF_CLASS_DEMOMAN)
	{
		// Full reset: no live sequence without the toggle, the class, and
		// a swing in flight. Otherwise a mid-track toggle-off strands a
		// busy state that suppresses every future instant swing.
		m_eChargeState = ChargeState::Idle;
		m_iChargeTarget = -1;
	}

	// Tail compression: swing already in flight with a long smack delay. The
	// head played in normal time (or via coverage hold); once the remaining
	// delay fits the available dump, shift through the tail so the smack
	// lands instantly. Self-limiting: the dump consumes the bank below the
	// condition. Skips trivial tails; Doubletap() validates downstream.
	// Stood down while a charge is sequenced: a fresh dump mid-track
	// collapses the swing window the sequence was timed against. The
	// Charge-press tick latches its own window-sized collapse instead.
	if (pWeapon->m_flSmackTime() > 0.f)
	{
		const int iTicksToSmack = TIME_TO_TICKS(std::max(pWeapon->m_flSmackTime() - I::GlobalVars->curtime, 0.f));
		if (iTicksToSmack >= 2 && iTicksToSmack <= F::Ticks.GetTicks(pWeapon) && !IsChargeBusy())
			F::Ticks.m_bDoubletap = true;
	}

	static int iStaticAimType = Vars::Aimbot::General::AimType.Value;
	const int iLastAimType = iStaticAimType;
	const int iRealAimType = Vars::Aimbot::General::AimType.Value;

	if (pWeapon->m_flSmackTime() > 0.f && !iRealAimType && iLastAimType)
		Vars::Aimbot::General::AimType.Value = iLastAimType;
	iStaticAimType = Vars::Aimbot::General::AimType.Value;

	if (F::AimbotGlobal.ShouldHoldAttack(pWeapon))
		pCmd->buttons |= IN_ATTACK;
	if (!Vars::Aimbot::General::AimType.Value
		|| !Vars::Aimbot::Melee::Rage.Value && !F::AutoTrickstab.IsCommitting() && !F::AimbotGlobal.ShouldAim() && pWeapon->m_flSmackTime() < 0.f)
		return;

	if (RunSapper(pLocal, pWeapon, pCmd))
		return;

	auto vTargets = F::AimbotGlobal.ManageTargets(GetTargets, pLocal, pWeapon, Vars::Aimbot::General::TargetSelectionEnum::Distance);
	if (vTargets.empty())
		return;

	//if (!G::AimTarget.m_iEntIndex)
	//	G::AimTarget = { vTargets.front().m_pEntity->entindex(), I::GlobalVars->tickcount, 0 };

	UpdateInfo(pLocal, pWeapon, pCmd, vTargets);

	// Crit refill gating: only farm when no close simulated combat exists.
	bool bHasCombatSimulation = false;
	// Plain distance safety first: any live, transmitted enemy within 700u
	// blocks the refill even when nothing was simulated yet (fresh fight,
	// culled targets), so farming never kicks in mid-fight. Dormant
	// enemies are skipped (stale origins would block forever).
	for (auto pEntity : H::Entities.GetGroup(EntityEnum::PlayerEnemy))
	{
		auto pEnemy = pEntity->As<CTFPlayer>();
		if (!pEnemy || !pEnemy->IsAlive() || pEnemy->IsAGhost() || pEnemy->IsDormant())
			continue;
		if (pLocal->GetAbsOrigin().DistTo(pEnemy->GetAbsOrigin()) <= kCritRefillSafeDistance)
		{
			bHasCombatSimulation = true;
			break;
		}
	}
	for (const auto& tTarget : vTargets)
	{
		if (tTarget.m_pEntity->m_iTeamNum() == pLocal->m_iTeamNum())
			continue;
		auto it = m_mRecordMap.find(tTarget.m_pEntity->entindex());
		if (it == m_mRecordMap.end() || it->second.empty())
			continue;
		if (pLocal->GetAbsOrigin().DistTo(tTarget.m_pEntity->GetAbsOrigin()) <= kCritRefillCombatDistance)
		{
			bHasCombatSimulation = true;
			break;
		}
	}

	if (Vars::Aimbot::Melee::CritRefill.Value && pWeapon->m_flSmackTime() < 0.f && !bHasCombatSimulation)
	{
		if (F::CritHack.GetAvailableCrits() < Vars::Aimbot::Melee::CritRefillAmount.Value && G::CanPrimaryAttack)
		{
			F::CritHack.m_bCritRefillActive = true;
			pCmd->buttons |= IN_ATTACK;
			return;
		}
	}
	F::CritHack.m_bCritRefillActive = false;

	// Charge-ready predicate (target-free half of the arm gate): lets the
	// press-tick dump stand down BEFORE Tracking arms, while the swing is
	// still decided. Evaluated at call time (captures by ref). Also excludes
	// a latched trickstab commit (mutual exclusion: the commit owns movement
	// and the warp would collapse the track window either way).
	auto ChargeWillArm = [&]() -> bool
	{
		return (pCmd->buttons & IN_ATTACK) && G::CanPrimaryAttack && pWeapon->m_flSmackTime() < 0.f
			&& Vars::Aimbot::Melee::ChargeReach.Value
			&& pLocal->m_iClass() == TF_CLASS_DEMOMAN
			&& pLocal->m_flChargeMeter() >= 100.f
			&& !pLocal->InCond(TF_COND_SHIELD_CHARGE)
			&& m_eChargeState == ChargeState::Idle
			&& (!(F::AutoTrickstab.IsCommitting() && !F::AutoTrickstab.IsWarpDirect()));
	};
	auto ArmChargeTracking = [&](const Target_t& tTarget)
	{
		if (ChargeWillArm())
		{
			m_eChargeState = ChargeState::Tracking;
			m_iChargeTicks = 0;
			m_iChargeTarget = tTarget.m_pEntity->entindex();
			m_vChargeAim = pCmd->viewangles;
		}
	};

	for (auto& tTarget : vTargets)
	{
		const auto iResult = CanHit(tTarget, pLocal, pWeapon);
		if (!iResult) continue;
		if (iResult == 2)
		{
			G::AimTarget = { tTarget.m_pEntity->entindex(), I::GlobalVars->tickcount, 0 };
			Aim(pCmd, tTarget.m_vAngleTo);
			ArmChargeTracking(tTarget);
			break;
		}

		if (Vars::Aimbot::General::AutoShoot.Value && pWeapon->m_flSmackTime() < 0.f)
		{
			// Instant damage needs the bank to span the whole smack delay.
			// Partial bank: hold the swing while fakelag banks up (max ~15
			// ticks), then fire anyway so strafers aren't missed. Empty bank
			// or full coverage: fire immediately (normal / instant swing).
			const int iBank = F::Ticks.m_iShiftedTicks;
			const bool bCover = iBank >= m_iSwingTicks;
			bool bPress = m_bShouldSwing || bCover;
			if (bPress && !bCover && iBank > 0)
			{
				if (!m_iSwingHoldTick)
					m_iSwingHoldTick = I::GlobalVars->tickcount;
				if (I::GlobalVars->tickcount - m_iSwingHoldTick <= 15)
					bPress = false;
				else
					m_iSwingHoldTick = 0;
			}
			else
				m_iSwingHoldTick = 0;
			if (bPress)
				pCmd->buttons |= IN_ATTACK;
			// Melee dump policy: bank >= smack window suffices (generic
			// GetTicks demands TickLimit and m_iWait refuses chained swings,
			// so banked re-presses would never go instant). No dump on a
			// charge-arming press: the swing then plays in normal time so
			// the sequence counts deterministically; impact still lands
			// instantly via the Charge-press collapse above.
			const bool bDumpUsable = Vars::Doubletap::Doubletap.Value
				&& !F::Ticks.m_bWarp && !F::Ticks.m_bRecharge && !F::Ticks.m_bSpeedhack
				&& !F::AutoRocketJump.IsRunning();
			const bool bDumpReady = m_iDoubletapTicks > 0
				|| (F::Ticks.m_iShiftedTicks >= m_iSwingTicks && bDumpUsable);
			if (bDumpReady && !ChargeWillArm() && !IsChargeBusy())
				F::Ticks.m_bDoubletap = true;
		}

		G::Attacking = SDK::IsAttacking(pLocal, pWeapon, pCmd, true);
		ArmChargeTracking(tTarget);
		if (G::Attacking == 1)
		{
			if (tTarget.m_bBacktrack)
				pCmd->tick_count = TIME_TO_TICKS(tTarget.m_pRecord->m_flSimTime + F::Backtrack.GetFakeInterp());
			// bug: fast old records seem to be progressively more unreliable ?
		}
		else
		{
			m_vEyePos = pLocal->GetShootPos();
		Aim(G::CurrentUserCmd->viewangles, Math::CalcAngle(m_vEyePos, tTarget.m_vPos), tTarget.m_vAngleTo);
		}
		DrawVisuals(pLocal, pWeapon, pCmd, tTarget, m_mPaths);

		Aim(pCmd, tTarget.m_vAngleTo);
		// Smack in flight (Silent/Locking): keep the lock written every tick
		// through the delay, recomputed above (CanHit ran this tick). DT
		// dumps replay pre-press backlog ticks; if those carry camera angles
		// the smack resolves unlocked -> systematic miss. Overshoot past the
		// smack is harmless once every tick is locked. Plain/Smooth/Assistive
		// keep their visible-aim contracts (camera owned by the user there).
		if (pWeapon->m_flSmackTime() > 0.f
			&& (Vars::Aimbot::General::AimType.Value == Vars::Aimbot::General::AimTypeEnum::Silent
			|| Vars::Aimbot::General::AimType.Value == Vars::Aimbot::General::AimTypeEnum::Locking))
		{
			SDK::FixMovement(pCmd, tTarget.m_vAngleTo);
			pCmd->viewangles = tTarget.m_vAngleTo;
			G::PSilentAngles = true;
		}
		break;
	}
	// Auto-warp press (Normal: with the claim; Full: arrival solved the same
	// way, server resolves the smack inside stopped time). Additive only,
	// after the hold logic so banking never vetoes it. Stamps AimTarget so
	// crosshair-backtrack yields the tick the claim already set.
	if (F::AutoTrickstab.WarpPressDue() && G::CanPrimaryAttack && pWeapon->m_flSmackTime() < 0.f)
	{
		pCmd->buttons |= IN_ATTACK;
		G::Attacking = SDK::IsAttacking(pLocal, pWeapon, pCmd, true);
		G::AimTarget = { F::AutoTrickstab.WarpEnt(), I::GlobalVars->tickcount, 0 };
		G::AimPoint = { F::AutoTrickstab.WarpDest() + pLocal->m_vecViewOffset(), I::GlobalVars->tickcount };
		if (F::AutoTrickstab.WarpWantsCharge())
			ArmChargeForWarp(pCmd, F::AutoTrickstab.WarpEnt());
	}
}

void CAimbotMelee::ArmChargeForWarp(CUserCmd* pCmd, int iEntIndex)
{
	if (m_eChargeState != ChargeState::Idle)
		return;
	m_eChargeState = ChargeState::Tracking;
	m_iChargeTicks = 0;
	m_iChargeTarget = iEntIndex;
	m_vChargeAim = pCmd->viewangles;
}

static inline int GetAttachment(CBaseObject* pBuilding, int i)
{
	int iAttachment = pBuilding->GetBuildPointAttachmentIndex(i);
	if (pBuilding->IsSentrygun() && pBuilding->m_iUpgradeLevel() > 1)
		iAttachment = 3; // idk why this is needed
	return iAttachment;
}
bool CAimbotMelee::FindNearestBuildPoint(CBaseObject* pBuilding, CTFPlayer* pLocal, Vec3& vPoint)
{
	bool bFoundPoint = false;

	m_vEyePos = pLocal->GetShootPos();
	static auto tf_obj_max_attach_dist = H::ConVars.FindVar("tf_obj_max_attach_dist");
	float flNearestPoint = tf_obj_max_attach_dist->GetFloat();

	for (int i = 0; i < pBuilding->GetNumBuildPoints(); i++)
	{
		Vector vOrigin;
		if (pBuilding->GetAttachment(GetAttachment(pBuilding, i), vOrigin))
		{
			if (!SDK::VisPos(pLocal, pBuilding, m_vEyePos, vOrigin))
				continue;

			float flDist = (vOrigin - pLocal->m_vecOrigin()).Length();
			if (flDist < flNearestPoint)
			{
				flNearestPoint = flDist;
				vPoint = vOrigin;
				bFoundPoint = true;
			}
		}
	}

	return bFoundPoint;
}

bool CAimbotMelee::RunSapper(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CUserCmd* pCmd)
{
	if (pWeapon->GetWeaponID() != TF_WEAPON_BUILDER)
		return false;

	const Vec3 vLocalPos = F::Ticks.GetShootPos();
	const Vec3 vLocalAngles = I::EngineClient->GetViewAngles();

	std::vector<Target_t> vTargets;
	for (auto pEntity : H::Entities.GetGroup(EntityEnum::BuildingEnemy))
	{
		if (F::AimbotGlobal.ShouldIgnore(pEntity, pLocal, pWeapon))
			continue;

		auto pBuilding = pEntity->As<CBaseObject>();
		if (pBuilding->m_bHasSapper() || !pBuilding->IsInValidTeam())
			continue;

		Vec3 vPoint;
		if (!FindNearestBuildPoint(pBuilding, pLocal, vPoint))
			continue;

		Vec3 vAngleTo = Math::CalcAngle(vLocalPos, vPoint);
		const float flFOVTo = Math::CalcFov(vLocalAngles, vAngleTo);
		const float flDistTo = vLocalPos.DistToSqr(vPoint);
		if (flFOVTo > Vars::Aimbot::General::AimFOV.Value)
			continue;

		vTargets.emplace_back(pBuilding, TargetEnum::Unknown, vPoint, vAngleTo, flFOVTo, flDistTo);
	}
	F::AimbotGlobal.SortTargetsPre(vTargets, Vars::Aimbot::General::TargetSelectionEnum::Distance);
	if (vTargets.empty())
		return true;

	auto& tTarget = vTargets.front();

	bool bShouldAim = true;
	if (Vars::Aimbot::General::AutoShoot.Value)
		pCmd->buttons |= IN_ATTACK;
	else
		bShouldAim = pCmd->buttons & IN_ATTACK;
	if (Vars::Aimbot::General::AimType.Value == Vars::Aimbot::General::AimTypeEnum::Silent)
		bShouldAim &= !I::ClientState->chokedcommands && F::Ticks.CanChoke(true);
		
	if (bShouldAim)
	{
		G::AimTarget = { tTarget.m_pEntity->entindex(), I::GlobalVars->tickcount };
		G::AimPoint = { tTarget.m_vPos, I::GlobalVars->tickcount };

		G::Attacking = true;

		Aim(pCmd->viewangles, Math::CalcAngle(m_vEyePos, tTarget.m_vPos), tTarget.m_vAngleTo);
		tTarget.m_vAngleTo.x = pCmd->viewangles.x; // we don't need to care about pitch
		Aim(pCmd, tTarget.m_vAngleTo);
	}

	return true;
}