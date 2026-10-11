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
#include "../../../Utils/DtTrace/DtTrace.h"

static constexpr float kChargeReachDistance = 128.f;
static constexpr float kCritRefillCombatDistance = 500.f;
static constexpr float kCritRefillSafeDistance = 700.f;
// Frozen-world ghost rejection: a record whose origin disagrees with the
// live origin by more than this never resolves (the server has no such
// position to rewind to usefully). 24u ~= several frames of fast movement;
// a truly frozen foe coincides exactly, any age.
static constexpr float kFrozenRecordEpsilon = 24.f;
// Frozen-world angle veto: a history-only winner diverging this far (deg,
// 2D) from the live-foe direction is a ghost, never a solution.
static constexpr float kFrozenAngleVetoDeg = 10.f;

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

	// Literal engine range, evaluated HERE (not after the early return):
	// drain ticks early-return on cooldown, and a post-return extension
	// would collapse 128→48 mid-flight — press validates long, drain
	// validates short, miss. 128 iff the swing genuinely resolves
	// charged: condition set, M2 already fired for this swing (condition
	// lands 1-2 ticks later), or the machine is sequencing one (fires 2
	// ticks after swing start). Idle + uncharged = base, always.
	if (Vars::Aimbot::Melee::ChargeReach.Value &&
		pLocal->m_iClass() == TF_CLASS_DEMOMAN &&
		(pLocal->InCond(TF_COND_SHIELD_CHARGE)
		|| (m_bChargeFired && pLocal->m_flChargeMeter() > 0.f)
		|| (m_eChargeState != ChargeState::Idle && pLocal->m_flChargeMeter() > 0.f)))
	{
		m_flRange = kChargeReachDistance;
	}
	// Press-hold: while a validated swing flies inside its dump, never
	// shrink the range below what the press validated (machine idles,
	// manual charge released, meter drained — the swing still flies and
	// its aim must keep tracking, not freeze). Undumped tails keep the
	// honest recomputed range instead.
	if (pWeapon->m_flSmackTime() > 0.f && F::Ticks.m_bDoubletap)
		m_flRange = std::max(m_flRange, m_flPressRange);
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

	// Coverage-gated frozen world: validation runs frozen ONLY when the
	// dump truly covers the full smack window. Pre-press: bank spans it
	// and the dump will fire (mirrors the arming gates below, so a swing
	// the dump can't take never validates frozen). Mid-swing: a dump is
	// executing for this swing, so the batch is atomic and no fresh
	// updates arrive inside it. Short bank = honest live prediction
	// instead: the swing fires NOW with a partial instant head and a live
	// tail, validated for the world it resolves in — never frozen-now and
	// live-later. (m_bDoubletap clears post-drain, so a partial dump's
	// live tail automatically re-validates live on following frames.)
	const bool bPreSwing = pWeapon->m_flSmackTime() < 0.f;
	const bool bDumpWillFire = Vars::Doubletap::Doubletap.Value && !F::Ticks.m_bSpeedhack
		&& !F::Ticks.m_bWarp && !F::AutoRocketJump.IsRunning()
		&& !(IsChargeBusy() && m_eChargeState != ChargeState::Charge);
	m_bDtValidated = bPreSwing
		? (bDumpWillFire && F::Ticks.m_iShiftedTicks >= m_iSwingTicks)
		: (Vars::Doubletap::Doubletap.Value && !F::Ticks.m_bSpeedhack && F::Ticks.m_bDoubletap);
	if (m_bDtValidated && bPreSwing)
		m_iSimulatedTicks = m_iSwingTicks = std::max(GetSwingTime(pWeapon, false), 1);

	m_bShouldSwing = m_iDoubletapTicks <= m_iSwingTicks || Vars::Doubletap::AntiWarp.Value && pLocal->m_hGroundEntity();
	m_bSimulatedLocal = false;

	if ((!Vars::Aimbot::Melee::SwingPrediction.Value || !m_iSimulatedTicks) && !m_iDoubletapTicks && !m_bDtValidated || !G::CanPrimaryAttack || (pWeapon->m_flSmackTime() > 0.f && !bChargeTrackingSwing))
	{
		m_iSimulatedTicks = m_iSwingTicks = 0;
		return;
	}

	std::unordered_map<int, MoveStorage> mMoveStorage;

	// Charging local: coast, not full AI. Locks the charge course (facing
	// yaw, full forward, charge speed cap) so predictions match the locked
	// ~750 straight instead of strafing fiction.
	const bool bCharging = pLocal->m_iClass() == TF_CLASS_DEMOMAN
		&& (pLocal->InCond(TF_COND_SHIELD_CHARGE)
		|| (m_bChargeFired && pLocal->m_flChargeMeter() > 0.f));
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
	// Frozen world (DT on): the foe is never simulated. Only we are.
	if (!m_bDtValidated)
		for (auto& tTarget : vTargets)
			F::MoveSim.Initialize(tTarget.m_pEntity, mMoveStorage[tTarget.m_pEntity->entindex()], false, true, false);

	int iMax = std::max(m_iSimulatedTicks, m_iDoubletapTicks), iTicks = iMax; bool bSwung = false;
	Vec3 vLocalOrigin = mMoveStorage[pLocal->entindex()].m_MoveData.m_vecAbsOrigin;
	// Resolve eye, not drain-end eye: the smack lands iSmackLeft ticks
	// from now, but the walk above runs the whole bank window. Aiming
	// from the overshoot aims past the foe whenever bank > delay.
	const int iResolveAt = std::clamp(iSmackLeft, 0, iMax);
	Vec3 vResolveOrigin = vLocalOrigin;
	for (int i = 0; i < iTicks; i++) // intended for plocal to collide with targets
	{
		// Local always simulates: we walk during the drain, so the resolve
		// eye is where we end up, not where we stand now.
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
			// AntiWarp mirrors the dump: dumped ticks execute with
			// AntiWarp-rewritten moves (reverse/zero/restore ≈ stopped
			// time), so the sim must walk the same pattern. Gate is
			// dump-POSSIBLE (validated), not dump-armed (doubletapTicks
			// is 0 pre-press, exactly when validation needs it) — else
			// validation aims from 60u of fictional walking. Exclusions
			// mirror live arming exactly: never under warp (teleport
			// owns position), trickstab commits (live doesn't arm
			// there), or speedhack (DT disarmed, live never applies it).
			const bool bSimAntiWarp = Vars::Doubletap::AntiWarp.Value && pLocal->m_hGroundEntity()
				&& !F::Ticks.m_bWarp && !F::Ticks.m_bSpeedhack && !F::AutoTrickstab.IsCommitting()
				&& !(IsChargeBusy() && m_eChargeState != ChargeState::Charge)
				&& !pLocal->InCond(TF_COND_SHIELD_CHARGE);
			if ((m_iDoubletapTicks || m_bDtValidated) && bSimAntiWarp)
				F::Ticks.AntiWarp(pLocal, pCmd->viewangles.y, tMoveStorage.m_MoveData.m_flForwardMove, tMoveStorage.m_MoveData.m_flSideMove, iMax - i - 1);

			F::MoveSim.RunTick(tMoveStorage);

			vLocalOrigin = tMoveStorage.m_MoveData.m_vecAbsOrigin;
			if (i + 1 == iResolveAt)
				vResolveOrigin = vLocalOrigin;
			m_bSimulatedLocal = true;
		}

		if (i < m_iSimulatedTicks - m_iDoubletapTicks && !m_bDtValidated)
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
	m_vEyePos = vResolveOrigin + pLocal->m_vecViewOffset();
	// Base range only here; the charge extension lives at UpdateInfo top
	// so it also holds on early-return drain ticks.
	m_flRange = pWeapon->GetSwingRange();

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

	// No-FOV aim: rage killaura, a latched trickstab commit, or a warp
	// solution being verified. Same condition as target selection so the
	// exec target always flows through and the stab can't be starved by
	// a small AimFOV.
	const bool bNoFov = Vars::Aimbot::Melee::Rage.Value
		|| F::AutoTrickstab.IsCommitting() || m_bWarpSolving;

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
		// Frozen world (DT drain): the foe has not moved. Time-filtering
		// records out mid-drain makes CanHit fail, aim stops updating,
		// player drifts, smack resolves at the wrong angle. So the age
		// filter is skipped — but age is replaced by a POSITION check: a
		// record whose origin disagrees with the live origin is a ghost
		// (history the server cannot rewind to usefully), never a solution.
		// A truly frozen foe coincides exactly, at any age.
		if (!m_bDtValidated)
			for (auto iTimeMod : vTimeMods)
				vRecords = F::Backtrack.GetValidRecords(vRecords, pLocal, true, -TICKS_TO_TIME(iTimeMod));
		else
		{
			const Vec3 vLiveOrigin = tTarget.m_pEntity->GetAbsOrigin();
			for (auto it = vRecords.begin(); it != vRecords.end();)
				if (!*it || (*it)->m_bInvalid || ((*it)->m_vOrigin - vLiveOrigin).Length() > kFrozenRecordEpsilon)
					it = vRecords.erase(it);
				else
					++it;
		}
	}
	// The live position always contests: it is the server truth, needs no
	// rewind, and a hittable live foe is never passed over for history.
	// Bones are borrowed from the newest record (melee traces hulls, not
	// bones); a full setup runs only when no record exists at all.
	m_tLiveRecord = { tTarget.m_pEntity->m_flSimulationTime(), tTarget.m_pEntity->m_vecOrigin(), tTarget.m_pEntity->m_vecMins(), tTarget.m_pEntity->m_vecMaxs() };
	if (!vRecords.empty())
		memcpy(m_tLiveRecord.m_aBones, vRecords.front()->m_aBones, sizeof(m_tLiveRecord.m_aBones));
	else if (!tTarget.m_pEntity->SetupBones(m_tLiveRecord.m_aBones, MAXSTUDIOBONES, BONE_USED_BY_ANYTHING, tTarget.m_pEntity->m_flSimulationTime()))
		return false;
	vRecords.insert(vRecords.begin(), &m_tLiveRecord);

	// Resolve eye: the smack lands after the swing delay (and inside a DT
	// dump, after the drain walk), so the firing solution must originate
	// at the simulated resolve eye. Validating from the live eye while
	// tracing from the sim eye tests a line the swing never takes — a
	// systematic miss whenever we move during the delay.
	const Vec3& vEye = m_vEyePos;
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
	// Live candidacy: the live record always contests, and when hittable
	// it wins outright (below) — a hittable live foe is never passed over
	// for history. Tracked at pass time (knife re-verified at override).
	bool bLiveHit = false;
	Vec3 vLivePos = {}, vLiveAngle = {};
	for (auto pRecord : vRecords)
	{
		// Box CENTER, not the closest corner: corner-grazing rays carry no
		// perturbation margin — client prediction clangs on the graze
		// while the server misses it by 5u (clang with no damage). Center
		// rays keep the full hull margin on every side, so eye/record
		// staleness is absorbed, not fatal. Point-blank (eye inside the
		// box) degenerates to a zero-vector angle: keep the selection
		// bone pos instead. Center-based angles also satisfy the knife
		// backstab path (2D dots), so no special case is needed.
		const Vec3 vCenter = pRecord->m_vOrigin + (pRecord->m_vMins + pRecord->m_vMaxs) * 0.5f;
		if (vCenter.DistToSqr(vEye) >= 1.f)
			tTarget.m_vPos = vCenter;
		Aim(G::CurrentUserCmd->viewangles, Math::CalcAngle(vEye, tTarget.m_vPos), tTarget.m_vAngleTo);

		Vec3 vRestoreOrigin = tTarget.m_pEntity->GetAbsOrigin();
		Vec3 vRestoreMins = tTarget.m_pEntity->m_vecMins();
		Vec3 vRestoreMaxs = tTarget.m_pEntity->m_vecMaxs();

		tTarget.m_pEntity->SetAbsOrigin(pRecord->m_vOrigin);
		tTarget.m_pEntity->m_vecMins() = pRecord->m_vMins + PLAYER_ORIGIN_COMPRESSION;
		tTarget.m_pEntity->m_vecMaxs() = pRecord->m_vMaxs - PLAYER_ORIGIN_COMPRESSION;

		Vec3 vForward; Math::AngleVectors(tTarget.m_vAngleTo, &vForward);
		Vec3 vTraceEnd = vEye + vForward * flRange;

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
			if (pRecord == &m_tLiveRecord)
				bLiveHit = true, vLivePos = tTarget.m_vPos, vLiveAngle = tTarget.m_vAngleTo;
			const float flFov = Math::CalcAngle(vEye, tTarget.m_vPos).DeltaAngle(vViewAngles).Length2DSqr();
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

			auto vAngle = Math::CalcAngle(vEye, tTarget.m_vPos);

			Math::AngleVectors(vAngle, &vForward);
			vTraceEnd = vEye + vForward * flRange;

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
				const Vec3 vAngle = Math::CalcAngle(vEye, vPoint);
				if (!bNoFov && !F::AimbotGlobal.ShouldAimAtAngle(vAngle))
					continue;

				Vec3 vForward; Math::AngleVectors(vAngle, &vForward);
				Vec3 vTraceEnd = vEye + vForward * flRange;

				SDK::TraceHull(m_vEyePos, vTraceEnd, {}, {}, MASK_SOLID, &filter, &trace);
				bool bHit = trace.m_pEnt == tTarget.m_pEntity;
				if (!bHit)
				{
					SDK::TraceHull(m_vEyePos, vTraceEnd, vSwingMins, vSwingMaxs, MASK_SOLID, &filter, &trace);
					bHit = trace.m_pEnt == tTarget.m_pEntity;
				}
				if (!bHit)
					continue;

				if (pRecord == &m_tLiveRecord)
					bLiveHit = true, vLivePos = vPoint, vLiveAngle = vAngle;
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

	// Live priority: a hittable live foe wins outright (knife re-verified;
	// the loop already verified any bid it tracks, this is the same check
	// for the override path). History only decides when live can't hit.
	if (bLiveHit && (!bKnifeStab || CanBackstab(tTarget.m_pEntity, pLocal, vLiveAngle)))
	{
		pBest = &m_tLiveRecord;
		vBestPos = vLivePos;
		vBestAngle = vLiveAngle;
	}
	// Frozen-world sanity: a history-only solution must agree with the
	// live foe. Divergent angles mean a ghost (stale position the server
	// will not reproduce); holding beats wasting the swing. Exempt live
	// winners (they ARE the agreement) and live play (rewind is the point
	// of backtrack there). Point-blank degenerates the angle: pass it.
	if (pBest && pBest != &m_tLiveRecord && m_bDtValidated)
	{
		const Vec3 vLiveCenter = tTarget.m_pEntity->GetAbsOrigin()
			+ (tTarget.m_pEntity->m_vecMins() + tTarget.m_pEntity->m_vecMaxs()) * 0.5f;
		if (vLiveCenter.DistToSqr(vEye) >= 1.f
			&& Math::CalcAngle(vEye, vLiveCenter).DeltaAngle(vBestAngle).Length2D() > kFrozenAngleVetoDeg)
			pBest = nullptr;
	}

	if (pBest)
	{
		tTarget.m_vPos = vBestPos;
		tTarget.m_vAngleTo = vBestAngle;
		tTarget.m_pRecord = pBest;
		// Live win = no rewind: the position IS the server truth, so no
		// tick_count stamp (stamping a backtrack tick for a live aim would
		// rewind the foe away from the aimed point).
		tTarget.m_bBacktrack = pBest != &m_tLiveRecord && tTarget.m_iTargetType == TargetEnum::Player;

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

// Firing solution FROM a warp-dest eye: runs the target through CanHit so
// the warp inherits identical gates, clamp-point angle math and record
// choice as the swing that will execute (frozen target assumed, same as
// the walk sim). Stale swing-pred sims would poison it (last tick's
// positions), so they go first; UpdateInfo rebuilds them later this tick
// anyway, as do m_vEyePos/m_flRange/horizons (set fresh here).
bool CAimbotMelee::SolveWarpAim(Target_t& tTarget, CTFPlayer* pLocal, CTFWeaponBase* pWeapon, const Vec3& vEye, float flRangeOverride)
{
	m_mRecordMap.clear();
	m_vEyePos = vEye;
	// Charged swings resolve at charge range: without the override the
	// 70-128u band verifies in the walk but fails here, silently dropping
	// claims that would have hit.
	m_flRange = (flRangeOverride >= 0.f) ? flRangeOverride : pWeapon->GetSwingRange();
	m_iSimulatedTicks = m_iSwingTicks = GetSwingTime(pWeapon);
	m_iDoubletapTicks = F::Ticks.GetTicks(pWeapon);
	m_bWarpSolving = true;
	const int iResult = CanHit(tTarget, pLocal, pWeapon);
	m_bWarpSolving = false;
	return iResult > 0;
}

void CAimbotMelee::Run(CTFPlayer* pLocal, CTFWeaponBase* pWeapon, CUserCmd* pCmd)
{
	m_bFiringSolution = false; // reset every tick; set below only by validation
	// Aimbot-swing latch: no smack in flight = previous swing (if any)
	// resolved; a press below re-latches. Manual swings never latch.
	if (!pWeapon || pWeapon->m_flSmackTime() <= 0.f)
		m_bAimbotSwing = false;
	static bool s_bSessionLogged = false;
	if (!s_bSessionLogged)
	{
		s_bSessionLogged = true;
		DtTrace("SESSION build %s %s", __DATE__, __TIME__);
	}
	if (Vars::Doubletap::Doubletap.Value && pWeapon)
	{
		const float flSmack = pWeapon->m_flSmackTime();
		const int iSmackLeft = flSmack > 0.f ? TIME_TO_TICKS(std::max(flSmack - I::GlobalVars->curtime, 0.f)) : -1;
		// Idle-throttle: RUN+HIT flood burns the 60k session cap in minutes
		// and blinds later forensics. Full rate while a swing lives, armed,
		// or warping; 1-in-8 otherwise.
		const bool bBusy = iSmackLeft >= 0 || F::Ticks.m_bDoubletap || F::Ticks.m_bWarp;
		if (bBusy || (I::GlobalVars->tickcount & 7) == 0)
		{
		const Vec3 vOrg = pLocal->m_vecOrigin();
		DtTrace("T%d RUN bank=%d goal=%d dtArm=%d warp=%d rech=%d smackLeft=%d canAtk=%d cs=%d chg=%d cf=%d rng=%.0f pos=(%.1f,%.1f,%.1f) vel=%.1f",
			I::GlobalVars->tickcount, F::Ticks.m_iShiftedTicks, F::Ticks.m_iShiftedGoal, F::Ticks.m_bDoubletap,
			F::Ticks.m_bWarp, F::Ticks.m_bRecharge, iSmackLeft, G::CanPrimaryAttack, (int)m_eChargeState,
			(int)pLocal->InCond(TF_COND_SHIELD_CHARGE), (int)m_bChargeFired, m_flRange,
			vOrg.x, vOrg.y, vOrg.z, pLocal->m_vecVelocity().Length2D());
		}
		// Swing-start edge: press logged but smack never begins = phantom
		// press (IN_ATTACK eaten/choked, never executed) vs smack with no
		// DMG = genuine miss. One line per swing, always on.
		static float s_flLastSmack = -1.f;
		if (s_flLastSmack < 0.f && flSmack > 0.f)
			DtTrace("T%d SWINGSTART", I::GlobalVars->tickcount);
		s_flLastSmack = flSmack;
	}
	// Auto trickstab warp-fan viz (phase 1: simulate + draw only, no writes).
	F::AutoTrickstab.Run(pLocal, pWeapon, pCmd);
	// AntiWarp on charge-sequenced DT dumps: the generic DT driver (the
	// normal m_bAntiWarp writer) yields while charge is busy, so our dumps
	// ran without it. Same lifecycle as the driver's: set only while a
	// dump is armed (cleared post-drain, trackers reset on normal ticks),
	// never while a warp owns movement (travel executes the sim verbatim).
	// Never while CHARGING: the charge must fly through the dump (750
	// coast, travel the batch) — freezing it mid-charge stalls the reach
	// the swing was timed against. Evaluated per tick, so a charge that
	// starts mid-batch (Early +2) freezes only its pre-charge sub-ticks.
	if (F::Ticks.m_bDoubletap && !F::AutoTrickstab.IsCommitting())
		F::Ticks.m_bAntiWarp = Vars::Doubletap::AntiWarp.Value && pLocal->m_hGroundEntity()
			&& !pLocal->InCond(TF_COND_SHIELD_CHARGE);
	// Charge bot (toggle): while actively charging, steer yaw toward the
	// current target's intercept point so the unstoppable charge arrives
	// instead of flying the stale M2-time heading. FOV-gated (no view
	// yanks past the limit), yaw-only (swing owns pitch; later lock ticks
	// overwrite with validated precision when they agree — both aim at
	// the same foe). Silent-safe: command angles only, camera untouched.
	// Target = latched charge target, else the aimbot's current target.
	if (Vars::Aimbot::Melee::ChargeBot.Value
		&& pLocal->m_iClass() == TF_CLASS_DEMOMAN
		&& pLocal->InCond(TF_COND_SHIELD_CHARGE))
	{
		CBaseEntity* pFoe = nullptr;
		auto fValidFoe = [&](IClientEntity* pE) -> CBaseEntity*
		{
			if (!pE || pE->IsDormant())
				return nullptr;
			auto pP = pE->As<CTFPlayer>();
			if (!pP || !pP->IsAlive() || pP->m_iTeamNum() == pLocal->m_iTeamNum())
				return nullptr;
			return pP;
		};
		if (m_iChargeTarget > 0)
		{
			if (auto pE = I::ClientEntityList->GetClientEntity(m_iChargeTarget))
				pFoe = fValidFoe(pE);
		}
		if (!pFoe && G::AimTarget.m_iEntIndex > 0
			&& I::GlobalVars->tickcount - G::AimTarget.m_iTickCount <= 66)
		{
			if (auto pE = I::ClientEntityList->GetClientEntity(G::AimTarget.m_iEntIndex))
				pFoe = fValidFoe(pE);
		}
		if (pFoe)
		{
			const Vec3 vC = pFoe->GetAbsOrigin() + (pFoe->m_vecMins() + pFoe->m_vecMaxs()) * 0.5f;
			const Vec3 vEyeNow = pLocal->GetShootPos();
			const float flDist = vEyeNow.DistTo(vC);
			// Intercept lead: foe velocity over our time-to-arrive at
			// charge speed, capped (no sight-unseen heroics).
			const float flLeadT = std::clamp(flDist / 750.f, 0.f, 0.5f);
			Vec3 vAimAt = vC;
			if (auto pPF = pFoe->As<CTFPlayer>())
				vAimAt = vAimAt + pPF->m_vecVelocity() * flLeadT;
			const Vec3 vWant = Math::CalcAngle(vEyeNow, vAimAt);
			if (vWant.DeltaAngle(pCmd->viewangles).Length2D() <= Vars::Aimbot::Melee::ChargeBotFOV.Value)
				pCmd->viewangles.y = vWant.y;
		}
	}
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
				// New swing while armed (stale approach-arm): recount from
				// THIS smack so Early/Late land on it, not the old count
				// (which would insta-fire or insta-timeout the sequence).
				if (m_flLastSmackTime != pWeapon->m_flSmackTime())
					m_iChargeTicks = 0;
				m_iChargeTicks++; // counts every execution: live ticks AND dump-batch ticks
				const int iTicksToSmack = TIME_TO_TICKS(std::max(pWeapon->m_flSmackTime() - I::GlobalVars->curtime, 0.f));
				const int iSmackFull = GetSwingTime(pWeapon, false);
				// Timing law: Early = 2 ticks after swing start; Late = 2
				// ticks before smack lands. Count-based (not smack-domain):
				// curtime is frozen inside a dump batch, so the countdown
				// can't distinguish batch ticks — the execution count can.
				// Aim+Charge merged to 1 tick (snap + press same tick), so
				// trigger→press = 1 tick. Press lands at iPressAt+1.
				const bool bEarly = Vars::Aimbot::Melee::ChargeTiming.Value == Vars::Aimbot::Melee::ChargeTimingEnum::Early;
				const int iPressAt = bEarly ? 2 : std::max(iSmackFull - 2, 1);
				// Compressed time (dump executing): counts run faster than
				// observations, so fire now. Same swing + countdown jumped
				// >1 = a dump ran. DT-armed also counts.
				const bool bSmackJumped = m_flLastSmackTime == pWeapon->m_flSmackTime()
					&& m_iLastSmackTicks - iTicksToSmack > 1;
				m_flLastSmackTime = pWeapon->m_flSmackTime();
				m_iLastSmackTicks = iTicksToSmack;
				const bool bCompressed = bSmackJumped || F::Ticks.m_bWarp
					|| (F::Ticks.m_bDoubletap && F::Ticks.GetTicks(pWeapon) > 0);
				// Minimum tick: Early needs ≥2 ticks after swing start
				// (swing registered first); Late can fire from tick 1.
				const int iMinTicks = bEarly ? 2 : 1;
				if ((m_iChargeTicks >= iPressAt || bCompressed) && m_iChargeTicks >= iMinTicks)
					m_eChargeState = ChargeState::Aim;
				else if (m_iChargeTicks > 25)
				{
					m_eChargeState = ChargeState::Idle;
					m_iChargeTarget = -1;
				}
			}
			break;
		case ChargeState::Aim:
			// Merged Aim+Charge: snap view AND press on the same tick.
			// Camera locks to the swing's angles (smack resolves with these,
			// not drifted camera). Press fires immediately — trigger→press
			// is 1 tick total. Collapse latch: if no dump is armed, start
			// one now so the smack tail rides the fresh charge instantly.
			if (pWeapon->m_flSmackTime() <= 0.f)
			{
				m_eChargeState = ChargeState::Idle;
				m_iChargeTarget = -1;
			}
			else
			{
				pCmd->viewangles.x = m_vChargeAim.x;
				pCmd->viewangles.y = m_vChargeAim.y;
				pCmd->buttons |= IN_ATTACK2;
				// Latch the trigger: the condition lands 1-2 ticks later,
				// but range + coast-sim must treat the swing as charged
				// from THIS tick (cleared at swing end below; meter gating
				// auto-clears it at charge end).
				m_bChargeFired = true;
				// Force-send ONLY on live ticks. Inside a dump/warp batch
				// (drained ticks re-run this hook) a force-send flushes the
				// packet mid-batch: the remaining ticks spill to the next
				// server frame, time unfreezes, and the smack resolves
				// against a moved target with stale angles — the systematic
				// bash-lands-but-swing-misses. Batch delivery needs no help.
				if (!F::Ticks.m_bDoubletap && !F::Ticks.m_bWarp)
					G::SendPacket = true;
				// Top-up latch only: a dump already running (press/tail set
				// it on fresh validation) gets its goal fitted to the exact
				// remaining tail. Never initiates: an unvalidated swing must
				// not spend bank, and the tail path covers bank-now-covers
				// with this tick's frozen validation instead.
				if (F::Ticks.m_bDoubletap && F::Ticks.GetTicks(pWeapon) > 0 && !F::AutoTrickstab.IsCommitting())
				{
					const int iTicksToSmack = TIME_TO_TICKS(std::max(pWeapon->m_flSmackTime() - I::GlobalVars->curtime, 0.f));
					F::Ticks.m_bDoubletap = true;
					F::Ticks.m_iShiftedGoal = std::min(F::Ticks.m_iShiftedGoal, F::Ticks.ReleaseGoal(std::max(iTicksToSmack, 1)));
				}
				m_iChargeTarget = -1;
				m_eChargeState = ChargeState::Idle;
			}
			break;
		case ChargeState::Charge:
			// Dead: Aim merged snap+press into one tick. Fall through to Idle.
			m_eChargeState = ChargeState::Idle;
			m_iChargeTarget = -1;
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
		m_bChargeFired = false;
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
		// No record-map requirement: sim records are never stored while
		// DT-validated and backtrack records are cleared for dormant
		// foes, so requiring them fails the guard open and the farm
		// swings mid-fight with a close target. Live origin is enough
		// to know there is someone to hit.
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
		if (Vars::Doubletap::Doubletap.Value
			&& (iResult || pWeapon->m_flSmackTime() > 0.f || F::Ticks.m_bDoubletap))
		{
			const Vec3 vFoe = tTarget.m_pEntity->m_vecOrigin();
			DtTrace("T%d HIT res=%d foe=(%.1f,%.1f,%.1f) rec=%d frozen=%d",
				I::GlobalVars->tickcount, (int)iResult, vFoe.x, vFoe.y, vFoe.z, tTarget.m_bBacktrack, m_bDtValidated);
		}
		if (!iResult) continue;
		m_bFiringSolution = true;
		G::AimTarget = { tTarget.m_pEntity->entindex(), I::GlobalVars->tickcount, 0 };
		if (iResult == 2)
		{
			Aim(pCmd, tTarget.m_vAngleTo);
			ArmChargeTracking(tTarget);
			break;
		}

		if (Vars::Aimbot::General::AutoShoot.Value && pWeapon->m_flSmackTime() < 0.f && G::CanPrimaryAttack)
		{
			// (Press AND dump gated on CanPrimaryAttack: pressing during
			// refire cooldown is ignored by the game but a dump would still
			// drain the bank — swing starts late with an empty bank, misses,
			// then pays full recovery. No press, no dump, no arm until the
			// swing can actually start this tick. Stable and predictable.)
			// Fire the instant validation passes — never wait for bank.
			// Covered: full instant dump. Short: dump whatever is banked
			// (partial instant head, ReleaseGoal floors at 0) and ride the
			// remainder live, which re-validates live every tick by
			// construction (unvalidated mode above). Waiting hugs the
			// enemy, then fires a fully-live swing on stale validation:
			// the worst of both worlds. (Rijin works exactly this way:
			// its melee fires on validation unconditionally; the DT
			// system shifts separately when charged.)
			const int iBank = F::Ticks.m_iShiftedTicks;
			const bool bCover = iBank >= m_iSwingTicks;
			const bool bPress = m_bDtValidated || m_bShouldSwing || bCover;
			if (bPress)
			{
				pCmd->buttons |= IN_ATTACK;
				m_bAimbotSwing = true;
				// Snapshot the validated range: mid-swing recomputation can
				// only shrink it (machine idles, manual charge released,
				// meter drained) while the swing still flies — collapsing
				// aim updates onto a stale base angle. The snapshot holds
				// the validated reach for THIS swing; a live (undumped)
				// tail keeps the honest recomputed range instead.
				m_flPressRange = m_flRange;
				// Stamp the target: downstream consumers (CritHack force,
				// resolver, visuals) key off G::AimTarget. The aim-only
				// branch stamps it, but the press path never did — forced
				// crits then depended on stale leftovers (sometimes set,
				// sometimes not). We ARE attacking this entity.
				G::AimTarget = { tTarget.m_pEntity->entindex(), I::GlobalVars->tickcount, 0 };
			}
			DtTrace("T%d PRESS press=%d cover=%d bank=%d need=%d dtFrozen=%d shouldSwing=%d",
				I::GlobalVars->tickcount, bPress, bCover, iBank, m_iSwingTicks, m_bDtValidated, m_bShouldSwing);
			// Melee dump policy: bank >= smack window suffices (generic
			// GetTicks demands TickLimit and m_iWait refuses chained swings,
			// so banked re-presses would never go instant). Fires on
			// charge-arming presses too: the swing collapses instantly,
			// charge sequences off the execution count. Flag AND goal:
			// Move() drains only on shifted>goal, a flag alone never dumps.
			// min() + fresh-only so a deeper tailored dump is never cut.
			// m_bRecharge is NOT a blocker: it is set on every tick the bank
			// is still filling, which is exactly when a covered swing must
			// dump. Recharge() stands down once m_bDoubletap is armed, so the
			// dump preempts it cleanly.
			const bool bDumpUsable = Vars::Doubletap::Doubletap.Value
				&& !F::Ticks.m_bWarp && !F::Ticks.m_bSpeedhack
				&& !F::AutoRocketJump.IsRunning();
			const bool bDumpReady = m_iDoubletapTicks > 0
				|| (F::Ticks.m_iShiftedTicks >= m_iSwingTicks && bDumpUsable)
				// Partial: spend whatever is banked. Anything spendable
				// shortens the live tail by that many instant ticks;
				// firing now with 6 instant + 8 live beats waiting 8 ticks
				// for full cover and swinging fully live into a moved foe.
				|| (bDumpUsable && F::Ticks.m_iShiftedTicks > F::Ticks.m_iShiftedGoal);
			DtTrace("T%d DUMP ready=%d busy=%d armed=%d bank=%d goal=%d", I::GlobalVars->tickcount,
				bDumpReady, IsChargeBusy(), F::Ticks.m_bDoubletap, F::Ticks.m_iShiftedTicks, F::Ticks.m_iShiftedGoal);
			if (bDumpReady && (!IsChargeBusy() || m_eChargeState == ChargeState::Charge) && !F::Ticks.m_bDoubletap)
			{
				F::Ticks.m_bDoubletap = true;
				F::Ticks.m_iShiftedGoal = std::min(F::Ticks.m_iShiftedGoal, F::Ticks.ReleaseGoal(std::max(m_iSwingTicks, 1)));
			}
		}
		else
			m_iSwingHoldTick = 0;

		G::Attacking = SDK::IsAttacking(pLocal, pWeapon, pCmd, true);
		ArmChargeTracking(tTarget);
		if (G::Attacking == 1)
		{
			// Rewind target = aimed position's time, EVERY attacking tick.
			// Sub-tick commands carry advancing tick_counts; without the
			// overwrite the server rewinds to (or past) process time and a
			// moving foe walks out of the aimed line — client prediction
			// clangs on batch-start positions while the server misses
			// (clang with no damage). Live winners stamp the live sim time:
			// static across the dump, same guarantee as record stamps.
			// (Reached only post-validation: failed targets `continue`
			// above, so manual swings keep CreateMove's default.)
			if (tTarget.m_pRecord)
			{
				const bool bHist = tTarget.m_bBacktrack;
				const float flAimTime = bHist ? tTarget.m_pRecord->m_flSimTime
					: tTarget.m_pEntity->m_flSimulationTime();
				pCmd->tick_count = TIME_TO_TICKS(flAimTime + F::Backtrack.GetFakeInterp());
				if (Vars::Doubletap::Doubletap.Value)
					DtTrace("T%d STAMP tick=%d recSim=%.3f live=%d", I::GlobalVars->tickcount,
						pCmd->tick_count, flAimTime, bHist ? 0 : 1);
			}
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
	// Swing-in-flight failsafe: ours (validated press) but NOTHING validates
	// this tick (stale records, blipped dormancy) — the resolve command
	// would keep camera angles and the smack is donated. Re-aim at the
	// nearest live enemy IN REACH instead. Manual swings (wall tests) are
	// never touched: no latch, no re-aim, no stamp. Silent/Locking only
	// (visible-aim contracts untouched).
	if (!m_bFiringSolution && m_bAimbotSwing && pWeapon->m_flSmackTime() > 0.f
		&& (Vars::Aimbot::General::AimType.Value == Vars::Aimbot::General::AimTypeEnum::Silent
		|| Vars::Aimbot::General::AimType.Value == Vars::Aimbot::General::AimTypeEnum::Locking))
	{
		CBaseEntity* pBestFoe = nullptr; float flBestDistSqr = FLT_MAX;
		const float flEmergRange = m_flRange + 32.f; // hull + re-approach slack
		for (auto& tEmerg : vTargets)
		{
			if (!tEmerg.m_pEntity || tEmerg.m_pEntity->m_iTeamNum() == pLocal->m_iTeamNum())
				continue;
			const Vec3 vEC = tEmerg.m_pEntity->GetAbsOrigin() + (tEmerg.m_pEntity->m_vecMins() + tEmerg.m_pEntity->m_vecMaxs()) * 0.5f;
			const float flD = vEC.DistToSqr(m_vEyePos);
			if (flD < flBestDistSqr && flD <= flEmergRange * flEmergRange)
				flBestDistSqr = flD, pBestFoe = tEmerg.m_pEntity;
		}
		if (pBestFoe)
		{
			const Vec3 vFC = pBestFoe->GetAbsOrigin() + (pBestFoe->m_vecMins() + pBestFoe->m_vecMaxs()) * 0.5f;
			if (vFC.DistToSqr(m_vEyePos) >= 1.f)
			{
				Vec3 vEmergency = Math::CalcAngle(m_vEyePos, vFC);
				Math::ClampAngles(vEmergency);
				SDK::FixMovement(pCmd, vEmergency);
				pCmd->viewangles = vEmergency;
				G::PSilentAngles = true;
				// Same rewind rule as aimed swings: the written angles
				// target this foe's live position, so stamp its time —
				// else movers walk out server-side (clang, no damage).
				pCmd->tick_count = TIME_TO_TICKS(pBestFoe->m_flSimulationTime() + F::Backtrack.GetFakeInterp());
				if (Vars::Doubletap::Doubletap.Value)
					DtTrace("T%d STAMP tick=%d recSim=%.3f live=2", I::GlobalVars->tickcount,
						pCmd->tick_count, pBestFoe->m_flSimulationTime());
			}
		}
	}
	// Verdict line: the resolve tick. Final eye + aim angles leaving this
	// tick, so a miss is judged against the stamped record and DMG lines.
	// (Tail compression was deleted: it converted live-validated swings
	// mid-flight, resolving against a different world than validated.)
	if (Vars::Doubletap::Doubletap.Value && pWeapon->m_flSmackTime() > 0.f
		&& TIME_TO_TICKS(std::max(pWeapon->m_flSmackTime() - I::GlobalVars->curtime, 0.f)) == 0)
	{
		const Vec3 vEyeNow = pLocal->GetShootPos();
		DtTrace("T%d SMACK eye=(%.1f,%.1f,%.1f) ang=(%.1f,%.1f,%.1f)", I::GlobalVars->tickcount,
			vEyeNow.x, vEyeNow.y, vEyeNow.z, pCmd->viewangles.x, pCmd->viewangles.y, pCmd->viewangles.z);
	}
	// Charge cooperation lives entirely in the Tracking machine above
	// (execution count: Early = 2 ticks after swing start, Late = 2 ticks
	// before damage; dump-batch ticks count identically to live ones)
	// plus the warp claim/replay schedule. No standalone proximity
	// charging: charge never fires without a swing in flight.
	// Auto-warp press (Normal: with the claim; Full: arrival solved the same
	// way, server resolves the smack inside stopped time). Additive only,
	// after the hold logic so banking never vetoes it. Stamps AimTarget so
	// crosshair-backtrack yields the tick the claim already set.
	if (F::AutoTrickstab.WarpPressDue() && G::CanPrimaryAttack && pWeapon->m_flSmackTime() < 0.f)
	{
		// Aimbot aims the decisive tick: re-solve live from the current eye
		// through the real firing path. Accept ONLY a verified solution
		// (record stamped under the same backtrack rule as the loop); else
		// the claim-frozen one, whose record the claim already stamped.
		// Verified-only keeps the plan's guarantee under live aim — never
		// drifted travel angles, never center-mass guesswork.
		Vec3 vPressAim = F::AutoTrickstab.WarpAim();
		if (auto pWE = I::ClientEntityList->GetClientEntity(F::AutoTrickstab.WarpEnt()))
		{
			if (!pWE->IsDormant())
			{
				if (CTFPlayer* pWT = pWE->As<CTFPlayer>(); pWT && pWT->IsAlive())
				{
					const Vec3 vEyeNow = pLocal->GetShootPos();
					Target_t tLive{ pWT, TargetEnum::Player, pWT->GetCenter(), Math::CalcAngle(vEyeNow, pWT->GetCenter()), 0.f, 0.f, 0 };
					if (SolveWarpAim(tLive, pLocal, pWeapon, vEyeNow))
					{
						vPressAim = tLive.m_vAngleTo;
						if (tLive.m_bBacktrack && tLive.m_pRecord)
							pCmd->tick_count = TIME_TO_TICKS(tLive.m_pRecord->m_flSimTime + F::Backtrack.GetFakeInterp());
					}
				}
			}
		}
		SDK::FixMovement(pCmd, vPressAim);
		pCmd->viewangles = vPressAim;
		G::PSilentAngles = true;
		pCmd->buttons |= IN_ATTACK;
		G::Attacking = SDK::IsAttacking(pLocal, pWeapon, pCmd, true);
		G::AimTarget = { F::AutoTrickstab.WarpEnt(), I::GlobalVars->tickcount, 0 };
		G::AimPoint = { F::AutoTrickstab.WarpDest() + pLocal->m_vecViewOffset(), I::GlobalVars->tickcount };
		// NOTE: no charge press here — charge sequences inside stopped time
		// via the replay schedule. Same cmd carrying swing+charge breaks
		// engine-side (one eats the other).
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