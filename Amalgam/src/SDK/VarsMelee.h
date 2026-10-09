#pragma once
#include "Vars.h" // CVar machinery only. Vars.h must NEVER include this file.

// Melee aimbot options, isolated from Vars.h so editing them recompiles
// ~3 files (AimbotMelee, Aimbot, Menu) instead of the world. MOVED verbatim
// out of Vars.h: same namespace, names, values, defaults. Only melee users
// include this file: AimbotMelee.cpp, Menu.cpp.
namespace Vars {
namespace Aimbot {
	NAMESPACE_BEGIN(Melee)
			CVar(AutoBackstab, "Auto backstab", true);
			CVar(IgnoreRazorback, "Ignore razorback", false);
			CVar(SwingPrediction, "Swing prediction", true);
			CVar(WhipTeam, "Whip team", false);
			CVar(Rage, "Rage", false);
			CVar(ChargeReach, "Charge reach", false);
			CVarEnum(ChargeTiming, "Charge timing", 1, NONE, nullptr,
				VA_LIST("Early", "Late"),
				Early, Late);
			CVar(CritRefill, "Auto crit refill", false);
			CVar(CritRefillAmount, "Crit refill amount", 5, SLIDER_CLAMP, 1, 25);

			CVar(SwingTicks, "Swing ticks", 13, NOSAVE | DEBUGVAR | SLIDER_MIN, 0, 14);
			CVar(SwingPredictLag, "Swing predict lag", true, NOSAVE | DEBUGVAR);
			CVarEnum(SwingValidateMode, "Swing validate mode", 0, NOSAVE | DEBUGVAR, nullptr,
				VA_LIST("Both", "Swing", "Simulated"),
				Both, Swing, Simulated);
			CVarEnum(BackstabFlags, "Backstab flags", 0b11, NOSAVE | DEBUGVAR | DROPDOWN_MULTI, nullptr,
				VA_LIST("Account ping", "Double test"),
				AccountPing = 1 << 0, DoubleTest = 1 << 1);
		NAMESPACE_END(Melee)
	NAMESPACE_END(Aimbot)
NAMESPACE_END(Vars)
