#pragma once
#include "Vars.h" // machinery only; Vars.h must NEVER include this (VarsMelee.h pattern)

// Auto-peek extras, isolated: editing here recompiles PacketBuffer, Menu and
// Misc-publish instead of the world. Vars::AutoPeek::Section stays in Vars.h;
// this file plain-reopens the namespace (never NAMESPACE_BEGIN it).
namespace Vars {
namespace AutoPeek {
	CVar(FakePeek, "Fake peek", false);
	CVar(FakePeekWindow, "Fake peek max window", 150, SLIDER_CLAMP, 50, 500);
	CVar(WarpBack, "Warp back on commit", false);
}
}
