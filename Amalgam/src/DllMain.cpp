#include <Windows.h>
#include "Core/Core.h"
#include "Utils/ExceptionHandler/ExceptionHandler.h"
#include "Utils/BootLog/BootLog.h"

// Cooperative unload protocol for manual-mapped images (Xenos Modules
// dialog -> Unload on a "Manual map" entry):
//  1. Xenos remote-calls AmalgamRequestUnload() (returns immediately).
//  2. Xenos polls AmalgamUnloadState() via thread exit code until 2.
//  3. Xenos frees the image memory. Re-inject lands on a clean target.
// State: 0 = running, 1 = teardown in progress, 2 = done, safe to free.
// x64 cdecl exports are undecorated; called as remote thread entries.
static volatile LONG s_lUnloadState = 0;

extern "C" __declspec(dllexport) DWORD AmalgamRequestUnload(void*)
{
	U::Core.m_bUnload = true;
	if (InterlockedCompareExchange(&s_lUnloadState, 1, 0) == 2)
		return 2; // already torn down (F11 raced us)
	return 1;
}

extern "C" __declspec(dllexport) DWORD AmalgamUnloadState(void*)
{
	return (DWORD)s_lUnloadState;
}

DWORD WINAPI MainThread(LPVOID lpParam)
{
	BootLog("BOOT main-thread");
	U::ExceptionHandler.Initialize(lpParam);

	U::Core.Load();
	U::Core.Loop();
	U::Core.Unload();

	U::ExceptionHandler.Unload();

	InterlockedExchange(&s_lUnloadState, 2);

	// Ldr-loaded images self-free; manual-mapped ones aren't in the loader
	// tables, so FreeLibrary would no-op (leak) or worse. The injector frees
	// us after polling AmalgamUnloadState() == 2.
	HMODULE hSelf = nullptr;
	if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			reinterpret_cast<LPCSTR>(lpParam), &hSelf)
		&& hSelf == static_cast<HMODULE>(lpParam))
		FreeLibraryAndExitThread(static_cast<HMODULE>(lpParam), EXIT_SUCCESS);
	ExitThread(EXIT_SUCCESS);
}

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved)
{
	if (fdwReason == DLL_PROCESS_ATTACH)
	{
		if (const auto hThread = CreateThread(nullptr, 0, MainThread, hinstDLL, 0, nullptr))
			CloseHandle(hThread);
	}

	return TRUE;
}