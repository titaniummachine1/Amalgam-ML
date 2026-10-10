# Amalgam incremental build guard — NEVER full rebuild without explicit user order.
# Usage: powershell -File build-fast.ps1
# If objects are missing, this REFUSES to build and tells you what was lost.

$ErrorActionPreference = 'Stop'
$proj = 'C:\gitProjects\amalgam\Amalgam\Amalgam.vcxproj'
$objDir = 'C:\gitProjects\amalgam\Amalgam\build\x64\Release'
$msbuild = 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe'
$srcDir = 'C:\gitProjects\amalgam\Amalgam\src'

# Count existing .obj files
$objs = Get-ChildItem -Path $objDir -Filter '*.obj' -ErrorAction SilentlyContinue
$objCount = ($objs | Measure-Object).Count

# Count .cpp files
$cppCount = (Get-ChildItem -Path $srcDir -Recurse -Filter '*.cpp' | Measure-Object).Count

Write-Host "Objects: $objCount | Sources: $cppCount"

if ($objCount -eq 0) {
    Write-Host "REFUSING TO BUILD: zero .obj files = full rebuild required." -ForegroundColor Red
    Write-Host "This is a ~10 min wave. Explicit user approval required." -ForegroundColor Red
    Write-Host "Run manually if approved: & '$msbuild' '$proj' /p:Configuration=Release /p:Platform=x64 /p:WholeProgramOptimization=false /m /v:minimal /fl"
    exit 1
}

# Check for stale/missing objects (sources newer than their .obj)
$stale = @()
Get-ChildItem -Path $srcDir -Recurse -Filter '*.cpp' | ForEach-Object {
    $objName = $_.BaseName + '.obj'
    $objPath = Join-Path $objDir $objName
    if (-not (Test-Path $objPath)) {
        $stale += $objName
    } elseif ((Get-Item $objPath).LastWriteTime -lt $_.LastWriteTime) {
        $stale += $objName
    }
}

if ($stale.Count -gt 10) {
    Write-Host "REFUSING TO BUILD: $($stale.Count) objects stale/missing = near-full rebuild." -ForegroundColor Red
    Write-Host "First 10: $($stale[0..9] -join ', ')" -ForegroundColor Yellow
    Write-Host "Explicit user approval required for waves >10 files." -ForegroundColor Red
    exit 1
}

if ($stale.Count -gt 0) {
    Write-Host "Incremental: rebuilding $($stale.Count) file(s): $($stale -join ', ')" -ForegroundColor Yellow
}

# Blessed command — flags frozen, never change
& $msbuild $proj /p:Configuration=Release /p:Platform=x64 /p:WholeProgramOptimization=false /m /v:minimal /fl "/flp:LogFile=C:\gitProjects\amalgam\Amalgam\build-fast.log;Verbosity=minimal"

if ($LASTEXITCODE -eq 0) {
    Copy-Item 'C:\gitProjects\amalgam\Amalgam\output\x64\Release\Amalgamx64Release.dll' 'C:\gitProjects\amalgam\output\x64\Release\' -Force
    $ts = (Get-Item 'C:\gitProjects\amalgam\output\x64\Release\Amalgamx64Release.dll').LastWriteTime
    Write-Host "DLL copied, timestamp: $ts" -ForegroundColor Green
} else {
    Write-Host "BUILD FAILED" -ForegroundColor Red
}
