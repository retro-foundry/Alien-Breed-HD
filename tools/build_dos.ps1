param([string]$BuildDir = 'build/dos', [switch]$Indexed,
      [switch]$ProfileGuided, [string]$DosBox = '')
$ErrorActionPreference = 'Stop'
if ($ProfileGuided -and $Indexed) { throw 'Compiler feedback currently targets native RGB565' }
if ($ProfileGuided -and !$DosBox) { throw '-ProfileGuided requires -DosBox <dosbox.exe>' }
$repo = Split-Path -Parent $PSScriptRoot
$build = Join-Path $repo $BuildDir
$config = Join-Path $repo 'build/dos-deps/toolchain.json'
if (!(Test-Path $config)) { throw 'Run tools/setup_dos.ps1 first' }
$toolchain = Get-Content $config -Raw | ConvertFrom-Json
$env:Path = "$($toolchain.djgpp)/bin;$env:Path"
function Check([string]$operation) { if ($LASTEXITCODE -ne 0) { throw "$operation failed ($LASTEXITCODE)" } }
$assetArgs = @('--build-dir', $build)
if ($Indexed) { $assetArgs += '--indexed' }
python (Join-Path $repo 'tools/package_dos.py') @assetArgs
Check 'DOS asset packaging'
cmake -S $repo -B $build -G Ninja "-DCMAKE_TOOLCHAIN_FILE=$repo/cmake/djgpp.cmake" `
  "-DDJGPP_ROOT=$($toolchain.djgpp)" "-DDOS_SDL_ROOT=$($toolchain.sdl)" "-DAB3D_DOS_INDEXED=$($Indexed.IsPresent)" `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DAB3D_DOS_PROFILE_DIR=
Check 'DOS configure'
cmake --build $build --parallel 4
Check 'DOS build'
$stubedit = Join-Path $toolchain.djgpp 'i586-pc-msdosdjgpp/bin/stubedit.exe'
& $stubedit (Join-Path $build 'ABHD.exe') minstack=128K
Check 'DOS stack configuration'
Copy-Item -LiteralPath (Join-Path $build 'ABHD.exe') -Destination (Join-Path $build 'game/ABHD.EXE') -Force
Copy-Item -LiteralPath (Join-Path $toolchain.dpmi 'bin/CWSDPMI.EXE') -Destination (Join-Path $build 'game/CWSDPMI.EXE') -Force
Copy-Item -LiteralPath (Join-Path $toolchain.dpmi 'bin/cwsdpmi.doc') -Destination (Join-Path $build 'game/CWSDPMI.DOC') -Force
Copy-Item -LiteralPath (Join-Path $toolchain.sdl 'SDL/LICENSE.txt') -Destination (Join-Path $build 'game/SDL-LIC.TXT') -Force
if ($ProfileGuided) {
  python (Join-Path $repo 'tools/profile_dos.py') --build-dir $build --dosbox $DosBox
  Check 'DOS compiler feedback and validation'
}
python (Join-Path $repo 'tools/package_dos.py') @assetArgs --archive
Check 'DOS distribution archive'
Write-Output "Built $build/game/ABHD.EXE"
