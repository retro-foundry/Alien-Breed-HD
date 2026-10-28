param([string]$DjgppRoot = '', [string]$SdlCache = '')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$deps = Join-Path $repo 'build/dos-deps'
$sdlRoot = Join-Path $deps 'sdl'
$sdl = Join-Path $sdlRoot 'SDL'
$sdlBuild = Join-Path $sdlRoot 'SDL-build'
$revision = 'c699512adcc139ec1a355ff97cb2e5dbab3c9ac2'
function Check([string]$operation) {
  if ($LASTEXITCODE -ne 0) { throw "$operation failed ($LASTEXITCODE)" }
}
New-Item -ItemType Directory -Force $deps | Out-Null
# Build with the host compiler before adding the DOS compiler to PATH.
$assetBuild = Join-Path $deps 'asset-tools'
cmake -S (Join-Path $repo 'tools/dos-assets') -B $assetBuild
Check 'Host asset tool configure'
cmake --build $assetBuild --config Release
Check 'Host asset tool build'

$dpmi = Join-Path $deps 'csdpmi7b'
if (!(Test-Path "$dpmi/bin/CWSDPMI.EXE")) {
  $dpmiArchive = Join-Path $deps 'csdpmi7b.zip'
  if (!(Test-Path $dpmiArchive)) {
    Invoke-WebRequest 'https://ftp.gwdg.de/pub/msdos/gcc/djgpp/v2misc/csdpmi7b.zip' -UseBasicParsing -OutFile $dpmiArchive
  }
  if ((Get-FileHash $dpmiArchive -Algorithm SHA256).Hash -ne 'DEACDA0488E1CDD7C4A9F32FAB45662B34C0ED6B2D7D4D13BC07041B62004A8C') {
    throw 'CWSDPMI archive checksum mismatch'
  }
  Expand-Archive -LiteralPath $dpmiArchive -DestinationPath $dpmi -Force
}
if (!$DjgppRoot) {
  $DjgppRoot = Join-Path $deps 'djgpp-mingw/djgpp'
  if (!(Test-Path "$DjgppRoot/bin/i586-pc-msdosdjgpp-gcc.exe")) {
    $archive = Join-Path $deps 'djgpp.zip'
    if (!(Test-Path $archive)) {
      Invoke-WebRequest 'https://github.com/andrewwutw/build-djgpp/releases/download/v3.4/djgpp-mingw-gcc1220-standalone.zip' -UseBasicParsing -OutFile $archive
    }
    if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne '6F88B531D216F4D92668C960B5CDE9A829B5611E06D2C3E431041E33F01C1A52') {
      throw 'DJGPP archive checksum mismatch'
    }
    Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $deps 'djgpp-mingw') -Force
  }
}
$DjgppRoot = (Resolve-Path $DjgppRoot).Path
if (!(Test-Path "$DjgppRoot/bin/i586-pc-msdosdjgpp-gcc.exe")) { throw "Missing DJGPP compiler: $DjgppRoot" }
$env:Path = "$DjgppRoot/bin;$env:Path"
if (!(Test-Path "$sdl/.git")) {
  New-Item -ItemType Directory -Force $sdlRoot | Out-Null
  if ($SdlCache) {
    git clone --no-hardlinks --no-checkout $SdlCache $sdl
  } else {
    git clone https://github.com/libsdl-org/SDL.git $sdl
  }
  Check 'SDL clone'
  git -C $sdl checkout --detach $revision
  Check 'SDL pinned checkout'
}
if ((git -C $sdl rev-parse HEAD).Trim() -ne $revision) { throw 'DOS SDL is not at the pinned revision' }
$patch = Join-Path $repo 'cmake/dos-sdl.patch'
$previousErrorAction = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
git -C $sdl apply --reverse --check $patch 2>$null
$alreadyPatched = $LASTEXITCODE -eq 0
$ErrorActionPreference = $previousErrorAction
if (!$alreadyPatched) {
  git -C $sdl apply --check $patch
  Check 'DOS SDL patch validation'
  git -C $sdl apply $patch
  Check 'DOS SDL patch'
}
cmake -S $sdl -B $sdlBuild -G Ninja `
  "-DCMAKE_TOOLCHAIN_FILE=$sdl/build-scripts/i586-pc-msdosdjgpp.cmake" `
  '-DCMAKE_C_FLAGS=-DSDL_DISABLE_STB -march=i386 -mtune=i486 -flto -ffat-lto-objects -ffunction-sections -fdata-sections' `
  '-DCMAKE_CXX_FLAGS=-DSDL_DISABLE_STB -march=i386 -mtune=i486 -flto -ffat-lto-objects -ffunction-sections -fdata-sections' `
  '-DCMAKE_C_FLAGS_RELEASE=-Os -DNDEBUG' '-DCMAKE_CXX_FLAGS_RELEASE=-Os -DNDEBUG' `
  -DCMAKE_BUILD_TYPE=Release -DSDL_SHARED=OFF -DSDL_STATIC=ON -DSDL_TESTS=OFF `
  -DSDL_EXAMPLES=OFF -DSDL_OPENGL=OFF -DSDL_OPENGLES=OFF -DSDL_ASSEMBLY=OFF `
  -DSDL_RENDER=OFF -DSDL_GPU=OFF -DSDL_CAMERA=OFF -DSDL_SENSOR=OFF -DSDL_HAPTIC=OFF -DSDL_DIALOG=OFF `
  -DSDL_POWER=OFF -DSDL_JOYSTICK=OFF -DSDL_HIDAPI=OFF -DSDL_VIRTUAL_JOYSTICK=OFF `
  -DSDL_DISKAUDIO=OFF -DSDL_DUMMYAUDIO=OFF -DSDL_DUMMYVIDEO=OFF -DSDL_TEST_LIBRARY=OFF `
  -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON
Check 'DOS SDL configure'
cmake --build $sdlBuild --parallel 6
Check 'DOS SDL build'
@{ djgpp = $DjgppRoot; sdl = $sdlRoot; dpmi = $dpmi; revision = $revision } | ConvertTo-Json | Set-Content (Join-Path $deps 'toolchain.json')
Write-Output "DOS dependencies ready: $deps"
