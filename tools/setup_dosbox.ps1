param([string]$BuildDir = 'build/dos-deps/dosbox-staging')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$destination = Join-Path $repo $BuildDir
$version = '0.82.2'
$archive = Join-Path $destination "dosbox-staging-windows-x64-v$version.zip"
$executable = Join-Path $destination "dosbox-staging-v$version/dosbox.exe"
$expectedHash = '9FA55DCBACC2E5565AC4BE63B5CFA4E7D50DC9ADB9F9CB53F506D973AF567F3A'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
if (!(Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest "https://github.com/dosbox-staging/dosbox-staging/releases/download/v$version/dosbox-staging-windows-x64-v$version.zip" -UseBasicParsing -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expectedHash) {
    throw 'DOSBox Staging archive checksum mismatch'
}
if (!(Test-Path -LiteralPath $executable)) {
    Expand-Archive -LiteralPath $archive -DestinationPath $destination
}
if (!(Test-Path -LiteralPath $executable)) {
    throw "DOSBox Staging executable missing after extraction: $executable"
}
Write-Output (Resolve-Path -LiteralPath $executable).Path
