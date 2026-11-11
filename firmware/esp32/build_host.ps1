param([Parameter(Mandatory=$true)][string]$OutDir, [switch]$Probe, [switch]$Foundation)
$ErrorActionPreference = 'Stop'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$compilerPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe'
if (-not $compilerPath) { throw 'MSVC x64 build tools are required for Windows host tests.' }
$toolRoot = Split-Path (Split-Path (Split-Path (Split-Path $compilerPath)))
$sdkRoot = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10'
$sdkVersion = Get-ChildItem -LiteralPath (Join-Path $sdkRoot 'Include') -Directory | Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'ucrt\stdio.h') } | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty Name
if (-not $sdkVersion) { throw 'Windows SDK with UCRT is required.' }
$outPath = [IO.Path]::GetFullPath($OutDir)
New-Item -ItemType Directory -Path $outPath -Force | Out-Null
if ($Probe -and $Foundation) { throw 'Choose one host target.' }
$programName = if ($Foundation) { 'foundation_host' } elseif ($Probe) { 'usb_probe_host' } else { 'receiver_host' }
$sourcePath = Join-Path $PSScriptRoot ('host\'+$programName+'.cpp')
$compileArgs = @('/nologo','/std:c++17','/EHsc','/W4','/WX','/MT',('/I'+(Join-Path $toolRoot 'include')),('/I'+(Join-Path $sdkRoot ('Include\'+$sdkVersion+'\ucrt'))),('/Fo'+(Join-Path $outPath ($programName+'.obj'))),('/Fe'+(Join-Path $outPath ($programName+'.exe'))),$sourcePath,'/link',('/LIBPATH:'+(Join-Path $toolRoot 'lib\x64')),('/LIBPATH:'+(Join-Path $sdkRoot ('Lib\'+$sdkVersion+'\ucrt\x64'))),('/LIBPATH:'+(Join-Path $sdkRoot ('Lib\'+$sdkVersion+'\um\x64'))))
& $compilerPath @compileArgs
if ($LASTEXITCODE -ne 0) { throw 'C++ host compilation failed.' }
