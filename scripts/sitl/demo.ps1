param(
    [ValidateRange(1024,65535)][int]$Port = 8765,
    [string]$VerifyRecording
)

$ErrorActionPreference = 'Stop'
$repoPath = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$dockerOptions = @('run', '--rm', '--init', '--stop-timeout', '15', '--cpus', '4', '--memory', '2g',
    '--workdir', '/work', '--mount', "type=bind,source=$repoPath,target=/work,readonly",
    '--env', 'PYTHONDONTWRITEBYTECODE=1')
$image = 'chong-fly-sitl:neural-v1'
if ($VerifyRecording) {
    $recordingPath = (Resolve-Path -LiteralPath $VerifyRecording).Path
    $recordingName = if ($recordingPath.EndsWith('.gz')) { '/recording.json.gz' } else { '/recording.json' }
    docker @dockerOptions --network none --mount "type=bind,source=$recordingPath,target=$recordingName,readonly" `
        $image python -u -m demo --verify-recording $recordingName
} else {
    docker @dockerOptions --publish "127.0.0.1:${Port}:${Port}" `
        $image python -u -m demo --container-sitl --port $Port
}
exit $LASTEXITCODE
