param([switch]$Apply)

# One-time, reversible cleanup. Preview by default; never deletes files.
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$archiveRelative = 'archive/legacy-artifacts-20261006'
$manifestRelative = 'data/artifact_archive_20261006.json'
$targets = @(
    'data/reduced_models',
    'data/raw_connectome',
    'data/filtered_models_manifest.json',
    'data/reflex_dataset.pt',
    'data/reflex_dataset_v2.pt',
    'data/reflex_dataset_v2_report.json',
    'data/reflex_validation_v2.json',
    'drone_optimization.db'
)
$protected = @(
    'data/raw_connectome_flywire783_interface_v1',
    'data/annotations_flywire783_v2_1_0',
    'data/research_sources',
    'data/reflex_dataset_v3.pt',
    'data/reflex_dataset_v3_report.json',
    'data/reflex_validation_v3.json'
)

function Get-WorkspacePath([string]$relative) {
    $absolute = [IO.Path]::GetFullPath((Join-Path $repoRoot $relative))
    if (-not $absolute.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar,
                                [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside workspace: $relative"
    }
    # Reject junctions/symlinks in both the target and its ancestors.
    $cursor = $absolute
    while ($cursor -and $cursor.Length -ge $repoRoot.Length) {
        if (Test-Path -LiteralPath $cursor) {
            if ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Reparse point is not allowed: $cursor"
            }
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $absolute
}

function Get-CheckedFiles([string]$relative) {
    $path = Get-WorkspacePath $relative
    if (-not (Test-Path -LiteralPath $path)) { return }
    $item = Get-Item -LiteralPath $path -Force
    if ($item.PSIsContainer) {
        $children = @(Get-ChildItem -LiteralPath $path -Recurse -Force)
        foreach ($child in $children) {
            if ($child.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Reparse point in artifact directory: $($child.FullName)"
            }
        }
        return $children | Where-Object { -not $_.PSIsContainer }
    }
    return $item
}

function Get-RelativeName([string]$absolute) {
    return $absolute.Substring($repoRoot.Length + 1).Replace('\', '/')
}

$archivePath = Get-WorkspacePath $archiveRelative
$manifestPath = Get-WorkspacePath $manifestRelative
if (Test-Path -LiteralPath $archivePath) { throw 'Archive already exists; inspect the existing manifest instead of repeating cleanup.' }
if (Test-Path -LiteralPath $manifestPath) { throw 'Manifest already exists; refusing to overwrite recovery information.' }

# SQLite sidecars or an open database may mean a running study; stop before moving anything.
foreach ($suffix in @('-wal', '-shm', '-journal')) {
    if (Test-Path -LiteralPath (Get-WorkspacePath ('drone_optimization.db' + $suffix))) {
        throw 'SQLite sidecar exists. Close the study cleanly before cleanup.'
    }
}
$databasePath = Get-WorkspacePath 'drone_optimization.db'
if (Test-Path -LiteralPath $databasePath) {
    $handle = [IO.File]::Open($databasePath, [IO.FileMode]::Open,
                            [IO.FileAccess]::Read, [IO.FileShare]::None)
    $handle.Dispose()
}

$files = @()
$moves = @()
foreach ($relative in $targets) {
    $sourcePath = Get-WorkspacePath $relative
    $destinationPath = Get-WorkspacePath ($archiveRelative + '/' + $relative)
    if (-not (Test-Path -LiteralPath $sourcePath)) { continue }
    foreach ($file in @(Get-CheckedFiles $relative)) {
        $name = Get-RelativeName $file.FullName
        $files += [ordered]@{
            original_path = $name
            archive_path = $archiveRelative + '/' + $name
            bytes = $file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            verified = $false
        }
    }
    $moves += [ordered]@{ original_path = $relative; state = 'planned' }
}
$keptFiles = @()
foreach ($relative in $protected) {
    foreach ($file in @(Get-CheckedFiles $relative)) {
        $keptFiles += [ordered]@{
            path = Get-RelativeName $file.FullName
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
}
$totalBytes = [long]0
foreach ($file in $files) { $totalBytes += $file.bytes }
$manifest = [ordered]@{
    format_version = 1
    operation = 'archive_legacy_artifacts'
    status = 'planned'
    created_at_utc = [DateTime]::UtcNow.ToString('o')
    archive_path = $archiveRelative
    file_count = $files.Count
    total_bytes = $totalBytes
    moves = $moves
    files = $files
    preserved_files = $keptFiles
}
Write-Output "Preview: $($manifest.file_count) files, $($manifest.total_bytes) bytes; archive=$archiveRelative"
if (-not $Apply) { return }
if (-not $files.Count) { throw 'No artifacts found; nothing to archive.' }

function Save-Manifest {
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8
}

# Save recovery information before the first move, and after every completed group.
Save-Manifest
try {
    foreach ($move in $moves) {
        $sourcePath = Get-WorkspacePath $move.original_path
        $destinationPath = Get-WorkspacePath ($archiveRelative + '/' + $move.original_path)
        if (Test-Path -LiteralPath $destinationPath) { throw "Destination exists: $destinationPath" }
        New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($destinationPath)) -Force | Out-Null
        Move-Item -LiteralPath $sourcePath -Destination $destinationPath
        $move.state = 'moved'
        Save-Manifest
    }
    foreach ($file in $files) {
        $path = Get-WorkspacePath $file.archive_path
        if ((Get-Item -LiteralPath $path).Length -ne $file.bytes -or
            (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) {
            throw "Archived file failed verification: $($file.original_path)"
        }
        $file.verified = $true
    }
    foreach ($file in $keptFiles) {
        if ((Get-FileHash -LiteralPath (Get-WorkspacePath $file.path) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) {
            throw "Preserved file changed: $($file.path)"
        }
    }
    $manifest.status = 'complete'
    Save-Manifest
    Write-Output "Complete: archived $($files.Count) files; SHA256 verified; preserved $($keptFiles.Count) files."
} catch {
    $manifest.status = 'incomplete'
    Save-Manifest
    throw
}
