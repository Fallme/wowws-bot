param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\dist")
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$destination = [IO.Path]::GetFullPath((Join-Path $OutputDirectory "wowws_bot_windows.zip"))
[void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination))

# Package current sources, including a newly added bootstrap script, instead
# of silently shipping an older HEAD. Git's ignore rules exclude local data,
# environments and credentials; files missing from the working tree stay out.
$files = @(git -C $projectRoot -c core.quotepath=false ls-files --cached --others --exclude-standard)
if ($LASTEXITCODE -ne 0) { throw 'Cannot list release sources; run this from a Git checkout.' }
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$temporary = Join-Path ([IO.Path]::GetDirectoryName($destination)) ("release_" + [guid]::NewGuid().ToString('N') + '.zip')
$archive = [IO.Compression.ZipFile]::Open($temporary, [IO.Compression.ZipArchiveMode]::Create)
$count = 0
try {
    foreach ($relative in @($files | Sort-Object -Unique)) {
        $source = [IO.Path]::GetFullPath((Join-Path $projectRoot $relative))
        if (-not $source.StartsWith($projectRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Release source escapes project directory: $relative"
        }
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
        [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive, $source, $relative.Replace('\', '/'), [IO.Compression.CompressionLevel]::Optimal
        )
        $count++
    }
} finally { $archive.Dispose() }
Move-Item -LiteralPath $temporary -Destination $destination -Force
Write-Host "Release package created from current sources: $destination ($count files)"
