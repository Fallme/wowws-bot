param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\dist")
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$destination = Join-Path $OutputDirectory "wowws_bot_windows.zip"
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
Remove-Item -LiteralPath $destination -Force -ErrorAction SilentlyContinue

git -C $projectRoot archive --format=zip --output=$destination HEAD
if ($LASTEXITCODE -ne 0) {
    throw "Release package creation failed. Commit the project changes first."
}

Write-Host "Release package created: $destination"
