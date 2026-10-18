[CmdletBinding()]
param([switch]$CheckPythonOnly)

$script:ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$script:StartLog = $null
$script:Utf8 = New-Object System.Text.UTF8Encoding($false)

function Write-StartupLog([string]$Message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host $line
    if ($script:StartLog) {
        [IO.File]::AppendAllText($script:StartLog, "$line`r`n", $script:Utf8)
    }
}

function ConvertTo-NativeArgument([string]$Value) {
    # Windows PowerShell 5.1 has no ProcessStartInfo.ArgumentList.
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Invoke-PythonCommand {
    param([string]$Executable, [string[]]$Arguments, [int]$TimeoutSeconds = 15)
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $Executable
    $info.Arguments = ($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' '
    $info.WorkingDirectory = $script:ProjectRoot
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8'
    $info.StandardOutputEncoding = $script:Utf8
    $info.StandardErrorEncoding = $script:Utf8
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    try {
        [void]$process.Start()
        $output = $process.StandardOutput.ReadToEndAsync()
        $errors = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $process.Kill()
            $process.WaitForExit()
            return [pscustomobject]@{ ExitCode = -1; Output = ''; Error = '执行超时' }
        }
        return [pscustomobject]@{
            ExitCode = $process.ExitCode; Output = $output.Result; Error = $errors.Result
        }
    } finally {
        $process.Dispose()
    }
}

function Test-PythonInterpreter {
    param([string]$Executable, [string[]]$PrefixArguments = @())
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $null }
    # WindowsApps aliases can open the Store instead of running an interpreter.
    if ($Executable -match '\\Microsoft\\WindowsApps\\[^\\]+\.exe$') { return $null }
    $probe = 'import json,struct,sys; print(json.dumps(dict(executable=sys.executable,version=list(sys.version_info[:3]),bits=struct.calcsize(''P'')*8)))'
    try {
        $result = Invoke-PythonCommand $Executable ($PrefixArguments + @('-I', '-c', $probe))
        if ($result.ExitCode -ne 0) {
            Write-StartupLog "不可用的 Python：$Executable；$($result.Error.Trim())"
            return $null
        }
        $details = $result.Output.Trim() | ConvertFrom-Json
        if ($details.version[0] -ne 3 -or $details.version[1] -ne 11 -or $details.bits -ne 64) {
            Write-StartupLog "跳过 $Executable：Python $($details.version -join '.') / $($details.bits) 位，需要 3.11 / 64 位"
            return $null
        }
        return [string]$details.executable
    } catch {
        Write-StartupLog "无法检查 Python：$Executable；$($_.Exception.Message)"
        return $null
    }
}

function Get-PythonCandidates {
    foreach ($name in @('py.exe', 'python3.11.exe', 'python311.exe', 'python.exe', 'python3.exe')) {
        foreach ($command in @(Get-Command $name -CommandType Application -All -ErrorAction SilentlyContinue)) {
            [pscustomobject]@{ Path = $command.Source; Prefix = $(if ($name -eq 'py.exe') { @('-3.11') } else { @() }) }
        }
    }
    foreach ($path in @(
        "$env:WINDIR\py.exe", "$env:LOCALAPPDATA\Programs\Python\Launcher\py.exe"
    )) {
        [pscustomobject]@{ Path = $path; Prefix = @('-3.11') }
    }
    # PEP 514 registrations cover per-user and all-user installs without PATH.
    foreach ($hive in @([Microsoft.Win32.RegistryHive]::CurrentUser, [Microsoft.Win32.RegistryHive]::LocalMachine)) {
        foreach ($view in @([Microsoft.Win32.RegistryView]::Registry64, [Microsoft.Win32.RegistryView]::Registry32)) {
            $base = $null
            $pythonKey = $null
            try {
                $base = [Microsoft.Win32.RegistryKey]::OpenBaseKey($hive, $view)
                $pythonKey = $base.OpenSubKey('Software\Python')
                if (-not $pythonKey) { continue }
                foreach ($company in $pythonKey.GetSubKeyNames()) {
                    $companyKey = $pythonKey.OpenSubKey($company)
                    try {
                        foreach ($tag in $companyKey.GetSubKeyNames()) {
                            $install = $companyKey.OpenSubKey("$tag\InstallPath")
                            if (-not $install) { continue }
                            try {
                                $path = $install.GetValue('ExecutablePath')
                                if (-not $path -and $install.GetValue('')) { $path = Join-Path $install.GetValue('') 'python.exe' }
                                if ($path) { [pscustomobject]@{ Path = [Environment]::ExpandEnvironmentVariables($path); Prefix = @() } }
                            } finally { $install.Dispose() }
                        }
                    } finally { $companyKey.Dispose() }
                }
            } catch {
                Write-Verbose "Python 注册表读取跳过：$($_.Exception.Message)"
            } finally {
                if ($pythonKey) { $pythonKey.Dispose() }
                if ($base) { $base.Dispose() }
            }
        }
    }
    foreach ($path in @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:ProgramFiles\Python311\python.exe", "${env:ProgramFiles(x86)}\Python311\python.exe",
        "$env:SystemDrive\Python311\python.exe"
    )) {
        [pscustomobject]@{ Path = $path; Prefix = @() }
    }
    if (Get-Command Get-AppxPackage -ErrorAction SilentlyContinue) {
        foreach ($package in @(Get-AppxPackage -Name 'PythonSoftwareFoundation.Python.3.11' -ErrorAction SilentlyContinue)) {
            [pscustomobject]@{ Path = (Join-Path $package.InstallLocation 'python3.11.exe'); Prefix = @() }
            [pscustomobject]@{ Path = (Join-Path $package.InstallLocation 'python.exe'); Prefix = @() }
        }
    }
}

function Find-BasePython {
    if ($env:WOWS_PYTHON) {
        $configured = [Environment]::ExpandEnvironmentVariables($env:WOWS_PYTHON.Trim('"'))
        $selected = Test-PythonInterpreter $configured
        if ($selected) { return $selected }
        throw "WOWS_PYTHON 指定的解释器不可用：$configured。请指定 64 位 Python 3.11 的 python.exe。"
    }
    $seen = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($candidate in Get-PythonCandidates) {
        if (-not $seen.Add($candidate.Path)) { continue }
        $selected = Test-PythonInterpreter $candidate.Path @($candidate.Prefix)
        if ($selected) { return $selected }
    }
    throw "没有找到可用的 64 位 Python 3.11（已检查 py、python、注册表和常见目录）。请从 https://www.python.org/downloads/windows/ 安装 Python 3.11 的 Windows installer (64-bit)，然后重新双击 start_project.bat；也可通过 WOWS_PYTHON 指定已有 python.exe 的完整路径。"
}

function Resolve-ProjectPython([switch]$CheckOnly) {
    $venv = Join-Path $script:ProjectRoot '.venv'
    $python = Join-Path $venv 'Scripts\python.exe'
    $existing = Test-PythonInterpreter $python
    if ($existing) {
        Write-StartupLog "使用项目 Python：$existing（3.11 / 64 位）"
        return $existing
    }
    $basePython = Find-BasePython
    Write-StartupLog "找到 Python：$basePython（3.11 / 64 位）"
    if ($CheckOnly) { return $basePython }
    if (Test-Path -LiteralPath $venv) {
        $backup = Join-Path $script:ProjectRoot ('.venv_backup_' + [guid]::NewGuid().ToString('N'))
        $rootPrefix = $script:ProjectRoot.TrimEnd('\') + '\'
        foreach ($target in @($venv, $backup)) {
            if (-not [IO.Path]::GetFullPath($target).StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "虚拟环境备份路径超出项目目录：$target"
            }
        }
        Move-Item -LiteralPath $venv -Destination $backup
        Write-StartupLog "旧虚拟环境不可用，已保留到 $backup"
    }
    Write-StartupLog '正在创建项目虚拟环境 .venv'
    $created = Invoke-PythonCommand $basePython @('-I', '-m', 'venv', $venv) 120
    if ($created.ExitCode -ne 0) { throw "虚拟环境创建失败：$($created.Error.Trim())" }
    $selected = Test-PythonInterpreter $python
    if (-not $selected) { throw '创建后的虚拟环境无法运行，请查看上方 Python 检查信息。' }
    return $selected
}

function Install-ProjectDependencies([string]$Python) {
    $imports = 'import mss,cv2,numpy,pathfinding,yaml,win32api,vgamepad,rapidocr,onnxruntime'
    Write-StartupLog '正在检查项目依赖'
    $check = Invoke-PythonCommand $Python @('-I', '-c', $imports) 60
    if ($check.ExitCode -eq 0) { return }
    Write-StartupLog "依赖需要安装或修复：$($check.Error.Trim())"
    $pip = Invoke-PythonCommand $Python @('-I', '-m', 'pip', '--version')
    if ($pip.ExitCode -ne 0) {
        $ensure = Invoke-PythonCommand $Python @('-I', '-m', 'ensurepip', '--upgrade') 120
        if ($ensure.ExitCode -ne 0) { throw "pip 修复失败：$($ensure.Error.Trim())" }
    }
    Write-StartupLog '首次安装需要联网，下面会显示安装进度；失败后再次双击可继续。'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUNBUFFERED = '1'
    $env:PIP_NO_INPUT = '1'
    # Keep native stderr as log text in Windows PowerShell 5.1.
    $ErrorActionPreference = 'Continue'
    & $Python -m pip install --disable-pip-version-check --prefer-binary --progress-bar off -r (Join-Path $script:ProjectRoot 'requirements.txt') 2>&1 | ForEach-Object {
        Write-StartupLog $_.ToString()
    }
    $installExit = $LASTEXITCODE
    if ($installExit -ne 0) { throw '依赖安装失败。请检查上方 pip 错误及网络连接后重试。' }
    $check = Invoke-PythonCommand $Python @('-I', '-c', $imports) 60
    if ($check.ExitCode -ne 0) { throw "安装后依赖仍无法加载：$($check.Error.Trim())" }
}

function Test-ProjectPanel([string]$Url) {
    try {
        $status = Invoke-RestMethod -Uri "$Url/api/status" -TimeoutSec 2 -Proxy $null
        $names = @($status.PSObject.Properties.Name)
        return ('running' -in $names -and 'state' -in $names -and 'run_id' -in $names)
    } catch { return $false }
}

function Start-Project {
    if ($CheckPythonOnly) {
        [void](Resolve-ProjectPython -CheckOnly)
        Write-StartupLog 'Python 检查完成；未创建环境、安装依赖或启动控制台。'
        return
    }
    [void][IO.Directory]::CreateDirectory((Join-Path $script:ProjectRoot 'data'))
    $script:StartLog = Join-Path $script:ProjectRoot 'data\startup.log'
    Write-StartupLog "启动入口：$script:ProjectRoot"
    $port = 8899
    if ($env:WOWS_PANEL_PORT -and (-not [int]::TryParse($env:WOWS_PANEL_PORT, [ref]$port) -or $port -lt 1 -or $port -gt 65535)) {
        throw 'WOWS_PANEL_PORT 必须是 1 到 65535 之间的端口号。'
    }
    $url = "http://127.0.0.1:$port"
    if (Test-ProjectPanel $url) {
        Write-StartupLog '已有控制台正在运行，打开网页'
        Start-Process $url
        return
    }
    $python = Resolve-ProjectPython
    Install-ProjectDependencies $python
    Write-StartupLog '正在启动本地控制台'
    $stdout = Join-Path $script:ProjectRoot 'data\control_server.stdout.log'
    $stderr = Join-Path $script:ProjectRoot 'data\control_server.stderr.log'
    $process = Start-Process -FilePath $python -ArgumentList @('control_server.py', '--no-browser') -WorkingDirectory $script:ProjectRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    $deadline = [DateTime]::UtcNow.AddSeconds(45)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-ProjectPanel $url) {
            Write-StartupLog "控制台已启动：$url"
            Start-Process $url
            return
        }
        if ($process.HasExited) {
            $tail = (Get-Content -LiteralPath $stderr -Tail 15 -ErrorAction SilentlyContinue) -join "`n"
            throw "控制台提前退出（退出码 $($process.ExitCode)）：`n$tail"
        }
        Start-Sleep -Milliseconds 500
    }
    throw "等待控制台超过 45 秒。请查看 $stderr；如果端口 $port 被占用，可设置 WOWS_PANEL_PORT 后重试。"
}

# Dot sourcing loads functions for the focused offline checks only.
if ($MyInvocation.InvocationName -ne '.') {
    $ErrorActionPreference = 'Stop'
    try { Start-Project; exit 0 }
    catch { Write-StartupLog "启动失败：$($_.Exception.Message)"; exit 1 }
}
