"""Focused Windows deployment checks; no pip downloads, panel or game launch."""

import base64
import json
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows bootstrap")
ROOT = Path(__file__).resolve().parents[1]


def ps_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def run_ps(code, *, timeout=30):
    script = (
        "[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false); "
        "$ErrorActionPreference = 'Stop'; "
        f". {ps_literal(ROOT / 'tools/start_project.ps1')}; "
        "$env:WOWS_PYTHON = ''; " + code
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-EncodedCommand", encoded],
        capture_output=True, encoding="utf-8", timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_python_on_path_is_accepted_without_py_launcher():
    # Exercise PATH discovery and the real interpreter probe while hiding py.
    python = ps_literal(sys.executable)
    output = run_ps(
        "function Get-Command { param($Name, $CommandType, [switch]$All, $ErrorAction) "
        f"if ($Name -eq 'python.exe') {{ [pscustomobject]@{{Source={python}}} }} }}; "
        "Find-BasePython"
    )
    assert Path(output).resolve() == Path(sys.executable).resolve()


@pytest.mark.parametrize("version,bits", [([3, 13, 0], 64), ([3, 11, 9], 32)])
def test_other_version_and_32_bit_python_cannot_create_environment(version, bits):
    details = json.dumps({"executable": sys.executable, "version": version, "bits": bits})
    output = run_ps(
        "function Invoke-PythonCommand { [pscustomobject]@{ExitCode=0; Error=''; Output="
        + ps_literal(details) + "} }; "
        f"$result = Test-PythonInterpreter {ps_literal(sys.executable)}; "
        "if ($result) { throw 'wrong interpreter accepted' }; 'rejected'"
    )
    assert output == "rejected"


def test_store_placeholder_never_executes_or_opens_store(tmp_path):
    alias = tmp_path / "Microsoft" / "WindowsApps" / "python.exe"
    alias.parent.mkdir(parents=True)
    alias.touch()
    assert run_ps(
        "function Invoke-PythonCommand { throw 'Store alias executed' }; "
        f"$result = Test-PythonInterpreter {ps_literal(alias)}; "
        "if ($result) { throw 'placeholder accepted' }; 'skipped'"
    ) == "skipped"


def test_custom_interpreter_and_check_only_do_not_create_environment(tmp_path):
    output = run_ps(
        f"$script:ProjectRoot = {ps_literal(tmp_path)}; "
        f"$env:WOWS_PYTHON = {ps_literal(sys.executable)}; "
        "$CheckPythonOnly = $true; Start-Project; 'checked'"
    )
    assert output == "checked"
    assert list(tmp_path.iterdir()) == []


def test_broken_environment_is_backed_up_and_rebuilt_in_non_ascii_path(tmp_path):
    project = tmp_path / "部署 space's! [project]"
    broken = project / ".venv"
    broken.mkdir(parents=True)
    (broken / "keep_marker").write_text("preserved")
    output = run_ps(
        f"$script:ProjectRoot = {ps_literal(project)}; "
        f"$env:WOWS_PYTHON = {ps_literal(sys.executable)}; "
        "$first = Resolve-ProjectPython; $second = Resolve-ProjectPython; "
        "if ($first -ne $second) { throw 'environment not reused' }; $second",
        timeout=60,
    )
    assert Path(output) == project / ".venv" / "Scripts" / "python.exe"
    backups = list(project.glob(".venv_backup_*"))
    assert len(backups) == 1
    assert (backups[0] / "keep_marker").read_text() == "preserved"


def test_python_probe_passes_spaces_quotes_unicode_and_trailing_slashes():
    arguments = ["空 格", "a'b!c", 'quote"value', "trailing\\"]
    args = ",".join(ps_literal(value) for value in arguments)
    output = run_ps(
        f"$result = Invoke-PythonCommand {ps_literal(sys.executable)} "
        f"@('-I','-c','import json,sys; print(json.dumps(sys.argv[1:]))',{args}); "
        "if ($result.ExitCode -ne 0) { throw $result.Error }; $result.Output.Trim()"
    )
    assert json.loads(output) == arguments


def test_bad_explicit_override_reports_path_without_using_another_python(tmp_path):
    assert run_ps(
        f"$env:WOWS_PYTHON = {ps_literal(tmp_path / 'missing.exe')}; "
        "function Get-PythonCandidates { throw 'override silently ignored' }; "
        "try { Find-BasePython; throw 'missing interpreter accepted' } "
        "catch { if ($_.Exception.Message -notlike '*WOWS_PYTHON*') { throw }; 'explicit_error' }"
    ) == "explicit_error"
