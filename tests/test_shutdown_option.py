from unittest.mock import patch
from runtime_control import RunLimits, shutdown_after_plan


def test_shutdown_requires_opt_in_normal_completion_and_no_stop(tmp_path):
    with patch('runtime_control.subprocess.run') as run:
        assert not shutdown_after_plan(RunLimits(), True)
        assert not shutdown_after_plan(RunLimits(shutdown_when_done=True), False)
        stop = tmp_path / 'stop'
        stop.touch()
        assert not shutdown_after_plan(RunLimits(shutdown_when_done=True, stop_file=stop), True)
        run.assert_not_called()
        assert shutdown_after_plan(RunLimits(shutdown_when_done=True), True)
        assert run.call_args.args[0] == ['shutdown.exe', '/s', '/t', '0']


def test_shutdown_environment_defaults_off():
    with patch.dict('os.environ', {}, clear=True):
        assert not RunLimits.from_env().shutdown_when_done
    with patch.dict('os.environ', {'WOWS_SHUTDOWN_WHEN_DONE': '1'}, clear=True):
        assert RunLimits.from_env().shutdown_when_done
