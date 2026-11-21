import importlib.util
import pathlib


def test_pid_depth_hold_reaches_reference():
    path = pathlib.Path(__file__).parent.parent / "episodes" / "L4-02-pid-depth-hold" / "run.py"
    spec = importlib.util.spec_from_file_location("ep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    log, card = mod.main()
    assert abs(log["eta"][-1, 2] - 5.0) < 0.05
    assert card["saturation_pct"] < 50
