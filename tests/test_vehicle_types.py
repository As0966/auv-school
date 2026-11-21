import importlib.util
import pathlib


def test_hover_vehicle_pushes_every_way_and_torpedo_travels_cheaply():
    path = pathlib.Path(__file__).parent.parent / "episodes" / "L1-01-vehicle-types" / "run.py"
    spec = importlib.util.spec_from_file_location("ep_types", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hover, torp = mod.main()
    assert hover["rank"] == 6 and len(hover["directions"]) == 6
    assert torp["directions"] == ["forward"]
    assert torp["power"][1.5] < hover["power"][1.5] / 3
    assert torp["distance"] > hover["distance"]
