import json

from frsbot.baseline import route_follow


def test_main_runs_without_changing_baseline(tmp_path, monkeypatch, capsys):
    expected = json.loads((route_follow.SIM / "sim-baseline-254.json").read_text())
    monkeypatch.setattr(route_follow, "SIM", tmp_path)
    route_follow.main()
    actual = json.loads((tmp_path / "sim-baseline-254.json").read_text())
    assert actual == expected
    output = capsys.readouterr().out
    assert "robot Overload  hopper capacity 72" in output
    assert "waypoints reached" in output
