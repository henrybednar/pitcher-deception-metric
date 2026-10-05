import argparse
import os
import time

import pytest

import run_pipeline as rp


def args(**kw):
    return argparse.Namespace(only=kw.get("only"), start=kw.get("start"), pull=False, list=False)


def test_steps_are_selected_by_only_and_from_and_unknown_names_are_refused():
    names = [s.name for s in rp.STEPS]

    assert [s.name for s in rp.select_steps(args(only="projection"))] == ["projection"]
    assert [s.name for s in rp.select_steps(args(start="projection"))] == names[names.index("projection"):]
    assert [s.name for s in rp.select_steps(args())] == names
    with pytest.raises(SystemExit):
        rp.select_steps(args(only="nonsense"))


def test_every_step_script_exists_and_each_step_name_is_unique():
    names = [s.name for s in rp.STEPS]

    assert len(names) == len(set(names))
    for step in rp.STEPS:
        for script in step.scripts:
            assert (rp.ROOT / script).exists(), script


def make_project(tmp_path, monkeypatch, body):
    (tmp_path / "w.py").write_text(body, encoding="utf-8")
    monkeypatch.setattr(rp, "ROOT", tmp_path)
    monkeypatch.setattr(rp, "LOG_DIR", tmp_path / "logs")


def test_a_step_that_writes_its_output_passes(tmp_path, monkeypatch):
    make_project(tmp_path, monkeypatch, "open('out.txt', 'w').write('x')\n")

    rp.run_step(rp.Step("t", ("w.py",), ("out.txt",), "note"))

    assert (tmp_path / "out.txt").exists()


def test_a_step_that_leaves_an_old_output_in_place_fails_instead_of_passing_on_the_stale_file(tmp_path, monkeypatch):
    make_project(tmp_path, monkeypatch, "pass\n")
    old = tmp_path / "out.txt"
    old.write_text("from an earlier run")
    long_ago = time.time() - 3600
    os.utime(old, (long_ago, long_ago))

    with pytest.raises(SystemExit, match="did not update"):
        rp.run_step(rp.Step("t", ("w.py",), ("out.txt",), "note"))


def test_a_step_whose_script_fails_or_writes_nothing_stops_the_run(tmp_path, monkeypatch):
    make_project(tmp_path, monkeypatch, "raise SystemExit(3)\n")
    with pytest.raises(SystemExit, match="exited with code 3"):
        rp.run_step(rp.Step("t", ("w.py",), ("out.txt",), "note"))

    make_project(tmp_path, monkeypatch, "pass\n")
    with pytest.raises(SystemExit, match="did not write"):
        rp.run_step(rp.Step("t", ("w.py",), ("out.txt",), "note"))
