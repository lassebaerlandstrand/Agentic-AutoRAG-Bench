"""``--resume`` must imply ``--no-clean`` so the documented resume command
works on its own (and so the launcher can pass ``--resume`` after a crash).
Only an EXPLICIT ``--clean`` alongside ``--resume`` is a conflict.
"""

from __future__ import annotations

from typer.testing import CliRunner

import agentic_autorag_bench.run as run_module
from agentic_autorag_bench.cli import app

runner = CliRunner()


def _capture_clean(monkeypatch) -> dict:
    captured: dict = {}

    def fake_run_cli(config, *, methods=None, clean=True, resume=False, force=False):  # noqa: ANN001
        captured.update(config=config, methods=methods, clean=clean, resume=resume, force=force)

    monkeypatch.setattr(run_module, "run_cli", fake_run_cli)
    return captured


def test_resume_implies_no_clean(monkeypatch) -> None:
    captured = _capture_clean(monkeypatch)
    result = runner.invoke(app, ["run", "-c", "x.yaml", "--resume"])
    assert result.exit_code == 0, result.output
    assert captured["resume"] is True
    assert captured["clean"] is False  # implied


def test_default_is_clean(monkeypatch) -> None:
    captured = _capture_clean(monkeypatch)
    result = runner.invoke(app, ["run", "-c", "x.yaml"])
    assert result.exit_code == 0, result.output
    assert captured["clean"] is True
    assert captured["resume"] is False


def test_no_clean_without_resume(monkeypatch) -> None:
    captured = _capture_clean(monkeypatch)
    result = runner.invoke(app, ["run", "-c", "x.yaml", "--no-clean"])
    assert result.exit_code == 0, result.output
    assert captured["clean"] is False
    assert captured["resume"] is False


def test_explicit_clean_with_resume_errors(monkeypatch) -> None:
    _capture_clean(monkeypatch)
    result = runner.invoke(app, ["run", "-c", "x.yaml", "--resume", "--clean"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_force_flag_passed_through(monkeypatch) -> None:
    captured = _capture_clean(monkeypatch)
    result = runner.invoke(app, ["run", "-c", "x.yaml", "--force"])
    assert result.exit_code == 0, result.output
    assert captured["force"] is True
