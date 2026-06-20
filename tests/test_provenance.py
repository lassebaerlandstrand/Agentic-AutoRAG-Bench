"""Tests for the optimizer-provenance stamp recorded in bench_metadata.json.

Every bench run records which ``agentic-autorag`` build produced it (package
version + git commit), so a results directory self-documents the exact code.
These tests mock ``importlib``/git so they stay deterministic and network-free.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from agentic_autorag_bench import run


def _patch_version(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    monkeypatch.setattr(run, "metadata", SimpleNamespace(version=lambda _name: version))


def _patch_spec(monkeypatch: pytest.MonkeyPatch, origin: str | None) -> None:
    spec = None if origin is None else SimpleNamespace(origin=origin)
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: spec)


def _patch_git(monkeypatch: pytest.MonkeyPatch, *, commit: str, describe: str, porcelain: str) -> None:
    def _fake_git(_repo_root, *args: str) -> str:
        if args[0] == "rev-parse":
            return commit
        if args[0] == "describe":
            return describe
        if args[0] == "status":
            return porcelain
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(run, "_git", _fake_git)


def test_clean_checkout_records_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: version + commit + describe captured, dirty False on a clean tree."""
    _patch_version(monkeypatch, "0.1.0")
    _patch_spec(monkeypatch, "/x/agentic_autorag/__init__.py")
    _patch_git(monkeypatch, commit="abc123", describe="v0.1.0-paper", porcelain="")

    assert run._optimizer_provenance() == {
        "version": "0.1.0",
        "commit": "abc123",
        "describe": "v0.1.0-paper",
        "dirty": False,
    }


def test_dirty_worktree_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-empty ``status --porcelain`` marks the run as built from a dirty tree."""
    _patch_version(monkeypatch, "0.1.0")
    _patch_spec(monkeypatch, "/x/agentic_autorag/__init__.py")
    _patch_git(monkeypatch, commit="abc123", describe="abc123-dirty", porcelain=" M run.py")

    assert run._optimizer_provenance()["dirty"] is True


def test_non_git_install_falls_back_to_version_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Installed from a wheel (no .git): git calls fail, version still recorded."""
    _patch_version(monkeypatch, "0.1.0")
    _patch_spec(monkeypatch, "/site-packages/agentic_autorag/__init__.py")

    def _raise(_repo_root, *_args: str) -> str:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(run, "_git", _raise)

    assert run._optimizer_provenance() == {
        "version": "0.1.0",
        "commit": None,
        "describe": None,
        "dirty": None,
    }


def test_missing_spec_falls_back_to_version_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """No import spec at all: never raises, returns version-only."""
    _patch_version(monkeypatch, "0.1.0")
    _patch_spec(monkeypatch, None)

    assert run._optimizer_provenance() == {
        "version": "0.1.0",
        "commit": None,
        "describe": None,
        "dirty": None,
    }
