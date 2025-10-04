import pytest
from pathlib import Path

from execution.risk_management import RiskManager


@pytest.fixture
def killswitch_file():
    """Ensure kill_switch.txt does not exist before/after each test.

    Uses the project root (repository root) where RiskManager expects the file.
    """
    file_path = Path("kill_switch.txt")
    if file_path.exists():
        file_path.unlink()
    try:
        yield file_path
    finally:
        if file_path.exists():
            file_path.unlink()


def test_killswitch_inactive_by_default(killswitch_file):
    rm = RiskManager()
    assert rm.is_killswitch_active() is False
    assert not killswitch_file.exists()


def test_activate_killswitch(killswitch_file):
    rm = RiskManager()
    rm.activate_killswitch()
    assert killswitch_file.exists()
    assert rm.is_killswitch_active() is True


def test_deactivate_killswitch(killswitch_file):
    # Manually create the file first
    killswitch_file.touch(exist_ok=True)
    assert killswitch_file.exists()

    rm = RiskManager()
    rm.deactivate_killswitch()
    assert not killswitch_file.exists()
    assert rm.is_killswitch_active() is False


def test_is_killswitch_active_when_file_exists(killswitch_file):
    killswitch_file.touch(exist_ok=True)
    assert killswitch_file.exists()

    rm = RiskManager()
    assert rm.is_killswitch_active() is True
