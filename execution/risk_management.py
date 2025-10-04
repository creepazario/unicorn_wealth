from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


class RiskManager:
    """System-wide manual kill-switch manager.

    This component provides a simple, file-based kill-switch that can be used to
    immediately prevent the system from initiating any new trades. The presence
    of a file named ``kill_switch.txt`` at the project root directory indicates
    that the kill-switch is active.

    Usage:
        rm = RiskManager()
        if rm.is_killswitch_active():
            # Skip placing new orders
            ...

        # Activate or deactivate as needed
        rm.activate_killswitch()
        rm.deactivate_killswitch()
    """

    KILL_SWITCH_FILENAME: str = "kill_switch.txt"

    def __init__(self, project_root: Optional[Path] = None) -> None:
        """Initialize the RiskManager.

        Args:
            project_root: Optional explicit path to the project root. If not
                provided, it defaults to the repository root inferred from this
                file's location (two directories up from this file, i.e., the
                package root). This keeps the path robust in various
                environments.
        """
        # Determine project root robustly using this file's path if not provided
        if project_root is None:
            # this file: <root>/execution/risk_management.py
            project_root = Path(__file__).resolve().parents[1]
        self.project_root: Path = project_root
        self.kill_switch_path: Path = self.project_root / self.KILL_SWITCH_FILENAME
        self._log = logging.getLogger(__name__)

    def is_killswitch_active(self) -> bool:
        """Return True if the kill-switch file exists, False otherwise."""
        return self.kill_switch_path.exists()

    def activate_killswitch(self) -> None:
        """Create the kill-switch file and log a CRITICAL message.

        If the file already exists, this is a no-op aside from logging.
        """
        try:
            # Touch creates the file if it doesn't exist
            self.kill_switch_path.touch(exist_ok=True)
            self._log.critical(
                "Kill-switch ACTIVATED: new trades must be halted. Path=%s",
                str(self.kill_switch_path),
            )
        except Exception as exc:
            # Log the exception but do not raise to avoid cascading failures
            self._log.exception(
                "Failed to activate kill-switch at %s: %s",
                str(self.kill_switch_path),
                exc,
            )

    def deactivate_killswitch(self) -> None:
        """Remove the kill-switch file if it exists and log a WARNING message."""
        try:
            if self.kill_switch_path.exists():
                self.kill_switch_path.unlink()
            self._log.warning(
                "Kill-switch DEACTIVATED: system may resume opening new trades. Path=%s",
                str(self.kill_switch_path),
            )
        except Exception as exc:
            self._log.exception(
                "Failed to deactivate kill-switch at %s: %s",
                str(self.kill_switch_path),
                exc,
            )
