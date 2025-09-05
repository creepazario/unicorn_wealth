"""
Minimal stub of the aiosqlite module for test environments where the actual
package is not installed. This stub is only intended to satisfy SQLAlchemy's
import during engine creation in tests that do not perform real DB I/O.

If any function here is actually invoked, a clear error is raised instructing
users to install the real dependency.
"""

from __future__ import annotations

__all__ = [
    "connect",
    "Row",
    "Connection",
    "Cursor",
    # exceptions
    "Error",
    "DatabaseError",
    "IntegrityError",
    "NotSupportedError",
    "OperationalError",
    "ProgrammingError",
    # version
    "sqlite_version",
    "sqlite_version_info",
]


# Define exception hierarchy expected by SQLAlchemy's aiosqlite adapter
class Error(Exception):  # pragma: no cover - placeholder
    pass


class DatabaseError(Error):  # pragma: no cover - placeholder
    pass


class IntegrityError(DatabaseError):  # pragma: no cover - placeholder
    pass


class NotSupportedError(DatabaseError):  # pragma: no cover - placeholder
    pass


class OperationalError(DatabaseError):  # pragma: no cover - placeholder
    pass


class ProgrammingError(DatabaseError):  # pragma: no cover - placeholder
    pass


# Provide sqlite version attributes
sqlite_version = "3.45.0"  # arbitrary placeholder
sqlite_version_info = (3, 45, 0)


class _NotImplementedProxy:
    def __getattr__(self, name: str):  # pragma: no cover - defensive
        raise RuntimeError(
            "aiosqlite stub in use. Install 'aiosqlite' to use SQLite async DB access."
        )


class Row:  # pragma: no cover - placeholder type
    pass


class Connection(_NotImplementedProxy):  # pragma: no cover - placeholder type
    pass


class Cursor(_NotImplementedProxy):  # pragma: no cover - placeholder type
    pass


async def connect(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise RuntimeError(
        "aiosqlite stub in use. Install 'aiosqlite' to establish async SQLite "
        "connections."
    )
