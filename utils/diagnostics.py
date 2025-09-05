from __future__ import annotations

import asyncio
import importlib
import os
import sys
from typing import List, Optional

# Ensure project root is on sys.path when running as a script
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from sqlalchemy import text as sa_text
except Exception:  # pragma: no cover - sqlalchemy import issues surfaced at runtime
    sa_text = None  # type: ignore

# Local imports with robust fallbacks (works in flat or packaged modes)
try:
    from core.config_loader import load_settings, Settings  # type: ignore
except Exception:  # pragma: no cover - fallback when imported as a package
    _cfg_mod = importlib.import_module("unicorn_wealth.core.config_loader")
    load_settings = getattr(_cfg_mod, "load_settings")  # type: ignore
    Settings = getattr(_cfg_mod, "Settings")  # type: ignore

try:
    from database.models.base import get_async_engine  # type: ignore
except Exception:  # pragma: no cover - fallback when imported as a package
    _db_mod = importlib.import_module("unicorn_wealth.database.models.base")
    get_async_engine = getattr(_db_mod, "get_async_engine")  # type: ignore


def _print_ok(message: str) -> None:
    print(f"✅ {message}")


def _print_fail(message: str) -> None:
    print(f"❌ {message}")


def _is_placeholder(value: Optional[str]) -> bool:
    if value is None:
        return True
    v = str(value).strip()
    if v == "":
        return True
    lowered = v.lower()
    if lowered in {"changeme", "your_api_key", "placeholder", "none", "n/a"}:
        return True
    if v.startswith("...") or "<insert" in lowered or "<replace" in lowered:
        return True
    return False


def check_gpu() -> bool:
    """Check GPU availability via NVML and CatBoost GPU initialization."""
    # NVML check
    try:
        import pynvml  # type: ignore
    except Exception as e:  # pragma: no cover - environment dependent
        _print_fail(f"pynvml not installed or failed to import: {e}")
        return False

    try:
        pynvml.nvmlInit()
        try:
            drv = pynvml.nvmlSystemGetDriverVersion()
            driver_version = (
                drv.decode() if isinstance(drv, (bytes, bytearray)) else str(drv)
            )
        except Exception:
            driver_version = "unknown"
        try:
            gpu_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            gpu_count = "unknown"
        _print_ok(
            "NVIDIA NVML initialized. " f"Driver: {driver_version}; GPUs: {gpu_count}"
        )
    except Exception as e:  # pragma: no cover - environment dependent
        _print_fail(f"Failed to initialize NVML: {e}")
        return False

    # CatBoost GPU init
    try:
        from catboost import CatBoostClassifier  # type: ignore
    except Exception as e:  # pragma: no cover - environment dependent
        _print_fail(f"CatBoost not installed or failed to import: {e}")
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return False

    try:
        # Instantiation with task_type='GPU' should validate GPU support
        # in the CatBoost build
        _ = CatBoostClassifier(task_type="GPU")
        _print_ok("CatBoost GPU interface is available.")
    except Exception as e:  # pragma: no cover - environment dependent
        _print_fail(f"CatBoost failed to initialize with GPU: {e}")
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return False

    # Shutdown NVML
    try:
        pynvml.nvmlShutdown()
    except Exception:
        # Not fatal
        pass

    return True


async def check_database(settings: Settings) -> bool:
    """Attempt to connect to the database and run basic checks.

    - Executes `SELECT 1` to confirm connection.
    - If using PostgreSQL, confirms TimescaleDB extension is active.
    """
    if sa_text is None:
        _print_fail(
            "sqlalchemy could not be imported; " "cannot perform database checks."
        )
        return False

    engine = None
    try:
        engine = get_async_engine(settings)
        async with engine.connect() as conn:
            # Basic connectivity
            try:
                res = await conn.execute(sa_text("SELECT 1"))
                val = res.scalar()
                if val == 1:
                    _print_ok("Database connection successful (SELECT 1).")
                else:
                    _print_fail("Database connection check returned unexpected result.")
                    return False
            except Exception as e:  # pragma: no cover - environment dependent
                _print_fail(f"Failed to execute connectivity check (SELECT 1): {e}")
                return False

            # TimescaleDB extension check for PostgreSQL
            try:
                dialect = getattr(conn.engine, "dialect", None)
                dialect_name = getattr(dialect, "name", None)
            except Exception:
                dialect_name = None

            if dialect_name == "postgresql":
                try:
                    res = await conn.execute(
                        sa_text(
                            "SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'"
                        )
                    )
                    present = res.scalar()
                    if present == 1:
                        _print_ok("TimescaleDB extension is active.")
                    else:
                        _print_fail(
                            "TimescaleDB extension not active " "in target database."
                        )
                        return False
                except Exception as e:  # pragma: no cover - environment dependent
                    _print_fail(
                        "Failed to verify TimescaleDB extension "
                        "(are you using PostgreSQL?): "
                        f"{e}"
                    )
                    return False
            else:
                _print_ok(
                    "Non-PostgreSQL database detected; "
                    "skipping TimescaleDB extension check."
                )

        return True
    except Exception as e:  # pragma: no cover - environment dependent
        _print_fail(f"Database engine/connection error: {e}")
        return False
    finally:
        if engine is not None:
            try:
                await engine.dispose()
            except Exception:
                pass


def check_api_keys(settings: Settings) -> bool:
    """Validate API-related settings using granular exchange/account toggles.

    Rules:
    - Global provider and Telegram keys are always required.
    - For each exchange in settings.exchange_connection_settings, only check
      credentials for account types explicitly enabled (flag True).
    - For Bitget and Kucoin, include passphrase when checking an enabled account.
    - Disabled accounts are ignored entirely (no warnings).
    """
    # Always-required global credentials
    global_required: List[str] = [
        "coinapi_api_key",
        "santiment_api_key",
        "coinmarketcap_api_key",
        "finnhub_api_key",
        "telegram_api_id",
        "telegram_api_hash",
        "telegram_bot_token",
        "telegram_admin_channel_id",
        "telegram_trade_channel_id",
    ]

    failed_required: List[str] = []

    # Check globals
    for attr in global_required:
        try:
            value = getattr(settings, attr)
        except Exception:
            value = None
        if _is_placeholder(value):
            failed_required.append(attr)

    # Exchanges that require a passphrase for accounts
    passphrase_exchanges = {"bitget", "kucoin"}

    # Iterate exchanges and account types per toggles
    try:
        toggles = settings.exchange_connection_settings or {}
    except Exception:
        toggles = {}

    for exchange, accounts in toggles.items():
        for account in ("unicorn", "personal"):
            if not isinstance(accounts, dict):
                continue
            enabled = bool(accounts.get(account))
            if not enabled:
                continue

            # Build required attributes for this enabled account
            base = f"{exchange}_{account}"
            required_attrs = [
                f"{base}_api_key",
                f"{base}_api_secret",
            ]
            if exchange in passphrase_exchanges:
                required_attrs.append(f"{base}_api_passphrase")

            for attr in required_attrs:
                try:
                    value = getattr(settings, attr)
                except Exception:
                    value = None
                if _is_placeholder(value):
                    failed_required.append(attr)

    if failed_required:
        for k in failed_required:
            _print_fail(f"Required credential not set or placeholder: {k}")
        return False

    _print_ok("All required API credentials (based on toggles) are present and valid.")
    return True


def check_filesystem(settings: Settings) -> bool:
    """Ensure the application can write logs to the configured path."""
    log_path = settings.log_file_path
    dir_path = os.path.dirname(log_path) or "."

    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:  # pragma: no cover - environment dependent
        _print_fail(f"Failed to create log directory '{dir_path}': {e}")
        return False

    if not os.access(dir_path, os.W_OK):
        _print_fail(f"No write permission for log directory: {dir_path}")
        return False

    # Attempt to open the log file for append
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("")
    except Exception as e:  # pragma: no cover - environment dependent
        _print_fail(f"Cannot write to log file '{log_path}': {e}")
        return False

    _print_ok(f"Filesystem permissions OK for logs at: {log_path}")
    return True


async def main() -> None:
    any_failures = False

    # Load settings
    try:
        settings = load_settings()
        _print_ok("Settings loaded from environment (.env).")
    except Exception as e:  # pragma: no cover - configuration dependent
        _print_fail(f"Failed to load settings: {e}")
        sys.exit(1)
        return

    print("\n--- Running Environment Diagnostics ---")

    # GPU check
    print("\n[GPU]")
    gpu_ok = check_gpu()
    any_failures = any_failures or (not gpu_ok)

    # Database check
    print("\n[Database]")
    db_ok = await check_database(settings)
    any_failures = any_failures or (not db_ok)

    # API keys check
    print("\n[API Keys]")
    api_ok = check_api_keys(settings)
    any_failures = any_failures or (not api_ok)

    # Filesystem check
    print("\n[Filesystem]")
    fs_ok = check_filesystem(settings)
    any_failures = any_failures or (not fs_ok)

    print("\n--- Diagnostics Summary ---")
    if not any_failures:
        _print_ok("All environment checks passed. System is ready to run.")
        sys.exit(0)
    else:
        _print_fail("One or more checks failed. Please review the messages above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
