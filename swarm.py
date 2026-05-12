"""
=============================================================================
utils/logger.py  –  Structured Flight Logger
=============================================================================
Color-coded, structured logging for the UAV system using Rich + Loguru.
Supports console, file (rotating), and telemetry sinks.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Try loguru, fallback to stdlib logging
try:
    from loguru import logger as _loguru_logger
    _HAS_LOGURU = True
except ImportError:
    _HAS_LOGURU = False

import logging

# Try rich console
try:
    from rich.logging import RichHandler
    from rich.console import Console
    _HAS_RICH = True
    _console = Console()
except ImportError:
    _HAS_RICH = False


# ──────────────────────────────────────────────────────────────────────────────
# UAVLogger
# ──────────────────────────────────────────────────────────────────────────────

class UAVLogger:
    """
    Structured logger for UAV subsystems.
    - Color-coded by subsystem (SENSORS, NAV, RL, VISION, etc.)
    - Rotating file handler (10 MB, 7 days retention)
    - Telemetry event hooks
    """

    SUBSYSTEM_COLORS = {
        "SENSORS":   "cyan",
        "FUSION":    "bright_cyan",
        "NAV":       "green",
        "RL":        "yellow",
        "VISION":    "magenta",
        "SWARM":     "blue",
        "DASHBOARD": "white",
        "MISSION":   "bright_green",
        "SYSTEM":    "red",
    }

    def __init__(
        self,
        name: str,
        log_dir: Optional[str] = None,
        level: str = "INFO",
        enable_file: bool = True,
    ):
        self.name = name.upper()
        self._level = level
        self._log_dir = Path(log_dir) if log_dir else Path("logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._logger = self._build_logger(enable_file)

    def _build_logger(self, enable_file: bool) -> logging.Logger:
        log = logging.getLogger(self.name)
        log.setLevel(getattr(logging, self._level, logging.INFO))
        log.handlers.clear()

        if _HAS_RICH:
            handler = RichHandler(
                console=_console,
                rich_tracebacks=True,
                markup=True,
                show_time=True,
                show_level=True,
                show_path=False,
            )
        else:
            handler = logging.StreamHandler(sys.stdout)
            fmt = logging.Formatter(
                f"%(asctime)s | [{self.name}] %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(fmt)

        log.addHandler(handler)

        if enable_file:
            log_file = self._log_dir / f"uav_{self.name.lower()}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=7
            ) if hasattr(logging, "handlers") else logging.FileHandler(log_file)
            file_fmt = logging.Formatter(
                "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s"
            )
            file_handler.setFormatter(file_fmt)
            log.addHandler(file_handler)

        return log

    # ── Logging interface ─────────────────────────────────────────────────────

    def debug(self, msg: str, **kwargs) -> None:
        self._logger.debug(f"[{self.name}] {msg}", **kwargs)

    def info(self, msg: str, **kwargs) -> None:
        self._logger.info(f"[{self.name}] {msg}", **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        self._logger.warning(f"[{self.name}] {msg}", **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        self._logger.error(f"[{self.name}] {msg}", **kwargs)

    def critical(self, msg: str, **kwargs) -> None:
        self._logger.critical(f"[{self.name}] {msg}", **kwargs)

    def flight_event(self, event: str, data: dict) -> None:
        """Log a structured flight event (waypoint reached, collision, etc.)."""
        parts = " | ".join(f"{k}={v}" for k, v in data.items())
        self.info(f"[EVENT:{event}] {parts}")

    def telemetry(self, pos, vel, att, battery_pct: float) -> None:
        """Log condensed telemetry line."""
        self.debug(
            f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) "
            f"vel=({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}) "
            f"att=({att[0]:.2f},{att[1]:.2f},{att[2]:.2f}) "
            f"bat={battery_pct:.1f}%"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level factory
# ──────────────────────────────────────────────────────────────────────────────

import logging.handlers  # noqa: E402  (needed for RotatingFileHandler)

_loggers: dict[str, UAVLogger] = {}


def get_logger(name: str, level: str = "INFO") -> UAVLogger:
    """Return a cached UAVLogger for the given subsystem name."""
    key = name.upper()
    if key not in _loggers:
        _loggers[key] = UAVLogger(name=key, level=level)
    return _loggers[key]
