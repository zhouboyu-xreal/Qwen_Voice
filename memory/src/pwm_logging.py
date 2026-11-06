"""Instance-owned, independently configurable log files for PWM modules."""

from __future__ import annotations

import logging
import sys
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping, Optional


class PWMLoggingManager:
    """Create module loggers without changing root or application-owned handlers."""

    MODULES = (
        "pwm", "memory_runtime", "memory_manager", "user_state",
        "user_prediction", "user_proactive_service", "task_scheduler", "llm_client",
    )

    def __init__(
        self, *, memory_db_path: Path, config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        settings = dict(config or {})
        base = memory_db_path.expanduser().resolve().parent
        directory = Path(settings["directory"]).expanduser() if settings.get("directory") else base / "logs"
        self.directory = (base / directory).resolve() if not directory.is_absolute() else directory.resolve()
        modules = settings.get("modules", {})
        if not isinstance(modules, Mapping) or set(modules) - set(self.MODULES):
            raise ValueError("PWM logging.modules must contain only known module names")
        self._loggers: dict[str, logging.Logger] = {}
        self._closed = False
        namespace = f"pwm.{uuid.uuid4().hex}"
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        paths: set[Path] = set()
        try:
            for module in self.MODULES:
                options = modules.get(module, {})
                if not isinstance(options, Mapping):
                    raise ValueError(f"PWM logging.modules.{module} must be a mapping")
                level = str(options.get("level", settings.get("level", "INFO"))).upper()
                if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
                    raise ValueError(f"Invalid logging level for {module}: {level}")
                # A directly owned Logger avoids global getLogger registry retention,
                # as well as handler collisions between users and successive runs.
                logger = logging.Logger(f"{namespace}.{module}", level=level)
                logger.propagate = False
                self._loggers[module] = logger
                if not self._boolean(options.get("enabled", True), "enabled"):
                    logger.addHandler(logging.NullHandler())
                    continue
                filename = str(options.get("filename") or f"{module}.log")
                if Path(filename).name != filename or filename in {".", ".."}:
                    raise ValueError(f"Module log filename must be a basename: {module}")
                path = (self.directory / filename).resolve()
                if path.parent != self.directory or path in paths:
                    raise ValueError(f"Module log paths must be distinct and inside directory: {module}")
                paths.add(path)
                max_bytes = int(options.get("max_bytes", settings.get("max_bytes", 20 * 1024 * 1024)))
                backup_count = int(options.get("backup_count", settings.get("backup_count", 5)))
                if max_bytes < 0 or backup_count < 0 or (max_bytes > 0 and backup_count == 0):
                    raise ValueError(f"Invalid logging rotation settings for {module}")
                self.directory.mkdir(parents=True, exist_ok=True)
                handler = RotatingFileHandler(
                    path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                if self._boolean(options.get("console_enabled", settings.get("console_enabled", False)), "console_enabled"):
                    console = logging.StreamHandler(sys.stderr)
                    console.setFormatter(formatter)
                    logger.addHandler(console)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _boolean(value: Any, field: str) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"PWM logging.{field} must be a boolean")
        return value

    def get_logger(self, module: str) -> logging.Logger:
        if self._closed:
            raise RuntimeError("PWM loggers are closed")
        return self._loggers[module]

    def close(self) -> None:
        """Close only handlers owned by this instance, after its tasks have stopped."""
        if self._closed:
            return
        self._closed = True
        for logger in self._loggers.values():
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                try:
                    handler.flush()
                finally:
                    handler.close()
