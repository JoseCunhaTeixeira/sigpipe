from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sigpipe.base.transformer import Transformer


class Pipeline:
    def __init__(self, steps: Iterable[Transformer[Any, Any]]) -> None:
        self.steps: list[Transformer[Any, Any]] = list(steps)

    def __rshift__(self, other: Transformer[Any, Any] | Pipeline) -> Pipeline:
        if isinstance(other, Pipeline):
            return Pipeline([*self.steps, *other.steps])
        return Pipeline([*self.steps, other])

    def __repr__(self) -> str:
        return " >> ".join(step.name for step in self.steps)

    def run(
        self,
        data: Any = None,  # noqa: ANN401
        save_log: bool = False,
        show_log: bool = True,
    ) -> Any:  # noqa: ANN401

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{timestamp}.log"
        logger = self._create_logger(log_path, save_log)

        if show_log:
            logger.info("=" * 80)
            logger.info("Pipeline started")
            logger.info("Pipeline: %s", self)

        total_start = perf_counter()

        for i, step in enumerate(self.steps):
            if show_log:
                logger.info(
                    "[%d/%d] Running %s",
                    i + 1,
                    len(self.steps),
                    step.name,
                )

            start = perf_counter()

            try:
                data = step.transform(data)

                duration = perf_counter() - start

                if show_log:
                    logger.info(
                        "%s completed in %.2f s",
                        step.name,
                        duration,
                    )

            except Exception:
                if show_log:
                    logger.exception(
                        "%s failed",
                        step.name,
                    )
                raise

        total_duration = perf_counter() - total_start

        if show_log:
            logger.info(
                "Pipeline completed in %.2f s",
                total_duration,
            )

            logger.info("=" * 80)

        return data

    @staticmethod
    def _create_logger(log_path: Path, save_log: bool = True) -> logging.Logger:

        logger = logging.getLogger("pipeline")

        logger.handlers.clear()

        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        if save_log:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger
