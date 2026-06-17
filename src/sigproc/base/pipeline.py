import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any


class Pipeline:
    def __init__(self, steps):
        self.steps = list(steps)

    def __rshift__(
        self,
        other,
    ) -> Pipeline:
        return Pipeline(
            [
                *self.steps,
                other,
            ]
        )

    def run(
        self,
        data: Any = None,
        save_log: bool = False,
        show_log: bool = True,
    ) -> Any:

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

    def __repr__(self):
        return " >> ".join(step.name for step in self.steps)
