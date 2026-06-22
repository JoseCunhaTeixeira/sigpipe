from pathlib import Path

from sigproc.base.inversion import InversionResult
from sigproc.dataio.velocity_model.saving import save_velocity_model


def save_inversion_result(result: InversionResult, path: Path) -> None:
    save_velocity_model(result.best, path.with_name(f"{path.stem}_best{path.suffix}"))
    save_velocity_model(result.median, path.with_name(f"{path.stem}_median{path.suffix}"))
    save_velocity_model(result.ensemble, path.with_name(f"{path.stem}_ensemble{path.suffix}"))
