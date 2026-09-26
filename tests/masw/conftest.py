from pathlib import Path

import pytest
from synthetic import write_profiles

from sigpipe.masw.workspace import Folders


@pytest.fixture(scope="session")
def input_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("input")
    write_profiles(root)
    return root


@pytest.fixture
def workspace(input_dir: Path, tmp_path: Path) -> Folders:
    return Folders(input_dir=input_dir, output_dir=tmp_path / "output", workers=2)
