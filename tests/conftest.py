import pytest
from pathlib import Path

@pytest.fixture
def project_root():
    return Path(__file__).parent.parent

@pytest.fixture
def scenarios_dir(project_root):
    return project_root / "scenarios"

@pytest.fixture
def config_dir(project_root):
    return project_root / "config"
