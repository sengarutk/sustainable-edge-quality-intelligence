import numpy as np
import pytest

from src.params import load_scenario
from src.paths import SCENARIOS


@pytest.fixture(scope="session", params=SCENARIOS)
def scenario(request):
    return load_scenario(request.param)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
