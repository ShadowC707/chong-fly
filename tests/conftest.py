import pytest

from tests.synthetic_graphs import write_tiny_model


@pytest.fixture
def tiny_model_meta(tmp_path):
    return write_tiny_model(tmp_path)
