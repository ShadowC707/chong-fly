import pytest

@pytest.fixture
def tiny_model_meta(tmp_path):
    # Hardware protocol tests need only NumPy/pytest, not the reducer's research dependencies.
    from tests.synthetic_graphs import write_tiny_model
    return write_tiny_model(tmp_path)
