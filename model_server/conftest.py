import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


def pytest_configure(config):
    # Milestone 4's grader is grouped by TODO so each one can be run on its own
    # while you work through batching.py: `pytest test_batching.py -m todo3`.
    # Registering them here keeps pytest from warning about unknown markers.
    for n in range(1, 7):
        config.addinivalue_line("markers", f"todo{n}: milestone 4 batching.py TODO {n}")
    config.addinivalue_line("markers", "todoe2e: milestone 4 end-to-end tests")
