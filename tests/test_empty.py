"""Placeholder Test."""

import pytest


def test_unit():
    """Pass non-integration tests in pre-commit.

    For pytest to pass during pre-commit, at least one test must be found and
    passed.
    """
    pass


@pytest.mark.integration_test
def test_integration():
    """Pass integration tests in Coverage.

    For pytest to pass for Coverage, at least one test must be found and
    passed.
    """
    pass
