import pytest
from pytest_mock import MockerFixture

from src.celery import DocumentStorePreflight, celery_app
from src.document_store.preflight import PreflightError


def test_preflight_bootstep_is_registered_on_the_worker() -> None:
    assert DocumentStorePreflight in celery_app.steps["worker"]


def test_bootstep_start_runs_preflight(mocker: MockerFixture) -> None:
    run_preflight = mocker.patch(
        "src.document_store.preflight.run_preflight", return_value=None
    )
    step = DocumentStorePreflight(mocker.Mock())
    step.start(mocker.Mock())
    run_preflight.assert_called_once()


def test_bootstep_propagates_preflight_failure(mocker: MockerFixture) -> None:
    # Bootstep exceptions must propagate (aborting worker startup with a nonzero
    # exit), unlike signal receivers which may only log.
    mocker.patch(
        "src.document_store.preflight.run_preflight",
        side_effect=PreflightError("bad store config"),
    )
    step = DocumentStorePreflight(mocker.Mock())
    with pytest.raises(PreflightError, match="bad store config"):
        step.start(mocker.Mock())
