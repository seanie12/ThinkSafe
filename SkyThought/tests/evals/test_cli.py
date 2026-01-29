import json
import os

import pytest
from typer.testing import CliRunner

from skythought.evals.cli import (
    Backend,
    SamplingParameters,
    app,
)

runner = CliRunner(mix_stderr=False)


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    yield
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)


@pytest.fixture
def tmp_result_dir(tmp_path):
    return tmp_path / "results"


def test_evaluate_missing_required_args():
    """Test that the `evaluate` command fails when required arguments are missing."""
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code != 0
    assert "Missing option" in result.stderr


def test_evaluate_invalid_task():
    """Test that providing an invalid task raises a ValueError."""
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--task",
            "invalid_task",
            "--model",
            "openai/gpt-3",
            "--backend",
            "VLLM",
        ],
    )
    assert result.exit_code != 0
    assert "invalid value" in result.stderr.lower()


def test_evaluate_invalid_batch_size():
    """Test that providing batch size for a non-vllm backend raises a ValueError."""
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--task",
            "amc23",
            "--model",
            "openai/gpt-3",
            "--backend",
            "openai",  # not VLLM
            "--batch-size",
            "128",  # Invalid for non-VLLM backend
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "Batch size is only supported for the vllm backend." in str(result.exception)


def test_evaluate_success(tmp_result_dir, mocker):
    """Test a successful execution of the `evaluate` command."""
    # Mock helper functions to avoid actual processing
    mocker.patch(
        "skythought.evals.cli.parse_common_args",
        return_value=(
            "amc23",
            {},
            "openai/gpt-3",
            "VLLM",
            {},
            {},
            {},
            SamplingParameters.from_dict(Backend.VLLM, {}),
            1,
            64,
            None,
            None,
        ),
    )
    mocker.patch("skythought.evals.cli.get_run_config", return_value={})
    mocker.patch("skythought.evals.cli.get_output_dir", return_value=tmp_result_dir)
    mocker.patch("skythought.evals.cli.generate_and_score")

    # Ensure the result directory does not exist initially
    assert not tmp_result_dir.exists()

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--task",
            "amc23",
            "--model",
            "openai/gpt-3",
            "--backend",
            "VLLM",
            "--result-dir",
            str(tmp_result_dir),
            "--n",
            "1",
        ],
    )
    assert result.exit_code == 0, f"Run failed with exception {result.exception}"
    assert tmp_result_dir.exists()


def test_generate_invalid_sampling_params():
    """Test that providing invalid sampling parameters raises an error."""
    result = runner.invoke(
        app,
        [
            "generate",
            "--task",
            "amc23",
            "--model",
            "openai/gpt-3",
            "--sampling-params",
            "invalid_param",
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert (
        "got invalid argument" in str(result.exception).lower()
    )  # Adjust based on actual error message


def test_generate_success(tmp_result_dir, mocker):
    """Test a successful execution of the `generate` command."""
    # Mock helper functions to avoid actual processing
    mocker.patch(
        "skythought.evals.cli.parse_common_args",
        return_value=(
            "amc23",
            {},
            "openai/gpt-3",
            "VLLM",
            {},
            {},
            {},
            SamplingParameters.from_dict(Backend.VLLM, {}),
            1,
            64,
            None,
            None,
        ),
    )
    mocker.patch("skythought.evals.cli.get_run_config", return_value={})
    mocker.patch("skythought.evals.cli.get_output_dir", return_value=tmp_result_dir)
    mocker.patch("skythought.evals.cli.generate_and_save")

    result = runner.invoke(
        app,
        [
            "generate",
            "--task",
            "amc23",
            "--model",
            "openai/gpt-3",
            "--backend",
            "VLLM",
            "--result-dir",
            str(tmp_result_dir),
            "--n",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert tmp_result_dir.exists()


def test_score_missing_run_dir():
    """Test that the `score` command fails when the run directory does not exist."""
    result = runner.invoke(
        app,
        [
            "score",
            "--run-dir",
            "non_existent_dir",
            "--task",
            "amc23",
        ],
    )
    assert result.exit_code != 0
    assert "Run directory non_existent_dir does not exist." in str(result.exception)


def test_score_invalid_task():
    """
    Test that providing an invalid task to `score` raises a ValueError.
    """
    # Create a temporary run directory with a summary.json
    with runner.isolated_filesystem():
        os.makedirs("run_dir")
        with open("run_dir/summary.json", "w") as f:
            json.dump({}, f)

        result = runner.invoke(
            app,
            [
                "score",
                "--run-dir",
                "run_dir",
                "--task",
                "invalid_task",
            ],
        )
        assert result.exit_code != 0
        # error to be raised by Click/Typer
        assert "invalid value" in result.stderr.lower()


def test_score_missing_summary_file():
    """
    Test that missing summary.json in the run directory raises a ValueError.
    """
    with runner.isolated_filesystem():
        os.makedirs("run_dir")
        # Do not create summary.json

        result = runner.invoke(
            app,
            [
                "score",
                "--run-dir",
                "run_dir",
                "--task",
                "aime24",
            ],
        )
        assert result.exit_code != 0
        assert "Run summary file" in str(result.exception)
