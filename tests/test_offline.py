"""Feedlytics must be fully usable without Anthropic API access.

These tests prove that the non-AI functionality never needs an API key or the
SDK, that the AI path fails clearly when requested without access, and that
the safety net in conftest.py blocks any live network call.
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from app.llm_client import ClaudeJsonLLM, LLMError, has_credentials
from scripts import evaluate_labels

ROOT = Path(__file__).resolve().parent.parent


def test_safety_net_removes_credentials_and_blocks_network():
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert not has_credentials()
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        socket.create_connection(("api.anthropic.com", 443), timeout=1)


def test_real_client_cannot_be_built_without_credentials():
    with pytest.raises(LLMError, match="No API key") as err:
        ClaudeJsonLLM()
    assert err.value.fatal


def test_ai_evaluation_without_access_fails_clearly(tmp_path, capsys):
    code = evaluate_labels.main(["--ai", "--yes", "--out-dir", str(tmp_path / "out")])
    assert code == 2
    message = capsys.readouterr().err
    assert "no Anthropic API access" in message and "Run without --ai" in message
    assert not (tmp_path / "out").exists()  # nothing was processed or written


def test_non_ai_evaluation_runs_without_key_and_documents_it(tmp_path):
    out = tmp_path / "out"
    assert evaluate_labels.main(["--out-dir", str(out)]) == 0
    report = (out / "RESULTS_food_delivery_demo.md").read_text()
    assert "Claude (AI) was not evaluated" in report
    assert "Baseline: VADER + keywords" in report and "Rating rule (no text)" in report


def test_local_evaluation_is_reproducible(tmp_path):
    for run in ("a", "b"):
        assert evaluate_labels.main(["--out-dir", str(tmp_path / run)]) == 0
    files = sorted(p.name for p in (tmp_path / "a" / "results").glob("*.json"))
    assert files
    for name in files:
        first = json.loads((tmp_path / "a" / "results" / name).read_text())
        second = json.loads((tmp_path / "b" / "results" / name).read_text())
        assert first == second


def test_non_ai_pipeline_never_imports_the_sdk():
    """Run cleaning, baseline labelling, analytics and evaluation in a fresh
    process where importing `anthropic` is impossible and no key is set."""
    code = r"""
import sys
sys.modules["anthropic"] = None  # any `import anthropic` would now fail
from app.data_processing import process_reviews
from app.ai_analysis import BaselineLabeller, label_reviews, sample_for_ai
from app.profiles import load_profile
from app import analytics
df, report = process_reviews("data/sample_reviews_SYNTHETIC.csv", reference_date="2026-10-08")
labelled, run = label_reviews(df, BaselineLabeller(load_profile(report.profile_name)))
assert run.from_baseline > 3000
weights = sample_for_ai(df, max_rows=500)
assert abs(weights.sum() - df["has_text"].sum()) < 1e-6
analytics.priority_table(labelled, "issue_ai", sentiment_col="sentiment_ai")
from scripts import evaluate_labels, profile_dataset, make_labeling_sample, generate_synthetic_data
print("OK")
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")


@pytest.mark.parametrize("value", ["your-api-key-here", "", "   "])
def test_placeholder_key_counts_as_no_access(monkeypatch, tmp_path, capsys, value):
    """An unedited .env (copied from .env.example) must not unlock the paid path."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    assert not has_credentials()
    with pytest.raises(LLMError, match="No API key"):
        ClaudeJsonLLM()
    assert evaluate_labels.main(["--ai", "--yes", "--out-dir", str(tmp_path / "out")]) == 2
    assert "no Anthropic API access" in capsys.readouterr().err
