"""Tests for canary_transcribe.py"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
SCRIPT_PATH = SCRIPT_DIR / "canary_transcribe.py"
TEST_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "tests/data"
AUDIO_FILE = TEST_DATA_DIR / "AMI-Corpus-IB4002.Mix-Headset-clip.wav"


def load_canary_module():
    spec = importlib.util.spec_from_file_location("canary_transcribe_test_module", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_transformers_backend_reexecs_in_isolated_env(monkeypatch, tmp_path):
    """The low-memory backend must not install into the shared NeMo env."""
    module = load_canary_module()
    nvidia_project = tmp_path / "parakeet"
    virtual_env = nvidia_project / ".venv"
    virtual_env.mkdir(parents=True)

    monkeypatch.setenv("VIRTUAL_ENV", str(virtual_env))
    monkeypatch.setenv("SCRIBERR_CANARY_BACKEND", "transformers")
    monkeypatch.delenv("SCRIBERR_CANARY_TRANSFORMERS_CHILD", raising=False)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "audio.wav", "--output", "result.json"])

    captured = {}

    def fake_run(cmd, env):
        captured["cmd"] = cmd
        captured["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc:
        module.transcribe_audio("audio.wav")

    assert exc.value.code == 0
    transformers_project = nvidia_project / "canary-transformers"
    assert (transformers_project / "pyproject.toml").exists()
    assert str(transformers_project) in captured["cmd"]
    assert "--system-certs" in captured["cmd"]
    assert "--native-tls" not in captured["cmd"]
    assert captured["env"]["SCRIBERR_CANARY_TRANSFORMERS_CHILD"] == "1"
    assert captured["env"]["HF_HOME"] == str(nvidia_project / "hf-cache")


def test_transformers_chunk_plan_uses_quiet_boundaries_and_overlap():
    module = load_canary_module()
    sample_rate = 100
    audio = np.ones(75 * sample_rate, dtype=np.float32)
    audio[2850:2870] = 0.0

    chunks = module._plan_transformers_chunks(
        audio,
        sample_rate=sample_rate,
        max_chunk_seconds=30,
        overlap_seconds=1,
        boundary_search_seconds=2,
        min_chunk_seconds=10,
    )

    assert 2850 <= chunks[0][1] <= 2870
    assert chunks[1][0] == chunks[0][1] - sample_rate
    assert all(
        current[0] == previous[1] - sample_rate
        for previous, current in zip(chunks, chunks[1:])
    )
    assert chunks[-1][1] == len(audio)
    assert all(end > start for start, end in chunks)
    assert all(end - start <= 30 * sample_rate for start, end in chunks)


def test_transformers_chunk_merge_deduplicates_overlap_and_preserves_empty_chunks():
    module = load_canary_module()
    chunk_ranges = [
        (0, 30 * module.SAMPLE_RATE),
        (29 * module.SAMPLE_RATE, 59 * module.SAMPLE_RATE),
        (58 * module.SAMPLE_RATE, 75 * module.SAMPLE_RATE),
    ]
    decoded_chunks = [
        "Hello, brave new world.",
        "NEW world! This continues.",
        "",
    ]

    text, segments = module._assemble_transformers_chunks(
        decoded_chunks,
        chunk_ranges,
        total_samples=75 * module.SAMPLE_RATE,
    )

    assert text == "Hello, brave new world. This continues."
    assert len(segments) == 3
    assert segments[-1]["segment"] == ""
    assert segments[0]["start_offset"] == 0
    assert segments[-1]["end_offset"] == 75 * module.SAMPLE_RATE
    assert all(
        previous["end_offset"] == current["start_offset"]
        for previous, current in zip(segments, segments[1:])
    )


def test_transformers_generation_has_loop_guards():
    module = load_canary_module()

    assert module.TRANSFORMERS_GENERATION_KWARGS["do_sample"] is False
    assert module.TRANSFORMERS_GENERATION_KWARGS["repetition_penalty"] > 1.0
    assert module.TRANSFORMERS_GENERATION_KWARGS["no_repeat_ngram_size"] > 0
    assert module.TRANSFORMERS_GENERATION_KWARGS["renormalize_logits"] is True


def test_transformers_inputs_keep_integer_tensors_and_cast_floats_to_fp16():
    module = load_canary_module()

    class FakeTensor:
        def __init__(self, floating):
            self.floating = floating
            self.move = None

        def is_floating_point(self):
            return self.floating

        def to(self, **kwargs):
            self.move = kwargs
            return self

    class FakeTorch:
        Tensor = FakeTensor

    features = FakeTensor(floating=True)
    token_ids = FakeTensor(floating=False)
    inputs = {"input_features": features, "input_ids": token_ids}

    module._move_inputs_to_model(
        inputs,
        torch=FakeTorch,
        device="cuda",
        dtype="float16",
    )

    assert features.move == {"device": "cuda", "dtype": "float16"}
    assert token_ids.move == {"device": "cuda"}


def test_transformers_child_dispatches_without_nemo(monkeypatch):
    module = load_canary_module()
    monkeypatch.setenv("SCRIBERR_CANARY_BACKEND", "transformers")
    monkeypatch.setenv("SCRIBERR_CANARY_TRANSFORMERS_CHILD", "1")

    called = {}

    def fake_transformers(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(module, "_transcribe_transformers", fake_transformers)
    module.transcribe_audio("audio.wav", source_lang="en", target_lang="en")

    assert called["audio_path"] == "audio.wav"
    assert called["source_lang"] == "en"


def test_nemo_backend_remains_default(monkeypatch):
    module = load_canary_module()
    monkeypatch.delenv("SCRIBERR_CANARY_BACKEND", raising=False)

    called = {}

    def fake_nemo(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(module, "_transcribe_nemo", fake_nemo)
    module.transcribe_audio("audio.wav", source_lang="en", target_lang="en")

    assert called["audio_path"] == "audio.wav"
    assert called["timestamps"] is True


def test_canary_transcription_output():
    """Verify the existing NeMo Canary transcription output remains intact."""
    assert AUDIO_FILE.exists(), f"Audio file not found: {AUDIO_FILE}"

    project_root = Path(__file__).resolve().parents[6]
    env_path = project_root / "data/whisperx-env/parakeet"
    assert env_path.exists(), f"Environment not found at: {env_path}"

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
        output_file = tmp_file.name

    try:
        cmd = [
            "uv", "run",
            "--project", str(env_path),
            "python", str(SCRIPT_PATH),
            str(AUDIO_FILE),
            "--timestamps",
            "--output", output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        if result.returncode != 0:
            pytest.fail(f"Script failed with error:\n{result.stderr}")

        assert os.path.exists(output_file), "Output file was not created"

        with open(output_file, "r") as f:
            data = json.load(f)

        assert data["source_language"] == "en"
        assert data["target_language"] == "en"
        assert data["task"] == "transcribe"
        assert data["model"] == "canary-1b-v2"
        assert data["backend"] == "nemo"

        assert "Most of us" in data["transcription"]
        assert "desktop computers" in data["transcription"]

        assert "word_timestamps" in data
        assert len(data["word_timestamps"]) > 0
        first_word = data["word_timestamps"][0]
        assert first_word["word"] == "Most"
        assert abs(first_word["start"] - 0.0) < 0.1

        assert "segment_timestamps" in data
        assert len(data["segment_timestamps"]) > 0

    finally:
        if os.path.exists(output_file):
            os.remove(output_file)
