#!/usr/bin/env python3
"""NVIDIA Canary multilingual transcription and translation.

The default backend remains NeMo for full word/segment timestamps. Set
SCRIBERR_CANARY_BACKEND=transformers to use Hugging Face Transformers and
avoid loading Canary's auxiliary CTC timestamp model. The Transformers path
runs in an isolated sibling uv environment and uses FP16 on CUDA. It emits
coarse chunk-level segment timestamps, not NeMo NFA word timestamps.
"""

import argparse
import json
import os
import re
import subprocess
import sys

MODEL_ID = "nvidia/canary-1b-v2"
SAMPLE_RATE = 16000
TRANSFORMERS_MAX_CHUNK_SECONDS = 30.0
TRANSFORMERS_CHUNK_OVERLAP_SECONDS = 1.0
TRANSFORMERS_BOUNDARY_SEARCH_SECONDS = 2.0
TRANSFORMERS_MIN_CHUNK_SECONDS = 10.0
TRANSFORMERS_BOUNDARY_FRAME_SECONDS = 0.02
TRANSFORMERS_GENERATION_KWARGS = {
    "max_new_tokens": 512,
    "do_sample": False,
    "num_beams": 1,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 6,
    "renormalize_logits": True,
}
_TRANSFORMERS_CHILD_ENV = "SCRIBERR_CANARY_TRANSFORMERS_CHILD"

_TRANSFORMERS_PYPROJECT = r'''[project]
name = "canary-transformers"
version = "0.1.0"
description = "Low-memory NVIDIA Canary inference using Hugging Face Transformers"
requires-python = ">=3.11"
dependencies = [
    "torch",
    "torchaudio",
    "librosa",
    "soundfile",
    "transformers @ git+https://github.com/huggingface/transformers.git",
]

[tool.uv.sources]
torch = [
    { index = "pytorch", marker = "platform_machine == 'x86_64' and sys_platform == 'linux'" },
    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
]
torchaudio = [
    { index = "pytorch", marker = "platform_machine == 'x86_64' and sys_platform == 'linux'" },
    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
'''


def _write_output(output_data, output_file):
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")
    else:
        print(json.dumps(output_data, indent=2, ensure_ascii=False))


def _project_root_from_virtual_env():
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if not virtual_env:
        raise RuntimeError("VIRTUAL_ENV is not set; run this script with uv run")
    return os.path.dirname(virtual_env)


def _reexec_in_transformers_env():
    """Run the Transformers backend in an isolated uv project.

    The environment and HF model cache both live under /app/whisperx-env (or
    the configured NVIDIA env root), so multiple Scriberr containers can share
    them without modifying the NeMo/Parakeet environment.
    """
    project_root = _project_root_from_virtual_env()
    transformers_project = os.path.join(project_root, "canary-transformers")
    os.makedirs(transformers_project, exist_ok=True)

    pyproject_path = os.path.join(transformers_project, "pyproject.toml")
    current = None
    if os.path.exists(pyproject_path):
        with open(pyproject_path, "r", encoding="utf-8") as f:
            current = f.read()
    if current != _TRANSFORMERS_PYPROJECT:
        with open(pyproject_path, "w", encoding="utf-8") as f:
            f.write(_TRANSFORMERS_PYPROJECT)

    env = os.environ.copy()
    env[_TRANSFORMERS_CHILD_ENV] = "1"
    env.setdefault("HF_HOME", os.path.join(project_root, "hf-cache"))
    env.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(project_root, "hf-cache", "hub"))

    cmd = [
        "uv",
        "run",
        "--system-certs",
        "--project",
        transformers_project,
        "python",
        os.path.abspath(__file__),
        *sys.argv[1:],
    ]
    print(f"Launching isolated Transformers backend: {transformers_project}")
    completed = subprocess.run(cmd, env=env)
    raise SystemExit(completed.returncode)


def _transcribe_nemo(
    audio_path,
    source_lang,
    target_lang,
    task,
    timestamps,
    output_file,
    include_confidence,
):
    # Import NeMo lazily so the low-memory backend never initializes NeMo.
    import nemo.collections.asr as nemo_asr

    project_root = _project_root_from_virtual_env()
    model_filename = "canary-1b-v2.nemo"
    model_path = os.path.join(project_root, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Can't find {model_filename} in project root: {project_root}"
        )

    print(f"Loading NVIDIA Canary NeMo model from: {model_path}")
    asr_model = nemo_asr.models.ASRModel.restore_from(model_path)
    output = asr_model.transcribe(
        [audio_path],
        source_lang=source_lang,
        target_lang=target_lang,
        timestamps=timestamps,
    )

    result_data = output[0]
    text = result_data.text
    word_timestamps = []
    segment_timestamps = []
    if timestamps and getattr(result_data, "timestamp", None):
        word_timestamps = result_data.timestamp.get("word", [])
        segment_timestamps = result_data.timestamp.get("segment", [])

    output_data = {
        "transcription": text,
        "source_language": source_lang,
        "target_language": target_lang,
        "task": task,
        "word_timestamps": word_timestamps,
        "segment_timestamps": segment_timestamps,
        "audio_file": audio_path,
        "model": "canary-1b-v2",
        "backend": "nemo",
        "timestamp_precision": "word" if timestamps else "none",
    }

    if include_confidence:
        confidence = getattr(result_data, "confidence", None)
        if confidence:
            output_data["confidence"] = confidence

    print(f"Result: {text}")
    _write_output(output_data, output_file)


def _move_inputs_to_model(inputs, torch, device, dtype):
    """Move Canary processor outputs to the model device/dtype safely.

    BatchFeature.to(device) moves tensors but preserves floating-point dtype.
    When the model is loaded in FP16 this leaves the acoustic features in
    FP32, which fails in the FastConformer convolution frontend with
    "Input type (float) and bias type (c10::Half) should be the same".
    Cast only floating tensors; token IDs and masks must remain integral.
    """
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                inputs[key] = value.to(device=device, dtype=dtype)
            else:
                inputs[key] = value.to(device=device)
    return inputs


def _quietest_boundary(audio, target, lower, upper, frame_samples):
    """Choose a low-energy split point, preferring one close to the target."""
    half_frame = max(1, frame_samples // 2)
    first = max(lower, half_frame)
    last = min(upper, len(audio) - half_frame)
    if first >= last:
        return target

    candidates = range(first, last + 1, max(1, frame_samples))

    def boundary_score(sample):
        frame = audio[sample - half_frame : sample + half_frame]
        energy = float((frame * frame).mean()) if len(frame) else float("inf")
        return energy, abs(target - sample)

    return min(candidates, key=boundary_score)


def _plan_transformers_chunks(
    audio,
    sample_rate=SAMPLE_RATE,
    max_chunk_seconds=TRANSFORMERS_MAX_CHUNK_SECONDS,
    overlap_seconds=TRANSFORMERS_CHUNK_OVERLAP_SECONDS,
    boundary_search_seconds=TRANSFORMERS_BOUNDARY_SEARCH_SECONDS,
    min_chunk_seconds=TRANSFORMERS_MIN_CHUNK_SECONDS,
):
    """Plan bounded, silence-aware chunks with acoustic overlap.

    Chunk ends move backwards by at most ``boundary_search_seconds`` to a
    quiet frame. The next chunk starts one second before that boundary, which
    gives Canary context on both sides without increasing peak GPU memory.
    """
    total_samples = len(audio)
    if total_samples == 0:
        return []

    max_samples = max(1, int(max_chunk_seconds * sample_rate))
    overlap_samples = max(0, int(overlap_seconds * sample_rate))
    search_samples = max(0, int(boundary_search_seconds * sample_rate))
    min_samples = max(1, int(min_chunk_seconds * sample_rate))
    frame_samples = max(1, int(TRANSFORMERS_BOUNDARY_FRAME_SECONDS * sample_rate))

    if overlap_samples >= max_samples:
        raise ValueError("Canary chunk overlap must be shorter than the chunk")

    chunks = []
    start = 0
    while start < total_samples:
        target_end = min(total_samples, start + max_samples)

        if target_end == total_samples:
            chunks.append((start, total_samples))
            break

        lower = max(start + min_samples, target_end - search_samples)
        end = _quietest_boundary(
            audio,
            target=target_end,
            lower=lower,
            upper=target_end,
            frame_samples=frame_samples,
        )
        if end <= start + overlap_samples:
            end = target_end

        chunks.append((start, end))
        next_start = end - overlap_samples
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _normalized_word(word):
    return re.sub(r"^\W+|\W+$", "", word, flags=re.UNICODE).casefold()


def _deduplicate_chunk_prefix(previous_text, current_text, max_overlap_words=64):
    """Remove a matching word prefix produced by the acoustic overlap."""
    current_words = current_text.split()
    if not previous_text or not current_words:
        return current_text.strip()

    previous_words = previous_text.split()
    limit = min(len(previous_words), len(current_words), max_overlap_words)
    previous_normalized = [_normalized_word(word) for word in previous_words]
    current_normalized = [_normalized_word(word) for word in current_words]

    for word_count in range(limit, 1, -1):
        previous_suffix = previous_normalized[-word_count:]
        current_prefix = current_normalized[:word_count]
        if all(current_prefix) and previous_suffix == current_prefix:
            return " ".join(current_words[word_count:]).strip()

    return current_text.strip()


def _assemble_transformers_chunks(decoded_chunks, chunk_ranges, total_samples):
    """Merge overlapping text and assign gap-free ownership timestamps.

    Every planned chunk becomes a segment, including chunks that decode to an
    empty string. This keeps silence and decode failures visible instead of
    silently dropping time from the result.
    """
    if len(decoded_chunks) != len(chunk_ranges):
        raise ValueError("Decoded Canary chunks must match the chunk plan")

    boundaries = [0]
    for previous, current in zip(chunk_ranges, chunk_ranges[1:]):
        boundaries.append((previous[1] + current[0]) // 2)
    boundaries.append(total_samples)

    merged_text = ""
    segments = []
    for index, ((chunk_start, chunk_end), decoded_text) in enumerate(
        zip(chunk_ranges, decoded_chunks)
    ):
        decoded_text = decoded_text.strip()
        contribution = _deduplicate_chunk_prefix(merged_text, decoded_text)
        if contribution:
            merged_text = f"{merged_text} {contribution}".strip()

        owned_start = boundaries[index]
        owned_end = boundaries[index + 1]
        segments.append(
            {
                "segment": contribution,
                "start": owned_start / SAMPLE_RATE,
                "end": owned_end / SAMPLE_RATE,
                "start_offset": owned_start,
                "end_offset": owned_end,
                "chunk_start": chunk_start / SAMPLE_RATE,
                "chunk_end": chunk_end / SAMPLE_RATE,
            }
        )

    return merged_text, segments


def _transcribe_transformers(
    audio_path,
    source_lang,
    target_lang,
    task,
    output_file,
    preserve_formatting,
):
    # Imports are deliberately local. This child environment contains no NeMo.
    import librosa
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading {MODEL_ID} with Transformers on {device} ({dtype})")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype=dtype)
    model.to(device)
    model.eval()

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if task == "transcribe":
        target_lang = source_lang

    chunk_ranges = _plan_transformers_chunks(audio)
    decoded_chunks = []

    for start_sample, end_sample in chunk_ranges:
        chunk = audio[start_sample:end_sample]

        inputs = processor.apply_transcription_request(
            audio=chunk,
            source_language=source_lang,
            target_language=target_lang,
            punctuation=preserve_formatting,
        )
        inputs = _move_inputs_to_model(inputs, torch, device, dtype)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **TRANSFORMERS_GENERATION_KWARGS)

        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        text = decoded[0].strip() if decoded else ""
        start = start_sample / SAMPLE_RATE
        end = end_sample / SAMPLE_RATE
        print(f"[{start:.2f}-{end:.2f}] {text or '[no speech]'}")
        decoded_chunks.append(text)

    text, segment_timestamps = _assemble_transformers_chunks(
        decoded_chunks,
        chunk_ranges,
        len(audio),
    )

    output_data = {
        "transcription": text,
        "source_language": source_lang,
        "target_language": target_lang,
        "task": task,
        "word_timestamps": [],
        "segment_timestamps": segment_timestamps,
        "audio_file": audio_path,
        "model": "canary-1b-v2",
        "backend": "transformers",
        "timestamp_precision": "chunk",
    }
    _write_output(output_data, output_file)


def transcribe_audio(
    audio_path: str,
    source_lang: str = "en",
    target_lang: str = "en",
    task: str = "transcribe",
    timestamps: bool = True,
    output_file: str = None,
    include_confidence: bool = True,
    preserve_formatting: bool = True,
):
    backend = os.environ.get("SCRIBERR_CANARY_BACKEND", "nemo").strip().lower()
    print(f"Canary backend: {backend}")
    print(f"Processing: {audio_path}")
    print(f"Task: {task}")
    print(f"Source language: {source_lang}")
    print(f"Target language: {target_lang}")

    if backend == "transformers":
        if os.environ.get(_TRANSFORMERS_CHILD_ENV) != "1":
            _reexec_in_transformers_env()
        _transcribe_transformers(
            audio_path=audio_path,
            source_lang=source_lang,
            target_lang=target_lang,
            task=task,
            output_file=output_file,
            preserve_formatting=preserve_formatting,
        )
        return

    if backend != "nemo":
        raise ValueError(
            f"Unsupported SCRIBERR_CANARY_BACKEND={backend!r}; use 'nemo' or 'transformers'"
        )

    _transcribe_nemo(
        audio_path=audio_path,
        source_lang=source_lang,
        target_lang=target_lang,
        task=task,
        timestamps=timestamps,
        output_file=output_file,
        include_confidence=include_confidence,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe or translate audio using NVIDIA Canary"
    )
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument(
        "--source-lang",
        default="en",
        choices=["en", "de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "zh"],
        help="Source language (default: en)",
    )
    parser.add_argument(
        "--target-lang",
        default="en",
        choices=["en", "de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "zh"],
        help="Target language (default: en)",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task to perform (default: transcribe)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        default=True,
        help="Include timestamps when supported by the selected backend",
    )
    parser.add_argument(
        "--no-timestamps",
        dest="timestamps",
        action="store_false",
        help="Disable timestamps",
    )
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument(
        "--include-confidence",
        action="store_true",
        default=True,
        help="Include confidence scores when available",
    )
    parser.add_argument(
        "--no-confidence",
        dest="include_confidence",
        action="store_false",
        help="Exclude confidence scores",
    )
    parser.add_argument(
        "--preserve-formatting",
        action="store_true",
        default=True,
        help="Request punctuation and capitalization",
    )

    args = parser.parse_args()
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    try:
        transcribe_audio(
            audio_path=args.audio_file,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            task=args.task,
            timestamps=args.timestamps,
            output_file=args.output,
            include_confidence=args.include_confidence,
            preserve_formatting=args.preserve_formatting,
        )
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error during transcription: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
