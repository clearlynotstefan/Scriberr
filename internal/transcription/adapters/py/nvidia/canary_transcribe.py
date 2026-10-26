#!/usr/bin/env python3
"""NVIDIA Canary multilingual transcription and translation.

The default backend remains NeMo for full word/segment timestamps. Set
SCRIBERR_CANARY_BACKEND=transformers to use Hugging Face Transformers and
avoid loading Canary's auxiliary CTC timestamp model. The Transformers path
uses FP16 on CUDA and emits coarse chunk-level segment timestamps only.
"""

import argparse
import json
import os
import sys

MODEL_ID = "nvidia/canary-1b-v2"
SAMPLE_RATE = 16000
TRANSFORMERS_CHUNK_SECONDS = 30.0


def _write_output(output_data, output_file):
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")
    else:
        print(json.dumps(output_data, indent=2, ensure_ascii=False))


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

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if not virtual_env:
        raise RuntimeError("VIRTUAL_ENV is not set; run this script with uv run")

    project_root = os.path.dirname(virtual_env)
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


def _transcribe_transformers(
    audio_path,
    source_lang,
    target_lang,
    task,
    output_file,
    preserve_formatting,
):
    # Imports are deliberately local. This path should not import NeMo at all.
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

    chunk_samples = int(TRANSFORMERS_CHUNK_SECONDS * SAMPLE_RATE)
    texts = []
    segment_timestamps = []

    for start_sample in range(0, len(audio), chunk_samples):
        end_sample = min(len(audio), start_sample + chunk_samples)
        chunk = audio[start_sample:end_sample]
        if chunk.size == 0:
            continue

        inputs = processor.apply_transcription_request(
            audio=chunk,
            source_language=source_lang,
            target_language=target_lang,
            punctuation=preserve_formatting,
        ).to(device)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        text = decoded[0].strip() if decoded else ""
        if not text:
            continue

        start = start_sample / SAMPLE_RATE
        end = end_sample / SAMPLE_RATE
        print(f"[{start:.2f}-{end:.2f}] {text}")
        texts.append(text)
        segment_timestamps.append(
            {
                "segment": text,
                "start": start,
                "end": end,
                "start_offset": start_sample,
                "end_offset": end_sample,
            }
        )

    output_data = {
        "transcription": " ".join(texts).strip(),
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
    except Exception as exc:
        print(f"Error during transcription: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
