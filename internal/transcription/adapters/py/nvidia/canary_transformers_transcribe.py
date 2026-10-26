#!/usr/bin/env python3
"""Low-memory NVIDIA Canary inference using Hugging Face Transformers.

This backend intentionally avoids the auxiliary CTC timestamp model bundled in
Canary's .nemo archive. It therefore provides coarse chunk-level timestamps,
not NeMo NFA word timestamps.
"""

import argparse
import json
import os
import sys

import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_ID = "nvidia/canary-1b-v2"
SAMPLE_RATE = 16000
DEFAULT_CHUNK_SECONDS = 30.0


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading {MODEL_ID} with Transformers on {device} ({dtype})")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, dtype=dtype)
    model.to(device)
    model.eval()
    return processor, model, device


def transcribe_audio(
    audio_path: str,
    source_lang: str = "en",
    target_lang: str = "en",
    task: str = "transcribe",
    output_file: str | None = None,
    preserve_formatting: bool = True,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
):
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    processor, model, device = load_model()

    if task == "transcribe":
        target_lang = source_lang

    chunk_samples = max(1, int(chunk_seconds * SAMPLE_RATE))
    segment_timestamps = []
    texts = []

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
        texts.append(text)
        segment_timestamps.append({
            "segment": text,
            "start": start,
            "end": end,
            "start_offset": start_sample,
            "end_offset": end_sample,
        })
        print(f"[{start:.2f}-{end:.2f}] {text}")

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

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")
    else:
        print(json.dumps(output_data, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Low-memory Canary transcription using Transformers")
    parser.add_argument("audio_file", help="Path to a mono 16 kHz audio file")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--output", "-o")
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--preserve-formatting", action="store_true", default=True)
    # Accepted for CLI compatibility with the NeMo backend. Transformers does
    # not expose Canary's auxiliary CTC/NFA word timestamps.
    parser.add_argument("--timestamps", action="store_true", default=True)
    parser.add_argument("--no-timestamps", dest="timestamps", action="store_false")
    parser.add_argument("--include-confidence", action="store_true", default=True)
    parser.add_argument("--no-confidence", dest="include_confidence", action="store_false")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    try:
        transcribe_audio(
            audio_path=args.audio_file,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            task=args.task,
            output_file=args.output,
            preserve_formatting=args.preserve_formatting,
            chunk_seconds=args.chunk_seconds,
        )
    except Exception as exc:
        print(f"Error during Transformers Canary transcription: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
