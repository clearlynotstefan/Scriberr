#!/usr/bin/env python3
"""
PyAnnote speaker diarization script.
Processes audio files to identify and separate different speakers.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from pyannote.audio import Pipeline
import torch

# Fix for PyTorch 2.6+ which defaults weights_only=True
# We need to allowlist PyAnnote's custom classes
try:
    from pyannote.audio.core.task import Specifications, Problem, Resolution
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([Specifications, Problem, Resolution])
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Could not add safe globals: {e}")


def diarize_audio(
    audio_path: str,
    output_file: str,
    hf_token: str,
    model: str = "pyannote/speaker-diarization-community-1",
    min_speakers: int = None,
    max_speakers: int = None,
    output_format: str = "rttm",
    device: str = "auto",
    segmentation_onset: float = None,
    segmentation_offset: float = None,
):
    """
    Perform speaker diarization on audio file using PyAnnote.
    """
    print(f"Loading PyAnnote speaker diarization pipeline: {model}")

    runtime_device = "cpu"

    try:
        # Initialize the diarization pipeline
        pipeline = Pipeline.from_pretrained(
            model,
            token=hf_token
        )

        # Move to specified device
        # if device == "auto" or device == "cuda":
        try:
            if torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))
                runtime_device = "cuda"
                print("Using CUDA for diarization")
            elif device == "cuda":
                print("CUDA requested but not available, falling back to CPU")
            else:
                print("CUDA not available, using CPU")
        except ImportError:
            print("PyTorch not available for CUDA, using CPU")
        except Exception as e:
            print(f"Error moving to device: {e}, using CPU")

        # Apply segmentation thresholds if provided
        if segmentation_onset is not None or segmentation_offset is not None:
            try:
                # Get current parameters
                params = pipeline.parameters(instantiated=True)

                changed = False
                if segmentation_onset is not None:
                    changed |= set_existing_parameter(
                        params,
                        [("segmentation", "threshold"), ("segmentation", "onset")],
                        segmentation_onset,
                        "segmentation onset",
                    )
                if segmentation_offset is not None:
                    changed |= set_existing_parameter(
                        params,
                        [("segmentation", "min_duration_off"), ("segmentation", "offset")],
                        segmentation_offset,
                        "segmentation offset",
                    )

                if changed:
                    pipeline.instantiate(params)
            except Exception as e:
                print(f"Warning: Could not set segmentation thresholds: {e}")
                print("Continuing with default thresholds")

        print("Pipeline loaded successfully")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Make sure you have a valid Hugging Face token and have accepted the model's license")
        sys.exit(1)

    print(f"Processing audio file: {audio_path}")

    try:
        # Run diarization
        diarization_params = {}
        if min_speakers is not None:
            diarization_params["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_params["max_speakers"] = max_speakers

        if diarization_params:
            print(f"Using speaker constraints: {diarization_params}")
            diarization = pipeline(audio_path, **diarization_params)
        else:
            print("Using automatic speaker detection")
            diarization = pipeline(audio_path)

        annotation, diarization_type = select_diarization_annotation(diarization)
        print(f"Diarization completed ({diarization_type} output). Saving results to: {output_file}")

        if output_format == "rttm":
            # Save the diarization output to RTTM format
            with open(output_file, "w") as rttm:
                annotation.write_rttm(rttm)
        else:
            # Save as JSON format
            save_json_format(
                annotation,
                output_file,
                audio_path,
                model,
                diarization_type,
                runtime_device,
            )

        # Print summary
        speakers = set()
        total_speech_time = 0.0

        for turn, speaker in iter_annotation(annotation):
            speakers.add(speaker)
            total_speech_time += turn.duration

        print(f"\nDiarization Summary:")
        print(f"  Speakers detected: {len(speakers)}")
        print(f"  Speaker labels: {sorted(speakers)}")
        print(f"  Total speech time: {total_speech_time:.2f} seconds")
        print(f"  Output file saved: {output_file}")

    except Exception as e:
        print(f"Error during diarization: {e}")
        sys.exit(1)


def set_existing_parameter(params, candidates, value, label):
    """Set a tuning value only when that pipeline actually exposes the key."""
    for parent, key in candidates:
        section = params.get(parent)
        if isinstance(section, dict) and key in section:
            section[key] = value
            print(f"Set {label} ({parent}.{key}): {value}")
            return True
    print(f"Warning: {label} is not supported by this pipeline; using its default")
    return False


def select_diarization_annotation(output):
    """Prefer Community-1's exclusive output for transcript reconciliation."""
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    if exclusive is not None:
        return exclusive, "exclusive"
    regular = getattr(output, "speaker_diarization", None)
    if regular is not None:
        return regular, "regular"
    return output, "regular"


def iter_annotation(annotation):
    if hasattr(annotation, "itertracks"):
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            yield segment, speaker
        return
    for segment, speaker in annotation:
        yield segment, speaker


def save_json_format(
    annotation,
    output_file: str,
    audio_path: str,
    model: str,
    diarization_type: str,
    runtime_device: str,
):
    """Save diarization results in JSON format."""
    segments = []
    speakers = set()

    for turn, speaker in iter_annotation(annotation):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
            "confidence": 1.0,
            "duration": turn.duration
        })
        speakers.add(speaker)

    # Sort segments by start time
    segments.sort(key=lambda x: x["start"])

    results = {
        "audio_file": audio_path,
        "model": model,
        "diarization_type": diarization_type,
        "runtime": {"device": runtime_device},
        "segments": segments,
        "speakers": sorted(speakers),
        "speaker_count": len(speakers),
        "total_duration": max(seg["end"] for seg in segments) if segments else 0,
        "processing_info": {
            "total_segments": len(segments),
            "total_speech_time": sum(seg["duration"] for seg in segments)
        }
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Perform speaker diarization using PyAnnote.audio"
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "--hf-token",
        required=True,
        help="Hugging Face access token"
    )
    parser.add_argument(
        "--model",
        default="pyannote/speaker-diarization-community-1",
        help="PyAnnote model to use"
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers"
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers"
    )
    parser.add_argument(
        "--output-format",
        choices=["rttm", "json"],
        default="rttm",
        help="Output format"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--segmentation-onset",
        type=float,
        help="Voice activity detection onset threshold (0.0-1.0). Lower values detect quieter speech."
    )
    parser.add_argument(
        "--segmentation-offset",
        type=float,
        help="Voice activity detection offset/min_duration_off (0.0-1.0). Lower values are more sensitive to speech endings."
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    # Validate speaker constraints
    if args.min_speakers is not None and args.min_speakers < 1:
        print("Error: min_speakers must be at least 1")
        sys.exit(1)

    if args.max_speakers is not None and args.max_speakers < 1:
        print("Error: max_speakers must be at least 1")
        sys.exit(1)

    if (args.min_speakers is not None and args.max_speakers is not None and
        args.min_speakers > args.max_speakers):
        print("Error: min_speakers cannot be greater than max_speakers")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        diarize_audio(
            audio_path=args.audio_file,
            output_file=args.output,
            hf_token=args.hf_token,
            model=args.model,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            output_format=args.output_format,
            device=args.device,
            segmentation_onset=args.segmentation_onset,
            segmentation_offset=args.segmentation_offset,
        )
    except Exception as e:
        print(f"Error during diarization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
