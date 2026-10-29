package transcription

import (
	"encoding/json"
	"strconv"

	"scriberr/internal/models"
	"scriberr/internal/transcription/interfaces"
)

// effectiveExecutionParameters removes Whisper-only defaults from fixed-model
// backends. Exact device and precision are recorded after the adapter runs.
func effectiveExecutionParameters(params models.WhisperXParams) models.WhisperXParams {
	effective := params

	switch params.ModelFamily {
	case FamilyNvidiaCanary:
		effective.Model = "canary-1b-v2"
		effective.BatchSize = 1
		effective.Device = "runtime-selected"
		effective.ComputeType = "runtime-selected"
		if effective.CanaryBackend == "" {
			effective.CanaryBackend = "nemo"
		}
	case FamilyNvidiaParakeet:
		effective.Model = "parakeet-tdt-0.6b-v3"
		effective.BatchSize = 1
		effective.Device = "runtime-selected"
		effective.ComputeType = "runtime-selected"
	case FamilyMistralVoxtral:
		effective.Model = "mistralai/Voxtral-mini"
		effective.BatchSize = 1
		effective.Device = "runtime-selected"
		effective.ComputeType = "runtime-selected"
	case FamilyOpenAI:
		effective.Device = "provider-managed"
		effective.ComputeType = "provider-managed"
		effective.BatchSize = 1
	}

	return effective
}

func buildRuntimeMetadata(transcript *interfaces.TranscriptResult, diarization *interfaces.DiarizationResult) *string {
	metadata := map[string]interface{}{
		"transcription_model":    transcript.ModelUsed,
		"transcription_segments": len(transcript.Segments),
		"word_segments":          len(transcript.WordSegments),
	}

	for _, key := range []string{
		"backend",
		"device",
		"compute_type",
		"batch_size",
		"timestamp_precision",
		"chunk_size_seconds",
		"chunk_overlap_seconds",
		"decoder_guardrails",
		"word_timestamps",
		"diarization_output",
	} {
		if value := transcript.Metadata[key]; value != "" {
			metadata[key] = parseRuntimeValue(value)
		}
	}

	if diarization != nil {
		metadata["diarization_model"] = diarization.ModelUsed
		metadata["diarization_segments"] = len(diarization.Segments)
		metadata["speaker_count"] = diarization.SpeakerCount
		if value := diarization.Metadata["diarization_type"]; value != "" {
			metadata["diarization_output"] = value
		}
		if value := diarization.Metadata["device"]; value != "" {
			metadata["diarization_device"] = value
		}
	}

	encoded, err := json.Marshal(metadata)
	if err != nil {
		return nil
	}
	value := string(encoded)
	return &value
}

func parseRuntimeValue(value string) interface{} {
	if integer, err := strconv.Atoi(value); err == nil {
		return integer
	}
	if number, err := strconv.ParseFloat(value, 64); err == nil {
		return number
	}
	if boolean, err := strconv.ParseBool(value); err == nil {
		return boolean
	}
	return value
}
