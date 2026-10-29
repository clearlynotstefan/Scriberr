package transcription

import (
	"strings"
	"testing"

	"scriberr/internal/models"
	"scriberr/internal/transcription/interfaces"
)

func TestExclusiveDiarizationSplitsCoarseCanaryChunk(t *testing.T) {
	service := &UnifiedTranscriptionService{}
	transcript := &interfaces.TranscriptResult{
		Text: "one two three four five six seven eight nine ten",
		Segments: []interfaces.TranscriptSegment{
			{Start: 0, End: 10, Text: "one two three four five six seven eight nine ten"},
			{Start: 10, End: 12, Text: ""},
		},
		Metadata: map[string]string{"backend": "transformers"},
	}
	diarization := &interfaces.DiarizationResult{
		Segments: []interfaces.DiarizationSegment{
			{Start: 0, End: 4, Speaker: "SPEAKER_00"},
			{Start: 4, End: 10, Speaker: "SPEAKER_01"},
		},
		Metadata: map[string]string{"diarization_type": "exclusive"},
	}

	merged := service.mergeDiarizationWithTranscription(transcript, diarization)
	if len(merged.Segments) != 3 {
		t.Fatalf("expected two speaker turns plus preserved empty chunk, got %d segments", len(merged.Segments))
	}
	if merged.Segments[0].Speaker == nil || *merged.Segments[0].Speaker != "SPEAKER_00" {
		t.Fatalf("first turn has wrong speaker: %#v", merged.Segments[0].Speaker)
	}
	if merged.Segments[1].Speaker == nil || *merged.Segments[1].Speaker != "SPEAKER_01" {
		t.Fatalf("second turn has wrong speaker: %#v", merged.Segments[1].Speaker)
	}
	if merged.Segments[2].Text != "" || merged.Segments[2].Start != 10 || merged.Segments[2].End != 12 {
		t.Fatalf("empty chunk was not preserved: %#v", merged.Segments[2])
	}
	if len(merged.WordSegments) != 10 {
		t.Fatalf("expected 10 estimated word timestamps, got %d", len(merged.WordSegments))
	}
	var mergedWords []string
	for _, segment := range merged.Segments[:2] {
		mergedWords = append(mergedWords, strings.Fields(segment.Text)...)
	}
	if strings.Join(mergedWords, " ") != transcript.Text {
		t.Fatalf("speaker split changed transcript text: %q", strings.Join(mergedWords, " "))
	}
	if merged.Metadata["word_timestamps"] != "estimated_from_exclusive_diarization" {
		t.Fatalf("estimated timestamps were not labeled honestly: %#v", merged.Metadata)
	}
}

func TestRegularDiarizationKeepsMajoritySpeakerMerge(t *testing.T) {
	service := &UnifiedTranscriptionService{}
	transcript := &interfaces.TranscriptResult{
		Segments: []interfaces.TranscriptSegment{{Start: 0, End: 10, Text: "unchanged segment"}},
		Metadata: map[string]string{},
	}
	diarization := &interfaces.DiarizationResult{
		Segments: []interfaces.DiarizationSegment{
			{Start: 0, End: 3, Speaker: "SPEAKER_00"},
			{Start: 3, End: 10, Speaker: "SPEAKER_01"},
		},
		Metadata: map[string]string{"diarization_type": "regular"},
	}

	merged := service.mergeDiarizationWithTranscription(transcript, diarization)
	if len(merged.Segments) != 1 || merged.Segments[0].Text != "unchanged segment" {
		t.Fatalf("regular diarization unexpectedly split the transcript: %#v", merged.Segments)
	}
	if merged.Segments[0].Speaker == nil || *merged.Segments[0].Speaker != "SPEAKER_01" {
		t.Fatalf("expected maximum-overlap speaker, got %#v", merged.Segments[0].Speaker)
	}
}

func TestCommunityOneUsesSeparateDiarizationForWhisper(t *testing.T) {
	service := &UnifiedTranscriptionService{}
	params := models.WhisperXParams{
		ModelFamily:  FamilyWhisper,
		Diarize:      true,
		DiarizeModel: ModelDiarizationCommunity,
	}

	if service.transcriptionIncludesDiarization(ModelWhisperX, params) {
		t.Fatal("Community-1 must run as a separate universal diarization stage")
	}
	converted := service.convertToWhisperXParams(params)
	if converted["diarize"] != false {
		t.Fatalf("WhisperX embedded diarization should be disabled for Community-1: %#v", converted)
	}

	params.DiarizeModel = ModelDiarization31
	if !service.transcriptionIncludesDiarization(ModelWhisperX, params) {
		t.Fatal("explicit Pyannote 3.1 should preserve legacy WhisperX behavior")
	}
}

func TestEffectiveCanaryExecutionParametersAreNotWhisperDefaults(t *testing.T) {
	params := models.WhisperXParams{
		ModelFamily: FamilyNvidiaCanary,
		Model:       "small",
		Device:      "cpu",
		ComputeType: "float32",
		BatchSize:   8,
	}
	effective := effectiveExecutionParameters(params)

	if effective.Model != "canary-1b-v2" || effective.BatchSize != 1 {
		t.Fatalf("wrong effective Canary parameters: %#v", effective)
	}
	if effective.Device != "runtime-selected" || effective.ComputeType != "runtime-selected" {
		t.Fatalf("Whisper device defaults leaked into Canary execution data: %#v", effective)
	}
	if effective.CanaryBackend != "nemo" {
		t.Fatalf("NeMo must remain the default Canary backend: %q", effective.CanaryBackend)
	}
}
