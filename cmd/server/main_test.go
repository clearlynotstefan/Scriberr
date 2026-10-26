package main

import (
	"testing"

	"scriberr/internal/config"
	"scriberr/internal/transcription/registry"
)

func TestCanaryTestModeRegistersIsolatedSortformer(t *testing.T) {
	registry.ClearRegistry()
	t.Cleanup(registry.ClearRegistry)
	t.Setenv("SCRIBERR_CANARY_TEST_ONLY", "1")

	registerAdapters(&config.Config{WhisperXEnv: t.TempDir()})

	transcriptionAdapters := registry.GetTranscriptionAdapters()
	if _, ok := transcriptionAdapters["canary"]; !ok {
		t.Fatal("Canary test mode did not register the Canary adapter")
	}
	if len(transcriptionAdapters) != 1 {
		t.Fatalf("Canary test mode registered %d transcription adapters; want 1", len(transcriptionAdapters))
	}

	diarizationAdapters := registry.GetDiarizationAdapters()
	if _, ok := diarizationAdapters["sortformer"]; !ok {
		t.Fatal("Canary test mode did not register the Sortformer adapter")
	}
	if len(diarizationAdapters) != 1 {
		t.Fatalf("Canary test mode registered %d diarization adapters; want 1", len(diarizationAdapters))
	}
}
