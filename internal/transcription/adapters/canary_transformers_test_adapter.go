package adapters

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"scriberr/pkg/logger"
)

// CanaryTransformersTestAdapter is a deliberately narrow test harness for the
// low-memory Transformers backend. It embeds the normal Canary adapter so the
// transcription/output path stays identical, but overrides environment
// preparation to avoid installing NeMo or downloading the .nemo archive.
type CanaryTransformersTestAdapter struct {
	*CanaryAdapter
}

// NewCanaryTransformersTestAdapter creates the isolated Transformers test
// adapter. It is only registered when SCRIBERR_CANARY_TEST_ONLY=1.
func NewCanaryTransformersTestAdapter(envPath string) *CanaryTransformersTestAdapter {
	return &CanaryTransformersTestAdapter{
		CanaryAdapter: NewCanaryAdapter(envPath),
	}
}

const canaryTransformersLauncherProject = `[project]
name = "scriberr-canary-transformers-launcher"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
`

// PrepareEnvironment creates only the tiny launcher environment required for
// canary_transcribe.py to re-exec itself into its isolated Transformers uv
// project. The actual Transformers dependencies are installed lazily there on
// first transcription. No NeMo environment or .nemo model is prepared here.
func (c *CanaryTransformersTestAdapter) PrepareEnvironment(ctx context.Context) error {
	logger.Info("Preparing isolated Canary Transformers test environment", "env_path", c.envPath)

	if err := os.MkdirAll(c.envPath, 0755); err != nil {
		return fmt.Errorf("failed to create Canary test environment directory: %w", err)
	}

	if err := c.copyTranscriptionScript(); err != nil {
		return fmt.Errorf("failed to copy Canary transcription script: %w", err)
	}

	pyprojectPath := filepath.Join(c.envPath, "pyproject.toml")
	if err := os.WriteFile(pyprojectPath, []byte(canaryTransformersLauncherProject), 0644); err != nil {
		return fmt.Errorf("failed to write Canary test launcher pyproject: %w", err)
	}

	c.initialized = true
	logger.Info("Isolated Canary Transformers test environment prepared")
	return nil
}
