package adapters

import "testing"

func TestResolveCanaryBackend(t *testing.T) {
	t.Setenv(canaryBackendEnv, canaryBackendTransformers)

	if got := resolveCanaryBackend(nil); got != canaryBackendTransformers {
		t.Fatalf("expected environment fallback %q, got %q", canaryBackendTransformers, got)
	}

	params := map[string]interface{}{"backend": canaryBackendNeMo}
	if got := resolveCanaryBackend(params); got != canaryBackendNeMo {
		t.Fatalf("expected job parameter to override environment with %q, got %q", canaryBackendNeMo, got)
	}

	t.Setenv(canaryBackendEnv, "")
	if got := resolveCanaryBackend(nil); got != canaryBackendNeMo {
		t.Fatalf("expected default backend %q, got %q", canaryBackendNeMo, got)
	}
}

func TestSetEnvironmentValueReplacesExistingValue(t *testing.T) {
	environment := []string{
		"PATH=/usr/bin",
		canaryBackendEnv + "=" + canaryBackendNeMo,
	}

	updated := setEnvironmentValue(environment, canaryBackendEnv, canaryBackendTransformers)
	backendEntries := 0
	for _, entry := range updated {
		if entry == canaryBackendEnv+"="+canaryBackendTransformers {
			backendEntries++
		}
		if entry == canaryBackendEnv+"="+canaryBackendNeMo {
			t.Fatalf("stale backend environment value was retained: %v", updated)
		}
	}

	if backendEntries != 1 {
		t.Fatalf("expected one backend environment entry, got %d: %v", backendEntries, updated)
	}
}
