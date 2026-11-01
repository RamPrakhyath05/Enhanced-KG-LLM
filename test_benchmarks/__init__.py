"""Shared evaluation utilities and metrics for Enhanced-KG-LLM."""
import sys

# Provide backward-compatibility alias for legacy scripts importing 'metrics'
if "metrics" not in sys.modules:
    sys.modules["metrics"] = sys.modules[__name__]
