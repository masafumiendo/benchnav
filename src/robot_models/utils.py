"""
Masafumi Endo, 2024.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    mode: Optional[str] = field(default=None)
    inference_metric: Optional[str] = field(default=None)
    confidence_value: Optional[float] = field(default=None)

    def __post_init__(self):
        valid_modes = ["inference", "observation"]
        valid_metrics = ["expected_value", "var", "cvar"]

        assert (
            self.mode in valid_modes
        ), f"mode must be either 'inference' or 'observation', got '{self.mode}'."

        if self.mode == "inference":
            assert (
                self.inference_metric in valid_metrics
            ), f"inference_metric must be one of {valid_metrics} when mode is 'inference', got '{self.inference_metric}'."
            if self.inference_metric in ["var", "cvar"]:
                assert (
                    self.confidence_value is not None
                    and 0.0 <= self.confidence_value <= 1.0
                ), "confidence_value must be set between 0 and 1 when inference_metric is 'var' or 'cvar'."
        else:
            # For 'observation' mode, ensure inference_metric and confidence_value are not set
            assert (
                self.inference_metric is None
            ), "inference_metric should not be set when mode is 'observation'."
            assert (
                self.confidence_value is None
            ), "confidence_value should not be set when mode is 'observation'."
