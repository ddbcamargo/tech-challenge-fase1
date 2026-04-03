from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class PipelineContext:
    df: pd.DataFrame | None = None
    data: dict[str, Any] = field(default_factory=dict)
    tuning_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    modeling_results: list[dict[str, Any]] = field(default_factory=list)
    evaluation_df: pd.DataFrame | None = None