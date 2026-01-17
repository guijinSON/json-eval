from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EvalItem:
    """Single evaluation example."""

    id: str
    prompt: str
    schema: Dict[str, Any]
    verify_schema: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)
