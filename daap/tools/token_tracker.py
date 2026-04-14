"""
DAAP Token Tracker — accumulates LLM token usage across agent calls.

Used by TrackedOpenAIChatModel to record input/output tokens per model call.
Attached to Session so the WebSocket handler can report usage to the client.
"""

from dataclasses import dataclass, field


@dataclass
class ModelCallRecord:
    model_id: str
    input_tokens: int
    output_tokens: int


class TokenTracker:
    """Accumulates token usage across multiple LLM calls."""

    def __init__(self):
        self._calls: list[ModelCallRecord] = []

    def add(self, model_id: str, input_tokens: int, output_tokens: int) -> None:
        self._calls.append(ModelCallRecord(
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ))

    def reset(self) -> None:
        self._calls.clear()

    @property
    def total_input(self) -> int:
        return sum(c.input_tokens for c in self._calls)

    @property
    def total_output(self) -> int:
        return sum(c.output_tokens for c in self._calls)

    @property
    def total_tokens(self) -> int:
        return self.total_input + self.total_output

    @property
    def models_used(self) -> list[str]:
        seen: list[str] = []
        for c in self._calls:
            if c.model_id not in seen:
                seen.append(c.model_id)
        return seen

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.total_input,
            "output_tokens": self.total_output,
            "total_tokens": self.total_tokens,
            "models_used": self.models_used,
        }
