"""
DAAP Tracked Model — OpenAIChatModel subclass that records token usage.

Wraps every model call and pushes input/output token counts to a
TokenTracker so the API layer can report consumption to the user.
"""

from agentscope.model import OpenAIChatModel

from daap.tools.token_tracker import TokenTracker


class TrackedOpenAIChatModel(OpenAIChatModel):
    """OpenAIChatModel that records token usage into a TokenTracker."""

    def __init__(self, *args, tracker: TokenTracker | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._tracker = tracker

    async def __call__(self, *args, **kwargs):
        result = await super().__call__(*args, **kwargs)
        # Qwen3 thinking mode emits thinking blocks. OpenAIChatFormatter warns and
        # skips them, corrupting conversation history on the next turn. Strip here
        # so they never enter BoundedMemory.
        if hasattr(result, "content") and isinstance(result.content, list):
            result.content = [
                b for b in result.content
                if not (isinstance(b, dict) and b.get("type") == "thinking")
            ]
        if self._tracker is not None:
            usage = getattr(result, "usage", None)
            if usage is not None:
                self._tracker.add(
                    model_id=self.model_name,
                    input_tokens=getattr(usage, "input_tokens", 0),
                    output_tokens=getattr(usage, "output_tokens", 0),
                )
        return result
