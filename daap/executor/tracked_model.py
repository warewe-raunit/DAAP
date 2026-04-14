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
        if self._tracker is not None:
            usage = getattr(result, "usage", None)
            if usage is not None:
                self._tracker.add(
                    model_id=self.model_name,
                    input_tokens=getattr(usage, "input_tokens", 0),
                    output_tokens=getattr(usage, "output_tokens", 0),
                )
        return result
