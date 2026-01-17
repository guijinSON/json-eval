import os
from typing import Any, Dict, List, Optional


class LiteLLMRunner:
    """
    Simple wrapper around LiteLLM completion/chat API.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        try:
            import litellm
        except ImportError as exc:
            raise ImportError("litellm is not installed. Install via `pip install litellm`.") from exc

        self.litellm = litellm
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        if base_url:
            os.environ.setdefault("OPENAI_API_BASE", base_url)
        self.model = model
        self.extra_kwargs = extra_kwargs or {}

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.litellm.completion(
            model=self.model,
            messages=messages,
            stop=stop,
            **self.extra_kwargs,
        )
        choice = response["choices"][0]
        message = choice.get("message") or choice.get("delta", {})
        return message.get("content", "") or ""
