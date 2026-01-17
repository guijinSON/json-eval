from typing import Any, Dict, List, Optional


class VLLMRunner:
    """
    Lightweight vLLM wrapper.
    """

    def __init__(self, model: str, max_tokens: int = 512, stop: Optional[List[str]] = None, **kwargs: Dict[str, Any]):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. Install via `pip install -r requirements-vllm.txt`."
            ) from exc

        self.sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop)
        self.llm = LLM(model=model, **kwargs)

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        params = self.sampling_params
        if stop:
            params = params.clone()
            params.stop = stop
        outputs = self.llm.generate([prompt], sampling_params=params)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text
