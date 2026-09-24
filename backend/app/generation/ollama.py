"""Single local Ollama provider using the existing HTTP client dependency."""

import httpx
from pydantic import ValidationError

from app.config import Settings, settings
from app.generation.models import ContextBlock, GeneratedAnswer, GenerationError
from app.generation.prompt import SYSTEM_PROMPT, user_message


class OllamaGroundedGenerator:
    """Reuse one client and validate the model's structured JSON response."""

    def __init__(self, config: Settings | None = None, client: httpx.Client | None = None):
        self.config = config or settings
        self.client = client or httpx.Client(timeout=self.config.ollama_timeout_seconds)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def generate(self, question: str, contexts: list[ContextBlock]) -> GeneratedAnswer:
        payload = {
            "model": self.config.ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message(question, contexts)},
            ],
            "stream": False,
            "think": False,
            "format": GeneratedAnswer.model_json_schema(),
            "options": {
                "temperature": self.config.generation_temperature,
                "num_predict": self.config.generation_max_tokens,
            },
        }
        try:
            response = self.client.post(
                f"{self.config.ollama_base_url}/api/chat",
                json=payload,
                timeout=self.config.ollama_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise GenerationError(f"Ollama request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise GenerationError(f"cannot connect to Ollama: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise GenerationError(
                    f"Ollama model {self.config.ollama_model!r} or endpoint was not found"
                ) from exc
            raise GenerationError(
                f"Ollama returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise GenerationError(f"Ollama request failed: {exc}") from exc
        try:
            envelope = response.json()
            if not isinstance(envelope, dict):
                raise GenerationError("Ollama response must be a JSON object")
            message = envelope["message"]
            if not isinstance(message, dict):
                raise GenerationError("Ollama response message is missing")
            if message.get("role") != "assistant":
                raise GenerationError("Ollama response message must be from assistant")
            content = message["content"]
            if not isinstance(content, str) or not content.strip():
                raise GenerationError("Ollama assistant content is missing")
            return GeneratedAnswer.model_validate_json(content)
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            raise GenerationError(f"invalid Ollama structured response: {exc}") from exc
