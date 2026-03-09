from datetime import datetime
from time import time
from typing import List
import os

import tiktoken
from openai import OpenAI

from resources.model_resource.model_provider import ModelProvider
from resources.model_resource.model_response import ModelResponse
from utils.logger import get_main_logger


logger = get_main_logger(__name__)


class RchatModels(ModelProvider):
    def __init__(self):
        self.client = self.create_client()

    def create_client(self) -> OpenAI:
        base_url = os.getenv("RCHAT_BASE_URL", "https://rchat.nist.gov/proxy/v1/")
        return OpenAI(api_key=self._api_key(), base_url=base_url)

    def request(
        self,
        model: str,
        message: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: List[str],
    ) -> ModelResponse:

        start_time = datetime.now()
        status_code = None

        try:
            # Extract base model name, e.g. "rchat/gpt-oss-120b" -> "gpt-oss-120b"
            model_name = model.split("/")[-1] if "/" in model else model

            params = {
                "model": model_name,
                "input": message,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }

            response = self.client.responses.create(**params)

            # For successful responses, we don't typically get HTTP status code
            # from OpenAI client, but could try to extract if available
            if hasattr(response, "response") and hasattr(
                response.response, "status_code"
            ):
                status_code = response.response.status_code

            output_tokens = response.usage.output_tokens

            logger.info(
                f"[Rchat] max output tokens: {max_tokens} - total output tokens: {output_tokens}"
            )
            return ModelResponse(
                content=response.output_text,
                input_tokens=response.usage.input_tokens,
                output_tokens=output_tokens,
                time_taken_in_ms=float(time()) - response.created_at,
                status_code=status_code,
            )
        except Exception as e:
            # Extract status code from OpenAI-style errors if possible
            try:
                if hasattr(e, "status_code"):
                    status_code = e.status_code
                elif hasattr(e, "response") and hasattr(e.response, "status_code"):
                    status_code = e.response.status_code
                elif "Error code:" in str(e):
                    error_parts = str(e).split("Error code:")
                    if len(error_parts) > 1:
                        code_part = error_parts[1].strip().split(" ")[0]
                        if code_part.isdigit():
                            status_code = int(code_part)
            except Exception:
                pass

            if status_code is not None:
                e.status_code = status_code
            raise

    def tokenize(self, model: str, message: str) -> List[int]:
        encoding = tiktoken.encoding_for_model("gpt-4o")
        return encoding.encode(message)

    def decode(self, model: str, tokens: List[int]) -> str:
        encoding = tiktoken.encoding_for_model("gpt-4o")
        return encoding.decode(tokens)

    def get_num_tokens(self, model: str, message: str) -> int:
        encoding = tiktoken.encoding_for_model("gpt-4o")
        return len(encoding.encode(message))

