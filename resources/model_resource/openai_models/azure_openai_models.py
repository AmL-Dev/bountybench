import os
from pathlib import Path
from time import time
from typing import List

from dotenv import load_dotenv
from openai import AzureOpenAI

from resources.model_resource.openai_models.openai_models import OpenAIModels
from resources.model_resource.model_response import ModelResponse


class AzureOpenAIModels(OpenAIModels):
    def __init__(self):
        self.client = self.create_client()

    def _endpoint(self) -> str:
        env_var = "AZURE_OPENAI_ENDPOINT"
        current_dir = Path(__file__).resolve().parent.parent.parent
        root_dir = current_dir.parent
        env_path = root_dir / ".env"
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path)
        endpoint = os.getenv(env_var)
        if not endpoint:
            if env_path.is_file():
                raise ValueError(
                    f"{env_var} is not set in the .env file or environment variables"
                )
            else:
                raise ValueError(
                    f"{env_var} is not set in environment variables and .env file not found at {env_path}"
                )
        return endpoint

    def create_client(self) -> AzureOpenAI:
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        return AzureOpenAI(
            api_key=self._api_key(),
            azure_endpoint=self._endpoint(),
            api_version=api_version,
        )

    def request(
        self,
        model: str,
        message: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: List[str],
    ) -> ModelResponse:
        """
        Azure currently supports chat.completions for many deployments where
        responses.create can return 404. Use chat.completions to match
        the verified working integration.
        """
        status_code = None
        model_name = model.split("/")[-1] if "/" in model else model

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": message}],
                max_completion_tokens=max_tokens,
                stop=stop_sequences if stop_sequences else None,
            )

            if hasattr(response, "response") and hasattr(
                response.response, "status_code"
            ):
                status_code = response.response.status_code

            content = response.choices[0].message.content or ""
            return ModelResponse(
                content=content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                time_taken_in_ms=float(time()) - response.created,
                status_code=status_code,
            )
        except Exception as e:
            try:
                if hasattr(e, "status_code"):
                    status_code = e.status_code
                elif hasattr(e, "response") and hasattr(e.response, "status_code"):
                    status_code = e.response.status_code
            except Exception:
                pass
            if status_code is not None:
                e.status_code = status_code
            raise
