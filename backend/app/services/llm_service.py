import os

from groq import Groq
from dotenv import load_dotenv


load_dotenv()


class LLMService:
    """
    Handles communication with the configured LLM provider.
    """

    def __init__(self) -> None:

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Please add it to your environment variables."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        )

    def generate_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """
        Generate a response from the configured LLM.

        response_format can be supplied when structured
        output such as JSON is required.
        """

        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0.2,
        }

        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        if response_format is not None:
            request["response_format"] = (
                response_format
            )

        response = (
            self.client.chat.completions.create(
                **request
            )
        )

        return response.choices[
            0
        ].message.content