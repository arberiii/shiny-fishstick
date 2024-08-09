import os
from typing import Union
import numpy as np
import openai
from llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = 'text-embedding-3-small'):
        self.model = model
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        openai.api_key = self.api_key

    async def get_embedding(self, text: Union[str, list[str]]) -> np.ndarray:
        try:
            response = await openai.Embedding.acreate(
                input=text,
                engine=self.model
            )
            if isinstance(text, str):
                return np.array(response['data'][0]['embedding'])
            return np.array([data['embedding'] for data in response['data']])
        except Exception as e:
            raise RuntimeError(f"Error getting embedding: {str(e)}")
