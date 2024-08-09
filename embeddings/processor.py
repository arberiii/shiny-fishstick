import numpy as np
from typing import List, Tuple
from llm.base import BaseLLM
from utils.math_operations import cosine_similarity


class EmbeddingProcessor:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    async def find_most_similar_word(self, words: List[str], concept: str) -> Tuple[str, float]:
        concept_embedding = await self.llm.get_embedding(concept)
        word_embeddings = await self.llm.get_embedding(words)

        similarities = [cosine_similarity(concept_embedding, word_embedding)
                        for word_embedding in word_embeddings]

        most_similar_index = np.argmax(similarities)
        return words[most_similar_index], similarities[most_similar_index]
