import asyncio
from dotenv import load_dotenv

from llm.openai_llm import OpenAILLM
from embeddings.processor import EmbeddingProcessor

load_dotenv()


async def main():
    llm = OpenAILLM()
    processor = EmbeddingProcessor(llm)

    words = ["apple", "car", "bus", "train"]
    concept = "fruit"

    try:
        most_similar, similarity = await processor.find_most_similar_word(words, concept)

        print(f"Words: {words}")
        print(f"Concept: {concept}")
        print(f"Most similar word: {most_similar}")
        print(f"Similarity score: {similarity:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
