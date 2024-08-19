import asyncio
from dotenv import load_dotenv

from llm.openai_llm import OpenAILLM
from embeddings.processor import EmbeddingProcessor
from matcher.dict_matcher import SourceDict, TargetDict

load_dotenv()


async def main():
    llm = OpenAILLM()
    processor = EmbeddingProcessor(llm)

    source = SourceDict({"fruit name": "apple", "colour": "red", "size": "M"})
    source_list = source.dict_to_list()
    target = TargetDict({"fruit": "apple", "color": "blue"})
    target_list = target.dict_to_list()

    try:
        most_similar = await processor.find_most_similar_word(source_list, target_list)
        print(f"Source: {source}")
        print(f"Target: {target}")

        for index in range(len(most_similar)):
            target_word = target_list[index].split(":")[0]
            source_word = most_similar[index][0].split(":")[0]
            print(f"Most similar word of {target_word} is {source_word}")
            print(f"Similarity score: {most_similar[index][1]:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
