import abc
from typing import Union
import numpy as np


class BaseLLM(abc.ABC):
    @abc.abstractmethod
    async def get_embedding(self, text: Union[str, list[str]]) -> np.ndarray:
        pass
