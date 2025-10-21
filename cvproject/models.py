from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from cvproject.shapes.insts import Insts


class Model(ABC):
    @abstractmethod
    def infer(
        self, 
        img: npt.NDArray[np.uint8]
    ) -> Insts:
        pass

class PromptModel(ABC):
    @abstractmethod
    def infer_from_prompt(
        self,
        img: npt.NDArray[np.uint8],
        prompt: Insts
    ) -> Insts:
        pass