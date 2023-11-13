import numpy as np
class Transform:
    def __init__(self) -> None:
        self.transform = np.identity

    @property
    def transform