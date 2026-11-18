from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    """A discrete-time state-feedback controller running at a fixed sample time `dt`."""

    name: str = "controller"

    def __init__(self, dt: float) -> None:
        self.dt = dt

    def reset(self) -> None:
        """Clear internal state (integrators, warm starts...). Called before every simulation."""

    @abstractmethod
    def compute(self, x: np.ndarray, x_ref: np.ndarray) -> float:
        """Return the force [N] for measured state `x` and reference state `x_ref`."""
