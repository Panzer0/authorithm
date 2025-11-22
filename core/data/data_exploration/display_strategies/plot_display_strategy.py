from abc import ABC, abstractmethod
from typing import List, Callable


class PlotDisplayStrategy(ABC):
    """Abstract base class for plot display strategies."""

    @abstractmethod
    def display_plots(
        self,
        feature_columns: List[str],
        plot_func: Callable,
        title: str,
        columns_per_row: int,
        **kwargs,
    ):
        """Display plots according to the strategy."""
        pass
