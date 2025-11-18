from typing import List, Callable

from matplotlib import pyplot as plt

from core.data.data_exploration.plot_display_strategy import PlotDisplayStrategy


class SequentialDisplayStrategy(PlotDisplayStrategy):
    """Display plots one after another in separate windows."""

    def display_plots(
        self,
        feature_columns: List[str],
        plot_func: Callable,
        title: str,
        columns_per_row: int = None,  # Not used in sequential mode
        **kwargs,
    ):
        for col in feature_columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_func(col, ax, **kwargs)
            ax.set_title(f"{title}: {col}")
            plt.tight_layout()
            plt.show()
