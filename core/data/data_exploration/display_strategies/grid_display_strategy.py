from matplotlib import pyplot as plt
from typing import List, Callable

from core.data.data_exploration.display_strategies.plot_display_strategy import (
    PlotDisplayStrategy,
)


class GridDisplayStrategy(PlotDisplayStrategy):
    """Display all plots in a single grid window."""

    def display_plots(
        self,
        feature_columns: List[str],
        plot_func: Callable,
        title: str,
        columns_per_row: int = 4,
        **kwargs,
    ):
        num_features = len(feature_columns)
        rows = (num_features + columns_per_row - 1) // columns_per_row
        fig, axes = plt.subplots(
            rows, columns_per_row, figsize=(4 * columns_per_row, 3 * rows)
        )
        axes = axes.flatten() if num_features > 1 else [axes]

        for i, col in enumerate(feature_columns):
            plot_func(col, axes[i], **kwargs)
            axes[i].set_title(col)

        # Remove empty subplots
        for i in range(num_features, len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
