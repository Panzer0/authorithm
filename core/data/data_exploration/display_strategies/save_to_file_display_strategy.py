import os
from datetime import datetime
from typing import List, Callable
from matplotlib import pyplot as plt
from core.data.data_exploration.display_strategies.plot_display_strategy import (
    PlotDisplayStrategy,
)


class SaveToFileDisplayStrategy(PlotDisplayStrategy):
    """Save plots to files."""

    def __init__(self, base_dir: str = "./graphs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(base_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📂 Saving graphs to: {self.output_dir}")

    def display_plots(
        self,
        feature_columns: List[str],
        plot_func: Callable,
        title: str,
        columns_per_row: int = None,
        **kwargs,
    ):
        subfolder = kwargs.get("folder_name", "other_plots")
        save_path = os.path.join(self.output_dir, subfolder)
        os.makedirs(save_path, exist_ok=True)

        for col in feature_columns:
            fig, ax = plt.subplots(figsize=(10, 8))

            plot_func(col, ax)

            ax.set_title(f"{col} - {title}")
            plt.tight_layout()

            safe_col_name = "".join(
                c for c in col if c.isalnum() or c in (" ", "_", "-")
            ).strip()
            file_path = os.path.join(save_path, f"{safe_col_name}.png")

            plt.savefig(file_path)
            print(f"Saving to {file_path}")
            plt.close(fig)
