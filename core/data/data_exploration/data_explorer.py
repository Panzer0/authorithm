from core.data.data_exploration.grid_display_strategy import GridDisplayStrategy
from core.data.data_exploration.plot_builder import PlotBuilder
from core.data.data_exploration.sequential_display_strategy import (
    SequentialDisplayStrategy,
)


class DataExplorer:
    """Main class for data exploration with flexible plot display strategies."""

    def __init__(self, df, feature_columns, display_mode: str = "grid"):
        """
        Initialize DataExplorer.

        Args:
            df: DataFrame to explore
            feature_columns: List of feature column names
            display_mode: Either "grid" (all plots in one window) or "sequential" (one by one)
        """
        self.df = df
        self.feature_columns = feature_columns
        self.plot_builder = PlotBuilder(df, feature_columns)
        self._set_display_strategy(display_mode)
        print(f"Display_strategy = {self._display_strategy}")

    def _set_display_strategy(self, mode: str):
        """Set the display strategy based on mode."""
        if mode == "grid":
            self._display_strategy = GridDisplayStrategy()
        elif mode == "sequential":
            self._display_strategy = SequentialDisplayStrategy()
        else:
            raise ValueError("Mode must be 'grid' or 'sequential'")

    @property
    def display_mode(self) -> str:
        """Get the current display mode."""
        if isinstance(self._display_strategy, GridDisplayStrategy):
            return "grid"
        elif isinstance(self._display_strategy, SequentialDisplayStrategy):
            return "sequential"
        else:
            return "unknown"

    @display_mode.setter
    def display_mode(self, mode: str):
        """Set the display mode for plots."""
        self._set_display_strategy(mode)

    def print_sample_rows(self, n=3):
        """Print sample rows from the dataframe."""
        print(self.df[self.feature_columns].iloc[:n].to_string())

    def _multi_feature_plot(
        self, plot_func, title: str, columns_per_row: int = 4, **kwargs
    ):
        """Generic method for creating multiple feature plots."""
        self._display_strategy.display_plots(
            self.feature_columns, plot_func, title, columns_per_row, **kwargs
        )

    def plot_feature_histograms(self, columns_per_row=4):
        """Plot standard histograms for all features."""
        self._multi_feature_plot(
            self.plot_builder.create_histogram,
            "Histograms",
            columns_per_row,
        )

    def plot_log_histograms(self, columns_per_row=4):
        """Plot histograms with log-scaled y-axis."""
        self._multi_feature_plot(
            self.plot_builder.create_log_histogram,
            "Log-Scaled Histograms",
            columns_per_row,
        )

    def plot_boxplots(self, columns_per_row=4):
        """Plot boxplots for all features."""
        self._multi_feature_plot(
            self.plot_builder.create_boxplot,
            "Boxplots of Stylometric Features",
            columns_per_row,
        )

    def plot_correlation_heatmap(self, method="pearson"):
        """
        Plot a lower-triangle correlation heatmap.

        Args:
            method: The method to be used. Accepts 'pearson' and 'spearman'
        """
        self.plot_builder.create_correlation_heatmap(method)

    def print_feature_extremes(self, top_n: int = 5):
        """Print out top-N and bottom-N rows per feature with value, author, ID and body preview."""
        for col in self.feature_columns:
            print(f"\n=== {col.upper()} ===")
            for direction, ascending in [
                ("🔺 Top values", False),
                ("🔻 Bottom values", True),
            ]:
                print(f"\n{direction}:")
                rows = self.df.sort_values(by=col, ascending=ascending).head(
                    top_n
                )
                for _, row in rows.iterrows():
                    print(f"Value: {row[col]:.4f}")
                    print(f"Author: {row['author']}")
                    print(f"ID: {row['id']}")
                    print(
                        f"Body: {row['body'][:300]}{'...' if len(row['body']) > 300 else ''}"
                    )
                    print("-" * 60)

    def run_full_exploration(self):
        """Run all exploratory methods in sequence."""
        self.print_sample_rows()
        self.print_feature_extremes()
        self.plot_feature_histograms()
        self.plot_log_histograms()
        self.plot_boxplots()
        self.plot_correlation_heatmap(method="pearson")
        self.plot_correlation_heatmap(method="spearman")
