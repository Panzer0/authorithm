import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


class DataExplorer:
    def __init__(self, df, feature_columns):
        self.df = df
        self.feature_columns = feature_columns

    def print_sample_rows(self, n=3):
        print(self.df[self.feature_columns].iloc[:n].to_string())

    def _multi_feature_plot(
        self, plot_func, title: str, columns_per_row: int = 4, **kwargs
    ):
        """Generic subplot grid for stylometric features using `plot_func`."""
        num_features = len(self.feature_columns)
        rows = (num_features + columns_per_row - 1) // columns_per_row
        fig, axes = plt.subplots(
            rows, columns_per_row, figsize=(4 * columns_per_row, 3 * rows)
        )
        axes = axes.flatten()

        for i, col in enumerate(self.feature_columns):
            plot_func(col, axes[i], **kwargs)
            axes[i].set_title(col)

        for i in range(num_features, len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_feature_histograms(self, bins=30):
        """Plot standard histograms for all features."""
        self.df[self.feature_columns].hist(
            figsize=(16, 12), bins=bins, edgecolor="black"
        )
        plt.suptitle("Stylometric Feature Distributions", fontsize=18)
        plt.tight_layout()
        plt.show()

    def plot_log_histograms(self, columns_per_row=4):
        """Plot histograms with log-scaled y-axis."""

        def _log_hist(col, ax, bins=100):
            self.df[col].hist(bins=bins, ax=ax)
            ax.set_yscale("log")
            ax.set_xlabel(col)
            ax.set_ylabel("Log Frequency")

        self._multi_feature_plot(
            _log_hist, "Log-Scaled Histograms", columns_per_row
        )

    def plot_boxplots(self, columns_per_row=4):
        """Plot boxplots for all features."""

        def _boxplot(col, ax):
            sns.boxplot(x=self.df[col], ax=ax, fliersize=2)

        self._multi_feature_plot(
            _boxplot, "Boxplots of Stylometric Features", columns_per_row
        )

    def plot_correlation_heatmap(self, method="pearson"):
        """Plot a lower-triangle correlation heatmap."""
        if method not in ["pearson", "spearman"]:
            raise ValueError("Method must be 'pearson' or 'spearman'.")
        corr = self.df[self.feature_columns].corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.75},
        )
        plt.title(
            f"{method.title()} Correlation Heatmap (Lower Triangle)",
            fontsize=16,
        )
        plt.tight_layout()
        plt.show()

    def print_feature_extremes(self, top_n: int = 5):
        """Print out top-N and bottom-N rows per feature with value, author, and body preview."""
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
