from typing import Any

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


class PlotBuilder:
    """Encapsulates plot creation logic."""

    def __init__(self, df, feature_columns):
        self.df = df
        self.feature_columns = feature_columns

    def create_histogram(self, col: str, ax: Any, bins: int = 100):
        """Create a histogram on the given axis."""
        self.df[col].hist(bins=bins, ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    def create_log_histogram(self, col: str, ax: Any, bins: int = 100):
        """Create a log-scaled histogram on the given axis."""
        self.df[col].hist(bins=bins, ax=ax)
        ax.set_yscale("log")
        ax.set_xlabel(col)
        ax.set_ylabel("Log Frequency")

    def create_boxplot(self, col: str, ax: Any):
        """Create a boxplot on the given axis."""
        sns.boxplot(x=self.df[col], ax=ax, fliersize=2)

    def create_standard_histograms(self, bins: int = 30):
        """Create standard histograms using pandas built-in."""
        self.df[self.feature_columns].hist(
            figsize=(16, 12), bins=bins, edgecolor="black"
        )
        plt.suptitle("Stylometric Feature Distributions", fontsize=18)
        plt.tight_layout()
        plt.show()

    def create_correlation_heatmap(self, method: str = "pearson"):
        """Create a correlation heatmap."""
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
