import pathlib
from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt


def custom_barplot(
        labels: List[str] or np.ndarray,
        values: List[float] or np.ndarray,
        xlabel: str,
        ylabel: str,
        ylim: Tuple[float, float],
        title: str,
        filepath: str or pathlib.Path
) -> None:
    """Create a barplot for class membership in a dataset"""
    fig = plt.figure(figsize=(5, 3.5), dpi=300)
    x = [2*i for i in list(range(len(labels)))]
    plt.bar(x, height=values, color='lightseagreen', align='center', width=1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, labels, rotation=90)
    plt.ylim(ylim)
    plt.grid(True)
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')
