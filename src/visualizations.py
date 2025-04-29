import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

import seaborn as sns

import plotly.graph_objects as go

from scipy.stats import pearsonr, spearmanr, kendalltau


def visualize_feature_dists(
        df: pd.DataFrame,
        target: pd.Series,
        bins: int = 40,
        figsize: tuple[int, int] = (18, 6)
    ) -> None:
    """Visualize the distribution of features in the DataFrame, split by class.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to visualize.
    target : pandas.Series
        The target variable to split the data by class.
    bins : int, default 40
        The number of bins to use for the histogram.
    figsize : tuple of int, default (15, 5)
        The size of the figure to create for the plots.

    Returns
    -------
    None
        Displays the distribution plots for each feature.
    """

    ys = target.values
    clsses = sorted(set(ys))
    ys = np.where(ys == "Malignant", 1, 0)


    def inner_plot_histogram(data, name, i, color, title):
        sns.histplot(
            data,
            kde=True, bins=bins, ax=axes[i],
            color=color, line_kws={'linewidth': 3}
        )
        axes[i].axvline(
            data.mean(),
            color="black", label="Mean",
            linewidth=2, linestyle="-", 
        )
        axes[i].axvline(
            data.median(),
            color="black", label="Median",
            linewidth=2, linestyle="--", 
        )
        axes[i].axvline(
            data.mean() + data.std(),
            color="black", label="Std",
            linewidth=2, linestyle=":",
        )
        axes[i].axvline(
            data.mean() - data.std(),
            color="black",
            linewidth=2, linestyle=":",
        )
        axes[i].set_title(title)
        axes[i].set_xlabel(name)
        axes[i].legend()
        pass

    for col in df.columns:

        class0 = df[col][ys == 0]
        class1 = df[col][ys == 1]

        _, axes = plt.subplots(1, 3, figsize=figsize)

        for i, (col_i, clss) in enumerate(zip([class0, class1], clsses)):
            if i == 1:  # Clever index swapping ;)
                i = 2
            inner_plot_histogram(
                data=col_i,
                name=col,
                i=i,
                color="#3486eb" if i == 0 else "#eb3434",
                title=f"Distribution of {col} for class {clss}"
            )

        # Clever index assigning, fill in the empty subplot
        i = 1
        inner_plot_histogram(
            data=df[col],
            name=col,
            i=i,
            color="#b734eb",
            title=f"Distribution of {col} for all classes"
        )

        plt.suptitle(
            f"Distribution of feature '{col}'",
            fontsize=16
        )
        plt.tight_layout()
        plt.show()


def plot_correlation_coefficients(
        corr_df: pd.DataFrame,
        title: str = "Correlation Coefficients",
    ) -> None:
    """Plot the correlation coefficients of the features against the target.

    Parameters
    ----------
    corr_df : pandas.DataFrame
        The DataFrame containing the correlation coefficients. The columns of
        the DataFrame are the features and the index is the correlation
        coefficient method.
    title : str, default "Correlation Coefficients"
        The title of the plot.

    Returns
    -------
    None
        Displays the plot of the correlation coefficients.
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=corr_df.columns,
        y=corr_df.loc["point_biserial"],
        mode="markers",
        marker=dict(size=12),
        name="Point Biserial"
    ))

    fig.add_trace(go.Scatter(
        x=corr_df.columns,
        y=corr_df.loc["spearman"],
        mode="markers",
        marker=dict(size=12),
        name="Spearman"
    ))

    fig.add_trace(go.Scatter(
        x=corr_df.columns,
        y=corr_df.loc["kendall"],
        mode="markers",
        marker=dict(size=12),
        name="Kendall"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Correlation Coefficient",
        template="plotly_white",
        height=700
    )

    fig.show()


def heatmap_correlations(
        matrix: pd.DataFrame,
        labels: list[str],
        title: str = "Pearson correlation Heatmap",
        figsize: tuple[int, int] = (16, 16),
        cmap: str = "crest"
    ) -> None:
    """Plot a heatmap of the correlations between the features in the given
    dataframe.

    Parameters
    ----------
    matrix : pd.DataFrame
        The dataframe containing the features to plot the correlations for. The
        columns of the dataframe are the features.
    labels : list of str
        The labels for the features to plot the correlations for.
    figsize : tuple of int, default (20, 20)
        The size of the figure to create for the heatmap.
    title : str, default "Correlation Heatmap"
        The title of the heatmap.
    cmap : str, default "crest"
        The colormap to use for the heatmap.

    Returns
    -------
    None
        Displays the heatmap of the correlations between the features.
    """

    plt.figure(figsize=figsize)

    sns.heatmap(
        matrix,
        annot=False,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={"shrink": .8},
        linewidths=0.5,
        linecolor="black",
        square=True
    )
    plt.xticks(
        ticks=np.arange(len(labels)) + 0.5,
        labels=labels,
        rotation=45,
        ha="right"
    )
    plt.yticks(
        ticks=np.arange(len(labels)) + 0.5,
        labels=labels,
        rotation=0,
        va="center"
    )
    plt.title(title, fontsize=20)
    plt.show()


def pairplot(
        data: pd.DataFrame,
        title: str,
        kde_color: str = "#421f6e",
        scatter_color: str = "#7a4db0",
        hue: pd.Series = None,
        cmap: str = "cool"
    ):
    """Plot a pairplot of the given data.

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot the pairplot for. The columns of the dataframe are the
        features.
    title : str
        The title of the pairplot.
    kde_color : str
        The color to use for the KDE plots.
    scatter_color : str
        The color to use for the scatter plots.
    hue : pd.Series
        The hue variable to use for the pairplot. If None, the pairplot will
        not be colored by hue.
    cmap : str
        The colormap to use for the pairplot. If hue is None, this will be
        ignored.

    Returns
    -------
    None
        Displays the pairplot of the given data.
    """

    def annotate_correlations(x, y, **kwargs):
        """Annotate the correlation coefficients on the pairplot."""

        pearson_coef, _ = pearsonr(x, y)
        spearman_coef, _ = spearmanr(x, y)
        kendall_coef, _ = kendalltau(x, y)

        text = "".join([
            f"Pearson: {pearson_coef:.2f}\n",
            f"Spearman: {spearman_coef:.2f}\n",
            f"Kendall: {kendall_coef:.2f}"
        ])
        
        plt.annotate(
            text,
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.75)
        )

    def plot_mean_median(x, **kwargs):
        """Plot the mean and median of the data on the diagonal of the
        pairplot."""

        mean_val = np.mean(x)
        plt.axvline(
            mean_val,
            color="blue",
            linestyle="-",
            label=f"Mean: {mean_val:.2f}"
        )

        median_val = np.median(x)
        plt.axvline(
            median_val,
            color="red",
            linestyle="--",
            label=f"Median {median_val:.2f}"
        )
        plt.legend()


    if hue is not None:
        plot_kws = {"hue": hue, "palette": cmap}
    else:
        plot_kws = {"color": scatter_color}

    g = sns.pairplot(
        data,
        diag_kind="kde",
        plot_kws=plot_kws,
        diag_kws={"color": kde_color}
    )

    g.map_upper(annotate_correlations)
    g.map_lower(sns.kdeplot, levels=4, color="black")
    g.map_diag(plot_mean_median)

    if hue is not None:
        # Create a normalized scalar mappable for the color bar
        norm = Normalize(vmin=hue.min(), vmax=hue.max())
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        g.figure.subplots_adjust(right=0.85)
        cbar_ax = g.figure.add_axes([0.88, 0.15, 0.02, 0.7])
        g.figure.colorbar(sm, cax=cbar_ax, label=hue.name)

    plt.suptitle(title, y=1.02)
    plt.show()