
import pandas as pd

import numpy as np

from scipy.stats import pearsonr, spearmanr, kendalltau, pointbiserialr

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap


def check_types(
        df: pd.DataFrame,
        print_summary: bool = True
    ) -> tuple[dict[str, set[type]], set[str], dict[str, int]]:
    """Check the types of columns and whether they have NaN values.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check.
    print_summary : bool, default True
        Whether to print a summary of the checks.

    Returns
    -------
    col_types : dict
        A dictionary mapping column names to sets of types.
    has_nan : set
        A set of column names that have NaN values.
    count_nan : dict
        A dictionary mapping column names to the count of NaN values.
    """
    col_types = {}
    count_nan = {}

    for col in df.columns:
        col_types[col] = set()
        count_nan[col] = 0

        for val in df[col]:
            col_types[col].add(type(val))
            if pd.isna(val):
                count_nan[col] += 1

    if print_summary:
        for col in df.columns:
            print(
                f"Column <{col}> is of type {col_types[col]} and",
                f"has NaN {count_nan[col]} values"
                if count_nan[col] > 0 else
                "does not have NaN values"
            )

    return col_types, count_nan


def fill_nans_with_median(
        df: pd.DataFrame,
        target: pd.Series,
        class_based: bool = True
    ) -> pd.DataFrame:
    """Fills NaN values in numeric columns of a DataFrame with the median.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    target : pd.Series
        The target variable. Must have the same index as df.
    class_based : bool, optional
        If True, fills NaN values based on the median of each class in the
        target variable. If False, fills NaN values based on the overall median
        of each column. Default is True.

    Returns
    -------
    pd.DataFrame
        The DataFrame with NaNs filled. Modifies the DataFrame in place.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns

    if class_based:
        if not df.index.equals(target.index):
            raise ValueError(
                "DataFrame and target Series must have the same index."
            )

        temp_df = df.copy()
        temp_df['_target_group'] = target

        for col in numeric_cols:
            if temp_df[col].isnull().any():
                filled_col = temp_df.groupby('_target_group')[col].transform(
                    lambda x: x.fillna(x.median())
                )

                if filled_col.isnull().any():
                     overall_median = temp_df[col].median()
                     filled_col.fillna(overall_median, inplace=True)

                df[col] = filled_col

    else:
        for col in numeric_cols:
            if df[col].isnull().any():
                col_median = df[col].median()
                df[col].fillna(col_median, inplace=True)

    return df


def get_correlations(
        df: pd.DataFrame,
        target: pd.Series,
        sort_by_abs: bool = True
    ) -> pd.DataFrame:
    """Calculate the spearman rho, kendall tau, and point biserial correlation
    coefficients of the given dataframe against the given target data.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the features to calculate the correlation
        coefficients. The columns of the dataframe are the features.
    target : pd.Series
        The target data to calculate the correlation coefficients against.
    sort_by_abs : bool, default=True
        If True, return the absolute value of the correlation coefficients,
        sorted by the absolute value. Otherwise, return the unsorted absolute
        value of the correlation coefficients. Sorting is based on the sum of
        the absolute values of the correlation coefficients for each feature.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the correlation coefficients of the features
        against the target data. The columns of the dataframe are the features
        and the index is the correlation coefficient method.
    """

    corr_coeffs = pd.DataFrame(
        columns=df.columns,
        index=["spearman", "kendall", "point_biserial"]
    )

    for col in df.columns:
        corr_coeffs.loc["spearman", col] = abs(spearmanr(
            target,
            df[col]
        )[0])
        corr_coeffs.loc["kendall", col] = abs(kendalltau(
            target,
            df[col]
        )[0])
        corr_coeffs.loc["point_biserial", col] = abs(pointbiserialr(
            target,
            df[col]
        )[0])

    if sort_by_abs:
        corr_coeffs = corr_coeffs.reindex(
            corr_coeffs.abs().sum(axis=0).sort_values(ascending=False).index,
            axis=1
        )

    return corr_coeffs


def get_correlations_pair_matrix(
        df: pd.DataFrame
    ):
    """Calculate the point pearson correlation coefficients for all pairs of
    columns in the given dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the features to calculate the correlation
        coefficients. The columns of the dataframe are the features.

    Returns
    -------
    np.ndarray
        A 2D numpy array containing the point biserial correlation coefficients
        for all pairs of columns in the dataframe. The shape of the array is
        (n_features, n_features), where n_features is the number of columns in
        the dataframe.
    """

    corr_matrix = np.zeros((df.shape[1], df.shape[1]))

    for i in range(df.shape[1]):
        for j in range(i + 1, df.shape[1]):
            corr_matrix[i, j] = abs(pearsonr(
                df.iloc[:, i],
                df.iloc[:, j]
            )[0])
            corr_matrix[j, i] = corr_matrix[i, j]
    
    return corr_matrix


def dim_red(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Reduce the dimensions of a dataset with either PCA, t_SNE, or UMAP

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to reduce.
    **kwargs : keyword arguments
        Additional arguments to pass to the dimensionality reduction method.
        - method : str
            The method to use for dimensionality reduction. Must be one of
            'pca', 'tsne', or 'umap'.
        - n_components : int, optional
            The number of components to keep. Default is 3 for t-SNE and 5
            for UMAP. For PCA, it can be specified as a percentage of the
            explained variance.
        - percent : float, optional
            The percentage of explained variance to keep. Must be used with
            PCA. Default is None.

    Returns
    -------
    pd.DataFrame
        The reduced DataFrame. The columns are named according to the method
        used for dimensionality reduction.
    """

    if "method" not in kwargs:
        raise ValueError("Method must be specified in kwargs")

    if kwargs["method"] == "pca":
        if "n_components" in kwargs:
            n_components = kwargs["n_components"]
            pca = PCA()
            reduced = pca.fit_transform(df)
            reduced = reduced[:, :n_components]

        elif "percent" in kwargs:
            pca = PCA()
            reduced = pca.fit_transform(df)
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            n_components = np.argmax(
                cumulative_variance >= kwargs["percent"]
            ) + 1
            reduced = reduced[:, :n_components]

        else:
            raise ValueError("Either n_components or percent must be specified")

        return pd.DataFrame(
            reduced,
            columns=[f"PC{i}" for i in range(1, n_components + 1)]
        )

    elif kwargs["method"] == "tsne":
        if "n_components" in kwargs:
            if kwargs["n_components"] > 3:
                raise ValueError("n_components must be <= 3 for t-SNE")
            n_components = kwargs["n_components"]

        else:
            n_components = 3

        tsne = TSNE(n_components)
        reduced = tsne.fit_transform(df)
    
        return pd.DataFrame(
            reduced,
            columns=[f"t-SNE{i}" for i in range(1, n_components + 1)]
        )

    elif kwargs["method"] == "umap":
        if "n_components" in kwargs:
            n_components = kwargs["n_components"]

        else:
            n_components = 5

        umap_model = umap.UMAP(n_components=n_components)
        reduced = umap_model.fit_transform(df)
    
        return pd.DataFrame(
            reduced,
            columns=[f"UMAP{i}" for i in range(1, n_components + 1)]
        )

    else:
        raise ValueError("Method must be one of 'pca', 'tsne', or 'umap'")