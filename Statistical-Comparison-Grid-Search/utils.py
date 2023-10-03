from __future__ import annotations

from math import factorial
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.svm import SVC
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

def make_data(noise: float, random_state: int, n_samples: int) -> tuple[np.ndarray, np.array]:
    X, y = make_moons(noise=noise, random_state=random_state, n_samples=n_samples)
    return X, y

def fit_search(X: np.ndarray, y: np.array, folds: int, repetitions: int) -> GridSearchCV:
    param_grid = [
        {"kernel": ["linear"]},
        {"kernel": ["poly"], "degree": [2, 3]},
        {"kernel": ["rbf"]},
    ]

    svc = SVC(random_state=0)

    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repetitions, random_state=0)

    search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring="roc_auc", cv=cv)
    search.fit(X, y)

    n = folds * repetitions
    n_train = len(list(cv.split(X, y))[0][0])
    n_test = len(list(cv.split(X, y))[0][1])

    return search, n, n_train, n_test

def get_results_df(clf: GridSearchCV) -> pd.DataFrame:
    results_df = pd.DataFrame(clf.cv_results_)
    results_df = results_df.sort_values(by=["rank_test_score"])
    results_df = results_df.set_index(
        results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
    ).rename_axis("kernel")

    return results_df

def corrected_std(differences: np.array, n_train: int, n_test: int) -> float:
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences: np.array, df: int, n_train: int, n_test: int) -> tuple[float, float]:
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val

def plot_dataset(X, y) -> go.Figure:
    fig = px.scatter(x=X[:, 0], y=X[:, 1], color=y, color_continuous_scale='bluered', title='Data')
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title='X1',
        yaxis_title='X2',
    )

    return fig

def plot_cv_test(model_scores: pd.DataFrame) -> go.Figure:
    fig = px.line(
        model_scores.transpose(),
        markers="o",
    )
    # Remove xaxis
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            title="CV test fold"
        ),
        yaxis_title="Model AUC",
        title="CV Test Fold AUC Scores"
    )

    return fig

def plot_correlation_heatmap(model_scores: pd.DataFrame) -> go.Figure:
    fig = px.imshow(model_scores.transpose().corr().round(2), color_continuous_scale="Viridis", text_auto=True)
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title='',
        yaxis_title='',
    )
    
    return fig

def get_model_scores_pairs(model_scores: pd.DataFrame, idx: tuple[int, int]) -> list[tuple[np.array, str]]:
        idx1, idx2 = idx
        model_1_scores = model_scores.iloc[idx1].values  # scores of the best model
        model_2_scores = model_scores.iloc[idx2].values  # scores of the second-best model
        # Getting Name of the models
        model_1_name = model_scores.index[idx1]
        model_2_name = model_scores.index[idx2]

        return (model_1_scores, model_1_name), (model_2_scores, model_2_name)

def frequentist_two_model(model_scores: pd.DataFrame, n: int, n_train: int, n_test: int) -> pd.DataFrame:
    (model_1_scores, model_1_name), (model_2_scores, model_2_name) = get_model_scores_pairs(model_scores, (0, 1))
    differences = model_1_scores - model_2_scores

    assert n == differences.shape[0], "Number of test sets does not match number of differences"

    df = n - 1
    t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
    t_stat_uncorrected = np.mean(differences) / np.sqrt(np.var(differences, ddof=1) / n)
    p_val_uncorrected = t.sf(np.abs(t_stat_uncorrected), df)

    # Creating DataFrame with columns Type, Model Name, t-value, p-value
    df = pd.DataFrame(
        [
            ["Corrected", model_1_name, t_stat, p_val],
            ["Uncorrected", model_2_name, t_stat_uncorrected, p_val_uncorrected],
        ],
        columns=["Type", "Model Name", "t-value", "p-value"],
    )
    return df.round(3)

def plot_bayesian_posterior(model_scores: pd.DataFrame, n: int, n_train: int, n_test: int) -> go.Figure:
    (model_1_scores, _), (model_2_scores, _) = get_model_scores_pairs(model_scores, (0, 1))
    differences = model_1_scores - model_2_scores
    df = n - 1

    # initialize random variable
    t_post = t(
        df, loc=np.mean(differences), scale=corrected_std(differences, n_train, n_test)
    )
    x = np.linspace(t_post.ppf(0.001), t_post.ppf(0.999), 10001)
    t_post_values = t_post.pdf(x)

    xl = [xc for xc in x if xc <= 0]
    xr = [xc for xc in x if xc >= 0]

    yl = t_post_values[:len(xl)]
    yr = t_post_values[len(xl):]

    xr_annot = np.quantile(xr, 0.2)
    yr_annot = t_post.pdf(xr_annot) / 2
    better_prob = 1 - t_post.cdf(0)

    xl_annot = np.quantile(xl, 0.7)
    yl_annot = t_post.pdf(xl_annot) / 2
    worse_prob = t_post.cdf(0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xr, 
            y=yr, 
            fill='tozeroy', 
            mode='none', 
            name=f"Probability of Being Better"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xl, 
            y=yl, 
            fill='tozeroy', 
            mode='none', 
            name=f"Probability of Being Worse"
        )
    )

    fig.add_vline(
        x=0,
        line_dash='dot'
    )

    fig.add_annotation(
        x=xr_annot,
        y=yr_annot,
        text=f"<b>{100*better_prob:.2f}%</b>",
        showarrow=False,
        font=dict(
            size=18
        )
    )

    fig.add_annotation(
        x=xl_annot,
        y=yl_annot,
        text=f"<b>{100*worse_prob:.2f}%</b>",
        showarrow=False,
        font=dict(
            size=18
        )
    )

    fig.update_layout(
        xaxis_title='Mean Difference (μ)',
        yaxis_title='Probability Density',
        title='Posterior Distribution'
    )


    return fig

def plot_rope(model_scores: pd.DataFrame, n: int, n_train: int, n_test: int, rope_interval: tuple[float, float]) -> go.Figure:
    (model_1_scores, _), (model_2_scores, _) = get_model_scores_pairs(model_scores, (0, 1))
    differences = model_1_scores - model_2_scores
    df = n - 1

    # initialize random variable
    t_post = t(
        df, loc=np.mean(differences), scale=corrected_std(differences, n_train, n_test)
    )
    x = np.linspace(t_post.ppf(0.001), t_post.ppf(0.999), 10001)
    t_post_values = t_post.pdf(x)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=t_post_values,
            mode='lines',
            name='Posterior Distribution',
            showlegend=False
        )
    )
    x_int = [xc for xc in x if xc >= rope_interval[0] and xc <= rope_interval[1]]
    y_int = t_post.pdf(x_int)

    x_annot = np.quantile(x_int, 0.5)
    y_annot = t_post.pdf(x_annot) / 2
    rope_prob = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])

    fig.add_trace(
        go.Scatter(
            x=x_int,
            y=y_int,
            mode='none',
            fill='tozeroy',
            fillcolor="rgba(99, 110, 250, 0.5)",
            
            showlegend=False
        )
    )

    fig.add_vline(
        x=rope_interval[0],
        line_dash='dot'
    )

    fig.add_vline(
        x=rope_interval[1],
        line_dash='dot'
    )

    fig.add_annotation(
        x=x_annot,
        y=y_annot,
        text=f"<b>{100*rope_prob:.2f}%</b>",
        showarrow=False,
        font=dict(
            size=18
        )
    )

    fig.update_layout(
        xaxis_title='Mean Difference (μ)',
        yaxis_title='Probability Density',
        title=f'Region of Practical Equivalence (ROPE)',
        yaxis_showticklabels=False
    )

    return fig

def get_cred_intervals(intervals: list[float], model_scores: pd.DataFrame, n: int, n_train: int, n_test: int) -> pd.DataFrame:
    (model_1_scores, _), (model_2_scores, _) = get_model_scores_pairs(model_scores, (0, 1))
    differences = model_1_scores - model_2_scores
    df = n - 1

    # initialize random variable
    t_post = t(
        df, loc=np.mean(differences), scale=corrected_std(differences, n_train, n_test)
    )
    cred_intervals = []

    for interval in intervals:
        cred_interval = list(t_post.interval(interval))
        cred_intervals.append([interval, cred_interval[0], cred_interval[1]])

    cred_int_df = pd.DataFrame(
        cred_intervals, columns=["interval", "lower value", "upper value"]
    ).set_index("interval")
    
    return cred_int_df.round(3).reset_index()

def get_pairwise_frequentist(model_scores: pd.DataFrame, n: int, n_train: int, n_test: int) -> pd.DataFrame:
    n_comparisons = factorial(len(model_scores)) / (
        factorial(2) * factorial(len(model_scores) - 2)
    )
    df = n - 1

    cmbs = list(combinations(range(len(model_scores)), 2))
    pairwise_t_test = []

    for cmb in cmbs:
        (model_i_scores, model_i_name), (model_k_scores, model_k_name) = get_model_scores_pairs(model_scores, cmb)
        differences = model_i_scores - model_k_scores
        
        assert n == differences.shape[0], "Number of samples is not equal to the number of differences"

        t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
        p_val *= n_comparisons  # implement Bonferroni correction
        # Bonferroni can output p-values higher than 1
        p_val = 1 if p_val > 1 else p_val
        pairwise_t_test.append(
            [model_i_name, model_k_name, t_stat, p_val]
        )
    pairwise_comp_df = pd.DataFrame(
        pairwise_t_test, columns=["model_1", "model_2", "t_stat", "p_val"]
    ).round(3)

    return pairwise_comp_df

def get_pairwise_bayesian(rope_interval: tuple[float, float], model_scores: pd.DataFrame, n: int, n_train: int, n_test: int) -> pd.DataFrame:
    n_comparisons = factorial(len(model_scores)) / (
        factorial(2) * factorial(len(model_scores) - 2)
    )
    df = n - 1

    cmbs = list(combinations(range(len(model_scores)), 2))
    pairwise_bayesian = []

    for cmb in cmbs:
        (model_i_scores, model_i_name), (model_k_scores, model_k_name) = get_model_scores_pairs(model_scores, cmb)
        differences = model_i_scores - model_k_scores

        assert n == differences.shape[0], "Number of samples is not equal to the number of differences"

        t_post = t(
            df, loc=np.mean(differences), scale=corrected_std(differences, n_train, n_test)
        )

        worse_prob = t_post.cdf(rope_interval[0])
        better_prob = 1 - t_post.cdf(rope_interval[1])
        rope_prob = t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0])
        t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
        p_val *= n_comparisons  # implement Bonferroni correction
        # Bonferroni can output p-values higher than 1
        p_val = 1 if p_val > 1 else p_val
        pairwise_bayesian.append([model_i_name, model_k_name, t_stat, p_val, worse_prob, better_prob, rope_prob])

    pairwise_bayesian_df = pd.DataFrame(
        pairwise_bayesian, columns=["model_1", "model_2", "t_stat", "p_val", "worse_prob", "better_prob", "rope_prob"]
    ).round(3)

    return pairwise_bayesian_df


