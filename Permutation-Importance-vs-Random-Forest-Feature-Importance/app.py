import numpy as np
import gradio as gr
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import utils




def app_fn(seed: int, n_cat: int, n_estimators: int, min_samples_leaf: int):
    X, y = fetch_openml(
        "titanic", version=1, as_frame=True, return_X_y=True, parser="pandas"
    )

    rng = np.random.RandomState(seed=seed)

    X["random_cat"] = rng.randint(n_cat, size=X.shape[0])
    X["random_num"] = rng.randn(X.shape[0])

    categorical_columns = ["pclass", "sex", "embarked", "random_cat"]
    numerical_columns = ["age", "sibsp", "parch", "fare", "random_num"]

    X = X[categorical_columns + numerical_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed)

    categorical_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
    )
    numerical_pipe = SimpleImputer(strategy="mean")

    preprocessing = ColumnTransformer(
        [
            ("cat", categorical_encoder, categorical_columns),
            ("num", numerical_pipe, numerical_columns),
        ],
        verbose_feature_names_out=False,
    )

    clf = Pipeline(
        [
            ("preprocess", preprocessing),
            ("classifier", RandomForestClassifier(
                    random_state=seed,
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf
                )
            ),
        ]
    )

    clf.fit(X_train, y_train)

    fig_mdi = utils.plot_rf_importance(clf)
    fig_perm_train = utils.plot_permutation_boxplot(clf, X_train, y_train, set_="train set")
    fig_perm_test = utils.plot_permutation_boxplot(clf, X_test, y_test, set_="test set")

    return fig_mdi, fig_perm_train, fig_perm_test


title = "Permutation Importance vs Random Forest Feature Importance (MDI)"
with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(
        """
        This demo compares the feature importances of a Random Forest classifier using the Mean Decrease Impurity (MDI) method and the Permutation Importance method. \
        To showcase the difference between the two methods, we add two random features to the Titanic dataset. \
        The first random feature is categorical and the second one is numerical. \
        The categorical feature can have its number of categories changed \
        and the numerical feature is sampled from a Standard Normal Distribution. \
        Random Forest hyperparameters can also be changed to verify the impact of model complexity on the feature importances.

        See the original scikit-learn example [here](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py).
        """
    )

    with gr.Row():
        seed = gr.inputs.Slider(0, 42, 1, default=42, label="Seed")
        n_cat = gr.inputs.Slider(2, 30, 1, default=3, label="Number of categories in random_cat")
        n_estimators = gr.inputs.Slider(5, 150, 5, default=100, label="Number of Trees")
        min_samples_leaf = gr.inputs.Slider(1, 30, 5, default=1, label="Minimum number of samples to create a leaf")

    
    fig_mdi = gr.Plot(label="Mean Decrease Impurity (MDI)")

    with gr.Row():
        fig_perm_train = gr.Plot(label="Permutation Importance (Train)")
        fig_perm_test = gr.Plot(label="Permutation Importance (Test)")
    
    seed.change(fn=app_fn, outputs=[fig_mdi, fig_perm_train, fig_perm_test], inputs=[seed, n_cat, n_estimators, min_samples_leaf])
    n_cat.change(fn=app_fn, outputs=[fig_mdi, fig_perm_train, fig_perm_test], inputs=[seed, n_cat, n_estimators, min_samples_leaf])
    n_estimators.change(fn=app_fn, outputs=[fig_mdi, fig_perm_train, fig_perm_test], inputs=[seed, n_cat, n_estimators, min_samples_leaf])
    min_samples_leaf.change(fn=app_fn, outputs=[fig_mdi, fig_perm_train, fig_perm_test], inputs=[seed, n_cat, n_estimators, min_samples_leaf])
    demo.load(fn=app_fn, outputs=[fig_mdi, fig_perm_train, fig_perm_test], inputs=[seed, n_cat, n_estimators, min_samples_leaf])

demo.launch()