import gradio as gr
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomTreesEmbedding

import utils

def app_fn(n_samples: int, n_estimators: int, max_depth: int): 
    # Create Data
    (X_train_ensemble, y_train_ensemble), \
    (X_train_linear, y_train_linear), \
    (X_test, y_test) = utils.create_and_split_dataset(n_samples)

    # Creating and fitting Random Forest
    random_forest = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=10
    )
    random_forest.fit(X_train_ensemble, y_train_ensemble)

    # Creating and fitting Gradient Boosting
    gradient_boosting = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=10
    )
    _ = gradient_boosting.fit(X_train_ensemble, y_train_ensemble)

    # Creating and fitting Pipeline of Random Tree Embedding w/ Logistic Regression
    random_tree_embedding = RandomTreesEmbedding(
        n_estimators=n_estimators, max_depth=max_depth, random_state=0
    )
    rt_model = make_pipeline(random_tree_embedding, LogisticRegression(max_iter=1000))
    rt_model.fit(X_train_linear, y_train_linear)

    # Creating and fitting Pipeline of Random Forest Embedding w/ Logistic Regression
    rf_leaves_yielder = FunctionTransformer(utils.rf_apply, kw_args={"model": random_forest})
    rf_model = make_pipeline(
        rf_leaves_yielder,
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=1000),
    )
    rf_model.fit(X_train_linear, y_train_linear)

    # Creating and fitting Pipeline of Gradient Boosting Embedding w/ Logistic Regression
    gbdt_leaves_yielder = FunctionTransformer(
        utils.gbdt_apply, kw_args={"model": gradient_boosting}
    )
    gbdt_model = make_pipeline(
        gbdt_leaves_yielder,
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=1000),
    )
    gbdt_model.fit(X_train_linear, y_train_linear)

    # Plotting ROC Curve
    models = [
        ("RT embedding -> LR", rt_model),
        ("RF", random_forest),
        ("RF embedding -> LR", rf_model),
        ("GBDT", gradient_boosting),
        ("GBDT embedding -> LR", gbdt_model),
    ]

    fig = utils.plot_roc(
        X_test,
        y_test,
        models
    )

    return fig

title="Feature Transformations with Ensembles of Trees 🌳"
with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(
        """
        This example shows how one can apply features transformations using ensembles of trees \
        on a synthetic dataset. The transformations are then used to train a linear model on the \
        transformed data. The plot shows the ROC curve of the different models trained on the \
        transformed data. The plot is interactive and you can zoom in and out.
        
        See original example [here](https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py).
        """
    )
    
    with gr.Row():
        n_samples = gr.inputs.Slider(50_000, 100_000, 1000, label="Number of Samples", default=80_000)
        n_estimators = gr.inputs.Slider(10, 100, 10, label="Number of Estimators", default=10)
        max_depth = gr.inputs.Slider(1, 10, 1, label="Max Depth", default=3)

    plot = gr.Plot(label="ROC Curve")

    n_samples.change(fn=app_fn, inputs=[n_samples, n_estimators, max_depth], outputs=[plot])
    n_estimators.change(fn=app_fn, inputs=[n_samples, n_estimators, max_depth], outputs=[plot])
    max_depth.change(fn=app_fn, inputs=[n_samples, n_estimators, max_depth], outputs=[plot])
    
    demo.load(fn=app_fn, inputs=[n_samples, n_estimators, max_depth], outputs=[plot])

demo.launch()