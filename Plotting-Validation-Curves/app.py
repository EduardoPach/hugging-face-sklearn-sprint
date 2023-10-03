from __future__ import annotations

import numpy as np
import gradio as gr
from sklearn.svm import SVC
import plotly.graph_objects as go
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve

def plot_validation_curve(x: np.array, ys: list[np.array], yerros: list[np.array], names: list[str], colors: list[str], log_x: bool=True, title: str=""):
    fig = go.Figure()

    for y, yerror, name, color in zip(ys, yerros, names, colors):
        y_upper = y + yerror
        y_lower = y - yerror
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.round(y, 3),
                name=name,
                line_color=color
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x.tolist()+x[::-1].tolist(), # x, then x reversed
                y=y_upper.tolist()+y_lower[::-1].tolist(), # upper, then lower reversed
                fill='toself',
                fillcolor=color,
                line=dict(color=color),
                hoverinfo="skip",
                showlegend=False,
                opacity=0.2
            )
        )

    if log_x:
        fig.update_xaxes(type="log")

    fig.update_layout(
        title=title, 
        xaxis_title="Hyperparameter", 
        yaxis_title="Accuracy",
        hovermode="x unified",
    )

    return fig



def app_fn(n_points: int, param_name: str):
    X, y = load_digits(return_X_y=True)
    subset_mask = np.isin(y, [1, 2])  # binary classification: 1 vs 2
    X, y = X[subset_mask], y[subset_mask]

    if param_name=="gamma":
        param_range = np.logspace(-6, -1, n_points)
        log_x = True
    elif param_name=="C":
        param_range = np.logspace(-2, 0, n_points)
        log_x = True
    elif param_name=="kernel":
        param_range = np.array(["rbf", "linear", "poly", "sigmoid"])
        log_x = False

    train_scores, test_scores = validation_curve(
        SVC(),
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        scoring="accuracy",
        n_jobs=-1,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig = plot_validation_curve(
        param_range, 
        [train_scores_mean, test_scores_mean], 
        [train_scores_std, test_scores_std], 
        ["Training score", "Cross-validation score"], 
        ["orange", "navy"], 
        title=f"Validation Curve with SVM for {param_name} Hyperparameter",
        log_x=log_x
    )

    return fig

title = "Plotting Validation Curve"
with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(
        """
        #### This example shows the usage of a validation curve to understand \
        how the performance of a model, SVM in this case, changes with varying hyperparameters. \
        The dataset used was the [digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits) \
        from scikit-learn. The hyperparameter varied was gamma. \
        
        [Original Example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py)
        """
    )
    with gr.Row():
        n_points = gr.inputs.Slider(5, 100, 5, 5,label="Number of points")
        param_name = gr.inputs.Dropdown(["gamma", "C", "kernel"], label="Hyperparameter", default="gamma")


    fig = gr.Plot(label="Validation Curve")

    n_points.release(fn=app_fn, inputs=[n_points, param_name], outputs=[fig])
    param_name.change(fn=app_fn, inputs=[n_points, param_name], outputs=[fig])
    # C.change(fn=app_fn, inputs=[n_points, param_name, C, gamma, kernel, degree], outputs=[fig])
    # gamma.change(fn=app_fn, inputs=[n_points, param_name, C, gamma, kernel, degree], outputs=[fig])
    # kernel.change(fn=app_fn, inputs=[n_points, param_name, C, gamma, kernel, degree], outputs=[fig])
    # degree.change(fn=app_fn, inputs=[n_points, param_name, C, gamma, kernel, degree], outputs=[fig])


    demo.load(fn=app_fn, inputs=[n_points, param_name], outputs=[fig])

demo.launch()