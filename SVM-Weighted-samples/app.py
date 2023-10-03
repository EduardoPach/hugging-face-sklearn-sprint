from __future__ import annotations

import numpy as np
import gradio as gr
from sklearn.svm import SVC
import plotly.graph_objects as go

def plot_decision(
        clf: SVC,
        X: np.ndarray,
        y: np.array,
        x_range: np.array,
        y_range: np.array,
        weights: np.array,
        title: str
    ):
    # plot the decision function
    xx, yy = np.meshgrid(x_range, y_range)

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)


    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=Z,
            colorscale="Viridis",
            opacity=0.75,        
            showscale=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(
                color=y,
                colorscale="viridis",
                size=(weights + 5) * 2
            ),
        )
    )

    # Remove x and y ticks
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    # Add title
    fig.update_layout(title=title)

    return fig

def app_fn(seed: int, weight_1: int, weight_2: int):
    # we create 20 points
    np.random.seed(seed)
    X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
    y = [1] * 10 + [-1] * 10

    sample_weight_last_ten = abs(np.random.randn(len(X)))
    sample_weight_constant = np.ones(len(X))

    sample_weight_last_ten[15:] *= weight_1
    sample_weight_last_ten[9] *= weight_2

    # This model does not take into account sample weights.
    clf_no_weights = SVC(gamma=1)
    clf_no_weights.fit(X, y)

    # This other model takes into account some dedicated sample weights.
    clf_weights = SVC(gamma=1)
    clf_weights.fit(X, y, sample_weight=sample_weight_last_ten)

    # Plotting
    x_range = np.arange(-4, 5, 0.1)

    fig_no_weights = plot_decision(
        clf_no_weights, 
        X,
        y,
        x_range, 
        x_range, 
        sample_weight_constant,  
        "SVM without Weights"
    )

    fig_weights = plot_decision(
        clf_weights,
        X,
        y,
        x_range,
        x_range,
        sample_weight_last_ten,
        "SVM with Weights"
    )

    return fig_no_weights, fig_weights

title = "SVM with Weighted Samples"

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(
        """
        ### This is a demo of how SVMs can be trained with weighted samples \
        and the impact on the decision boundary. To represent that a synthetic \
        dataset is generated with 20 points, 10 of which are assigned to the \
        positive class and 10 to the negative class. A weight is assigned to \
        each sample, which is the importance of that sample in the dataset. \
        A model with and without weights is trained and the decision boundary \
        is plotted. The size of the points is proportional to the weight of \
        the sample.

        Created by [@eduardopacheco](https://huggingface.co/EduardoPacheco) based on [scikit-learn-docs](https://scikit-learn.org/stable/auto_examples/svm/plot_weighted_samples.html#sphx-glr-auto-examples-svm-plot-weighted-samples-py)
        """
    )
    with gr.Row():
        seed = gr.inputs.Slider(0, 100, 1, default=0, label="Seed")
        weight_1 = gr.inputs.Slider(0, 20, 1, default=5, label="Weight for last 5 Samples")
        weight_2 = gr.inputs.Slider(0, 20, 1, default=15, label="Weight for Sample 10")
    # btn = gr.Button("Run")
    with gr.Row():
        fig_no_weights = gr.Plot(label="SVM without Weights")
        fig_weights = gr.Plot(label="SVM with Weights")
        
    seed.change(fn=app_fn, outputs=[fig_no_weights, fig_weights], inputs=[seed, weight_1, weight_2])
    weight_1.change(fn=app_fn, outputs=[fig_no_weights, fig_weights], inputs=[seed, weight_1, weight_2])
    weight_2.change(fn=app_fn, outputs=[fig_no_weights, fig_weights], inputs=[seed, weight_1, weight_2])
    demo.load(fn=app_fn, outputs=[fig_no_weights, fig_weights], inputs=[seed, weight_1, weight_2])

demo.launch()
            