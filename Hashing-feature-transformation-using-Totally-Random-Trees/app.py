import numpy as np
import gradio as gr
import plotly.graph_objects as go
from sklearn.datasets import make_circles
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomTreesEmbedding, ExtraTreesClassifier


def plot_scatter(X, y, title):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(color=y, size=10, colorscale="Viridis", line=dict(width=1)),
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )

    return fig

def plot_decision_boundary(X, y, model, data_preprocess=None, title=None):
    # Creating Grid
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Creating Contour
    if data_preprocess:
        grid = data_preprocess.transform(grid)
    y_grid_pred = model.predict_proba(grid)[:, 1]

    # Plotting
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=y_grid_pred.reshape(xx.shape),
            colorscale="Viridis",
            opacity=0.8,
            showscale=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(color=y, size=10, colorscale="Viridis", line=dict(width=1)),
        )
    )

    fig.update_layout(
        title=title if title else "Decision Boundary",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )

    return fig



def app_fn(
        factor: float, 
        random_state: int, 
        noise:float, 
        n_estimators: int, 
        max_depth: int
    ):
    # make a synthetic dataset
    X, y = make_circles(factor=factor, random_state=random_state, noise=noise)

    # use RandomTreesEmbedding to transform data
    hasher = RandomTreesEmbedding(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
    X_transformed = hasher.fit_transform(X)

    # Visualize result after dimensionality reduction using truncated SVD
    svd = TruncatedSVD(n_components=2)
    X_reduced = svd.fit_transform(X_transformed)

    # Learn a Naive Bayes classifier on the transformed data
    nb = BernoulliNB()
    nb.fit(X_transformed, y)

    # Learn an ExtraTreesClassifier for comparison
    trees = ExtraTreesClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state)
    trees.fit(X, y)

    # Plotting Original Data
    fig1 = plot_scatter(X, y, "Original Data")
    fig2 = plot_scatter(X_reduced, y, f"Truncated SVD Reduction (2D) of Transformed Data ({X_transformed.shape[1]})")
    fig3 = plot_decision_boundary(X, y, nb, hasher, "Naive Bayes Decision Boundary")
    fig4 = plot_decision_boundary(X, y, trees, title="Extra Trees Decision Boundary")

    return fig1, fig2, fig3, fig4

title = "Hashing Feature Transformation using Totally Random Trees"
with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(
        """
        ### RandomTreesEmbedding provides a way to map data to a very high-dimensional, \
        sparse representation, which might be beneficial for classification. \
        The mapping is completely unsupervised and very efficient.

        ### This example visualizes the partitions given by several trees and shows how \
        the transformation can also be used for non-linear dimensionality reduction \
        or non-linear classification.

        ### Points that are neighboring often share the same leaf of a \
        tree and therefore share large parts of their hashed representation. \
        This allows to separate two concentric circles simply based on \
        the principal components of the transformed data with truncated SVD.

        ### In high-dimensional spaces, linear classifiers often achieve excellent \
        accuracy. For sparse binary data, BernoulliNB is particularly well-suited. \
        The bottom row compares the decision boundary obtained by BernoulliNB in the \
        transformed space with an ExtraTreesClassifier forests learned on the original data.

        [Original Example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_random_forest_embedding.html#sphx-glr-auto-examples-ensemble-plot-random-forest-embedding-py)
        """
    )
    with gr.Row():
        factor = gr.inputs.Slider(minimum=0.05, maximum=1.0, step=0.01, default=0.5, label="Factor")
        noise = gr.inputs.Slider(minimum=0.0, maximum=1.0, step=0.01, default=0.05, label="Noise")
        n_estimators = gr.inputs.Slider(minimum=1, maximum=100, step=1, default=10, label="Number of Estimators")
        max_depth = gr.inputs.Slider(minimum=1, maximum=100, step=1, default=3, label="Max Depth")
        random_state = gr.inputs.Slider(minimum=0, maximum=100, step=1, default=0, label="Random State")
    with gr.Row():
        plot1 = gr.Plot(label="Origianl Data")
        plot2 = gr.Plot(label="Truncated Date")
    with gr.Row():
        plot3 = gr.Plot(label="Naive Bayes Decision Boundary")
        plot4 = gr.Plot(label="Extra Trees Decision Boundary")
    
    factor.change(app_fn, outputs=[plot1, plot2, plot3, plot4], inputs=[factor, random_state, noise, n_estimators, max_depth])
    noise.change(app_fn, outputs=[plot1, plot2, plot3, plot4], inputs=[factor, random_state, noise, n_estimators, max_depth])
    n_estimators.change(app_fn, outputs=[plot1, plot2, plot3, plot4], inputs=[factor, random_state, noise, n_estimators, max_depth])
    max_depth.change(app_fn, outputs=[plot1, plot2, plot3, plot4], inputs=[factor, random_state, noise, n_estimators, max_depth])
    random_state.change(app_fn, outputs=[plot1, plot2, plot3, plot4], inputs=[factor, random_state, noise, n_estimators, max_depth])
    demo.load(app_fn, inputs=[factor, random_state, noise, n_estimators, max_depth], outputs=[plot1, plot2, plot3, plot4])

demo.launch()