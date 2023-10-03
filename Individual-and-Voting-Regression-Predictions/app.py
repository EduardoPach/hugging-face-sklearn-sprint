from __future__ import annotations

import numpy as np
import gradio as gr
import plotly.graph_objects as go
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor


def plot_votes(preds: list[tuple[str, np.array]], markers: list[str]=None) -> go.Figure:
    fig = go.Figure()

    for idx, (name, pred) in enumerate(preds):
        if not markers:
            symbol = "diamond"
        else: 
            symbol = markers[idx]
        fig.add_trace(
            go.Scatter(
                y=pred,
                mode="markers",
                name=name,
                marker=dict(symbol=symbol, size=10, line=dict(width=2, color="DarkSlateGrey"))
            )
        )
    fig.update_layout(
        title="Regressor predictions and their average",
        yaxis_title="Predicted",
        xaxis_title="Training Samples",
        height=500,
        xaxis=dict(showticklabels=False),
        hovermode="x unified"
    )

    return fig


def app_fn(n: int) -> go.Figure:
    X, y = load_diabetes(return_X_y=True)

    # Train classifiers
    reg1 = GradientBoostingRegressor(random_state=1)
    reg2 = RandomForestRegressor(random_state=1)
    reg3 = LinearRegression()

    reg1.fit(X, y)
    reg2.fit(X, y)
    reg3.fit(X, y)

    ereg = VotingRegressor([("gb", reg1), ("rf", reg2), ("lr", reg3)])
    ereg.fit(X, y)

    xt = X[:n]

    pred1 = reg1.predict(xt)
    pred2 = reg2.predict(xt)
    pred3 = reg3.predict(xt)
    pred4 = ereg.predict(xt)

    preds = [
        ("Gradient Boosting", pred1),
        ("Random Forest", pred2),
        ("Linear Regression", pred3),
        ("Voting Regressor", pred4)
    ]
    markers = ["diamond-tall", "triangle-up", "square", "star"]
    fig = plot_votes(preds, markers)

    return fig

title="Individual and Voting Regression Predictions 🗳️"
with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(
        """
        A voting regressor is an ensemble meta-estimator that fits several base regressors, each on the whole dataset. \
        Then it averages the individual predictions to form a final prediction. This example will use three different regressors to \
        predict the data: GradientBoostingRegressor, RandomForestRegressor, and LinearRegression. Then the 3 regressors will be used for the VotingRegressor. \
        The dataset used consists of 10 features collected from a cohort of diabetes patients. The target is a quantitative measure of disease progression one year after baseline.

        [Original example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_regressor.html#sphx-glr-auto-examples-ensemble-plot-voting-regressor-py)
        """
    )
    n = gr.inputs.Slider(10, 30, 5, 20, "Number of training samples")
    plot = gr.Plot(label="Individual & Voting Predictions")
    
    n.change(fn=app_fn, inputs=[n], outputs=[plot])
    demo.load(fn=app_fn, inputs=[n], outputs=[plot])

demo.launch()