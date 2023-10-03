from __future__ import annotations

import numpy as np
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

import utils


def app_fn(
    formula_str: str, 
    n_samples: int, 
    lower: float,
    upper: float,
    learning_rate: float,
    n_estimators: int,
    max_depth: int,
) -> list[go.Figure, pd.DataFrame]:  
    # Generating Data
    x_range = [0, 10]
    seed = 42
    gen = utils.DataGenerator(formula_str, x_range=x_range, n_samples=n_samples, seed=seed)
    X = gen.X
    y = gen.y
    y_raw = gen.y_raw

    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    # Model Parameters
    model_kwargs = {
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }

    # Ftting Interval Model
    model_interval = utils.GradientBoostingCoverage(lower, upper, **model_kwargs)
    model_interval.fit(X_train, y_train)

    # Ftting Median Model
    model_median = utils.fit_gradientboosting(X_train, y_train, alpha=0.5, loss="quantile",**model_kwargs)

    # Ftting Mean Model
    model_mean = utils.fit_gradientboosting(X_train, y_train, loss="squared_error", **model_kwargs)

    # Calculating Train and Test Coverage
    expected_coverage = model_interval.expected_coverage
    coverage_train = model_interval.coverage_fraction(X_train, y_train)
    coverage_test = model_interval.coverage_fraction(X_test, y_test)

    # Plotting Predictions
    xx = np.atleast_2d(np.linspace(*x_range, 1000)).T
    y_lower, y_upper = model_interval.predict(xx)
    y_median = model_median.predict(xx)
    y_mean = model_mean.predict(xx)

    fig = utils.plot_interval(
        xx, X_test, y_test, y_upper, y_lower, y_median, y_mean, formula_str, f"{expected_coverage*100:.0f}"
    )

    # DataFrame with Coverage
    df_coverage = pd.DataFrame(
        {
            "Split": ["Train", "Test"],
            "Coverage": [f"{coverage_train*100:.0f}", f"{coverage_test*100:.0f}"],
            "Expected Coverage": [f"{expected_coverage*100:.0f}", f"{expected_coverage*100:.0f}"],
        }
    )

    return fig, df_coverage

title = "🤗 Prediction Intervals w/ Gradient Boosting Regression 🤗"
with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(
        """
        ## This app shows how to use Gradient Boosting Regression to predict intervals. \
        The app uses the [Quantile Loss](https://en.wikipedia.org/wiki/Quantile_regression#Quantile_loss_function) \
        to predict the lower and upper quantiles with Gradient Boosting Regression. The data used in this example \
        is generated through the equation passed in the Formula textbox heteroscedasticity noise is introduced to \
        make the data more realistic. The app also shows the coverage of the intervals on the train and test data.

        ## Write equations using x as the variable and Python notation. Other supported functions are sin, cos, tan, exp, log, sqrt, and abs.

        [Orignal Example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-quantile-py)
        """
    )
    with gr.Row():
        with gr.Column():
            formula_str = gr.inputs.Textbox(
                lines=1, 
                label="Formula", 
                default="x * sin(x)"
            )

            n_samples = gr.inputs.Slider(
                minimum=100, 
                maximum=10000, 
                step=100, 
                default=1000, 
                label="Number of Samples"
            )

        with gr.Column():
            lower = gr.inputs.Slider(
                minimum=0.01, 
                maximum=0.45, 
                step=0.01, 
                default=0.05, 
                label="Lower Quantile"
            )

            upper = gr.inputs.Slider(
                minimum=0.5, 
                maximum=0.99, 
                step=0.01, 
                default=0.95, 
                label="Upper Quantile"
            )  

        with gr.Column():
            learning_rate = gr.inputs.Slider(
                minimum=0.01, 
                maximum=1.0, 
                step=0.01, 
                default=0.05, 
                label="Learning Rate"
            )

            n_estimators = gr.inputs.Slider(
                minimum=1, 
                maximum=1000, 
                step=1, 
                default=200, 
                label="Number of Estimators"
            )

            max_depth = gr.inputs.Slider(
                minimum=1, 
                maximum=10, 
                step=1, 
                default=2, 
                label="Max Depth"
            )

    btn = gr.Button(label="Run")
    with gr.Row():
        with gr.Column():
            fig = gr.Plot(label="Coverage Plot")
            df_coverage = gr.Dataframe(label="Coverage DataFrame")
    
    btn.click(
        fn=app_fn, 
        inputs=[formula_str, n_samples, lower, upper, learning_rate, n_estimators, max_depth],
        outputs=[fig, df_coverage],
    )
    demo.load(
        fn=app_fn, 
        inputs=[formula_str, n_samples, lower, upper, learning_rate, n_estimators, max_depth],
        outputs=[fig, df_coverage],
    )

demo.launch()