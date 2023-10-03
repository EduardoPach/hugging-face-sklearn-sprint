from __future__ import annotations

import re
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor

class DataGenerator:
    def __init__(self, formula_str: str, x_range: list, n_samples: int, seed: int) -> None:
        self.formula_str = formula_str
        self.x_range = x_range
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @property
    def X(self) -> np.array:
        self.rng = np.random.RandomState(42)
        X = np.atleast_2d(self.rng.uniform(*self.x_range, size=self.n_samples)).T
        return X
    
    @property
    def y_raw(self) -> np.array:
        y_raw = self._eval_formula()
        return y_raw.ravel()
    
    @property
    def y(self) -> np.array:
        sigma = 0.5 + self.X.ravel() / 10
        noise = self.rng.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
        return self.y_raw + noise

    def _eval_formula(self) -> np.array:
        function_map = {
            'sin': "np.sin",
            'cos': "np.cos",
            'tan': "np.tan",
            'exp': "np.exp",
            'log': "np.log",
            'sqrt': "np.sqrt",
            'abs': "np.abs",
        }
        # Replace "x" in the formula string with "x_values"
        _formula_str = re.sub(r'\bx\b', '(self.X)', self.formula_str)
        # Replace any function calls in the formula string with the appropriate function object
        _formula_str = re.sub(r'(\w+)\(([^)]*)\)', lambda m: f'{function_map[m.group(1)]}({m.group(2)})', _formula_str)
        # Evaluate the formula using the updated string and return the result
        return eval(_formula_str)

class GradientBoostingCoverage:
    def __init__(self, lower: float, upper: float, **kwargs) -> None:
        self.lower = lower
        self.upper = upper
        self.kwargs = kwargs
        self.models = self._build_models()

    @property
    def expected_coverage(self) -> float:
        return self.upper - self.lower

    def _build_models(self) -> dict[str, GradientBoostingRegressor]:
        models = {}
        for name, alpha in [("lower", self.lower), ("upper", self.upper)]:
            models[f"{name}"] = GradientBoostingRegressor(loss="quantile", alpha=alpha, **self.kwargs)
        return models
    
    def fit(self, X: np.ndarray, y: np.array) -> None:
        for model in self.models.values():
            model.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.array, np.array]:
        lower = self.models["lower"].predict(X)
        upper = self.models["upper"].predict(X)
        return lower, upper
    
    def coverage_fraction(self, X: np.ndarray, y: np.array) -> float:
        y_low, y_high = self.predict(X)
        return np.mean(np.logical_and(y >= y_low, y <= y_high))


def fit_gradientboosting(X, y, **kwargs) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(**kwargs)
    model.fit(X, y)
    return model

def plot_interval(
    xx: np.array, 
    X_test: np.array, 
    y_test: np.array,
    y_upper: np.array, 
    y_lower: np.array, 
    y_med: np.array,
    y_mean: np.array,
    formula_str: Optional[str]=None,
    interval: Optional[Union[int, str]]=None,
) -> go.Figure:
    # Using plotly to plot an interval
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xx.ravel(), 
            y=y_upper, 
            fill=None, 
            mode="lines", 
            line_color="rgba(255,255,0,0)", 
            name=""
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xx.ravel(), 
            y=y_lower, 
            fill="tonexty", 
            mode="lines", 
            line_color="rgba(255,255,0,0)", 
            name=f"Predicted Interval"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xx.ravel(), 
            y=y_med, 
            mode="lines", 
            line_color="red", 
            name='Predicted Median',
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xx.ravel(), 
            y=y_mean, 
            mode="lines", 
            name='Predicted Mean',
            line=dict(color='red', dash='dash')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=X_test.ravel(),
            y=y_test,
            mode="markers",
            marker_color="blue",
            name="Test Observations",
            marker=dict(size=5, line=dict(width=2, color="DarkSlateGrey"))
        )
    )

    fig.update_layout(
        title=f"Predicted {interval}% Interval",
        xaxis_title="x",
        yaxis_title="f(x)" if not formula_str else formula_str,
        height=600
    )

    return fig