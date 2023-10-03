import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.inspection import permutation_importance

def plot_rf_importance(clf):
    feature_names = clf[:-1].get_feature_names_out()
    mdi_importances = pd.Series(
        clf[-1].feature_importances_, index=feature_names
    ).sort_values(ascending=True)


    fig = px.bar(mdi_importances, orientation="h", title="Random Forest Feature Importances (MDI)")
    fig.update_layout(showlegend=False, xaxis_title="Importance", yaxis_title="Feature")
    
    return fig

def plot_permutation_boxplot(clf, X: np.ndarray, y: np.array, set_: str=None):

    result = permutation_importance(
        clf, X, y, n_repeats=10, random_state=42, n_jobs=2
    )

    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=X.columns[sorted_importances_idx],
    )

    fig = px.box(
        importances.melt(),
        y="variable",
        x="value"
    )

    # Add dashed vertical line
    fig.add_shape(
        type="line",
        x0=0,
        y0=-1,
        x1=0,
        y1=len(importances.columns),
        opacity=0.5,
        line=dict(
            dash="dash"
        ),
    )
    # Adapt x-range
    x_min = importances.min().min() 
    x_min = x_min - 0.005 if x_min < 0 else -0.005
    x_max = importances.max().max() + 0.005
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_layout(
        title=f"Permutation Importances {set_ if set_ else ''}",
        xaxis_title="Importance",
        yaxis_title="Feature",
        showlegend=False
    )

    return fig