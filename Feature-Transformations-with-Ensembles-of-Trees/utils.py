from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomTreesEmbedding


def create_and_split_dataset(n_samples: int) -> list[tuple[np.ndarray, np.array]]:
    # Create Data
    X, y = make_classification(n_samples=n_samples, random_state=10)

    # Split Data
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=10
    )

    # Split Data for Ensemble and Linear Model
    X_train_ensemble, X_train_linear, y_train_ensemble, y_train_linear = train_test_split(
        X_full_train, y_full_train, test_size=0.5, random_state=10
    )

    return (X_train_ensemble, y_train_ensemble), (X_train_linear, y_train_linear), (X_test, y_test)

def rf_apply(X, model):
    return model.apply(X)

def gbdt_apply(X, model):
    return model.apply(X)[:, :, 0]

def plot_roc(X: np.ndarray, y:np.array, models: tuple[str, ClassifierMixin]):
    fig = go.Figure()

    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    for model_name, model in models: 
        y_score = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_score)
        auc_val = auc(fpr, tpr) 
        name = f"{model_name} (AUC={auc_val:.4f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
        
    fig.update_layout(
        title="Model ROC Curve Comparison", 
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=600
    )

    return fig
