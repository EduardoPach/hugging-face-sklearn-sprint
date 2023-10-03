import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, average_precision_score

def plot_multi_label_pr_curve(clf, X_test: np.ndarray, Y_test: np.ndarray):
    n_classes = Y_test.shape[1]
    y_score = clf.decision_function(X_test)

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        Y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
    
    # Plotting
    fig = go.Figure()

    
    # Plottin Precision-Recall Curves for each class
    colors = ["navy", "turquoise", "darkorange", "gold"]
    keys = list(precision.keys())

    for color, key in zip(colors, keys):
        if key=="micro":
            name = f"Micro-average(AP={average_precision[key]:.2f})"
        else:
            name = f"Class {key} (AP={average_precision[key]:.2f})"
        fig.add_trace(
            go.Scatter(
                x=recall[key],
                y=precision[key],
                mode="lines",
                name=name,
                line=dict(color=color),
                showlegend=True,
                line_shape="hv"
            )
        )

    # Creating Iso-F1 Curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for idx, f_score in enumerate(f_scores):
        if idx==0:
            name = "Iso-F1 Curves"
            showlegend = True
        else:
            name = ""
            showlegend = False
        x = np.linspace(0.01, 1, 1001)
        y = f_score * x / (2 * x - f_score)
        mask = y >= 0
        fig.add_trace(go.Scatter(x=x[mask], y=y[mask], mode='lines', line_color='gray', name=name, showlegend=showlegend))
        fig.add_annotation(x=0.9, y=y[900] + 0.02, text=f"<b>f1={f_score:0.1f}</b>", showarrow=False, font=dict(size=15))


    fig.update_yaxes(range=[0, 1.05])

    fig.update_layout(
        title='Extension of Precision-Recall Curve to Multi-Class', 
        xaxis_title='Recall', 
        yaxis_title='Precision'
    )

    return fig


def plot_binary_pr_curve(clf, X_test: np.ndarray, y_test:np.array):
    # make predictions on the test data
    y_pred = clf.decision_function(X_test)

    # calculate precision and recall for different probability thresholds
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    # calculate the average precision
    ap = average_precision_score(y_test, y_pred)

    # Plotting
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"LinearSVC (AP={ap:.2f})",
            line=dict(color="blue"),
            showlegend=True,
            line_shape="hv"
        )
    )

    # Make x-range slightly larger than max value
    fig.update_xaxes(range=[-0.05, 1.05])
    # Make Legend text size larger
    fig.update_layout(
        title='2-Class Precision-Recall Curve',
        xaxis_title='Recall (Positive label: 1)',
        yaxis_title='Precision (Positive label: 1)',
        legend=dict(
            x=0.009,
            y=0.05,
            font=dict(
                size=12,
            ),
        )
    )

    return fig
    