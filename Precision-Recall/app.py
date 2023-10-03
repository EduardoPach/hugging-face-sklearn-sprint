import numpy as np
import gradio as gr
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler

import utils


def app_fn(n_random_features: int, test_size: float, random_state_val: int):
    X, y = load_iris(return_X_y=True)

    # Add noisy features
    random_state = np.random.RandomState(random_state_val)
    n_samples, n_features = X.shape
    X = np.concatenate([X, random_state.randn(n_samples, n_random_features)], axis=1)

    # Solving Binary Problem
    X_train, X_test, y_train, y_test = train_test_split(
        X[y < 2], y[y < 2], test_size=test_size, random_state=random_state
    )

    clf_bin = make_pipeline(StandardScaler(), LinearSVC(random_state=random_state))
    clf_bin.fit(X_train, y_train)

    fig_bin = utils.plot_binary_pr_curve(clf_bin, X_test, y_test)

    # Solving Multi-Label Problem
    Y = label_binarize(y, classes=[0, 1, 2])
    X_train_multi, X_test_multi, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    clf = OneVsRestClassifier(
        make_pipeline(StandardScaler(), LinearSVC(random_state=random_state))
    )
    clf.fit(X_train_multi, Y_train)

    fig_multi = utils.plot_multi_label_pr_curve(clf, X_test_multi, Y_test)

    return fig_bin, fig_multi


title = "Precision-Recall Curves"
with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(
        """
        ### This demo shows the precision-recall curves on the Iris dataset \
        using a Linear SVM classifier + StandardScaler. \
        Noise is added to the dataset to make the problem more challenging. \
        The dataset is split into train and test sets. \
        The model is trained on the train set and evaluated on the test set. \
        Two separate problems are solved: 
        
        ### Binary classification: class 0 vs class 1
        ### Multi-label classification: class 0 vs class 1 vs class 2

        [Original Example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py)
        """
    )

    with gr.Row():
        n_random_features = gr.inputs.Slider(0, 1000, 50, 800,label="Number of Random Features")
        test_size = gr.inputs.Slider(0.1, 0.9, 0.01, 0.5, label="Test Size")
        random_state_val = gr.inputs.Slider(0, 100, 5, 0,label="Random State")


    with gr.Row():
        fig_bin = gr.Plot(label="Binary PR Curve")
        fig_multi = gr.Plot(label="Multi-Label PR Curve")

    n_random_features.change(fn=app_fn, inputs=[n_random_features, test_size, random_state_val], outputs=[fig_bin, fig_multi])
    test_size.change(fn=app_fn, inputs=[n_random_features, test_size, random_state_val], outputs=[fig_bin, fig_multi])
    random_state_val.change(fn=app_fn, inputs=[n_random_features, test_size, random_state_val], outputs=[fig_bin, fig_multi])

    demo.load(fn=app_fn, inputs=[n_random_features, test_size, random_state_val], outputs=[fig_bin, fig_multi])

demo.launch()
    

