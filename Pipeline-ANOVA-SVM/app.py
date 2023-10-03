import gradio as gr
import pandas as pd
import plotly.express as px
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif


def app_fn(k: int, n_features: int, n_informative: int, n_redundant: int):
    X, y = make_classification(
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        n_clusters_per_class=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    anova_filter = SelectKBest(f_classif, k=k)
    clf = LinearSVC()
    anova_svm = make_pipeline(anova_filter, clf)
    anova_svm.fit(X_train, y_train)

    y_pred = anova_svm.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.reset_index().rename(columns={"index": "class"}).round(2)
    report_df["accuracy"] = report_df.loc[report_df["class"]=="accuracy"].values.flatten()[-1]
    report_df = report_df.loc[report_df["class"]!="accuracy"]

    features = anova_svm[:-1].inverse_transform(anova_svm[-1].coef_).flatten()  > 0
    features = features.astype(int)
    fig = px.bar(y=features)
    # Changing y-axis ticks to show 0 and 1 instead of False and True
    fig.update_yaxes(ticktext=["False", "True"], tickvals=[0, 1])
    fig.update_layout(
        title="Selected Features",
        xaxis_title="Feature Index",
        yaxis_title="Selected",
        legend_title="Selected",
    )
    return report_df, fig

title = "🔥 Pipeline ANOVA SVM 🔥"
with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(
        """
        ## This example shows how a feature selection can be easily integrated within a machine learning pipeline.

        [Original Example](https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection_pipeline.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-pipeline-py)
        """
    )
    with gr.Row():
        k = gr.inputs.Slider(minimum=1, maximum=20, default=3, step=1, label="Number of Features to Select")
        n_features = gr.inputs.Slider(minimum=1, maximum=20, default=20, step=1, label="Total Features")
        n_informative = gr.inputs.Slider(minimum=1, maximum=20, default=3, step=1, label="Informative Features")
        n_redundant = gr.inputs.Slider(minimum=0, maximum=20, default=0, step=1, label="Redundant Features")
    btn = gr.Button(label="Run")
    with gr.Row():
        report = gr.DataFrame(label="Classification Report")
        features = gr.Plot(label="Selected Features")

    btn.click(
        fn=app_fn, 
        inputs=[k, n_features, n_informative, n_redundant],
        outputs=[report, features],
    )
    demo.load(
        fn=app_fn, 
        inputs=[k, n_features, n_informative, n_redundant],
        outputs=[report, features],
    )

demo.launch()
