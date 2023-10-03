from __future__ import annotations

import gradio as gr

from . import utils

def app_fn(
        noise, 
        random_state, 
        n_samples, 
        folds, 
        repetitions, 
        rope_val
    ):
    X, y = utils.make_data(noise=noise, random_state=random_state, n_samples=n_samples)
    search, n, n_train, n_test = utils.fit_search(X, y, folds, repetitions)
    results_df = utils.get_results_df(search)
    model_scores = results_df.filter(regex=r"split\d*_test_score")
    rope_interval = (-rope_val, rope_val)
    intervals = [0.5, 0.75, 0.95]
    
    fig_dataset = utils.plot_dataset(X, y)
    fig_cv_results = utils.plot_cv_test(model_scores)
    fig_corr = utils.plot_correlation_heatmap(model_scores)
    df_two_frequentist = utils.frequentist_two_model(model_scores, n, n_train, n_test)
    fig_bayesian_posterior = utils.plot_bayesian_posterior(model_scores, n, n_train, n_test)
    fig_rope = utils.plot_rope(model_scores, n, n_train, n_test, rope_interval)
    df_cred_interval = utils.get_cred_intervals(intervals, model_scores, n, n_train, n_test)
    df_pairwise_frequentist = utils.get_pairwise_frequentist(model_scores, n, n_train, n_test)
    df_pairwise_bayesian = utils.get_pairwise_bayesian(rope_interval, model_scores, n, n_train, n_test)

    return (
        fig_dataset,
        fig_cv_results,
        fig_corr,
        df_two_frequentist,
        fig_bayesian_posterior,
        fig_rope,
        df_cred_interval,
        df_pairwise_frequentist,
        df_pairwise_bayesian,
    )


title = "Statistical Comparison Grid Search"
with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(
        """
        #### This example illustrates how to statistically compare the performance of models trained and evaluated using GridSearchCV \
        using a synthetic dataset. We will compare the performance of SVC estimators that vary on their kernel parameter, to decide \
        which choice of this hyper-parameter predicts our simulated data best. We will evaluate the performance of the models using RepeatedStratifiedKFold, \
        The performance will be evaluated using roc_auc_score.

        Created by [eduardopacheco](https://huggingface.co/EduardoPacheco) based on [scikit-learn-docs](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html#sphx-glr-auto-examples-model-selection-plot-grid-search-stats-py)
        """
    )

    with gr.Accordion("Dataset", open=True):
        gr.Markdown(
            """
            This is the dataset we will use to train and evaluate our models.
            """
        )
        with gr.Row():
            noise = gr.inputs.Slider(minimum=0, maximum=1, step=0.001, default=0.352, label="Noise")
            random_state = gr.inputs.Slider(minimum=0, maximum=100, step=1, default=1, label="Random State")
            n_samples = gr.inputs.Slider(minimum=100, maximum=1000, step=10, default=100, label="Number of Samples")
        fig_dataset = gr.Plot(label="Dataset")
    
    with gr.Accordion("Models Results", open=False):
        gr.Markdown(
            """
            Here we evaluate the performance of the models using cross-validation by plotting the test scores for each fold. \
            We also plot the correlation between the test scores of each model showing that the models are not independent.
            """
        )
        with gr.Row():
            folds = gr.inputs.Slider(minimum=2, maximum=10, step=1, default=10, label="Number of Folds")
            repetitions = gr.inputs.Slider(minimum=1, maximum=10, step=1, default=10, label="Number of Repetitions")

        with gr.Row():
            fig_cv_results = gr.Plot(label="CV Results")
            fig_corr = gr.Plot(label="Correlation")

    with gr.Accordion("Comparing 2 models", open=False):
        gr.Markdown(
            """
            Since models aren't independent, we use the [Nadeau and Bengio's corrected t-test](https://proceedings.neurips.cc/paper_files/paper/1999/file/7d12b66d3df6af8d429c1a357d8b9e1a-Paper.pdf) \
            Usign two statistical frameworks: Frequentist and Bayesian. \
            In a nutshell:

            - Frequentist: tell us if the performance of one model is better than another with a degree of certainty above chance.
            - Bayesian: tell us the probabilities of one model being better, worse or practically equivalent than another also tell us how confident we are of knowing that the true differences of our models fall under a certain range of values.

            With Bayesian approach we can calculate the Credible Interval that show us the range of values that the true difference of our models fall under with a certain degree of confidence. \
            The ROPE (Region of Practical Equivalence) is a range of values that we consider that the true difference of our models is practically equivalent, \
            thus it's problem dependent since different problems may have different accuracy requirements and consequences. \
            """
        )
        rope_val = gr.inputs.Slider(minimum=0, maximum=0.1, step=0.01, default=0.01, label="ROPE Value")
        gr.Markdown("## Frequentist")
        df_two_frequentist = gr.DataFrame(label="Frequentist")
        gr.Markdown("## Bayesian")
        with gr.Row():
            fig_bayesian_posterior = gr.Plot(label="Bayesian Posterior")
            fig_rope = gr.Plot(label="ROPE")
        df_cred_interval = gr.DataFrame(label="Credible Interval")
    
    with gr.Accordion("Pairwise Comparisons", open=False):
        gr.Markdown(
            """
            We can also compare the performance of all models pairwise. \
            To do this we add yet another correction called [Bonferoni correction](https://en.wikipedia.org/wiki/Bonferroni_correction). \
            when calculating the p-values for Frequentist framework whereas the Bayesian framework doesn't require any correction. \
            """
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Frequentist")
                df_pairwise_frequentist = gr.DataFrame(label="Frequentist")
            with gr.Column():
                gr.Markdown("## Bayesian")
                df_pairwise_bayesian = gr.DataFrame(label="Bayesian")

    noise.release(
        fn=app_fn, 
        inputs=[        
            noise, 
            random_state, 
            n_samples, 
            folds, 
            repetitions, 
            rope_val
        ],
        outputs=[
            fig_dataset,
            fig_cv_results,
            fig_corr,
            df_two_frequentist,
            fig_bayesian_posterior,
            fig_rope,
            df_cred_interval,
            df_pairwise_frequentist,
            df_pairwise_bayesian,
        ]
    )

    random_state.release(
        fn=app_fn, 
        inputs=[        
            noise, 
            random_state, 
            n_samples, 
            folds, 
            repetitions, 
            rope_val
        ],
        outputs=[
            fig_dataset,
            fig_cv_results,
            fig_corr,
            df_two_frequentist,
            fig_bayesian_posterior,
            fig_rope,
            df_cred_interval,
            df_pairwise_frequentist,
            df_pairwise_bayesian,
        ]
    )

    n_samples.release(
        fn=app_fn, 
        inputs=[        
            noise, 
            random_state, 
            n_samples, 
            folds, 
            repetitions, 
            rope_val
        ],
        outputs=[
            fig_dataset,
            fig_cv_results,
            fig_corr,
            df_two_frequentist,
            fig_bayesian_posterior,
            fig_rope,
            df_cred_interval,
            df_pairwise_frequentist,
            df_pairwise_bayesian,
        ]
    )

    folds.release(
        fn=app_fn, 
        inputs=[        
            noise, 
            random_state, 
            n_samples, 
            folds, 
            repetitions, 
            rope_val
        ],
        outputs=[
            fig_dataset,
            fig_cv_results,
            fig_corr,
            df_two_frequentist,
            fig_bayesian_posterior,
            fig_rope,
            df_cred_interval,
            df_pairwise_frequentist,
            df_pairwise_bayesian,
        ]
    )

    repetitions.release(
        fn=app_fn, 
        inputs=[        
            noise, 
            random_state, 
            n_samples, 
            folds, 
            repetitions, 
            rope_val
        ],
        outputs=[
            fig_dataset,
            fig_cv_results,
            fig_corr,
            df_two_frequentist,
            fig_bayesian_posterior,
            fig_rope,
            df_cred_interval,
            df_pairwise_frequentist,
            df_pairwise_bayesian,
        ]
    )

    rope_val.release(
        fn=app_fn, 
        inputs=[        
            noise, 
            random_state, 
            n_samples, 
            folds, 
            repetitions, 
            rope_val
        ],
        outputs=[
            fig_dataset,
            fig_cv_results,
            fig_corr,
            df_two_frequentist,
            fig_bayesian_posterior,
            fig_rope,
            df_cred_interval,
            df_pairwise_frequentist,
            df_pairwise_bayesian,
        ]
    )

    demo.load(
        fn=app_fn, 
        inputs=[        
            noise, 
            random_state, 
            n_samples, 
            folds, 
            repetitions, 
            rope_val
        ],
        outputs=[
            fig_dataset,
            fig_cv_results,
            fig_corr,
            df_two_frequentist,
            fig_bayesian_posterior,
            fig_rope,
            df_cred_interval,
            df_pairwise_frequentist,
            df_pairwise_bayesian,
        ]
    )


demo.launch()
