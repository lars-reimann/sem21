instances = [
    ("coef0",  'It is only significant in "poly" and "sigmoid"'),
    ("random_state", 'ignored when probability is False'),
    ("l1_ratio", 'only used if penalty is "elasticnet"'),
    ("validation_fraction", 'only used if early_stopping is True'),
    ("normalize", 'this parameter is ignored when fit_intercept is set to False'),
    ("solver", 'It can be used only when positive is True'),
    ("positive", 'Only "lbfgs" solver is supported in this case'),
    ("random_state", 'used when solver == "sag" or "saga"'), 
    ("intercept_scaling", 'useful only when the solver "liblinear" is used and self.fit_intercept is set to True'),
    ("random_state", 'used when solver == "sag", "saga" or "liblinear"'), 
    ("n_jobs", 'This parameter is ignored when the solver is set to "liblinear"'),
    ("l1_ratio", 'only used if penalty="elasticnet"'),
    ("random_state", 'used when selection == "random"'),
    ("oob_score", 'only available if bootstrap=True'),
    ("random_state", 'it is only used when base_estimator exposes a random_state'),
    ("alpha", 'only if loss="huber" or loss="quantile"'),
    ("validation_fraction", 'Only used if n_iter_no_change is set to an integer'),
    ("random_state", 'Used when the "arpack" or "randomized" solvers are used'), 
    ("preprocessor", 'Only applies if analyzer is not callable'),
    ("tokenizer", 'Only applies if analyzer == "word"'),
    ("stop_words", 'Only applies if analyzer == "word"'),
    ("token_pattern", 'Only used if analyzer == "word"'),
    ("ngram_range", 'Only applies if analyzer is not callable'),
    ("digits", 'if output_dict is True, this will be ignored') 
]


patterns = {
    "IgnoredWhen": [{"LOWER": "ignored"}, {"LOWER": "when"}, {"POS": "NOUN"}]
}
