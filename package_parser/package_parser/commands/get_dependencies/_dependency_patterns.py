dependency_matcher_patterns = {
    "pattern_parameter_sconj": [
        {
            "RIGHT_ID": "applies",
            "RIGHT_ATTRS": {"POS": "VERB"}
        },
        {
            "LEFT_ID": "applies",
            "REL_OP": ">",
            "RIGHT_ID": "condition",
            "RIGHT_ATTRS": {"DEP": "advcl"},
        },
        {
            "LEFT_ID": "condition",
            "REL_OP": ">",
            "RIGHT_ID": "dependee_param",
            "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}},
        },
    ],
}
