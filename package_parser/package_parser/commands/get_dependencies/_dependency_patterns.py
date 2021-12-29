dependency_matcher_patterns = {
    "pattern_parameter_used_condition": [
        {
            "RIGHT_ID": "used",
            "RIGHT_ATTRS": {"ORTH": "used"}
        },
        {
            "LEFT_ID": "used",
            "REL_OP": ">",
            "RIGHT_ID": "condition",
            "RIGHT_ATTRS": {"DEP": "advcl"}
        },
        {
            "LEFT_ID": "condition",
            "REL_OP": ">",
            "RIGHT_ID": "dependee_param",
            "RIGHT_ATTRS": {"DEP": "nsubj"}
        }
    ]
}
