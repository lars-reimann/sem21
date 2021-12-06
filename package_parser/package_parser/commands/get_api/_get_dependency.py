import spacy
from spacy.matcher import Matcher

from typing import List

from ._dependency_patterns import patterns
from ._model import API


PIPELINE = "en_core_web_sm"


def get_dependency(api_endpoints: List[API]):
    nlp = spacy.load(PIPELINE)
    matcher = Matcher(nlp.vocab)
    for id, pattern in patterns.items():
        matcher.add(id, pattern)
    
    dependencies = {}

    for endpoint in api_endpoints:
        endpoint_functions = endpoint.functions

        for function_name, function in endpoint_functions.items():
            parameters = function.parameters
            dependencies[function_name] = {}
            
            for parameter in parameters:
                doc = nlp(parameter.docstring)
                dependency_matches = matcher(doc)
                dependencies[function_name][parameter.name] = dependency_matches
    
    return dependencies
