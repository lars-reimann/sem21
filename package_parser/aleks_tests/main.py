from strings import instances, patterns

import spacy

from spacy.matcher import Matcher


nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
for id, pattern in patterns.items():
    matcher.add(id, [pattern])

dependencies = []
for index, tuple in enumerate(instances):
    parameter, description = tuple[0], tuple[1]
    doc = nlp(description)
    if parameter == 'random_state':
        for token in doc:
            print(token, token.pos_)
    dependency_matches = matcher(doc)
    dependencies.append((parameter, dependency_matches))


print("\n MATCHES: \n")
for index, tuple in enumerate(dependencies):
    parameter, match = tuple[0], tuple[1]
    if match != []:
        print(parameter, match)
