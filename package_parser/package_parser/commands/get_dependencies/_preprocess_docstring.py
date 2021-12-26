from collections import OrderedDict


def preprocess_docstring(docstring: str) -> str:
    """
    1. Remove cluttered punctuation around parameter references
    2. Set '=', ==' to 'equals' and set '!=' to 'does not equal'
    3. Handle cases of step two where the signs are not separate tokens, e.g. "a=b".
    Note ordered dict since "=" is a substring of the other symbols.
    """
    
    docstring = docstring.replace('"', '')
    words = docstring.split()

    ordered_string_mapping = OrderedDict()
    ordered_string_mapping["!="] = "does not equal"
    ordered_string_mapping["=="] = "equals"
    ordered_string_mapping["="] =  "equals"

    for i, word in enumerate(words):
        if word in ordered_string_mapping:
            words[i] = ordered_string_mapping[word]
        else:
            for key, val in ordered_string_mapping.items():
                if key in word:
                    word = word.replace(key, f' {val} ')
            words[i] = word

    docstring = ' '.join(words)
    return docstring
