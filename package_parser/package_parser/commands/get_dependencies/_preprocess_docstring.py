def preprocess_docstring(docstring: str) -> str:
    """
    1. Remove cluttered punctuation around parameter references
    2. Set '=', ==' to word equals
    3. Set cases of 'a=b', 'a==b to 'a equals b' (not successful in step 2 due to no spaces)
    """
    
    docstring = docstring.replace('"', '')
    words = docstring.split()
    equals_strings = ["=", "=="]
    for i, word in enumerate(words):
        if word in equals_strings:
            words[i] = "equals"
        else:
            # Order here is important! Cleaner way?
            if "==" in word:
                word = word.replace("==", " equals ")
                words[i] = word
            if "=" in word:
                word = word.replace("=", " equals ")
                words[i] = word

    docstring = ' '.join(words)
    return docstring
