import re
def tokenize(text):
    """This function tokenizes the input text and returns the output.
     it is a custom tokenizer"""
    text = str(text)
    # remove non-ascii characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # make the text lowercase
    cleaned_text = cleaned_text.lower()
    
    # replace any letter repeated more than twice to a single one (eg: tiired to tired. oopppps -> oops)
    cleaned_text = re.sub(r'([a-zA-Z])\1{2,}', r'\1', cleaned_text)

    # replace can't with cannot
    cleaned_text = re.sub(r'can\'t', 'cannot', cleaned_text)

    # replace xn't with x + not
    cleaned_text = re.sub(r'n\'t', r' not', cleaned_text)

    # replace x'm with x + am
    cleaned_text = re.sub(r'\'m', r' am', cleaned_text)

    # replace x's with x + is
    cleaned_text = re.sub(r'\'s', r' is', cleaned_text)

    # replace x're with x + are
    cleaned_text = re.sub(r'\'re', r' are', cleaned_text)

    # replace x'll to x + will
    cleaned_text = re.sub(r'\'ll', r' will', cleaned_text)

    # replace x'd to x + would
    cleaned_text = re.sub(r'\'d', r' would', cleaned_text)

    # replace x've to x + have
    cleaned_text = re.sub(r'\'ve', r' have', cleaned_text)

    # replace any repeated punctuation to a blank space
    cleaned_text = re.sub(r'([.,!?:;><\'])+', r' ', cleaned_text)


    # remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # split words with hyphens
    cleaned_text = re.sub(r'(\w+)-(\w+)', r'\1 \2', cleaned_text)

    return cleaned_text