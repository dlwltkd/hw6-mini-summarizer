import re

def split_into_sentences(text):
    '''
    Splits text inn to array of sentences
    '''
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

def lead1_summary(text: str) -> str:
    sentences = split_into_sentences(text)
    if not sentences:
        return ""
    return sentences[0]
