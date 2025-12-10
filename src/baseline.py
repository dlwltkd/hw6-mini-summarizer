import re

def lead1_summary(text):
    '''
    Returns first sentence of paragraph of text.
    '''
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)[0]


