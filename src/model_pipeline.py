from transformers import pipeline


def load_model():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6" )
    return summarizer

def generate_summary(model,text):
    r