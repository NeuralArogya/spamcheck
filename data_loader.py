import pandas as pd
import string
from nltk.corpus import stopwords

def clean_text(text):
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.lower().split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_table(url, header=None, names=["label", "message"])
    df['cleaned'] = df['message'].apply(clean_text)
    return df
