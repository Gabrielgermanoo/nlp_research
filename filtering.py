import numpy as np
import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import csv

df = pd.read_csv('dados.csv', sep=',')
evento_df = df[df['evento'].str.contains("ENEPET")]
#print(evento_df)
from nltk import word_tokenize


def tokening(words):
    for w in words:
        word_tokenize(w);

def find_words(text):
    for t in text:
        t = ''.join(letter for letter in t if letter.isalnum())
    return text

nltk.download('stopwords')
words_lower = evento_df['descricao'].str.lower()
words = find_words(words_lower)
stopwords = nltk.corpus.stopwords.words('portuguese')
stop = set(stopwords)

no_stopwords = [w for w in words if w not in stop]

cleanText = " ".join(no_stopwords)
#print(cleanText)

nltk.download('punkt')

tokens = tokening(cleanText)
#word_tokenize(cleanText)
print(tokens)