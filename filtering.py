#import libres from tokenized, removing stowords, stemming and lemmatization
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import RSLPStemmer
nltk.download('rslp')
import spacy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def filter_data_event(data, evento=0):
  '''
  This function selecte the deliberations correspnding one each event type 
      0: INTERPET
      1: ENEPET
      2: ENAPET
  '''
  list_events = ['InterPET', 'ENEPET', 'ENAPET']
  event = list_events[evento]
  evento_df = data[data['evento'].str.contains(event)]

  return evento_df

# Functions for pré-processing

def tokenizing(data):
  #data: series with texts
  return data.apply(lambda x: word_tokenize(x))

def removed_ponctuation(data_tokenized):
  #data: series with texts tokenized
  return data_tokenized.apply(lambda x: [word for word in x if word.isalpha()])
  
def removed_stopwords(data_tokenized):
  #data: series with texts tokenized
  stop_words = set(stopwords.words('portuguese'))
  return data_tokenized.apply(lambda x: [word.lower() for word in x if word.lower() not in stop_words])

def stemming(data_tokenized):
  #data: series with texts tokenized
  stemmer = RSLPStemmer()
  return data_tokenized.apply(lambda x: [stemmer.stem(word) for word in x])

def lemmatizing(data_tokenized):
  #data: series with texts tokenized

  #load model from linguage portuguese
  nlp = spacy.load('pt_core_news_sm')

  #Realizing lematizing in words
  return data_tokenized.apply(lambda x: [token.lemma_ for token in nlp(' '.join(x))])

def pre_processing(df):

  data = df.copy()
  #Realizing pré-processing
  data['tokenized'] = tokenizing(data.descricao)
  data['removed_ponctuation'] = removed_ponctuation(data['tokenized'])
  data['removed_stopwords'] = removed_stopwords(data['removed_ponctuation'])
  data['stemming'] = stemming(data['removed_stopwords'])
  data['lemmatizing'] = lemmatizing(data['removed_stopwords'])

  return data


#Functions for calculating frequences of words and visualizing

def freq_words(data):
  ''' data: data series with tokens '''
  
  #Converting for Text object all deliberations and calculating frequence absolute the words
  deliberacoes_lematizing_text = nltk.Text([word for deliberacao in data for word in deliberacao])
  freq_plot = nltk.FreqDist(deliberacoes_lematizing_text)
  
  #Creating dataframe for save words and frequences
  df_freq = pd.DataFrame()
  df_freq['word'] = list(word for word, freq in freq_plot.items())
  df_freq['freq'] = list(freq for word, freq in freq_plot.items())

  #Organizing in descending order
  df_freq.sort_values(by='freq', ascending=False, inplace=True)

  return df_freq

def plot_freq_words(data, qtd):
  ''' data: dataframe with words and frequence in descending order
      qtd: quantity of words more often '''

  plt.bar(data.word[:qtd], data.freq[:qtd]) 
  plt.title(str(qtd) + ' palavras mais frequentes', fontsize=15, fontweight='bold')
  plt.xlabel('Palavras', fontweight='bold')
  plt.ylabel('Frequência', fontweight='bold')
  plt.xticks(rotation='vertical')
  plt.show()

  
def similarity_BOW(text1, text2):
    '''
    This function calculates the similarity between two texts
        text1: candidate text for comparison
        text2: reference text for comparison
    '''

    #Converting text in vector 
    vec = CountVectorizer(ngram_range=(1, 1)) #considering unigrams
    x1, x2 = vec.fit_transform([text1, text2])
    t1, t2 = x1.toarray(), x2.toarray()

    #Calculateted minimum frequency of each word (intersects between texts)
    min = np.amin([t1, t2], axis = 0)

    #counting number of intersections of words between texts
    sum = np.sum(min)

    #Counting unique words in vector of text1 (candidate for comparison)
    count = np.sum(t1)

    #Calculate 
    percentil_similar = np.around((sum/count)*100, 2)

    return percentil_similar

#Faz pré-processamento de um texto digitado
def preprocessing_text_input(text):
    
    '''Functions for realizing pre processing in new text '''

    #Created dataframe with text 
    df_text = pd.DataFrame()
    df_text['descricao'] = [text]

    #Apply prprocessing in text 
    df_text = pre_processing(df_text)
    df_text['lemma_text'] = (df_text.lemmatizing.apply(lambda x: ' '.join(x))).to_list()

    return df_text.lemma_text[0]