import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

dataframe = pd.read_csv('metadata.csv', usecols=['title','abstract'])

#Drop all rows containing a null value for the abstract
dataframe = dataframe[dataframe['abstract'].notnull()]

#persisting in a separate csv file instead of overriding the original
dataframe.to_csv("cleaned.csv")

#using sklearns tfidf vectorizer to extract features from text
dataframe = pd.read_csv('cleaned.csv')
vectorizer = TfidfVectorizer(stop_words={'english'})
X = vectorizer.fit_transform(dataframe['abstract'])

#print Inverse document frequency vector
print(vectorizer.idf_)

#print a mapping of terms to feature indices
print(vectorizer.vocabulary_)
