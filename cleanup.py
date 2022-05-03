import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

dataframe = pd.read_csv('metadata.csv', usecols=['title','abstract'])

#Drop all rows containing a null value for the abstract
dataframe = dataframe[dataframe['abstract'].notnull()]

#persisting in a separate csv file instead of overriding the original
dataframe.to_csv("cleaned.csv")
