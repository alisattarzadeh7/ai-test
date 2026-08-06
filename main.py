
import pandas as pd
import numpy as np
from cleantext import clean
import re
from transformers import XLNetTokenizer, XLNetForSequenceClassification, TrainingArguments, Trainer, pipeline
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import datasets
import evaluate
import random



data_train = pd.read_csv('./emotion-labels-train.csv')
data_test = pd.read_csv('./emotion-labels-test.csv')
data_val = pd.read_csv('./emotion-labels-val.csv')

data_train.head()

data = pd.concat([data_train, data_test, data_val], ignore_index=True)

data['text_clean']  = data['text'].apply(lambda x: clean(x,no_emoji=True))

data['text_clean'] = data['text_clean'].apply(lambda x: re.sub(r'@[^\s]+','',x))

data.head(20)

data['label'].value_counts().plot(kind='bar')

g = data.groupby('label')
data = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True))

data['label'].value_counts().plot(kind='bar')



