import pandas as pd
import numpy as np

df =  pd.read_csv('balanced_reviews.csv')

df.dropna(inplace =  True)
df = df [df['overall'] != 3]
df['Positivity'] = np.where(df['overall'] > 3 , 1 , 0)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(df['reviewText'], df['Positivity'], random_state = 42 )

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df = 5).fit(features_train)

features_train_vectorized = vect.transform(features_train)
#features_train_vectorized.toarray()

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(features_train_vectorized, labels_train)

predictions = model.predict(vect.transform(features_test))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(labels_test, predictions))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(labels_test, predictions))