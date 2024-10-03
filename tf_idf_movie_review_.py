

- You can perform ML and DL models on NLP data

- Read the NLP data

- Preprocess or text cleaning

- Apply the **word embeddings**

    - convert text to vector form
    
    - BoW
    
    - tf-idf
    
    - word2vec
    
- Develop the model

- Predictions

- Metrics
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import nltk
import re
import string
from nltk.stem import WordNetLemmatizer
import pandas as pd
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plot

pip install matplotlib

df=pd.read_excel("IMDB_Dataset_sample.xlsx")

df.head()

df.values[0]

"""**Text preprocessing**

**Convert text to vectors**

**apply ML model**

**metrics**

**prediction**
"""

print(df.value_counts('sentiment'))
df.value_counts('sentiment').plot(kind='bar')

df['sentiment_numeric'] = df.sentiment.map({'positive':1,
           'negative':0})
df.head()

### Preprocessing Function
ps = PorterStemmer()
corpus = set()
def preprocess(text):

    ## removing unwanted space
    text = text.strip()

    ## removing html tags
    text = re.sub("<[^>]*>", "",text)

    ## removing any numerical values
    text = re.sub('[^a-zA-Z]', ' ',text)

    ## lower case the word
    text = text.lower()

    text = text.split()

    ## stemming the word for sentiment analysis do not remove the stop word
    text = [ps.stem(word) for word in text]
    text = ' '.join(text)
    return text

df['Preprocessed_review'] = df.review.apply(preprocess)

df.head()

df.shape

df = df.drop(columns=["review","sentiment"])
df

x=df["Preprocessed_review"]
y=df["sentiment_numeric"]
x

### performing train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 test_size=0.2,
                                                 random_state=42,    # For reproducibility
                                                 stratify=y) # Preserve the distribution of the target variable

x_train.shape,x_test.shape

#tf idf

tf_idf = TfidfVectorizer()

tf_idf

#len(tf_idf.vocabulary_)

#applying tf idf to training data

X_train_tf = tf_idf.fit_transform(x_train)
X_train_tf

X_train_tf.shape

X_train_tf[0].toarray()

#applying tf idf to testing data

X_test_tf = tf_idf.transform(x_test)
X_test_tf

X_test_tf.shape

### Model creation

#naive bayes classifier

naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tf,y_train)

#predicted y

y_pred = naive_bayes_classifier.predict(X_test_tf)
y_pred

print(metrics.classification_report(y_test, y_pred,
                                            target_names=['Positive', 'Negative']))

print("Confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

print(x_test,type(x_test))

# Doing test prediction
[t]= ["delivers breathtaking visuals and epic action, set in a mythical kingdom. Directed by S.S. Rajamouli, it combines grand battles and intricate storytelling with strong performances by Prabhas and Rana Daggubati. The film's scale and emotional depth make it a landmark in Indian cinema"]
[t]

test_processed=preprocess(t)

test_input = tf_idf.transform([test_processed])

#0= bad review
#1= good review

res=naive_bayes_classifier.predict(test_input)[0]

if res==1:
    print("Good Review")

elif res==0:
    print("Bad Review")

y_test

### Testing all together
review=['Movie is good not a comedy movie']
test_processed=preprocess(review[0])
test_input = tf_idf.transform([test_processed])
#0= bad review
#1= good review

res=naive_bayes_classifier.predict(test_input)[0]

if res==0:
    print("Good Review")

elif res==1:
    print("Bad Review")

"""- Preprocess

- Vectorizer

- Pass into the model

- Then model give predictions 1 and 0

**Model Save**
"""

import joblib
joblib.dump(naive_bayes_classifier,"nb_classifier.pkl")

joblib.dump(tf_idf,"Tf_Idf.pkl")

import pickle
pickle.dump(naive_bayes_classifier,open("nb_classifier.pkl","wb"))

import pickle
pickle.dump(tf_idf,open("Tf_Idf.pkl","wb"))

import joblib
joblib.dump(naive_bayes_classifier,"nb_classifier.joblib")

import joblib
joblib.dump(tf_idf,"Tf_Idf.joblib")

