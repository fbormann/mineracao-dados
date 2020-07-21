from sklearn import tree
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import pandas as pd
import numpy as np
import graphviz 

stop_words = set(stopwords.words('english')) 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

df = pd.read_csv("./twitter_data/exploration_dataset.csv")

corpus = []

Y = np.zeros(len(df['id']), dtype=int)
vectorizer = CountVectorizer()
for idx, row in df.iterrows():
    word_tokens = word_tokenize(row['text']) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    corpus.append(" ".join(filtered_sentence))
    if row['label'] != 'none':
        Y[idx] = 1

X = vectorizer.fit_transform(corpus)

clf = tree.DecisionTreeClassifier(max_depth=10,criterion="entropy")
clf = clf.fit(X, Y)

dot_data = tree.export_graphviz(clf, out_file=None,
    class_names=['not','hate'],
    feature_names=vectorizer.get_feature_names(),
    filled=True, rounded=True,  special_characters=True) 
# print(dot_data)
graph = graphviz.Source(dot_data) 
graph.render("./twitter_data/tree") 