# -*- coding: utf-8 -*-
"""
Created on Sat May 12 19:13:12 2018

@author: anandrathi
"""
import os
import inspect
from inspect import currentframe, getframeinfo

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

path="C:/Users/anandrathi/Documents/DataScieince/Coursera/NLP/natural-language-processing-master/week1"
path="C:/temp/DataScience/TextParseNLP/natural-language-processing-master/week1/"
os.chdir(path)

import sys
sys.path.append("..")
from common.download_utils import download_week1_resources

download_week1_resources()


from grader import Grader
grader = Grader()

import nltk
nltk.set_proxy('http://he21061:Bollocks22@proxyserver.health.wa.gov.au:8181',)
nltk.download('stopwords')
from nltk.corpus import stopwords

from ast import literal_eval
import pandas as pd
import numpy as np

def read_data(filename):
    data = pd.read_csv(filename,  sep="\t")
    data['tags'] = data['tags'].apply(literal_eval)
    return data

train = read_data('data/train.tsv')
validation = read_data('data/validation.tsv')
test = pd.read_csv('data/test.tsv', sep='\t')

X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values

print("X_train {}".format( len(X_train)))
print("y_train  {}".format( len(y_train )))

import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set( [ w.strip().lower() for w in stopwords.words('english') ] )
REMOVE_ALLNUMBER_WORDS =  re.compile(r'\b[0-9]+\b')

def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower() # lowercase text
    #print("lowercase text {}".format(text) )
    text = REPLACE_BY_SPACE_RE.sub(" ", str(text)) # replace REPLACE_BY_SPACE_RE symbols by space in text
    #print("REPLACE_BY_SPACE_RE {}".format(text) )
    text = BAD_SYMBOLS_RE.sub("", text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = REMOVE_ALLNUMBER_WORDS.sub("", text) # delete numbers only words

    #print("BAD_SYMBOLS_RE {}".format(text) )
    text = text.strip()
    #print("strip {}".format(text) )
    text = " ".join( [ w.strip() for w in  text.split() if not w.strip() in STOPWORDS] )  # delete stopwords from text
    #print("STOPWORDS {}".format(text) )
    return text

#ttext= " this @!$#@#%$#$ 1675 is suppose%%% to be a really 8738273 ^&%&^%&^%^&%#@ b56564ad te&^$%^%$^%($xt)" 
#print( ttext)
#print( text_prepare(ttext))
#re.sub(r'\b[0-9]+\b', '', ttext)

def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function",
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


print(test_text_prepare())


prepared_questions = []
for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
    line = text_prepare(line.strip())
    prepared_questions.append(line)
text_prepare_results = '\n'.join(prepared_questions)

grader.submit_tag('TextPrepare', text_prepare_results)

X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]


print(" X_train {}".format( len(X_train) ))
print("y_train  {}".format( len(y_train )))



import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from ast import literal_eval
import pandas as pd
import numpy as np


# Dictionary of all tags from train corpus with their counts.
tags_counts = {}
# Dictionary of all words from train corpus with their counts.
words_counts = {}

######################################
######### YOUR CODE HERE #############
######################################

######################################
######### YOUR CODE HERE #############
######################################
from collections import Counter
text = ",".join( [ ",".join(l) for l in  list(train['tags'].values) ]).split(",")
tags_counts = dict(Counter( text))

merged = " ".join(list(X_train))
text =  merged.split()
words_counts = dict(Counter( text))


mctags = ",".join([ tag[0] for tag in list(Counter(words_counts).most_common(3)) ] )
mcwprds = ",".join([ word[0] for word in list(Counter(tags_counts).most_common(3)) ] )
print(mctags)
print(mcwprds)

most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]

grader.submit_tag('WordsTagsCount', '%s\n%s' % (','.join(tag for tag, _ in most_common_tags),
                                                ','.join(word for word, _ in most_common_words)))

DICT_SIZE = 5000
from sklearn.feature_extraction.text import CountVectorizer

WORDS_TO_INDEX = { word:index for index, word in enumerate( dict(Counter(words_counts).most_common(DICT_SIZE)).keys() ) }
INDEX_TO_WORDS = {v: k for k, v in WORDS_TO_INDEX.items()}

import numpy as np
result_vector = np.zeros(10)

def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary

        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    index=0
    for word in text.split():
      if word in words_to_index:
        index=words_to_index[word]
        result_vector[index]+=1
    return result_vector

def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

print(test_my_bag_of_words())

from scipy import sparse as sp_sparse

X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])

print( "This is line {} ".format( get_linenumber()))
print(" X_train {}".format( len(X_train)) )
print("y_train  {}".format( len(y_train )))

#Task 3 (BagOfWords)

for i in range(1,20):
  print(np.count_nonzero(X_train_mybag[i].toarray()[0]))
  
row = X_train_mybag[10].toarray()[0]
non_zero_elements_count = np.count_nonzero(row ) ####### YOUR CODE HERE #######
grader.submit_tag('BagOfWords', str(non_zero_elements_count))


from sklearn.feature_extraction.text import TfidfVectorizer


print( "This is line {} ".format( get_linenumber()))
print(" X_train {}".format( len(X_train)))
print("y_train  {}".format( len(y_train )))

def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result


    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    #tfidf_vectorizer = TfidfVectorizer()####### YOUR CODE HERE #######
    tfidf_vectorizer = TfidfVectorizer(token_pattern='(\S+)', ngram_range=(1,4) ,  
                                       analyzer='word', 
                                       min_df=2, 
                                       max_df = 0.98)
    xall=X_train.copy()
    xall.extend(X_val)
    xall.extend(X_test)
    #xall.extend(X_val)
    #xall.extend(X_test)
    tfidf_vectorizer =  tfidf_vectorizer.fit(xall)
    return tfidf_vectorizer.transform(X_train), tfidf_vectorizer.transform(X_val), tfidf_vectorizer.transform(X_test), tfidf_vectorizer.vocabulary_

len(X_train)
len(train['tags'].values)

print( "This is line {} ".format( get_linenumber()))


X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}
print( "This is line {} ".format( get_linenumber()))
print(" X_train {}".format( len(X_train)))
print("y_train  {}".format( len(y_train )))

X_train_tfidf.shape

tfidf_vocab["c++"]
tfidf_vocab["c#"]
tfidf_vocab["java"]

y_train = train['tags'].values
y_val = validation['tags'].values

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data

      return: trained classifier
    """
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    cls = OneVsRestClassifier(LogisticRegression())
    cls.fit(X_train, y_train)
    return cls

classifier_mybag = train_classifier(X_train_mybag, y_train)

classifier_tfidf = train_classifier(X_train_tfidf, y_train)

X_train_tfidf.shape
y_train.shape

print( "This is line {} ".format( get_linenumber()))
print(" X_train {}".format( len(X_train)))
print(" y_train  {}".format( len(y_train )))

print(" X_train_tfidf.shape {}".format( X_train_tfidf.shape ))
print("y_train  {}".format( y_train.shape))

y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)


y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(10):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))

y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_mybag)
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(10):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))



from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

#sklearn.metrics  MultiLabelBinarizer
def print_evaluation_scores(y_val, predicted):
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    print( "accuracy={}".format( accuracy_score(y_val, predicted)))
    print( "")
    print( "roc_auc_score={}".format( roc_auc_score(y_val, predicted)))
    print( "")
    print( "average_precision_score={}".format( average_precision_score(y_val, predicted)))
    print( "")

    print( "macro average_precision_score={}".format( average_precision_score(y_val, predicted, average = "macro")))
    print( "micro average_precision_score={}".format( average_precision_score(y_val, predicted, average = "micro")))
    print( "weighted average_precision_score={}".format( average_precision_score(y_val, predicted, average = "weighted")))

    print( "")

    print( "macro recall_score={}".format( recall_score(y_val, predicted, average = "macro")))
    print( "micro recall_score={}".format( recall_score(y_val, predicted, average = "micro")))
    print( "weighted recall_score={}".format( recall_score(y_val, predicted, average = "weighted")))
    print( "")

    print( "macro f1_score={}".format( f1_score(y_val, predicted, average = "macro")))
    print( "micro f1_score={}".format( f1_score(y_val, predicted, average = "micro")))
    print( "weighted f1_score={}".format( f1_score(y_val, predicted, average = "weighted")))


    #print( "f1_score={}".format( f1_score(y_val, predicted)))



print( "This is line {} ".format( get_linenumber()))
print(" X_train {}".format( len(X_train)))
print("y_train  {}".format( len(y_train )))

print('\nBag-of-words')
print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
print('\nTfidf')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)


from metrics import roc_auc
##matplotlib inline
n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_mybag, n_classes)


n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_tfidf, n_classes)



##**Task 4 (MultilabelClassification).** Once we have the evaluation set up,
## we suggest that you experiment a bit with training


##########################################################################################################

#REMOVE_ALLNUMBER_WORDS =  re.compile(r'\b[0-9]+\b')


def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result


    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    #tfidf_vectorizer = TfidfVectorizer()####### YOUR CODE HERE #######
    tfidf_vectorizer = TfidfVectorizer(token_pattern='(\S+)', ngram_range=(1,4) ,  
                                       analyzer='word', 
                                       min_df=2, 
                                       max_df = 0.98)
    xall=X_train.copy()
    xall.extend(X_val)
    xall.extend(X_test)
    #xall.extend(X_val)
    #xall.extend(X_test)
    tfidf_vectorizer =  tfidf_vectorizer.fit(xall)
    return tfidf_vectorizer.transform(X_train), tfidf_vectorizer.transform(X_val), tfidf_vectorizer.transform(X_test), tfidf_vectorizer.vocabulary_

len(X_train)
len(train['tags'].values)

print( "This is line {} ".format( get_linenumber()))

X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}



def train_classifierNew(X_train, y_train):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.
    
    ovsr = OneVsRestClassifier( LogisticRegression(penalty='l1', C=10) )
    #score_func = make_scorer(metrics.f1_score)
    
    best_model = ovsr.fit(X_train, y_train)
    return best_model

classifier_tfidf = train_classifierNew(X_train_tfidf, y_train)

classifier_tfidf.estimators_

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)


y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)

for i in range(10):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))


print('\nTfidf')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)


from metrics import roc_auc
##matplotlib inline

n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_tfidf, n_classes)


y_test_inversed = mlb.inverse_transform(y_test)

print('\nBag-of-words')
print_evaluation_scores(y_test, y_test_predicted_labels_mybag)
print('\nTfidf')
print_evaluation_scores(y_test, y_test_predicted_labels_tfidf)

test_pred_inversed = mlb.inverse_transform(test_predictions)

test_predictions = classifier_tfidf.predict(X_test_tfidf) ######### YOUR CODE HERE #############
test_pred_inversed = mlb.inverse_transform(test_predictions)

test_predictions_for_submission = '\n'.join('%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))
grader.submit_tag('MultilabelClassification', test_predictions_for_submission)

best_estimator = classifier_tfidf.best_estimator_
classifier_tfidf.estimator.coef_
lassifier_tfidf.best_estimator_.coef_

classifier_tfidf.coef_
classifier_tfidf.classes_
classifier_tfidf.classes_.s

classes=sorted(tags_counts.keys())

len(classes)
len(classifier_tfidf.coef_)
len(classifier_tfidf.coef_[0])

mlb.inverse_transform(classifier_tfidf.coef_[0])

CoeffDict = { "Coeff_Tag_" + aclass : classifier_tfidf.coef_[i]  for i, aclass in enumerate(classes) }  
import operator
sorted_tfidf_vocab = [ voca[0] for voca in list(sorted(tfidf_vocab.items(), key=operator.itemgetter(1)))]
sorted_tfidf_vocab[1:10]
CoeffDict["Features"] = sorted_tfidf_vocab
CoeffDF = pd.DataFrame(CoeffDict)

[tfidf_reversed_vocab[i] for i in np.argsort(classifier_tfidf.coef_[classes.index("c++")])[::-1][:5]]

def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
    """
        classifier: trained classifier
        tag: particular tag
        tags_classes: a list of classes names from MultiLabelBinarizer
        index_to_words: index_to_words transformation
        all_words: all words in the dictionary
        
        return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print('Tag:\t{}'.format(tag))
    
    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator. 
    
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    #[index_to_words[i] for i in np.argsort(classifier.coef_[tags_classes.index(tag)])[::-1][:5]]
    top_positive_words = [index_to_words[i] for i in np.argsort(classifier.coef_[tags_classes.index(tag)])[::-1][:5]] # top-5 words sorted by the coefficiens.
    top_negative_words = [index_to_words[i] for i in np.argsort(classifier.coef_[tags_classes.index(tag)])[:5]] # bottom-5 words  sorted by the coefficients.
    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))

classes=sorted(tags_counts.keys())
print_words_for_tag(classifier=classifier_tfidf, tag="c++", tags_classes=sorted(tags_counts.keys()), index_to_words=tfidf_reversed_vocab, all_words=tfidf_reversed_vocab.values())
print_words_for_tag(classifier=classifier_tfidf, tag="java", tags_classes=sorted(tags_counts.keys()), index_to_words=tfidf_reversed_vocab, all_words=tfidf_reversed_vocab.values())
print_words_for_tag(classifier=classifier_tfidf, tag="c#", tags_classes=sorted(tags_counts.keys()), index_to_words=tfidf_reversed_vocab, all_words=tfidf_reversed_vocab.values())

                    
print_words_for_tag(classifier_tfidf, 'c', mlb.classes, tfidf_reversed_vocab, tfidf_reversed_vocab.values())
print_words_for_tag(classifier_tfidf, 'c++', mlb.classes, tfidf_reversed_vocab, tfidf_reversed_vocab.values())
print_words_for_tag(classifier_tfidf, 'linux', mlb.classes, tfidf_reversed_vocab, tfidf_reversed_vocab.values())

grader.status()

import os
proxy = 'http://he21061:Bollocks22@proxyserver.health.wa.gov.au:8181'
proxys = 'https://he21061:Bollocks22@proxyserver.health.wa.gov.au:8181'
os.environ['http_proxy'] = proxy 
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxys
os.environ['HTTPS_PROXY'] = proxys 
STUDENT_EMAIL = "anandrathi.dev@gmail.com" # EMAIL 
STUDENT_TOKEN = "lWIdM47EVCt9MhRf" # TOKEN 
grader.status()
