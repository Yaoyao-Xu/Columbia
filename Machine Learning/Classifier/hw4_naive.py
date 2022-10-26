import os
from queue import PriorityQueue
import re
import string
from termios import VT1
from tkinter import E
from turtle import clear
import numpy as np
import math
import nltk
import matplotlib as mpl
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

DATA_DIR = '/home/xyy/Desktop/ML'
target_names = ['ham', 'spam']
def get_data(DATA_DIR):
    subfolder = 'enron'
    data = []
        # spam
    spam_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'spam'))
    for spam_file in spam_files:
        #print(1)
        with open(os.path.join(DATA_DIR, subfolder, 'spam', spam_file), encoding="latin-1") as f:
            data.append(f.read())
        # ham
    #print(len(target))
    ham_files = os.listdir(os.path.join(DATA_DIR, subfolder, 'ham'))
    for ham_file in ham_files:
        with open(os.path.join(DATA_DIR, subfolder, 'ham', ham_file), encoding="latin-1") as f:
            data.append(f.read())
    return data
 

def word_stem(data):
    pStemmer = PorterStemmer()
    List=[]
    for word in data:
        List.append(pStemmer.stem(word))
    return List

def clean(s):
        translator = str.maketrans("", "", string.punctuation)
        return s.translate(translator)

def to_vector(object):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(object)
    return vectorizer,X


class Naive_classifier:
    def __init__(self, a, b):
        self.train_spam = a
        self.train_ham = b
    def P_multi(self):
        word_size = np.size(self.train_spam,1)
        p_spam_Num = np.ones(word_size)   
        p_ham_Num = np.ones(word_size)
        p_spam_Denom = 2.0               ##Laplace
        p_ham_Denom = 2.0
        for i in range(0, 800):
            for j in range(0, word_size):
                p_spam_Num[j]+=self.train_spam[i][j]
                p_spam_Denom += self.train_spam[i][j]
        
        for i in range(0, 800):
            for j in range(0, word_size):
                p_ham_Num[j]+=self.train_ham[i][j]
                p_ham_Denom += self.train_ham[i][j]


        p_spam = np.log(p_spam_Num/p_spam_Denom)
        p_ham = np.log(p_ham_Num/p_ham_Denom)

        return p_spam, p_ham


def classifyNB(vec2Classify, p_spam, p_ham, prior_spam):
    p_conditonal_spam = sum(vec2Classify * p_spam) + np.log(prior_spam) 
    p_conditional_ham =sum(vec2Classify *p_ham)+np.log(1-prior_spam)

    if p_conditonal_spam>p_conditional_ham:
        return 0
    else:
        return 1

# test =["John likes to watch movies, Mary likes movies too.","John also likes to watch football games."]
# X=[]
# X_after_stem = word_stem(test)
# for word in X_after_stem:
#     X.append(clean(word))
# vectorizer, X = to_vector(X)    
# print(vectorizer.get_feature_names())
# print(X.toarray())

##Process the data
X_origin = get_data(DATA_DIR)#读取数据
X = word_stem(X_origin)
X_train=[]

for word in X:
    X_train.append(clean(word))
vectorizer, X_train = to_vector(X_train)

# print(vectorizer.get_feature_names())
# print(X.toarray())
X_train = X_train.toarray()
X_train = np.array(X_train)

## Apply the classifier
train_spam_size = 800
train_ham_size = 800
test_spam_size = 500
test_ham_size = 500
ham_end = 1500+train_ham_size
prior_spam = train_spam_size/(train_spam_size+train_ham_size)


X_train_spam = X_train[0:train_spam_size,:]
X_train_ham = X_train[1500:ham_end,:]


Naive_bayes = Naive_classifier(X_train_spam,X_train_ham)
v_spam, v_ham = Naive_bayes.P_multi()

test_spam = X_train[train_spam_size:(test_spam_size+train_spam_size),:]
test_ham = X_train[ham_end:(ham_end+test_ham_size),:]
Acc = 0
total_test_size = np.size(test_spam,0)+np.size(test_ham,0)
predict = []

test_result_spam=[]
for row in test_spam:
    result = classifyNB(row, v_spam, v_ham,prior_spam)
    if result == 0:
        Acc+=1
    predict.append(result)
    

for row1 in test_ham:
    result = classifyNB(row1, v_spam, v_ham,prior_spam)
    if result == 1:
        Acc+=1
    predict.append(result)

print(Acc)
print(total_test_size)

## Plot
# y_real= []
# plt.figure(figsize=(10, 10))
# x_plot = np.linspace(1,total_test_size,total_test_size)
# print(np.size(x_plot))
# test_spam_size = np.size(test_spam,0)
# for i in range(0,total_test_size):
#     if i<=test_spam_size:
#         y_real.append(0)
#     else:
#         y_real.append(1)

# plt.scatter(x=x_plot, y=y_real, color="r", label="test_data_real_label")
# plt.scatter(x=x_plot, y=predict, color="b", marker="x", label="predict_result")
# plt.xlabel("Test Data")
# plt.ylabel("Label")
# plt.title("Naive Bayes Results(test size = 800)")
# plt.legend(loc="best")
# plt.show()


















    


