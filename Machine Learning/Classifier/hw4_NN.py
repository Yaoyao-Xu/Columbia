import os
from queue import PriorityQueue
import string
import numpy as np
import math
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

class NN:
    def __init__(self, a, b, c):
        self.train_data = a
        self.train_label =b
        self.test_data = c

    def get_dist_L2(self):
        result =[]
        for x in self.test_data:
            dist = np.sqrt(np.sum((x - self.train_data) ** 2, axis=1))
            order = dist.argsort()
            index = order[0]
            result.append(index)
        return result

    
    def get_dist_L1(self):
        result=[]
        for x in self.test_data:
            dist = np.sum(abs(x - self.train_data), axis=1)
            order = dist.argsort()
            index = order[0]
            result.append(index)   
        return result 
    
    def get_dist_L_infi(self):
        result=[]
        for x in self.test_data:
            dist = np.amax(abs(x - self.train_data), axis=1)
            order = dist.argsort()
            index = order[0]
            result.append(index)   
        return result

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
X_train_spam = X_train[0:train_spam_size,:]
X_train_ham = X_train[1500:ham_end,:]
y_label_spam = np.zeros(train_spam_size)
y_label_ham = np.ones(train_ham_size)


X_input = np.append(X_train_spam,X_train_ham,axis=0)
y_label = np.append(y_label_spam, y_label_ham)


test_spam = X_train[train_spam_size:(train_spam_size+test_spam_size),:]
test_ham = X_train[ham_end:(ham_end+test_ham_size),:]
test_spam_size = np.size(test_spam,0)
total_test_size = np.size(test_spam,0)+np.size(test_ham,0)
X_test = np.append(test_spam,test_ham,axis=0)
y_test_label=[]
for i in range(0, total_test_size):
    if i<test_spam_size:
        y_test_label.append(0)
    else:
        y_test_label.append(1)


NN_classifier= NN(X_input, y_label, X_test)

## L2 dist
Acc_L2 = 0
Acc_L1=0
Acc_L_infi=0

L2_label = NN_classifier.get_dist_L2()
L1_label = NN_classifier.get_dist_L1()
L_infi_label = NN_classifier.get_dist_L_infi()


for i in range(0, len(L2_label)):
    if y_label[L2_label[i]]==y_test_label[i]:
        Acc_L2+=1

for i in range(0, len(L1_label)):
    if y_label[L1_label[i]]==y_test_label[i]:
        Acc_L1+=1

for i in range(0, len(L_infi_label)):
    if y_label[L_infi_label[i]]==y_test_label[i]:
        Acc_L_infi+=1

print(Acc_L2)
print(Acc_L1)
print(Acc_L_infi)
print(total_test_size)





