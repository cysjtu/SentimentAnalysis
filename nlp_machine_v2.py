# encoding=utf-8
'''
Created on 2015年12月23日

@author: nali
'''

import csv

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score

import jieba
import jieba.posseg as pseg
import jieba.analyse


from xml.dom import minidom

#import MySQLdb 

import nltk
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder  
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist 

import collections, itertools, re
import zhon.hanzi

import nlp

import xml.etree.ElementTree as ET
from __builtin__ import str




#加载XML文件
def load_file():
    #file_name="sample.txt"
    file_name_processed="sample-processed.txt"
    
    #file1=open(file_name,"r")
    #str=file1.read()
    #str=str.replace('&', '');
    
    #file2=open(file_name_processed,"w")
    #file2.write(str)
    #file2.close()
    """
    file1=open(file_name_processed,"r")
    xmldoc = minidom.parse(file1)
    reviews=xmldoc.getElementsByTagName("review")
    """
    tree = ET.ElementTree(file=file_name_processed)
    
    reviews=tree.getroot()
    
    
    data=[]
    target=[]
    """
    for node in reviews:
        data.append( (node.childNodes)[0].data.encode("utf-8") )
        
        id=node.getAttributeNode('id').nodeValue
        
        id=int(id)
        if id < 5000:
            target.append("positive")
        else :
            target.append("negative")
            
    """
    for node in reviews:
        data.append( node.text.encode("utf-8") )
        
        id=node.attrib['id']
        
        id=int(id)
        if id < 5000:
            target.append("positive")
        else :
            target.append("negative")
    
    
    assert len(data) == len(target) 
    
    #file1.close()
    print len(data)
    print len(target)
    
    return data,target

def load_test_file(filename):
    
    file_name_processed="processed_"+filename
    
    
    file1=open(filename,"r")
    str=file1.read()
    str=str.replace('&', '');
    
    file2=open(file_name_processed,"w")
    file2.write(str)
    file2.close()
    file1.close()
    
    

    
    tree = ET.ElementTree(file=file_name_processed)
    
    reviews=tree.getroot()
    
    
    data=[]
    ids=[]
   
    for node in reviews:
        data.append( node.text.encode("utf-8") )
        
        id=node.attrib['id']
        
        id=str(id)
       
        ids.append(id)
      
    assert len(data) == len(ids) 
    #file1.close()
    print len(data)
    print len(ids)
    
    return data,ids



def save_predict(data,ids,target,filename):
    root=ET.Element('weibos')
    for i in  range(len(data)):

        text=unicode(data[i],'utf-8')
        id=str(ids[i])
        polar=str(target[i])
    
        b=ET.SubElement(root, 'weibo', {'id':id,'polarity':polar})
        b.text=text
        
    
    tree = ET.ElementTree(root)
    tree.write(file_or_filename=filename, encoding='utf-8')
    print "save success!"
    
    #tree = ET.ElementTree(file=filename)
    
    
    
    
    

def bag_of_words(words):
    return dict([(word, True) for word in words])

def str_to_unicode(str):
    return unicode(str, 'utf-8').encode('unicode-escape').decode('unicode-escape')

def best_of_words(data_train, target_train):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    stop_words = stopwords.words("english")
    #jieba.analyse.set_stop_words("stopwords-utf8.txt")

    for i in range(len(data_train)):
            words = re.sub("[%s0-9,.=;\^*\/]" % zhon.hanzi.punctuation, " ", str_to_unicode(data_train[i]))
            words = jieba.cut(words, cut_all=False)
            # words = jieba.analyse.extract_tags(words, 55)
            tmp_words=[]
            for w in words:
                if w not in stop_words:
                    tmp_words.append(w)
            words=tmp_words
            bigram_finder = BigramCollocationFinder.from_words(words)
            bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 1000)
            for w in words:
                word_fd[w] += 1
                label_word_fd[target_train[i]][w] += 1
            for b in bigrams:
                word_fd[b[0]+b[1]] += 1
                label_word_fd[target_train[i]][b[0]+b[1]] += 1

            
    pos_word_count = label_word_fd["positive"].N()
    neg_word_count = label_word_fd["negative"].N()
    total_word_count = pos_word_count + neg_word_count
    print total_word_count
    word_scores = {}

    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd["positive"][word],
            (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd["negative"][word],
            (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]
    bestwords = set([w for w, s in best])

    return bestwords


def bigram(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    
    tmp_words=[]
    for w in words:
        tmp_words.append(w)
    words=tmp_words
    
    bigram_finder = BigramCollocationFinder.from_words(words)  #把文本变成双词搭配的形式
    bigrams = bigram_finder.nbest(score_fn, n) #使用了卡方统计的方法，选择排名前1000的双词
    #print type(words)
    
    res=[]
    for d in words:
        #print d
        res.append(d)
        
    for s in bigrams:
        #print type(s)
        #print s
        res.append(s[0]+s[1])
        
    
    
    return bag_of_words(res)
    #return res


    
def evaluate_model(target_true,target_predicted):
    print classification_report(target_true,target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))
    
    
def learn_model(data,target):
    bestwords = best_of_words(data, target)
    # preparing data for split validation. 80% training, 20% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.2,random_state=43)
    #classifier = BernoulliNB().fit(data_train,target_train)
    train_feature=[]
    test_feature=[]
    for i in range(len(data_train)):
        d=data_train[i]
        d=jieba.cut(d, cut_all=False)
        if (i == 1):
            print("Full Mode: " + "/ ".join(d))
        l=target_train[i]
        #tmp=[bigram(d),l]
        tmp = [dict([(word, True) for word in d if word in bestwords]), l]
        train_feature.append(tmp)
        
    for i in range(len(data_test)):
        d=data_test[i]
        d=jieba.cut(d, cut_all=False)
        l=target_test[i]
        #tmp=bigram(d)
        tmp = dict([(word, True) for word in d if word in bestwords])
        test_feature.append(tmp)
    
        
    classifier = SklearnClassifier(MultinomialNB())
    classifier.train(train_feature)
   
    predicted = classifier.classify_many(test_feature)
    
    evaluate_model(target_test,predicted)
    
def load_stop_words(filename):
    ofile = open(filename, "r")

    
def main():
    data,target = nlp.load_file()
    
    learn_model(data,target)    
    
main()





