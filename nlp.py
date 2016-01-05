# encoding=utf-8
'''
Created on 2015年12月5日

@author: nali
'''

import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score

import jieba
import jieba.posseg as pseg
from random import randrange

from xml.dom import minidom

import MySQLdb 

import test

#加载情感词典到数据库里面
def load_lexicon():
    conn=MySQLdb.connect(host="127.0.0.1",user="root",passwd="cy1993",port=3306,db="lexicon",charset='utf8')
    cur=conn.cursor()
    query="insert into lex(word,polar) values(%s,%s)"
    
    
    with open('lexicon1.csv') as csv_file:
        reader = csv.reader(csv_file,delimiter=",",quotechar='"')
        reader.next()
        data =[]
        target = []
        for row in reader:
            # skip missing data
            #print row[0]
            cur.execute(query,(row[0],row[6]))
            
        
        conn.commit()
        cur.close()
        conn.close()
            
        print "success!"
        
    
            




def token_fnt(str):
    return jieba.cut(str, cut_all=False)


def preprocess():
    data,target = load_file()
    
    count_vectorizer = CountVectorizer(binary=False,encoding="utf-8",analyzer='word',tokenizer=token_fnt,max_features=10000)

    data = count_vectorizer.fit_transform(data)

    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)
    
    feat=count_vectorizer.get_feature_names()
    print "feature length"
    print len(feat)
    
    for i in range(len(feat)-5):
        print feat[1-i]#.encode("utf-8")
        
    #fet=[]

    #for f  in feat:
    #    fet.append(f.encode("utf-8"))
    
   # print "tokens"
    #print "+".join(fet)
    
    return tfidf_data

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
    
    file1=open(file_name_processed,"r")
    xmldoc = minidom.parse(file1)
    reviews=xmldoc.getElementsByTagName("review")
    
    data=[]
    target=[]
    
    for node in reviews:
        data.append( (node.childNodes)[0].data.encode("utf-8") )
        
        id=node.getAttributeNode('id').nodeValue
        id=int(id)
        if id < 5000:
            target.append("positive")
        else :
            target.append("negative")
            
    
    assert len(data) == len(target) 
    
    file1.close()
    print len(data)
    print len(target)
    
    return data,target


        
    
    

    
def learn_model(data,target):
    # preparing data for split validation. 60% training, 40% test
    state=randrange(1,23432)+123
    print "statue 6857"
    print state

    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.2,random_state=state)
    classifier = BernoulliNB().fit(data_train,target_train)
    predicted = classifier.predict(data_test)
    evaluate_model(target_test,predicted)


# read more about model evaluation metrics here
# http://scikit-learn.org/stable/modules/model_evaluation.html
def evaluate_model(target_true,target_predicted):
    print classification_report(target_true,target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))

def main():
    data,target = load_file()
    state=randrange(1,23432)+123
    print "statue 6857"
    print state
    raw_input("sss")
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.2,random_state=state)
    predicted=test.allJudge(data_test,target_test)
    
    evaluate_model(target_test,predicted)
    
    
    #tf_idf = preprocess()
    #print tf_idf.toarray()
    #learn_model(tf_idf,target)


#main()





