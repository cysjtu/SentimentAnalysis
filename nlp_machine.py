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

import re 
from xml.dom import minidom

import MySQLdb 

import nltk
from nltk.collocations import BigramCollocationFinder ,TrigramCollocationFinder 
from nltk.metrics import BigramAssocMeasures ,TrigramAssocMeasures
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from random import randrange

from test import  *

import xml.etree.ElementTree as ET
from __builtin__ import str




hinfo_dict={}


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


def bigram(words, score_fn=BigramAssocMeasures.likelihood_ratio, n=500,freq=1):
    """
    tmp_words=[]
    for w in words:
        tmp_words.append(w)
    words=tmp_words
    """
    if len(words)<=0:
        return {}
    
    
    tmp_dict={}
        
    for w in words:
        tmp_dict[w]=1
        
    if len(tmp_dict.keys()) <2:
        return {}
 
    
    bigram_finder = BigramCollocationFinder.from_words(words)  #把文本变成双词搭配的形式
    bigram_finder.apply_freq_filter(freq)
    bigrams = bigram_finder.nbest(score_fn, n) #使用了卡方统计的方法，选择排名前1000的双词
    
    #print type(words)
    
    res=[]
        
    for s in bigrams:
        #print type(s)
        #print s
        res.append(s[0]+s[1])
        """
        if res.has_key(s[0]+s[1])==True:
            res[s[0]+s[1]]+=1
        else:
            res[s[0]+s[1]]=1
        """
    
    
    return res
    #return res




def trigram(words, score_fn=TrigramAssocMeasures.likelihood_ratio, n=1500,freq=1):
    """
    tmp_words=[]
    for w in words:
        tmp_words.append(w)
    words=tmp_words
    """
    if len(words)<=0:
        return {}
    
    
    tmp_dict={}
        
    for w in words:
        tmp_dict[w]=1
        
    if len(tmp_dict.keys()) <3:
        return {}


    
    trigram_finder = TrigramCollocationFinder.from_words(words)  #把文本变成双词搭配的形式
    trigram_finder.apply_freq_filter(freq)
    trigrams = trigram_finder.nbest(score_fn, n) #使用了卡方统计的方法，选择排名前1000的双词
    
    #print type(words)
    
    res={}
        
    for s in trigrams:
        
        if res.has_key(s[0]+s[1]+s[2])==True:
            res[s[0]+s[1]+s[2]]+=1
        else:
            res[s[0]+s[1]+s[2]]=1

    return res
    #return res







conn=MySQLdb.connect(host="127.0.0.1",user="root",passwd="cy1993",port=3306,db="lexicon",charset='utf8')

cur=conn.cursor()

def build_stop_word_dict():
    res=[]
    query2="select word from stop"
    cur.execute(query2)
    
    data=cur.fetchall()
    
    for s in data:
        res.append((s[0],True))
        
    
    return dict(res)


    
    

def build_sentiment_dict():
    res=[]
    query1="select word from sentiment"
    query2="select word from level"
    query3="select word from inverse"
    
    
    cur.execute(query1)
    data=cur.fetchall()
    for s in data:
        res.append((s[0],True))


    cur.execute(query2)
    data=cur.fetchall()
    for s in data:
        res.append((s[0],True))

    cur.execute(query3)
    data=cur.fetchall()
    for s in data:
        res.append((s[0],True))
        
    
    return dict(res)


def build_hinfo_dict(data_train,target_train):
 
    
   
    biaodian={u'，':1,u'。':1,u'；':1,u'\n':1,u';':1,u',':1,u'《':1,u'》':1,u'“':1,u'”':1,u' ':1,
              u'’':1,u'‘':1,u'、':1}
    
    #,u'.':1,u'...':1,u'"':1,u'/':1,u'（':1,u'）':1,u'(':1,u')':1
    pattern=re.compile(r'\w{1,}')
    

    pos_words=[]
    neg_words=[]
    
    pos_bigram=[]
    neg_bigram=[]
    
    pos_trigram=[]
    neg_trigram=[]


    for i in range(len(data_train)):
        d=data_train[i]
        d=jieba.cut(d)
        l=target_train[i]
            
 
        for ww in d:
            if pattern.match(ww)!=None:
                print "---"+ww
                continue
            if biaodian.has_key(ww)==True:
                print "---"+ww
                continue
            
            if biaodian.has_key(ww)==False:
                #cnt+=1
                #word_fd[ww]+=1
            
                if l=="positive":
                    pos_words.append(ww)
                    #label_word_fd['pos'][ww]+=1
                else:
                    neg_words.append(ww)
                    #label_word_fd['neg'][ww]+=1
    
    print "pos_words %d"%(len(bag_of_words(pos_words).keys()))
    print "neg_words %d"%(len(bag_of_words(neg_words).keys()))
    
    pos_words_d={}
    neg_words_d={}
    
    
    pos_bigram=bigram(pos_words,score_fn=BigramAssocMeasures.chi_sq, n=3000,freq=10)
    print "pos_bigram %d"%(len(pos_bigram))
    neg_bigram=bigram(neg_words,score_fn=BigramAssocMeasures.chi_sq, n=3000,freq=10)
    print "neg_bigram %d"%(len(neg_bigram))
    """
    pos_trigram=trigram(pos_words,score_fn=TrigramAssocMeasures.chi_sq, n=3000,freq=1)
    print "pos_trigram %d"%(len(pos_trigram.keys()))
    neg_trigram=trigram(neg_words,score_fn=TrigramAssocMeasures.chi_sq, n=3000,freq=1)
    print "neg_trigram %d"%(len(neg_trigram.keys()))
    """
    
    pos_words=pos_words+pos_bigram
    neg_words=neg_words+neg_bigram

    best_word_dict=get_best_words(pos_words,neg_words,9500)
    print "best_word_dict %d"%(len(best_word_dict.keys()))
    
    best_bigram_dict={}#get_best_words(pos_bigram,neg_bigram,2500)
    print "best_bigram_dict %d"%(len(best_bigram_dict.keys()))

    best_trigram_dict={}#get_best_words(pos_trigram,neg_trigram,2000)
    print "best_trigram_dict %d"%(len(best_trigram_dict.keys()))

      
    best_word_dict.update(best_bigram_dict)
    best_word_dict.update(best_trigram_dict)
   
    #pos=pos_words+pos_bigram.keys()
    #neg=neg_words+neg_bigram.keys()
    return best_word_dict


    
    
def get_best_words(pos,neg,limit):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()
    
    for word in pos:
        word_fd[word]+=1
        label_word_fd['pos'][word]+=1
    for word in neg:
        word_fd[word]+=1
        label_word_fd['neg'][word]+=1

    # n_ii = label_word_fd[label][word]
    # n_ix = word_fd[word]
    # n_xi = label_word_fd[label].N()
    # n_xx = label_word_fd.N()
    
    #for i in word_fd.keys():
    #    if word_fd[i] >20:
    #        print i
    #        print word_fd[i]
    
    pos_word_count = label_word_fd['pos'].N()
    neg_word_count = label_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count
    
    #print total_word_count
    #print cnt
    word_scores = {}
     
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
            (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
            (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
     
    best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:limit]
    
    
    bestwords = dict([(w,True) for w, s in best])
    
    pattern=re.compile(r'\w{1,}')
    new_bestwords={}
    for k in bestwords:
        res=pattern.match(k)
        if res ==None:
            new_bestwords[k]=True
            print "+++"+k.encode("utf-8")
        else: 
            print k.encode("utf-8")
    return new_bestwords
    
    

max_len=-1;
min_len=99999
sum=0
cnt=1

def get_line_feat(text):
    
    global hinfo_dict
    
    tokens=jieba.cut(text)
    
    words=[]
    flag=[]
    
    polars=["","postive","negtive"]
    
    levels=["","most","more","very","little","insufficiently"]
    sss=""
    
    ret={}
    
    for w in tokens:
        #print w
       
        words.append(w)
        
        stop=isstopWord(w)
        
        if 1==stop:
            flag.append(("none",""))
            continue
    
        
        inv=isInverseWord(w)
        level,lev_score=getLevel(w)
        
        if 1 == inv:
            flag.append(("inverse",""))
            continue
            #print ("%s----->[inverse]"%(w))
        elif -1 != level:
            flag.append(("level",levels[level]))
            continue
        else:
            polar=getPolar(w)
            
            if polar != -1:
                flag.append(("sentiment",polars[polar]))
                #print ("%s----->[sentiment]%s"%(w,polars[polar]))
            else:
                flag.append(("none",""))
                #print ("%s----->[none]"%(w))
              
            
    #pos_score=0.0
    #neg_score=0.0
    inv_cont=0
    lev_cont=0
    
    for i in range(len(flag)):
         
         if flag[i][0] == "sentiment":
             #postive
            sss = sss + "-" + words[i] + "[" + flag[i][1] + "]"
             
            tmp_score = 1.0
            
             # 往前找程度副词
            for k in range(i - 1, -1, -1):
                if flag[k][0] == "sentiment":
                    break
                elif flag[k][0] == "level":
                    level,lev_score= getLevel(words[k])
                    tmp_score = tmp_score * lev_score
                    break
            #往前计算否定词的数量
            tmp_inv_cnt=0
            for k in range(i - 1, -1, -1):
                if flag[k][0] == "sentiment":
                    break
                elif flag[k][0] == "inverse":
                    tmp_inv_cnt+=1
            
            #positive 
            if flag[i][1]==polars[1]:
                if tmp_inv_cnt%2 != 0 :
                    key="neg_"+words[i]
                    ret[key]=-1
                else:
                    key="pos_"+words[i]
                    ret[key]=1
            else:
                if tmp_inv_cnt%2 != 0 :
                    key="pos_"+words[i]
                    ret[key]=-1
                    #pos_score+=tmp_score
                else:
                    key="neg_"+words[i]
                    ret[key]=1
                    #neg_score+=tmp_score
                
                     
                 
         elif flag[i][0] == "inverse":
             sss = sss + "-" +words[i] + "[" + flag[i][0] + "]"
             inv_cont = inv_cont + 1
             key="inv_"+words[i]
             ret[key]=1
         elif flag[i][0] == "level":
             lev_cont+=1
             sss = sss + "-" + words[i] + "[" + flag[i][0] + "]"
             key="level_"+words[i]
             ret[key]=1
         else:
             #high info word
             if hinfo_dict.has_key(words[i]):
                 key="mid_"+words[i]
                 ret[key]=1
                 
    

    ##print sss
    print len(ret.keys())
    return ret
   



def get_text_feat(text):
    sp=',|，|。|\?|！|~|；|;|\n'
   
    texts=re.split(sp, text)
    
    ret={}
    
    
    for line in texts:
        w=line.strip()
        if w =="" :
            pass
            #print "null -->"+line
        else:
            t=get_line_feat(w)
            #print ("%s===>%d %f %f "%(t[3],t[0],t[1],t[2])) 
            ret.update(t)
            
            
            
    return ret





def best_word_feats2(text,stop_word_dict,sentiment_dict,hinfo_dict):
    return get_text_feat(text)





def best_word_feats(text,stop_word_dict,sentiment_dict,hinfo_dict):
    
    
    biaodian={u'，':1,u'。':1,u'；':1,u'\n':1,u';':1,u',':1,u'《':1,u'》':1,u'“':1,u'”':1,u' ':1,
              u'’':1,u'‘':1,u'、':1}
    
    #,u'.':1,u'...':1,u'"':1,u'/':1,u'（':1,u'）':1,u'(':1,u')':1
    
    #print biaodian.keys()
    
    word_dict_total={}
    
  
    words=jieba.cut(text)
    pattern=re.compile(r'\w{1,}')
    
    tmp_words=[]
    for w in words:
        if biaodian.has_key(w)==True:
            continue
        elif pattern.match(w)!=None:
            continue
        else:
            #print w
            tmp_words.append(w)
    words=tmp_words
    
    
    #,score_fn=BigramAssocMeasures.likelihood_ratio, n=90000
    """
    bigram_dict=bigram(words,score_fn=BigramAssocMeasures.chi_sq, n=500)
    word_dict_total.update(bigram_dict)   
    
    trigram_dict=trigram(words,score_fn=TrigramAssocMeasures.chi_sq, n=500)
    word_dict_total.update(trigram_dict)   
    """
    
#===========================
    bigram_dict=bigram(words,score_fn=BigramAssocMeasures.chi_sq, n=5000,freq=1)
    trigram_dict={}#trigram(words,score_fn=TrigramAssocMeasures.chi_sq, n=5000,freq=1)
    
    words+=bigram_dict;
    #words+=trigram_dict;
    
    

    res={}
            
    for w in words:
        #w=w.strip()
        
                
        if stop_word_dict.has_key(w):
            #print w
            #res.append((w,True))
            print "---"+w
            continue
        
        elif hinfo_dict.has_key(w):
            #print w
            res[w]=1
            continue
        else:
            #print w
            #res.append((w,True))
            print "---"+w
            continue
        
            
    word_dict=res
    
            
    word_dict_total.update(word_dict)
     
    """
    for k in word_dict_total.keys():
        print "[%s]->%d"%(k,word_dict_total[k])
    raw_input("==================================")
    """
    
    global max_len
    global min_len
    global sum
    global cnt
    
    
    tmp_len=len(word_dict_total.keys())
    
    print "feature length=%d"%(tmp_len)
    sum+=tmp_len
    cnt+=1
    
    if max_len<tmp_len:
        max_len=tmp_len
        
    if min_len>tmp_len:
        min_len=tmp_len
    
    
    return word_dict_total


    
def evaluate_model(target_true,target_predicted):
    print classification_report(target_true,target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))
    
    
def learn_model(data,target):
    # preparing data for split validation. 60% training, 40% test
    state=43#randrange(1,23432)+123
    print "statue 6857"
    print state
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.20,random_state=state)
    #classifier = BernoulliNB().fit(data_train,target_train)
    stop_word_dict={}#build_stop_word_dict()
    sentiment_dict={}#build_sentiment_dict()
    global hinfo_dict
    hinfo_dict=build_hinfo_dict(data,target)
    
        
        
    #print stop_word_dict.keys()
    raw_input("begin train")
    train_feature=[]
    test_feature=[]
    for i in range(len(data_train)):
        print i
        d=data_train[i]
        #d=jieba.cut(d, cut_all=False)
        l=target_train[i]
        tmp=[best_word_feats(d,stop_word_dict,sentiment_dict,hinfo_dict),l]
        train_feature.append(tmp)
        
    for i in range(len(data_test)):
        print i
        d=data_test[i]
        #d=jieba.cut(d, cut_all=False)
        l=target_test[i]
        tmp=best_word_feats(d,stop_word_dict,sentiment_dict,hinfo_dict)
        test_feature.append(tmp)
    
    #BernoulliNB MultinomialNB LogisticRegression  SVC LinearSVC
    print "max_len %d"%(max_len)
    print "min_len %d"%(min_len)
    
    print "avg_len %d"%(sum/cnt)
    
    print "BernoulliNB"
    classifier = SklearnClassifier(BernoulliNB())
    classifier.train(train_feature)
    print "--------------"
    print len(classifier._vectorizer.get_feature_names())
    
    for f in classifier._vectorizer.get_feature_names():
        print f.encode("utf-8")
    
    predicted = classifier.classify_many(test_feature)
    evaluate_model(target_test,predicted)
    
    
    ids=range(len(data_test))
    result=[]
    for p in predicted:
        if p =='positive':
            result.append('1')
        else:
            result.append('-1')
        
    save_predict(data_test, ids, result, "BernoulliNB.xml")
    
    
    """
    print "LogisticRegression"
    classifier = SklearnClassifier(LogisticRegression())
    classifier.train(train_feature)
    print "--------------"
    print len(classifier._vectorizer.get_feature_names())

    predicted = classifier.classify_many(test_feature)
    evaluate_model(target_test,predicted)
    
    
    print "MultinomialNB"
    classifier = SklearnClassifier(MultinomialNB())
    classifier.train(train_feature)
    print "--------------"
    print len(classifier._vectorizer.get_feature_names())

    predicted = classifier.classify_many(test_feature)
    evaluate_model(target_test,predicted)
    
    
    print "LinearSVC"
    classifier = SklearnClassifier(LinearSVC())
    classifier.train(train_feature)
    print "--------------"
    print len(classifier._vectorizer.get_feature_names())

    predicted = classifier.classify_many(test_feature)
    evaluate_model(target_test,predicted)
    """
    
    
def main():
    data,target = load_file()
    
    learn_model(data,target)    
    
main()

cur.close()
conn.close()




