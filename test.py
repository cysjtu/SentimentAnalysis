# encoding=utf-8
'''
Created on 2015年12月10日

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

from xml.dom import minidom

import MySQLdb 

import re 

import random



conn=MySQLdb.connect(host="127.0.0.1",user="root",passwd="cy1993",port=3306,db="lexicon",charset='utf8')

cur=conn.cursor()

#query="insert into lex(word,polar) values(%s,%s)"
#str="看了这本书，总体感觉老外带孩子比中国人要粗些。其实，本来就不用太过细致了，人家那样照样能把孩子带好，不能把孩子放在保险箱里养。这书挺好的，可以看看"
#cuts=pseg.cut(str)


    
def getPolar(word):
    query2="select polar from sentiment where word = %s"
    
    cur.execute(query2,(word))
    
    polar=cur.fetchone()
    
    if polar is not None:
        #print polar
        return polar[0]
    else :
        return -1
    
    
    
def getLevel(word):
    query2="select * from level where word = %s"
    
    cur.execute(query2,(word))
    
    res=cur.fetchone()
    
    if res is not None:
        #print float(res[3])
        return (res[2],float(res[3]) )
    else :
        return (-1,-1000.0)

def getSimilar(word):
    
    query2 = "select * from similar where word = %s"
     
    query3 = "select * from similar where class = %s"
     
    cur.execute(query2, (word))
     
    tmp = cur.fetchone()
     
    data = []
     
    if tmp is not None:
         word_class = tmp[2]
         #print word_class
         
         cur.execute(query3, (word_class))
         
         sim = cur.fetchall()
         
         if sim is not None:
             for s in sim:
                 data.append(s[1])
                 
                 
    return data




#判断一个词语是不是否定词
def isInverseWord(word):
    query2="select * from inverse where word = %s"
    
    cur.execute(query2,(word))
    
    res=cur.fetchone()
    
    if res is not None:
        #print polar
        return 1
    else :
        return -1



def isstopWord(word):
    query2="select * from stop where word = %s"
    
    cur.execute(query2,(word))
    
    res=cur.fetchone()
    
    if res is not None:
        #print polar
        return 1
    else :
        return -1






##===========================================================


def insert_stop(word):
    query="insert into stop(word) values(%s)"
    try:
        cur.execute(query,(word ) )
        conn.commit()
        
    except:
        print "insert fail-->"+word
        
    
#加载停用词
def load_stop():
    file_name="dict/stopword.txt" 
    file=open(file_name,"r")
    lines=file.readlines()
    
    for l in lines:
        insert_stop(l.strip())
        
    


     
def insert_similar(word,word_class):
    query3="insert into similar(word,class) values(%s,%s)"
    
    try:  
        cur.execute(query3,(word,word_class))
        conn.commit()
        
    except:
        print "insert fail"

#加载同义词
def load_similar():
    file_name="dict/similarWord.txt"
    file=open(file_name,"r")
    lines=file.readlines()
    
    for l in lines:
        words=l.split(' ')
        word_class=words[0]
        for i in range(1,len(words)):
            w=words[i]
            print w
            insert_similar(w.strip(),word_class.strip())
            
    print "load_similar finish"
    
    
    

def insert_level(word,word_class,score):
    query4="insert into level(word,class,score) values(%s,%s,%s)"
    
    try:  
        cur.execute(query4,(word,word_class,score))
        conn.commit()
        
    except:
        print "insert fail"

        
def load_level_sub(file_name,word_class,score):
    
    file=open(file_name,"r")
    lines=file.readlines()
    
    for l in lines:
        w=l.strip()
        print w 
        print word_class
        print score
        insert_level(w,word_class,score)
            
    print file_name+" load_level finish"
    file.close()
    
    
    
#加载程度副词   
def load_level():
    

    most="dict/most.txt"
    most_score=6.5
    
    load_level_sub(most,1,most_score)
    
    very="dict/very.txt"
    very_score=4.5
    
    load_level_sub(very,2,very_score)
    
    more="dict/more.txt"
    more_score=2.5
    load_level_sub(more,3,more_score)
    
    ish="dict/ish.txt"
    ish_score=1.5
    load_level_sub(ish,4,ish_score)
    
    
    insufficiently="dict/insufficiently.txt"
    
    insufficiently_score=0.75
    load_level_sub(insufficiently,5,insufficiently_score)
    
    
    #over="dict/over.txt"
    #over_score=0.05
    #load_level_sub(over,6,over_score)
    
    
    
    
    
    
    
def insert_inverse(word):
    query="insert into inverse(word) values(%s)"
    try:
        cur.execute(query,(word ) )
        conn.commit()
        
    except:
        print "insert fail-->"+word
        
    
#加载否定词
def load_inverse():
    file_name="dict/inverse.txt" 
    file=open(file_name,"r")
    lines=file.readlines()
    
    for l in lines:
        insert_inverse(l.strip())
        

        
        
           
    
    
    
    
    


   
def insert_sentiment(word,polar):
    query="insert into sentiment(word,polar) values(%s,%s)"
    
    try: 
        cur.execute(query,(word,polar))
        conn.commit()
    except:
        print "insert fail->>"+word
        
        

         
    
    
# 0代表中性，1代表褒义，2代表贬义，3代表兼有褒贬两性。
def load_sentiment():
    neg_file = "dict/negtive.txt"
    pos_file = "dict/postive.txt"
    
    fileneg = open(neg_file, "r")
    
    lines_neg = fileneg.readlines()
    
    for l in lines_neg:
        w = l.strip()
        if w != "" and w !=" ":
            insert_sentiment(w, 2)
        
        # ws=getSimilar(w)
        # for wss in ws:
        #    insert_sentiment(wss,2)
            
    ########################
    filepos = open(pos_file, "r")
    
    lines_pos = filepos.readlines()
    
    for l in lines_pos:
        w = l.strip()
        if w != "" and w !=" ":
            insert_sentiment(w, 1)
        # ws=getSimilar(w)
        # for wss in ws:
        #    insert_sentiment(wss,1)
         
         
"""         
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    raw_input("sssssssss")
     #======insert similar ==================    
         
         
         
    for l in lines_neg:
        w=l.strip()
        ws=getSimilar(w)
        for wss in ws:
            insert_sentiment(wss,2)
            
    ########################
    
    
    for l in lines_pos:
        w=l.strip()
        ws=getSimilar(w)
        for wss in ws:
            insert_sentiment(wss,1)
       
    
"""
        
        
        
        
        










def judgePolar(text):
    
    tokens=pseg.cut(text)
    
    words=[]
    flag=[]
    
    polars=["","postive","negtive"]
    
    levels=["","most","more","very","little","insufficiently"]
    sss=""
    
    for w,pos in tokens:
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
              
            
    pos_score=0.0
    neg_score=0.0
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
            
            
            if flag[i][1]==polars[1]:
                if tmp_inv_cnt%2 != 0 :
                    neg_score+=tmp_score
                else:
                    pos_score+=tmp_score
            else:
                if tmp_inv_cnt%2 != 0 :
                    pos_score+=tmp_score
                else:
                    neg_score+=tmp_score
                
                     
                 
         elif flag[i][0] == "inverse":
             sss = sss + "-" +words[i] + "[" + flag[i][0] + "]"
             inv_cont = inv_cont + 1
         elif flag[i][0] == "level":
             lev_cont+=1
             sss = sss + "-" + words[i] + "[" + flag[i][0] + "]"
         else:
            sss = sss + "-" + words[i]
    
            
                 
            
             
        
    
    #print ("pos_score=%f"%(pos_score))
    #print ("neg_score=%f"%(neg_score))
    #print sss
    pola=0
    #sum=pos_score+neg_score
    if pos_score<=0.000000001 and neg_score<=0.00000001:
        if inv_cont%2!=0:
            pola=2
            return (pola,0.0,0.5,sss)
        elif inv_cont!=0:
            pola=1
            return (pola,0.5,0.0,sss)
        else:
            if lev_cont !=0:
                return (1,0.2,0.0,sss)
            else:
                return (0,0.0,0.0,sss)
            

    elif pos_score < neg_score:
        pola=2
    elif pos_score > neg_score:
        pola=1
    else:
        if inv_cont%2!=0:
            pola=2
            return (pola,pos_score,neg_score,sss)
        elif inv_cont!=0:
            pola=1
            return (pola,pos_score,neg_score,sss)
        else:
            if lev_cont !=0:
                return (1,pos_score+0.25,neg_score,sss)
            else:
                return (0,pos_score,neg_score,sss)
        
    #print ("polar=%s"%(polars[pola]))
        
    
    return (pola,pos_score,neg_score,sss)
   





def process_text(text):
    sp=',|，|。|\?|！|~|；|;|\n'
   
    texts=re.split(sp, text)
    
    ret=[]
    
    for line in texts:
        w=line.strip()
        if w =="" :
            pass
            #print "null -->"+line
        else:
            t=judgePolar(w)
            print ("%s===>%d %f %f "%(t[3],t[0],t[1],t[2])) 
            ret.append(t)
            
            
            
    return ret




            
def calculate_score(data):
    pos=0.0
    neg=0.0
    cnt=len(data)
    pos_cnt=0
    neg_cnt=0
    for k in data:
        if k[0] !=0:
            pos+=k[1]
            neg+=k[2]
            if k[0]==1:
                pos_cnt+=1
            elif k[0]==2:
                neg_cnt+=1
    
        
    
    
    if pos>neg:
        return 1
    elif pos<neg:
        return -1
    else:
        if pos_cnt>neg_cnt:
            return 1
        elif pos_cnt<neg_cnt:
            return -1
        else:
            return random.randint(0,1)-1
        






    
def allJudge(data):

    ret=[]
    cnt=0
    for text in data:
        print ("%d================================================="%cnt)
        
        arr=process_text(text)
        result=calculate_score(arr)
        
        if result ==1:
            if cnt >4999:
                print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>error----------%d"%cnt) 
            ret.append("positive")
        else:
            if cnt <=4999:
                print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>error----------%d"%cnt) 
            ret.append("negative")
            
        cnt+=1
    return ret

#@@@@@@@@@@@@@@@@@@
#load_similar()
load_level()
load_inverse()
load_sentiment()
load_stop()
#@@@@@@@@@@@@@@@@@@@@







            
        
    
