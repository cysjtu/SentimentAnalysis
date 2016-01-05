# encoding=utf-8
'''
Created on 2015年12月12日

@author: nali
'''
import xml.etree.ElementTree as ET
import sys 


"""
tree = ET.ElementTree(file='sample-processed.txt')

for s in tree.getroot():
    #print s.attrib['id']
    #s.attrib['polar']=-1
    print s.text.encode('utf-8')
    s.set('polar','1')
    
    
    
tree.write(file_or_filename='sample-output.txt', encoding='utf-8')
""" 

print sys.getdefaultencoding()
print unicode("你好", 'utf-8').encode('unicode-escape').decode('unicode-escape')
print u"你好".encode('utf-8')

    
