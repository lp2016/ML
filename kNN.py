# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:06:41 2017

@author: Administrator
"""

from numpy import *
import matplotlib 
import matplotlib.pyplot as plt

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat =tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies =distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=None,reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat =zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnMat,classLabelVector
        
def drawing():
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingDataLabels),15.0*array(datingDataLabels))

#归一化数值
def autoNorm(dataSet):
    minVals =dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio=0.1                                #测试集数据占总数据的比例
    datingDataMat,datingDataLabels=file2matrix('E:\machine learning\KNN\datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult =classify0(normMat[i,:],normMat[numTestVecs:m,:],datingDataLabels[numTestVecs:m],3)
        print ("the classifier came back with :{},the real answer is : {}".format(classifierResult,datingDataLabels[i]))
        if classifierResult !=datingDataLabels[i]:
            errorCount+=1
    print("the total error rate is:{}".format(errorCount/float(numTestVecs)))
    
if __name__ =="__main__":
   datingClassTest()