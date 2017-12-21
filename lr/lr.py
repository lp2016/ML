from numpy import *

def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open(r'F:\machinelearning\ML\lr\dataset.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def GradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights.getA()         #getA()  matrix --> ndarray

def stoGradAscent(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    dataMatrix=array(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i] * weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights

def stoGradAscent2(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    dataMatrix=array(dataMatrix)
    weights=ones(n)
    for j in range(150):
        dataIndex = list(range(m))
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights

def stoGradAscent3(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    dataMatrix=array(dataMatrix)
    weights=ones(n)
    alpha = 0.001
    d=array([0.0,0.0,0.0])
    y=[ array([0.0,0.0,0.0])for i in range(m) ]
    for j in range(300):
        randIndex = int(random.uniform(0, m))
        h = sigmoid(sum(dataMatrix[randIndex] * weights))
        error = classLabels[randIndex] - h
        d=d-error*dataMatrix[randIndex]
        y[randIndex]=error*dataMatrix[randIndex]
        weights=weights+alpha*d/m
        print(weights)
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i]==1):
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
dataArr,labelMat=loadDataSet()
weights2=stoGradAscent3(dataArr,labelMat)
plotBestFit(weights2)
