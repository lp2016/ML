from numpy import *
import matplotlib.pyplot as plt
def loadDataSet(filename):
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append((float(lineArr[2])))
    return dataMat,labelMat

def showDataSet(dataMat, labelMat):
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) #负样本散点图
    plt.show()

def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    if L>aj:
        aj=L
    return aj

class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.eCache=mat(zeros((self.m,2)))
#格式化计算误差的函数，方便多次调用
def calcEk(oS,k):
    fXk=float(multiply(oS.alphas,oS.labelMat).T*\
        (oS.X*oS.X[k,:].T))+oS.b
    Ek=fXk-float(oS.labelMat[k])
    return Ek
#修改选择第二个变量alphaj的方法
def selectJ(i,oS,Ei):
    maxK=-1;maxDeltaE=-0;Ej=0
    #将误差矩阵每一行第一列置1，以此确定出误差不为0
    #的样本
    oS.eCache[i]=[1,Ei]
    #获取缓存中Ei不为0的样本对应的alpha列表
    validEcacheList=nonzero(oS.eCache[:,0].A)[0]
    #在误差不为0的列表中找出使abs(Ei-Ej)最大的alphaj
    if(len(validEcacheList)>0):
        for k in validEcacheList:
            if k ==i:continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if(deltaE>maxDeltaE):
                maxK=k;maxDeltaE=deltaE;Ej=Ek
        return maxK,Ej
    else:
    #否则，就从样本集中随机选取alphaj
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej
#更新误差矩阵
def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]


def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    #保存关键数据
    oS=optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter=0
    entireSet=True
    alphaPairsChanged=0
    #选取第一个变量alpha的三种情况，从间隔边界上选取或者整个数据集
    while(iter<maxIter)and((alphaPairsChanged>0)or(entireSet)):
        alphaPairsChanged=0
        #没有alpha更新对
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
                print("fullSet,iter: %d i:%d,pairs changed %d"%\
                (iter,i,alphaPairsChanged))
            iter+=1

        else:
            #统计alphas向量中满足0<alpha<C的alpha列表
            print(oS.alphas.A>0)
            print('...................')
            print(oS.alphas.A<C)
            print('fsdff')
            print((oS.alphas.A > 0)*(oS.alphas.A<C))
            nonBoundIs=nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print("non-bound,iter: %d i:%d,pairs changed %d"%\
                (iter,i,alphaPairsChanged))
            iter+=1
        if entireSet:entireSet=False
        #如果本次循环没有改变的alpha对，将entireSet置为true，
        #下个循环仍遍历数据集
        elif (alphaPairsChanged==0):entireSet=True
        print("iteration number: %d"%iter)
    return oS.b,oS.alphas

#内循环寻找alphaj
def innerL(i,oS):
    #计算误差
    Ei=calcEk(oS,i)
    #违背kkt条件
    if(((oS.labelMat[i]*Ei<-oS.tol)and(oS.alphas[i]<oS.C))or\
        ((oS.labelMat[i]*Ei>oS.tol)and(oS.alphas[i]>0))):
        j,Ej=selectJ(i,oS,Ei)
        alphaIold=oS.alphas[i].copy();alphaJold=oS.alphas[j].copy()
        #计算上下界
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H:print("L==H");return 0
        #计算两个alpha值
        eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-\
            oS.X[j,:]*oS.X[j,:].T
        if eta>=0:print("eta>=0");return 0
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        updateEk(oS,j)
        if(abs(oS.alphas[j]-alphaJold)<0.00001):
            print("j not moving enough");return 0
        oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]*\
            (alphaJold-oS.alphas[j])
        updateEk(oS,i)
        #在这两个alpha值情况下，计算对应的b值
        #注，非线性可分情况，将所有内积项替换为核函数K[i,j]
        b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
                    oS.X[i,:]*oS.X[i,:].T-\
                    oS.labelMat[j]*(oS.alphas[j]-alphaJold)*\
                    oS.X[i,:]*oS.X[j,:].T
        b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
                    oS.X[i,:]*oS.X[j,:].T-\
                    oS.labelMat[j]*(oS.alphas[j]-alphaJold)*\
                    oS.X[j,:]*oS.X[j,:].T
        if(0<oS.alphas[i])and (oS.C>oS.alphas[i]):oS.b=b1
        elif(0<oS.alphas[j])and (oS.C>oS.alphas[j]):oS.b=b2
        else:oS.b=(b1+b2)/2.0
        #如果有alpha对更新
        return 1
            #否则返回0
    else: return 0

def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = array(alphas), array(dataMat), array(labelMat)
    w = dot((tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()

def calcWs(alphas,dataArr,labelArr):
    X=mat(dataArr)
    labelMat=mat(labelArr).transpose()
    m,n=shape(X)
    w=zeros((n,1))
    for i in range(m):
        w+=multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def showClassifer(dataMat,labelMat, w, b):
    #绘制样本点
    data_plus = []                                  #正样本
    data_minus = []                                 #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = array(data_plus)              #转换为numpy矩阵
    data_minus_np = array(data_minus)            #转换为numpy矩阵
    plt.scatter(transpose(data_plus_np)[0], transpose(data_plus_np)[1], s=30, alpha=0.7)   #正样本散点图
    plt.scatter(transpose(data_minus_np)[0], transpose(data_minus_np)[1], s=30, alpha=0.7) #负样本散点图
    #绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()

if __name__ ==  '__main__':
    dataArr,labelArr=loadDataSet(r'F:\machinelearning\ML\svm\dataset.txt')
    # showDataSet(dataArr,labelArr)
    b,alphas=smoP(dataArr,labelArr,0.6,0.001,40)
    # w=calcWs(dataArr,labelArr,alphas)
    w=get_w(dataArr,labelArr,alphas)
    print(w)
    showClassifer(dataArr,labelArr,w,b)
    dataMat=mat(dataArr)
    print(dataMat[0]*mat(w)+b)
