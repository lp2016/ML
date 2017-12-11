# treePlotter.py
# coding = utf-8

import matplotlib.pyplot as plt
from pylab import mpl

# matplotlib添加中文字体，注意每个要显示的汉字串前要添加U
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False  # 由于更改了字体导致显示不出负号，则将axes设置为false，默认的是true

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):  # 绘制带箭头的注解
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', \
                            xytext=centerPt, textcoords='axes fraction', \
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])# 定义横纵坐标轴，无内容
    #createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) # 绘制图像,无边框,无坐标轴
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotTree.totalW = float(getNumLeafs(inTree))   #全局变量宽度 = 叶子数
    plotTree.totalD = float(getTreeDepth(inTree))  #全局变量高度 = 深度
    #图形的大小是0-1 ，0-1
    plotTree.xOff = -0.5/plotTree.totalW;  #例如绘制3个叶子结点，坐标应为1/3,2/3,3/3
    #但这样会使整个图形偏右因此初始的，将x值向左移一点。
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]  # myTree.keys是dict_key类型，不支持索引取值，需要转换成list
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":  # 测试节点的数据类型是否为字典
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):  # 获取树的层数
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":  # 测试节点的数据类型是否为字典
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):  # 预定义树的结构
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': \
                                                      {0: 'no', 1: 'yes'}}, 3: 'maybe'}},
                   {'no surfacing': {0: 'no', 1: {'flippers': \
                                                      {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):  # 在父子节点间填充文本信息
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)  # 计算宽与高
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD