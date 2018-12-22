import matplotlib.pyplot as plt# 定义文本框和箭头类型decisionNode = dict(boxstyle="sawtooth", fc="0.8")leafNode = dict(boxstyle="round4", fc="0.8")arrow_args = dict(arrowstyle="<-")def createPlot():    fig = plt.figure(1, facecolor='white')    fig.clf() # clear screen    createPlot.ax1 = plt.subplot(111, frameon=False)    plotNode("决策节点", (0.5, 0.1), (0.1, 0.5), decisionNode)    plotNode("叶节点", (0.8, 0.1), (0.3, 0.8), leafNode)    plt.show()def plotNode(nodeTxt, centerPt, parentPt, nodeType):    # 绘制带箭头的注解    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',                            xytext=centerPt, textcoords='axes fraction',                            va="center", bbox=nodeType, arrowprops=arrow_args)def getNumleafs(myTree):    """    遍历整棵树，累计叶子节点的个数，并返回    :param myTree:    :return:    """    numLeafs = 0    firstStr = list(myTree.keys())[0]    secondDict = myTree[firstStr]    for key in secondDict.keys():        # 测试节点的数据类型时候为字典        if type(secondDict[key]).__name__ == 'dict':            numLeafs += getNumleafs(secondDict[key])        else:            numLeafs += 1    return numLeafsdef getTreeDepth(myTree):    """    计算便利过程中遇到判断计算树深度的变量加１    :param myTree:    :return:    """    maxDepth = 0    firstStr = list(myTree.keys())[0]    secondDict = myTree[firstStr]    for key in secondDict.keys():        if type(secondDict[key]).__name__ == 'dict':            thisDepth = 1 + getTreeDepth(secondDict[key])        else:            thisDepth = 1        if thisDepth > maxDepth:            maxDepth = thisDepth    return maxDepthdef retrieveTree(i):    """    :i: 预先存储的树信息    :return:    """    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},                   {'no surfacing':                        {0: 'no',                         1:                             {'flippers':                                  {0:                                       {'head':                                            {0: 'no',                                             1: 'yes'}},                                   1: 'no'}}}}]    return listOfTrees[i]def plotMidText(cntrPt, parentPt, txtString):    """    在父子节间填充文本信息, 计算子节点和父节点的中间位置    :return:    """    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]    createPlot.ax1.text(xMid, yMid, txtString)def plotTree(myTree, parentPt, nodeTxt):    """    计算宽高    :param myTree:    :param parentPt:    :param nodeTxt:    :return:    """    numleafs = getNumleafs(myTree)  # width    depth = getTreeDepth(myTree)  # height    firstStr = list(myTree.keys())[0]    # x 属于 [0.0, 1.0], y 属于 [0.0, 1.0]    cntrPt = (plotTree.xOff + (1.0 + float(numleafs)) / 2.0 / plotTree.totalW, plotTree.yOff)    # 标记子节点属性    plotMidText(cntrPt, parentPt, nodeTxt)    plotNode(firstStr, cntrPt, parentPt, decisionNode)    # 减少y偏移量    secondDict = myTree[firstStr]    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD    for key in secondDict.keys():        # 如果是叶子节点，则在图形上画出叶子节点，否则就地柜调用plotTree        if type(secondDict[key]).__name__ == 'dict':            plotTree(secondDict[key], cntrPt, str(key))        else:            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),                     cntrPt, leafNode)            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))    # 绘制了所有子节点后，增加全局变量y的偏移    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalDdef createPlot(inTree):    # 自顶向下绘制图形    fig = plt.figure(1, facecolor='white')    fig.clf()    axprops = dict(xticks=[], yticks=[])    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    plotTree.totalW = float(getNumleafs(inTree))    plotTree.totalD = float(getTreeDepth(inTree))    plotTree.xOff = -0.5 / plotTree.totalW    plotTree.yOff = 1.0    plotTree(inTree, (0.5, 1.0), '')    plt.show()if __name__ == '__main__':    myTree = retrieveTree(1)    createPlot(myTree)