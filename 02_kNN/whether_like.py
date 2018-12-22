"""1. 计算已知类别数据集中的每个点依次执行以下操作2. 按照距离递增持续怕徐3. 选取与当前点距离最小的k个点4. 确定前k个点所在类别的出现频率5. 返回k个点出现频率最高的类别作为当前点的预测分类# 在约会网站上使用k-近邻算法(1) 收集数据(2) 准备数据: 解析文本文件(3) 分析数据: 使用Matplotlib画二维扩散图(4) 训练算法: 此步骤不适合k-近邻算法(5) 测试算法: 使用海伦提供的部分样本作为测试样本    测试样本和非测试样本的区别在于: 测试样本是已经完成分类的数据，如果测试分类与实际类别不同，则标记一个错误(6) 使用算法"""from numpy import *import operatordef createDataSet():    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])    labels = ['A', 'A', 'B', 'B']    return group, labelsdef classify0(inX, dataSet, labels, k):    """    res = classify0([0, 0], group, labels, 3)    :param inX: 分类的输入向量    :param dataSet: 训练样本集    :param labels: 标签向量 (标签向量的元素数目和矩阵dataSet的行数相同)    :param k: 选择最近邻居的数目    :return:    """    # 距离计算    dataSetSize = dataSet.shape[0]    diffMat = tile(inX, (dataSetSize, 1)) - dataSet    sqDiffMat = diffMat ** 2    sqDistances = sqDiffMat.sum(axis=1)    distances = sqDistances ** 0.5    sortedDistIndicies = distances.argsort()    # 选择距离最小的k个点    classCount = dict()    for i in range(k):        voteIlabel = labels[sortedDistIndicies[i]]        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1    # 排序    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)    return sortedClassCount[0][0]def file2matrix(filename):    fr = open(filename)    arraryOLines = fr.readlines()    # 得到文件行数    numberOfLines = len(arraryOLines)    # 创建返回矩阵    returnMat = zeros((numberOfLines, 3))    classLabelVector = []    index = 0    # 解析文件数据到列表    for line in arraryOLines:        line = line.strip('\n')        listFromLine = line.split('\t')        returnMat[index, :] = listFromLine[0: 3]        classLabelVector.append(int(listFromLine[-1]))        index += 1    return returnMat, classLabelVectordef autoNorm(dataSet):    """    归一化特征值    :param dataSet:    :return:    """    # 所有行中最小/大的    minVals = dataSet.min(0)    maxVals = dataSet.max(0)    ranges = maxVals - minVals    normDataSet = zeros(shape(dataSet))    m = dataSet.shape[0]    normDataSet = dataSet - tile(minVals, (m, 1))    # 特征值相处    normDataSet = normDataSet / tile(ranges, (m, 1))    return normDataSet, ranges, minValsdef datingClassTest():    hoRatio = 0.10    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')    normMat, ranges, minVals = autoNorm(datingDataMat)    m = normMat.shape[0]    numTestVecs = int(m * hoRatio)    errorCount = 0.0    for i in range(numTestVecs):        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],                                     datingLabels[numTestVecs:m], 7)        print("the classifier came back with: %d, the real answer is: %d" %              (classifierResult, datingLabels[i]))        if (classifierResult != datingLabels[i]):            errorCount += 1.0    print("the error number is: %f" % errorCount)    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))def classifyPerson():    resultList = ['not at all', 'in small doses', 'in large doses']    percentTats = float(input("percentage of time spent playing video games?"))    ffMiles = float(input("frequent flier miles earned per year?"))    iceCream = float(input("liters of ice cream consumed per year?"))    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')    normMat, ranges, minVals = autoNorm(datingDataMat)    inArr = array([ffMiles, percentTats, iceCream])    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)    print("you will probably like this person: ", resultList[classifierResult - 1])if __name__ == '__main__':    # classifyPerson()    datingClassTest()"""datingTestSet2.txt每年获得的飞行产科里程数玩视频游戏所耗时间百分比每周消费冰激凌公升数""""""训练样本矩阵类标签向量from numpy import *import operatorimport matplotlibfrom matplotlib.pylab import pltfig = plt.figure()ax = fig.add_subplot(111)# add_subplot(1,1,1)# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])# A scatter plot of *y* vs *x* with varying marker size and/or color# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))plt.show()"""