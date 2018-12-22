'''
Created on Oct 22, 2018
Logistic Regression Working Module
@author: shaowei

logistic回归的一般过程
1. 收集数据: 采用任意方法收集数据
2. 准备数据: 由于需要进行距离计算，因此要求数据你类型为数值型．另外结构化数据类型最佳
3. 分析数据: 采用任意方法对数据进行分析
4. 训练算法: 大部分时间用于训练，训练的目的是为了找到最佳的分类回归系数
5. 测试算法: 一旦训练步骤完成，分类将会很快
6. 使用算法: 首先，我们输入一些数据,并将其转换成对应的结构化数字
'''
from numpy import *


def loadDataSet():
    """
    dataMat: 1.0, x1, x2
    labelMat: 分类标签
    :return:
    """
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    # exp()    方法返回x的指数, ex。
    return 1.0 / (1 + exp(-inX))


def plotBestFit(weights):
    """
    画出数据集和Logistic回归最佳拟合直线的函数
    :param weights: 训练好的回归系数
    :return:
    """
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # 设置最佳拟合直线, 0是两个分类(类别1和类别0)的分界处，因此，我们设定0 = w0x0 + w1x1+w2x2,然后解出x2和x1的关系式(即分割线的方程，注意x0 = 1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def gradAscent(dataMatIn, classLabels):
    """
    训练回归系数
    :param dataMatIn: 样本数据
    :param classLabels: 标签向量
    :return: weigths 训练好的回归系数
    """
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix
    # 行变列
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix
    m, n = shape(dataMatrix)
    alpha = 0.001  # 向目标移动的步长
    maxCycles = 500  # 迭代次数
    weights = ones((n, 1))
    for k in range(maxCycles):  # heavy on matrix operations
        # 100 * 3 and 3 * 1
        # h是一个列向量, 列向量的元素个数等于样本个数, 这里是100
        h = sigmoid(dataMatrix * weights)  # matrix mult, 包含300次乘积
        error = (labelMat - h)  # vector subtraction，　计算真实类型和预测类型的差值,按照差值的方向调整回归系数
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
    # weigths 训练好的回归系数
    return weights


def stocGradAscent0(dataMatrix, classLabels):
    """
    随机梯度上升算法，和随机梯度上升算法相似,随机梯度里面h, error是数值,没有矩阵转换过程，所有数据类型都是Numpy数组
    h, error都是向量，
    :param dataMatrix:
    :param classLabels:
    :return:
    错了1/3的样本, 因为gradAscent0是在整个数据集上迭代了500次才得到的，一个判断优化算法优劣的可靠方法是看它时候收敛,也就是说参数时候达到了稳定值

    回归系数经过大量迭代才能到达稳定值
    在大的波动过后，还有一些小的周期性波动，产生这种现象的原因是存在一些不能正确分类的样本点(数据集并非线性可分),在每次迭代时会发生系数的剧烈波动．我们期待算法能避免来回波动，从而收敛到某个值
    """
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)  # initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    x0 x1 x2 (迭代次数)
    只经过50次迭代就达到了稳定值,
    :param dataMatrix:
    :param classLabels:
    :param numIter:
    :return:
    """
    m, n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            # alpha每次迭代都进行调整, 会缓解stocGradAscent0的数据波动否则高频波动,会随着alpha迭代，无限接近于0
            alpha = 4 / (1.0 + j + i) + 0.0001  # apha decreases with iteration, does not
            # 随机选择更新值
            randIndex = int(random.uniform(0, len(dataIndex)))  # go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
    dataArr, LabelMat = loadDataSet()
    weights = stocGradAscent0(array(dataArr), LabelMat)
    plotBestFit(weights)
