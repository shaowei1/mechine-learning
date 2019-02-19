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
    Logistic回归梯度上升优化算法
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
    随机梯度上升:
        一次仅用一个样本点来更新回归系数，是在线学习算法，相对应的一次处理所有数据被叫做批处理
        回归系数初始化为1
        对数据集中每个样本
            计算该样本的梯度
            使用alpha * gradient更新回归系数
        返回回归系数

    随机梯度上升算法，和随机梯度上升算法相似,随机梯度里面h, error是数值,没有矩阵转换过程，所有数据类型都是Numpy数组
    h, error都是向量，
    :param dataMatrix: 分类器的输入数据
    :param classLabels:
    :return:
    错了1/3的样本, 因为gradAscent0是在整个数据集上迭代了500次才得到的，
    一个判断优化算法优劣的可靠方法是看它时候收敛,也就是说参数时候达到了稳定值

    回归系数经过大量迭代才能到达稳定值
    在大的波动过后，还有一些小的周期性波动，
    产生这种现象的原因是存在一些不能正确分类的样本点(数据集并非线性可分),
    在每次迭代时会发生系数的剧烈波动．我们期待算法能避免来回波动，从而收敛到某个值
    """
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)  # initialize to all ones
    for i in range(m):
        # Signoid函数的输入记为 z = w0x0 + w1x1 + w2x2 +...+wnxn = w(T)x
        h = sigmoid(sum(dataMatrix[i] * weights))  # dataMatrix[i]一行三列 weights三行一列
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]  # 最佳参数系数
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    随机梯度下降更新参数w, 不仅要考虑当前的梯度值， 还要考虑上一次的参数更新

    计算回归系数向量
    如果完全收敛结果将确定

    我们仅仅对数据集进行了20次遍历，而之前是500次

    x0 x1 x2 (迭代次数)
    只经过50次迭代就达到了稳定值,
    :param dataMatrix:
    :param classLabels:
    :param numIter:
    :return:

    改进: 还可以增加一个迭代次数作为第三个参数，默认150次
    """
    m, n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # alpha每次迭代都进行调整, 会缓解stocGradAscent0的数据波动否则高频波动,
            # 会随着alpha迭代，无限接近于0, 但永远不会减小到0,因为有一个参数项,

            # 必须是这样做的原因是为了保证在多次迭代之后新数据依然有影响

            # 如果处理的问题是动态变化的，可以适当增大参数项，来确保新的值获得更大的回归系数.
            # 每次alpha每次减少1 / (j + i), j是迭代常数，i是样本点的下标,
            # 这样当 j << max(i), alpha就不是严格下降的,避免参数的严格下降也是常见于模拟退火算法等其他优化算法中
            alpha = 4 / (1.0 + j + i) + 0.0001  # apha decreases with iteration, does not
            # 通过随机样本来更新回归系数．减少周期性波动,　每次随机从列表欧总去除一个值,然后从列表中删掉该值

            randIndex = int(random.uniform(0, len(dataIndex)))  # go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    """

    :param inX: 特征向量
    :param weights: 回归系数
    :return: 计算 1 if signoid > 0.5 else 0
    """
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    """
    除了部分指标主观和难于预测外，该数据还存在一个问题，数据集中有30%的值是缺失的.
    # proble: 如果机器上某个传感器损坏了一个特征，怎么办
    # resolve:
        1. 使用可用特征的均值来填补缺失值
        2. 使用特殊值来填补缺失值, 如-1
        3. 忽略有缺失值的样本
        4. 使用相似样本的句子添补缺失值
        5. 使用另外的机器学习算法预测缺失值

    我们采用0替代缺失值，因为我们需要一个在更新时不会影响系数的值
    weights = weights + alpha * error * dataMatrix[randIndex] = weights
    sigmooid(0) = 0.5

    预处理2, 有一条数据中类别标签已经缺失，那么丢弃，　如果采用knn就不太可行

    :return:
    """
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
    """
    调用 colicTest()　10次求平均值
    :return:
    """
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))

if __name__ == '__main__':
    dataArr, LabelMat = loadDataSet()
    weights = stocGradAscent1(array(dataArr), LabelMat)
    plotBestFit(weights)
    # multiTest()
    # colicTest()
    """
    梯度下降: 300个样本28个特征(缺失30%数据)   the error rate of this test is: 0.298507, 怎么改结果都不变
    
    """
