from numpy import *


def loadSimapData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    通过阈值比较对数据进行分类
    所有在阈值一边的分到类别-1, 另一半的+1
    # 数组过滤实现


	 将最小错误率minError设为+∞
	 对数据集中的每一个特征(每一层循环):
	    对每个步长(第二层循环):
	        对每个不等号(第三层循环):
	            建立一棵单层决策树并利用加权平均数据集对它进行测试
	            如果错误率低于minError,则将当前单层决策树设为最佳决策树
    返回最佳决策树

    :param dataMatrix:
    :param dimen:
    :param threshVal:
    :param threshIneq: 不等于号
    :return:
    """
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    遍历stumpClassify()函数所有的可能输入值, 并找到数据集上最佳的单层决策树,这里的最佳是基于数据的权重向量D来定义的

    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    """
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0  # 用于在特征的所有可能值上进行遍历
    bestStump = {}  # 存储给定权重向量D时所得到的最佳单层决策树的相关信息,
    bestClassEst = mat(zeros((m, 1)))
    minError = inf  # init error sum, to +infinity # 一开始就初始化程无穷大,之后用于寻找可能的最小错误率
    count = 0
    for i in range(n):
        # loop ove all dimensions(维度) 在数据集上所有特征遍历
        rangeMin = dataMatrix[:, i].min()  # 考虑到数值型的特征,可以通过最大值,最小值来了解需要步长
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps

        for j in range(-1, int(numSteps) + 1):
            # 再在这些值上遍历.甚至将阈值设置为整个取值范围之外也是可以的
            # loop over all range in current dimension

            for inequal in ['lt', 'gt']:
                # go over less than and greater than # 在大于和小于之间切换不等式
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # call stump classify(分类) with i, j , lessThan
                errArr = mat(ones((m, 1)))  # 错误向量

                errArr[predictedVals == labelMat] = 0
                # 若果predictedVals中的值不等于labelMat中的真正类别标签值,那么errArr相应位置置为1

                # 权重向量D
                # 加权错误率 weightedError# 这就是AdaBoost和分类器交互的地方, 基于权重向量D来评价很猎奇
                weightedError = D.T * errArr  # calc total error multiplied by D 计算加权错误率
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                    i, threshVal, inequal, weightedError))
                count += 1
                if weightedError < minError:
                    # 当前错误率和已有的最小错误率进行比较,如果当前最小,那么就在吃点bestStump中保存该单层决策树.字典/错误率/类别估计都返回给AdaBoost
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    print(count)
    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # init D to all equal
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # build Stump
        print("D: ", D.T)
        alpha = float(0.5 * log(
            (1.0 - error) / max(error, 1e-16)))  # calc alpha, throw in max(error, eps) to account for error = 0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # store Stump Params in Array
        print("classEst: ", classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # exponent for D calc, getting messy
        D = multiply(D, exp(expon))
        D = D / D.sum()

        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(dataToClass, classifierArr):
    dataMatrix = mat(dataToClass)  # do stauff imilar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, not classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])  # call stump classify
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    """

    :param predStrengths: 分类器的预测强度
    :param classLabels:
    :return:
    """
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)  # cursor
    ySum = 0.0  # variable to calculate AUC
    numPosClas = sum(array(classLabels) == 1.0)  # calc positive
    yStep = 1 / float(numPosClas)  # step nums
    xStep = 1 / float(len(classLabels) - numPosClas)  # [0.0, 1.0]
    sortedIndicies = predStrengths.argsort()  # get sorted index , it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        # draw line from cur to (cur[0 - delX, cur[1] - delY)
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum * xStep)



if __name__ == '__main__':
    D = mat(ones((5, 1)) / 5)
    datMat, classLabels = loadSimapData()
    # bestStump, minError, bestClassEst = buildStump(datMat, classLabels, D)
    # print(bestStump, "\n", minError, "\n", bestClassEst, )
    print(datMat)
    print(classLabels)
    weakClassArr, aggClassEst = adaBoostTrainDS(datMat, classLabels, 9)
    print(weakClassArr)
    print(aggClassEst)
