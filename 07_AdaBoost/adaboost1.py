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
    :param threshIneq:
    :return:
    """
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == '1t':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minError = inf # init error sum, to +infinity

    for i in range(n):
        # loop ove all dimensions(维度)
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            # loop over all range in current dimension
            for inequal in ['lt', 'gt']:
                # go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # call stump classify(分类) with i, j , lessThan
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr # calc total error multiplied by D 计算加权错误率
                # print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

def