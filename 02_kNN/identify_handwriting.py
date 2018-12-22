"""算法需要为每个向量做2000次距离计算，每个距离计算包括了1024个维度浮点数运算，总共要执行900次，每个整型占2个字节，为此时向量准备2MB存储空间,"""from numpy import zeros, arrayfrom os import listdirfrom whether_like import classify0def img2vector(filename):    """    为了使用前面的分类器，将图像格式化为一个向量,    """    returnVect = zeros((1, 1024))    with open(filename) as fr:        for i in range(32):            lineStr = fr.readline()            for j in range(32):                returnVect[0, 32 * i + j] = int(lineStr[j])        return returnVectdef handwritingClass():    hwLabels = []    # 获取目录内容    trainingFileList = listdir('trainingDigits')    m = len(trainingFileList)    trainingMat = zeros((m, 1024))    for i in range(m):        # 从文件解析分类数字        fileNameStr = trainingFileList[i]        fileStr = fileNameStr.split('.')[0]        classNumStr = int(fileStr.split('_')[0])        hwLabels.append(classNumStr)        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)    testFileList = listdir('testDigits')    errorCount = 0.0    mTest = len(testFileList)    for i in range(mTest):        # 从文件解析分类数字        fileNameStr = testFileList[i]        fileStr = fileNameStr.split('.')[0]        classNumStr = int(fileStr.split('_')[0])        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)        print("the classifies came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))        if (classifierResult != classNumStr):            errorCount += 1    print("\nthe total number of error is: %d" % errorCount)    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))if __name__ == '__main__':    handwritingClass()