'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *
import feedparser


def loadDataSet():
    """
    postingList: 词条切分后的文档集合
    classVec: 类别标签的集合
    :return:
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    """
    创建一个包含所有文档中出现的不重复词的列表
    :param dataSet:
    :return:
    """
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    词集模型 set-of-words model
    使用词汇表或者想要检查的所有单词作为输入，然后为其中每一个单词构建一个特征．
    :param vocabList: 词汇
    :param inputSet: 文档
    :return: 文档向量，向量的每一个元素为1 或 0, 表示词汇表中单词在输入文档中是否出现
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    计算每个类别中的文档数目
    对每训练文档:
        对每个类别:
            如果词条出现在文档中--> 增加该词条的计数值
            增加所有词条的计数值
    对每个类别:
        对每个词条:
            将该词条的数目处于总词条数目得到条件概率
    返回每个类别的条件概率

    :param trainMatrix:　文档矩阵
    :param trainCategory: 由每篇文档类别标签构成的向量
    :return: 两个向量和一个概率
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 1. 初始化概率频率
    p0Num = ones(numWords)  # 分母变量: 是一个元素个数等于词汇表大小的NumPy数组，
    p1Num = ones(numWords)  # change to ones()
    p0Denom = 2.0
    p1Denom = 2.0  # change to 2.0
    for i in range(numTrainDocs):
        # 2.　向量相加
        if trainCategory[i] == 1:
            # 计算侮辱性文档的概率, 两类问题
            p1Num += trainMatrix[i]  # 侮辱性词数目
            p1Denom += sum(trainMatrix[i])  # 文档的总次数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 3. 对每个元素做除法
    # 通过求对数可以便面下溢出或者浮点数舍入导致的错误
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect, pAbusive


# if __name__ == '__main__':
#     listOPosts, listClasses = loadDataSet()
#     myVocabList = createVocabList(listOPosts)
#     trainMat = []
#     for postinDoc in listOPosts:
#         trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
#     print(listClasses)
#     print(listOPosts)
#     print(trainMat)
#     p0V, p1V, pAb = trainNB0(trainMat, listClasses)
#     print(p0V)
#     print(p1V)
#     print(pAb)  # 属于侮辱类的概率


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def bagOfWords2VecMN(vocabList, inputSet):
    """
    词袋模型    bag-of-words model
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


def textParse(bigString):  # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


import codecs


def spamTest():
    docList = []
    classList = []
    fullText = []

    for i in range(1, 26):
        wordList = textParse(codecs.open('email/spam/%d.txt' % i, "r", encoding="latin-1", errors='ignore').read())
        # wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(codecs.open('email/ham/%d.txt' % i, "r", encoding="latin-1", errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = list(range(50))
    testSet = []  # create test set

    for i in range(10):
        # 随机算着10封邮件为测试集，剩余部分作为测试集的过程称之为留存交叉验证(hold-out cross validation)
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    trainMat = []
    trainClasses = []

    # setofWords2Vec()用来构建词向量，
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    # trainNB0用于计算分类所需的概率
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))
    # return vocabList,fullText


def calcMostFreq(vocabList, fullText):
    """
    :return:　返回排序最高的30个单词
    """
    import operator
    freqDict = {}
    for token in vocabList:
        #     计算出现频率
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))

    for i in range(minLen):
        # 每次访问一条RSS源
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    # 去除出现次数最高的那些词(3行代码，注释去看效果)
    # 不仅仅移除高频词，同时从某个预定词表中移除结构中的辅助词．该词表称为停用词表(stop word list),
    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])

    trainingSet = list(range(2 * minLen))
    testSet = []  # create test set
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


if __name__ == '__main__':
    # testingNB()
    # 垃圾邮件
    # spamTest()

    # caclMostFreq()改变要移除的单词数目，然后观察错误率的变化情况
    ny = feedparser.parse('https://github.com/')
    sf = feedparser.parse('https://en.wikivoyage.org/wiki/Main_Page')
    vocabList, pSF, pNY = localWords(ny, sf)
    print(vocabList)
    print(pSF)
    print(pNY)
