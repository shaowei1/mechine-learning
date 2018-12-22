class TowPoints(object):
    """
    * @ Description: 查找某个单词是否在字典里出现
    * @ param dictionary - 排序后的字典, wordToFind - 待查的单词
    * @return boolean - 是否发现待查的单词
    """

    def search(self, dictionary, wordToFind):
        if (dictionary == ''):
            return False

        if (len(dictionary) == 0):
            return False

        left = 0
        right = len(dictionary) - 1

        while (left <= right):
            middle = (left + right) // 2

            if dictionary[middle] == wordToFind:
                return True
            else:
                if dictionary[middle] > wordToFind:
                    right = middle - 1
                else:
                    left = middle + 1
        return False


if __name__ == '__main__':
    dictionary = ["i", "am", "one", "of", "the", "authors", "in", "geekbang"]
    wordToFind = 'i'

    tp = TowPoints()
    res = tp.search(dictionary, wordToFind)
    if res:
        print(" 找到了单词 %s", wordToFind)
    else:
        print(" 未能找到单词 %s", wordToFind)
