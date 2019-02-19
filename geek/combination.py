"""
使用function 递归/嵌套调用，找出所有可能队伍组合
目前还剩多少队伍没有参加组合, result- 保存当前已经组合的队伍
"""

"""
combine: 从n个不同元素中去除m个(1 <=m<=n)个不同的元素，　

在自然语言处理中，我们需要用多元文法把临近几个单词合并起来，组合成一个新的词组．
普通的多元法定死了每个元组类单词出现的顺序．
但实时上多个单词出现时，我们可以不关心他们的顺序，而只关心他们的组合．
这样我们就可以对多元组类的单词进行某种形式的标准化．
即使原来的单词出现顺序有所不同，经过这个标准化之后，都会变成唯一的顺序.
"""


def combine(teams, result, m):
    # 挑选忘了m个元素，　output result
    if len(result) == m:
        print(result)
        return

    for i in range(len(teams)):
        # 从剩下的队伍中选择一队，　加入结果
        newResult = result.copy()
        newResult.append(teams[i])

        rest_teams = teams[i + 1: len(teams)]  # 组合不关心元素排列的顺序　

        combine(rest_teams, newResult, m)

    if __name__ == '__main__':
        teams = ["t1", "t2", "t3"]
        combine(teams, list(), 2)


def luck_draw():
    """
    抽奖系统，需要依次从１００个人中，抽取３等奖１０名，　二等奖３名和一等奖１名．请列出所有可能组合，　没有只能被抽中一次
    :return:
    """
    all = 100
    the_first_price = list()
    second_award = list()
    third_award = list()


comb = ['t1', 't2', 't3', 't4', 't5']
import copy


def combination(n, comb, result):
    if n == 0:
        print(result)
        return
    for i in comb:
        newResult = copy.copy(result)
        newComb = copy.copy(comb)
        newResult.append(i)
        newComb = list(set(newComb).difference(set(comb[:comb.index(i) + 1])))
        combination(n - 1, newComb, newResult)


# combination(4, comb, [])

