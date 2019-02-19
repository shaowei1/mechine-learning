import random


def conflict(state, nextx):
    """
    定义冲突函数，state为元组， nextx为下一个queen的水平位置
    """
    nexty = len(state)
    for i in range(nexty):
        if abs(state[i] - nextx) in (0, nexty - i):
            # 如果下一个皇后在同列，或者一条对角线，冲突
            return True
    return False


def queens(num=8, state=()):
    """
    num表示规模
    """
    for pos in range(num):
        if not conflict(state, pos):  # 位置不冲突
            if len(state) == num - 1:  # 最后一个皇后， 返回position
                yield (pos,)
            else:  # 将位置返回给state元组， 并传递给下一个queen
                for result in queens(num, state + (pos,)):
                    yield (pos,) + result


def prettyp(solution):
    def line(pos, length=len(solution)):
        return 'O' * (pos) + 'X' + 'O' * (length - pos - 1)

    for pos in solution:
        print(line(pos))


prettyp(random.choice(list(queens(8))))
