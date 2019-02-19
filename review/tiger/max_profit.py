def maxProfit(prices):
    if len(prices) <= 1:
        return 0
    low_price = prices[0]
    max_pro = 0
    for i in prices:
        low_price = min(low_price, i)
        max_pro = max(max_pro, i - low_price)
    return max_pro


def max_retreat(prices):
    if len(prices) <= 1:
        return 0
    max_price = prices[0]
    max_loss = 0
    for i in prices[1:]:
        max_price = max(i, max_price)
        max_loss = max(max_loss, max_price - i)
    return max_loss


def multiple(prices):
    max_pro = 0
    if not prices:
        return max_pro
    save = prices[0]
    for i in prices[1:]:
        if i > save:
            max_pro += i - save
        save = i
    return max_pro


print(maxProfit([2, 3, 4, 1, 10, 8, 1, 11, 4]))
print(maxProfit([9, 8, 7, 6, 5]))
print(max_retreat([2, 3, 4, 1, 10, 8, 1, 11, 4]))
print(max_retreat([9, 8, 7, 6, 5]))
print(multiple([2, 3, 4, 1, 10, 8, 1, 11, 4]))
print(multiple([9, 8, 7, 6, 5]))
