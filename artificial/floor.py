# floor.py

# 上n级台阶总共集中上法?


def n_count(n):
	if n == 1:
		return 1
	if n == 2:
		return 2
	return n_count(n - 1) + n_count(n - 2)

import math
import sys

n = int(sys.argv[1])	
an = (1 / math.sqrt(5)) * (((1 + math.sqrt(5)) / 2 ) ** n - ((1 - math.sqrt(5)) / 2) ** n)
print(int(an))
# print(n_count(5))
