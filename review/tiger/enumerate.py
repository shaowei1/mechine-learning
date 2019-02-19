def _enumerate(item):
	k = -1
	for i in item:
		k += 1
		yield k, i

items = ["a", "b", "c", "d"]

for i, j in _enumerate(items):
	print(i, j)		