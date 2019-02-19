"""
merge

while 
li = [] # 会创建新对象
li[:] = [] # 没有创建新对象，还是原来的地址
li　+= left[l_index:]
O(nlogn) # logn表示次数
bubble select insert      没有递归	分组
shell	insert	quick_sort
"""

"""
1. 若root 为None，　则直接挂到root上
2. 若root is not None, 根节点入队
3. 出队并判断是否有l, r child, if not 则挂tree
4. 依次入队l, r child　重复step3
"""
class Node(object):
	def __init__(self, item):
		self.item = item
		self.lchild = None
		self.rchild = None

class Tree():
	def __init__(self):
		self.root = None

	def add(self, item):
		node = Node(item)

		if self.root is None:
			self.root = node
			return
		que = []

		que.append(self.root)
		while len(que) > 0:
			cur = que.pop(0)	
			if cur.lchild is None:
				cur.lchild = node
				return
			elif cur.rchild is None:
				cur.rchild = node
				return
			else:
				que.append(cur.lchild)
				que.append(cur.rchild)

	def travel_tree(self):
		"""
		１．　根节点入队
		２．　出队打印，　依次入队左右孩子
		３．　重复step2, 直到队列空
		"""
		if self.root is None:
			return
		que = []
		que.append(self.root)

		while len(que) > 0:
			cur = que.pop(0)

			print(cur.item, end="--")
			if cur.lchild is not None:
				que.append(cur.lchild)
			if cur.rchild is not None:
				que.append(cur.rchild)

	def a(self):
		# (根左右) 先序遍历root --> lchild tree --> (下一层)lchild tree --> (上一层)rchild tree --> 
		# (左根右)：　压扁了读
		# (左右根)

if __name__=="__main__":
	tr  = Tree()
	tr.add(0)
	tr.add(1)
	tr.add(2)
	tr.add(3)
	tr.add(4)
	tr.add(5)
	tr.add(6)
	tr.add(7)
	tr.add(8)
	tr.add(9)
	tr.travel_tree()

"""
＃　用字典存树和树结构　优势劣势
	有

# 怎么通过.查找的
	怎么访问属性，属性就记录了一个偏移值

# 记录路径，　什么一层一层往下找
	记录路径不值得，因为树节点发生改变，路径就都需要换了

＃　顺序和非顺序存储
	顺序存储可以直接访问

# 二分查找 		应用场景
	任意查找，　比如对象可以采用某个属性来排序

树里面一般会存储序号:
	序号对应的位置存储变量
"""	
# 公共祖先

"""
ｋｍｐ算法

暴力匹配：　每次都需要比较模式串前面的每个字母，没有必要
找出 prefix table （找出公共前后缀）
ababc

-1
0	 a
0	ab
1	aba
2	abab
# 这段不要0	ababc

a　b　a　b　c
-1 0 0  1  2

"""

