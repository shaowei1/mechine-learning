#!/usr/bin/env python
# -*- coding:utf8 -*-

class Node(object):
    def __init__(self, item):
        self.item = item
        self.lchild = None
        self.rchild = None


class Tree(object):
    def __init__(self):
        self.root = None

    def add(self, item):
        node = Node(item)
        # 1 若树为空，挂到根节点
        if self.root is None:
            self.root = node
            return
        # 2 根节点入队
        que = []
        que.append(self.root)

        while len(que) > 0:
            # 3 出队依次判断是否有左右孩子，没有则挂树
            cur = que.pop(0)
            if cur.lchild is None:
                cur.lchild = node
                return
            elif cur.rchild is None:
                cur.rchild = node
                return
            else:
                # 4 依次入队左右孩子，重复步骤3
                que.append(cur.lchild)
                que.append(cur.rchild)

    def breath_travel(self):
        if self.root is None:
            return
        # 1 根节点入队
        que = []
        que.append(self.root)

        while len(que) > 0:
            # 2 出队打印，依次入队左右孩子
            cur = que.pop(0)
            print(cur.item, end="--")
            if cur.lchild is not None:
                que.append(cur.lchild)

            if cur.rchild is not None:
                que.append(cur.rchild)
            # 3 重复步骤2 直到队列空

        print()

    def preorder(self, root):
        if root is None:
            return

        print(root.item, end="--")
        self.preorder(root.lchild)
        self.preorder(root.rchild)

    def inorder(self, root):
        if root is None:
            return

        self.inorder(root.lchild)
        print(root.item, end="--")
        self.inorder(root.rchild)

    def postorder(self, root):
        if root is None:
            return

        self.postorder(root.lchild)
        self.postorder(root.rchild)
        print(root.item, end="--")

    def search(self, root, item):
        if root is None:
            return None

        if root.item == item:
            return root

        lres = self.search(root.lchild, item)
        if lres is not None:
            return lres
        rres = self.search(root.rchild, item)
        if rres is not None:
            return rres

        return None

    def pub_dad(self, root, nodeA, nodeB):
        if root is None:
            return None

        if root == nodeA or root == nodeB:
            return root

        ldad = self.pub_dad(root.lchild, nodeA, nodeB)
        rdad = self.pub_dad(root.rchild, nodeA, nodeB)

        if ldad is None:
            return rdad
        elif rdad is None:
            return ldad
        else:
            return root


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


class Solution:
    """
       @param: root: The root of the binary search tree.
       @param: A: A TreeNode in a Binary.
       @param: B: A TreeNode in a Binary.
       @return: Return the least common ancestor(LCA) of the two nodes.
       """

    def lowestCommonAncestor(self, root, A, B):
        # A&B=>LCA
        # !A&!B=>None
        # A&!B=>A
        # B&!A=>B
        if (root is None or root == A or root == B):
            return root  # 若root为空或者root为A或者root为B，说明找到了A和B其中一个
        left = self.lowestCommonAncestor(root.left, A, B)
        right = self.lowestCommonAncestor(root.right, A, B)
        if (left is not None and right is not None):
            return root  # 若左子树找到了A，右子树找到了B，说明此时的root就是公共祖先
        if (left is None):  # 若左子树是none右子树不是，说明右子树找到了A或B
            return right
        if (right is None):  # 同理
            return left
        return None

    def run(self):
        a = Tree = TreeNode(2)
        b = Tree.left = TreeNode(1)
        c = Tree.right = TreeNode(3)
        d = b.left = TreeNode(4)
        s = Solution()
        print(s.lowestCommonAncestor(a, b, d).val)


def graorder(root):
    if root is None:
        return ''
    queue = [root]
    while queue:
        res = []
        for i in queue:
            print(i.item)
            try:
                if i.left:
                    res.append(i.left)
                if i.right:
                    res.append(i.right)
            except:
                pass
        queue = res
    return res


if __name__ == '__main__':
    tr = Tree()
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
    print(graorder(tr.root))

    # tr.breath_travel()
    # tr.preorder(tr.root)
    # print()
    # tr.inorder(tr.root)
    # print()
    # tr.postorder(tr.root)
    # print()
    #
    # nodeA = tr.search(tr.root, 8)
    # nodeB = tr.search(tr.root, 9)
    #
    # node_dad = tr.pub_dad(tr.root, nodeA, nodeB)
    #
    # print(node_dad.item)

