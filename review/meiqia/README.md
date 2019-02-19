# 叙述

## 顺序表和链表的区别及应用场景

线性表可以分为:顺序表和链表. 其中, 顺序表又可分为动态和静态两种, 链表可分为单向链表,单向循环链表, 双向链表, 双向循环链表等.

- 区别
  - 顺序表可以实现下标的快速访问, 单链表则不可以, 单链表必须从头开始依次遍历查找.
  - 顺序表在中间或者头部插入节点时, 必须依次挪动节点到下一个节点的位置, 然而单链表却不用, 单链表插入, 删除节点比较方便
  - 顺序表每次增容固定大小的空间有可能造成空间浪费, 但不用每次插入时都都填开辟空间; 单链表插入节点时都必须动态开辟空间, 删除节点时必须释放掉动态开辟的空间
  - 由于计算机设计的多级缓存遵循局部性原理, 所有连续访问顺序表时缓存命中率较高, 而单链表本身存储比较分散, 连续访问时缓存命中率较低还会造成缓存污染.
- 应用场景
  - 顺序表: 尾插尾删较多使用
  - 单链表: 头部或中间插入较多
- 单链表题目

```
//1.从尾到头打印单链表
//2.删除一个无头单链表的非尾节点
//3.在无头单链表的一个节点前插入一个节点
//4.单链表实现约瑟夫环
//5.逆置 / 反转单链表
//6.单链表排序（冒泡排序&快速排序）
//7.合并两个有序链表, 合并后依然有序
//8.查找单链表的中间节点，要求只能遍历一次链表
//9.查找单链表的倒数第k个节点，要求只能遍历一次链表

https://blog.csdn.net/Mr_______zhang/article/details/73649707 
```

## 线程死锁是如何造成的, 如何避免

- 当线程A持有独占锁a, 并尝试去获取独占锁b的同时, 线程B持有独占锁b, 并尝试获取独占锁a的情况下, 就会发生AB两个线程由于互相持有对方所需要的锁, 而发生阻塞的现象, 我们成为死锁.

```python
#coding=utf-8
import time
import threading
class Account:
	def __init__(self, _id, balance, lock):
		self.id = _id
		self.balance = balance
		self.lock = lock

	def withdraw(self, amount):
		self.balance -= amount

	def deposit(self, amount):
		self.balance += amount

def transfer(_from, to, amount):
	if _from.lock.acquire():
		_from.withdraw(amount)
		time.sleep(1) # 让交易时间变长, 2个交易线程时间上重叠, 有足够的事件来产生死锁
		print('wait for lock')
		if to.lock.acquire(): # 锁住对方的账户
			to.deposit(amount)
			to.lock.release()
		_from.lock.release()
	print('finish')		

a = Account('a', 1000, threading.Lock())
b = Account('b', 1000, threading.Lock())

th1 = threading.Thread(target = transfer, args = (a, b, 100))
th2 = threading.Thread(target = transfer, args = (b, a, 100))
th1.start()
th2.start()
```

- 避免死锁

  哲学家就餐: 避免死锁就是破坏造成死锁的, 若干条件中的一个

  - 服务生解法: 引入一个餐厅服务生，哲学家必须经过他的允许才能拿起餐叉。因为服务生知道哪只餐叉正在使用，所以他能够作出判断避免死锁。
    为了演示这种解法，假设哲学家依次标号为A至E。如果A和C在吃东西，则有四只餐叉在使用中
    
  - 资源分级解法: 为资源（这里是餐叉）分配一个偏序或者分级的关系，并约定所有资源都按照这种顺序获取，按相反顺序释放，而且保证不会有两个无关资源同时被同一项工作所需要。

  - Chandy/Misra解法:


  1984年，K. Mani Chandy和J. Misra提出了哲学家就餐问题的另一个解法，允许任意的用户（编号P1, ..., Pn）争用任意数量的资源。与資源分級解法不同的是，这里编号可以是任意的。

    - 把餐叉湊成對，讓要吃的人先吃，沒餐叉的人得到一張換餐叉券。
    - 餓了，把換餐叉券交給有餐叉的人，有餐叉的人吃飽了會把餐叉交給有券的人。有了券的人不會再得到第二張券。
    - 保證有餐叉的都有得吃。
      这个解法允许很大的并行性，适用于任意大的问题。

- 造成死锁4个条件

  - 互斥条件: 一个资源每次只能被一个线程使用
  - 请求与保持条件: 一个线程因为请求资源而阻塞时, 已获得的资源保持不放
  - 不剥夺条件: 线程已获得的资源, 在未使用完之前, 不能强行剥夺
  - 循环等待条件: 若干线程之间形成一种头尾相接的循环等待资源关系

- 简述

  - 互斥条件 --> 独占锁的特点之一
  - 请求与保持条件 --> 独占锁的特点之一, 尝试获取锁时并不会释放已经持有的锁
  - 不剥夺条件 --> 独占锁的特点之一
  - 循环等待条件 --> 唯一需要记忆的造成死锁的条件

- 在并发程序中, 避免了逻辑中出现复数个线程互相持有对方线程所需要的独占锁的情况, 就可以避免死锁.

- 避免死锁的原则：

  \1. 一定要以一个固定的顺序来取得锁，这个列子中，意味着首先要取得alock, 然后再去block

  \2. 一定要按照与取得锁相反的顺序释放锁，这里，应该先释放block,然后是alock 

- 在多线程程序中, 死锁问题很大一部分是由于线程同时获取多个锁造成的.举一个例子: 一个线程获取了第一个锁, 然后在获取第二个锁的时候发生阻塞, 那么这个线程就可能阻塞其他线程的执行, 从而导致整个程序假死. 解决死锁问题的一种方案是为每一个锁分配一个唯一id, 然后只允许按照升序规则来使用多个锁, 这个规则使用上下文管理器很容易实现

```python
# coding=utf-8
from contextlib import contextmanager
import time


class Account:
    def __init__(self, _id, balance, lock):
        self.id = _id
        self.balance = balance
        self.lock = lock

    def withdraw(self, amount):
        self.balance -= amount

    def deposit(self, amount):
        self.balance += amount


import threading
from contextlib import contextmanager

# Thread-local state to stored information on locks already acquired
_local = threading.local()


@contextmanager
def acquire(*locks):
    # Sort locks by object identifier
    locks = sorted(locks, key=lambda x: id(x))

    # Make sure lock order of previously acquired locks is not violated
    acquired = getattr(_local, 'acquired', [])
    print(acquired)
    print(locks)
    if acquired and max(id(lock) for lock in acquired) >= id(locks[0]):
        raise RuntimeError('Lock Order Violation')

    # Acquire all of the locks
    acquired.extend(locks)
    _local.acquired = acquired
    print(_local.acquired)

    try:
        for lock in locks:
            lock.acquire()
            print("start")
        yield
    finally:
        # Release locks in reverse order of acquisition
        print("stop")
        for lock in reversed(locks):
            lock.release()
        del acquired[-len(locks):]


def transfer(_from, to, amount):
    with acquire(_from.lock, to.lock):
        _from.withdraw(amount)
        time.sleep(1)  # 让交易时间变长, 2个交易线程时间上重叠, 有足够的事件来产生死锁
        print('{} wait for lock'.format(_from.id))
        to.deposit(amount)
        print('{} finish'.format(_from.id))


a = Account('a', 1000, threading.Lock())
b = Account('b', 1000, threading.Lock())

th1 = threading.Thread(target=transfer, args=(a, b, 100))
th1.daemon = True
th1.start()

th2 = threading.Thread(target=transfer, args=(b, a, 200))
th2.daemon = True
th2.start()

time.sleep(3)
print(a.balance)
print(b.balance)

"""
[]
[<unlocked _thread.lock object at 0x7f9e90b7cdc8>, <unlocked _thread.lock object at 0x7f9e90b7ce40>]
[<unlocked _thread.lock object at 0x7f9e90b7cdc8>, <unlocked _thread.lock object at 0x7f9e90b7ce40>]
start
start
[]
[<locked _thread.lock object at 0x7f9e90b7cdc8>, <locked _thread.lock object at 0x7f9e90b7ce40>]
[<locked _thread.lock object at 0x7f9e90b7cdc8>, <locked _thread.lock object at 0x7f9e90b7ce40>]
a wait for lock
a finish
stop
start
start
b wait for lock
b finish
stop
1100
900
"""
```

[哲学家就餐wiki](!https://link.jianshu.com/?t=https://zh.wikipedia.org/wiki/%E5%93%B2%E5%AD%A6%E5%AE%B6%E5%B0%B1%E9%A4%90%E9%97%AE%E9%A2%98)

```python
import threading
from contextlib import contextmanager

# Thread-local state to stored information on locks already acquired
_local = threading.local()

@contextmanager
def acquire(*locks):
    # Sort locks by object identifier
    locks = sorted(locks, key=lambda x: id(x))

    # Make sure lock order of previously acquired locks is not violated
    acquired = getattr(_local,'acquired',[])
    if acquired and max(id(lock) for lock in acquired) >= id(locks[0]):
        raise RuntimeError('Lock Order Violation')

    # Acquire all of the locks
    acquired.extend(locks)
    _local.acquired = acquired


    try:
        for lock in locks:
            lock.acquire()
        yield
    finally:
        # Release locks in reverse order of acquisition
        for lock in reversed(locks):
            lock.release()
        del acquired[-len(locks):]

import threading
x_lock = threading.Lock()
y_lock = threading.Lock()

def thread_1():
    while True:
        with acquire(x_lock, y_lock):
            print('Thread-1')

def thread_2():
    while True:
        with acquire(y_lock, x_lock):
            print('Thread-2')

t1 = threading.Thread(target=thread_1)
t1.daemon = True
t1.start()

t2 = threading.Thread(target=thread_2)
t2.daemon = True
t2.start()

import time
while 1:
    time.sleep(10)
```

