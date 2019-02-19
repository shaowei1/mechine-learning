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