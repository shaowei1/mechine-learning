import threading
from contextlib import contextmanager

_local = threading.local()

@contextmanager
def acquire(*locks):
	locks = sorted(locks, key=lambda x: id(x))

	acquired = getattr(_local, 'acquired', [])
	if acquired and max(id(lock) for lock in acquired) >= id(locks[0]):
		raise RuntimeError("Lock Order violation")

	acquired.extend(locks)
	_local.acquired = acquired

	try:
		for lock in locks:
			lock.acquire()
		yield
	finally:
		for lock in reversed(locks):
			lock.release()
		del acquired[-len(locks):]

# The philosopher thread
def philosopher(left, right):
	while True:
		with acquire(left, right):
			print(threading.currentThread(), 'eatting')

# The chopsticks (represented by locks)			
NSTICKS = 5
chopsticks = [threading.Lock() for n in range(NSTICKS)]

# Create all of the philosophers
for n in range(NSTICKS):
	t = threading.Thread(target=philosopher, 
						args=(chopsticks[n], chopsticks[(n + 1) % NSTICKS]))
	t.start()