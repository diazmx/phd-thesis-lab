from time import time
def test():
	for i in range(100):
		print("Hello")

start_time = time()
test()
elapsed_time = time() - start_time
print("Elapsed time: %.10f seconds." % elapsed_time)
