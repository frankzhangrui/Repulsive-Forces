# target: dependencies
#  [tab] system command

all:
	nvcc test.cu

clean:
	rm -f *.o test

