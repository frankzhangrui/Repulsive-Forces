f=open("cpu_output","r");
cpu=[]
for line in f:
	cpu.append(float(line))
g=open("gpu_output","r");
gpu=[]
for line in g:
	gpu.append(float(line))
result=[abs(gpu[i]-cpu[i]) for i in range(len(cpu))]
print sum(result)
	
