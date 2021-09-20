all: sampleCode

ARCH=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86

sampleCode: sampleCode.o remesh.o
	nvcc $(ARCH) -O3 -o $@ sampleCode.o remesh.o -ltbb



sampleCode.o: sampleCode.cu
	nvcc -c $(ARCH) -O3 $< -o $@ -I . 

remesh.o: remesh.cu
	nvcc -c $(ARCH) -O3 $< -o $@ -I . 



