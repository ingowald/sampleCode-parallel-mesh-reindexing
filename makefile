all: sampleCode

#benchmark (in bash): for f in 8 64 1024 4096 8192; do ./sampleCode $f > result.$f.out ; done

ARCH=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86
CXXFLAGS=${ARCH} -g -O3 -DOWL_HAVE_TBB=1


sampleCode: sampleCode.o remesh.o
	nvcc $(CXXFLAGS) -o $@ sampleCode.o remesh.o -ltbb

sampleCode.o: sampleCode.cu
	nvcc -c $(CXXFLAGS) $< -o $@ -I . 

remesh.o: remesh.cu
	nvcc -c $(CXXFLAGS) $< -o $@ -I . 



