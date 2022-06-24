

all: Uni_SMptr Slc_SMptr 

Uni_SMptr: API.cu API.cuh exe.cu 
	nvcc exe.cu API.cu -rdc=true -Wno-deprecated-declarations -Xptxas -O3 -lcuda -lcudart --gpu-architecture=sm_61 --maxrregcount 63 -o Uni_SMptr

Slc_SMptr: API.cu API.cuh sliced.cu 
	nvcc sliced.cu API.cu -rdc=true -Wno-deprecated-declarations -Xptxas -O3 -lcuda -lcudart --gpu-architecture=sm_61 --maxrregcount 63 -o Slc_SMptr

clean:
	rm -f  Slc_SMptr Slc_SMptr
