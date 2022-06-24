all: Uni_SMall Slc_SMall

Uni_SMall: API.cu API.cuh exe.cu 
	nvcc exe.cu API.cu -rdc=true -Wno-deprecated-declarations -Xptxas -O3 -lcuda -lcudart --gpu-architecture=sm_61 -o Uni_SMall

Slc_SMall: API.cu API.cuh sliced.cu 
	nvcc sliced.cu API.cu -rdc=true -Wno-deprecated-declarations -Xptxas -O3 -lcuda -lcudart --gpu-architecture=sm_61 -o Slc_SMall

clean:
	rm -f  Uni_SMall Slc_SMall